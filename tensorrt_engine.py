import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

import time
import functools
from statistics import mean, median


def collect_stats(print_every=10, reset_after_print=False):
    def decorator(func):
        stats = {'calls': 0, 'times': [], 'errors': 0}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                stats['errors'] += 1
                raise
            finally:
                stats['times'].append(time.perf_counter() - start)
                stats['calls'] += 1
                
                if stats['calls'] % print_every == 0:
                    times = stats['times']
                    print(f"\n{func.__name__}() - Calls: {stats['calls']}, Errors: {stats['errors']}")
                    print(f"Times - Avg: {mean(times):.4f}s, Min: {min(times):.4f}s, Max: {max(times):.4f}s")
                    if reset_after_print:
                        stats['times'], stats['errors'] = [], 0
            
            return result
        
        wrapper.get_stats = lambda: stats
        return wrapper
    return decorator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRTLogger(trt.ILogger):
    """Custom TensorRT logger"""
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity <= trt.Logger.WARNING:
            print(f"TRT {severity}: {msg}")

class TRTEngineManager:
    def __init__(self, engine_file_path: str):
        """
        Initialize TensorRT Engine Manager
        
        Args:
            engine_file_path: Path to the serialized TensorRT engine file
        """
        self.engine_file_path = engine_file_path
        self.engine = None
        self.context = None
        self.runtime = None
        self.cuda_buffers = []
        self.host_inputs = []
        self.host_outputs = []
        self.bindings = []
        self.stream = None
        
    def __del__(self):
        """Cleanup CUDA resources"""
        self._cleanup()
    
    def _cleanup(self):
        """Free CUDA memory"""
        for buffer in self.cuda_buffers:
            if buffer:
                cuda.mem_free(buffer)
        self.cuda_buffers.clear()
        
        if self.stream:
            self.stream.synchronize()
    
    def initialize(self) -> bool:
        """
        Initialize the TensorRT engine and allocate memory
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load engine from file
            if not self._load_engine_from_file(self.engine_file_path):
                logger.error(f"Failed to load engine from file: {self.engine_file_path}")
                return False
            
            if not self.engine:
                logger.error("Engine is not initialized.")
                return False
            
            # Debug: Print binding information
            self._print_binding_info()
            
            # Create execution context
            logger.info("Creating execution context")
            self.context = self.engine.create_execution_context()
            if not self.context:
                logger.error("Failed to create execution context.")
                return False
            
            # Create CUDA stream
            self.stream = cuda.Stream()
            
            # Allocate CUDA buffers for inputs and outputs
            logger.info("Allocating CUDA buffers for inputs and outputs")
            self._allocate_buffers()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
    
    def _load_engine_from_file(self, file_path: str) -> bool:
        """
        Load TensorRT engine from serialized file
        
        Args:
            file_path: Path to the engine file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading engine file: {file_path}")
            
            # Read the serialized engine
            with open(file_path, 'rb') as f:
                engine_data = f.read()
            
            if not engine_data:
                logger.error(f"Failed to read engine file: {file_path}")
                return False
            
            logger.info(f"Engine file loaded successfully. Size: {len(engine_data)} bytes.")
            
            # Create TensorRT runtime
            logger.info("Creating Inference Runtime")
            trt_logger = TRTLogger()
            self.runtime = trt.Runtime(trt_logger)
            if not self.runtime:
                logger.error("Failed to create TensorRT runtime.")
                return False
            
            # Deserialize the engine
            logger.info("Creating CUDA Engine")
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if not self.engine:
                logger.error("Failed to deserialize CUDA engine.")
                return False
            
            logger.info(f"Engine {file_path} max batch size: {self.engine.max_batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading engine: {str(e)}")
            return False
    
    def _print_binding_info(self):
        """Print information about engine bindings for debugging"""
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            
            logger.info(f"Binding {i}: {binding_name}")
            logger.info(f"  Shape: {binding_shape}")
            logger.info(f"  Type: {binding_dtype}")
            logger.info(f"  Is Input: {is_input}")
    
    def _allocate_buffers(self):
        """Allocate CUDA memory for inputs and outputs"""
        self.cuda_buffers = []
        self.host_inputs = []
        self.host_outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = self.engine.get_binding_dtype(i)
            
            # Calculate size
            size = trt.volume(binding_shape)
            dtype = trt.nptype(binding_dtype)
            
            # Allocate host memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            
            # Allocate device memory
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to lists
            self.cuda_buffers.append(cuda_mem)
            self.bindings.append(int(cuda_mem))
            
            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem)
            else:
                self.host_outputs.append(host_mem)
    
    @collect_stats(print_every=20,reset_after_print=True)
    def do_inference(self, input_data: np.ndarray, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference using the TensorRT engine
        
        Args:
            input_data: Input data as numpy array with shape (1, 3, 224, 224)
            batch_size: Batch size (default: 1)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two output arrays
                - First output: shape (1, 1024)
                - Second output: shape (1, 1024, 14, 14)
        """
        try:
            # Ensure input data is the correct type and shape
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Validate input shape
            expected_input_shape = (1, 3, 224, 224)
            if input_data.shape != expected_input_shape:
                logger.error(f"Input shape mismatch. Expected: {expected_input_shape}, Got: {input_data.shape}")
                raise ValueError(f"Input must have shape {expected_input_shape}")
            
            # Copy input data to pagelocked memory
            np.copyto(self.host_inputs[0], input_data.ravel())
            
            # Transfer input data to GPU
            cuda.memcpy_htod_async(self.cuda_buffers[0], self.host_inputs[0], self.stream)
            
            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer output data from GPU (assuming we have 2 outputs)
            if len(self.host_outputs) < 2:
                logger.error(f"Expected 2 outputs, but found {len(self.host_outputs)}")
                raise ValueError("Model must have exactly 2 outputs")
            
            # Transfer both outputs
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_buffers[1], self.stream)
            cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_buffers[2], self.stream)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Reshape outputs to expected shapes
            output1_data = self.host_outputs[0].copy()
            output2_data = self.host_outputs[1].copy()
            
            # First output: (1, 1024)
            expected_output1_size = 1 * 1024
            if output1_data.size >= expected_output1_size:
                output1 = output1_data[:expected_output1_size].reshape((1, 1024))
            else:
                logger.error(f"Output 1 size mismatch. Expected: {expected_output1_size}, Got: {output1_data.size}")
                output1 = np.zeros((1, 1024), dtype=np.float32)
            
            # Second output: (1, 1024, 14, 14)
            expected_output2_size = 1 * 1024 * 14 * 14
            if output2_data.size >= expected_output2_size:
                output2 = output2_data[:expected_output2_size].reshape((1, 1024, 14, 14))
            else:
                logger.error(f"Output 2 size mismatch. Expected: {expected_output2_size}, Got: {output2_data.size}")
                output2 = np.zeros((1, 1024, 14, 14), dtype=np.float32)
            
            return output1, output2
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return np.array([]), np.array([])
    
    def get_output_dimensions(self) -> List[List[int]]:
        """
        Get output dimensions of the engine
        
        Returns:
            List[List[int]]: List of output dimensions for each output
        """
        output_dims = []
        if self.engine:
            # Find all output bindings
            for i in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(i):
                    binding_shape = self.engine.get_binding_shape(i)
                    output_dims.append(list(binding_shape))
        
        return output_dims
    
    def get_input_dimensions(self) -> List[int]:
        """
        Get input dimensions of the engine
        
        Returns:
            List[int]: List of input dimensions
        """
        input_dims = []
        if self.engine:
            # Find input binding (typically the first binding)
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    binding_shape = self.engine.get_binding_shape(i)
                    input_dims = list(binding_shape)
                    break
        
        return input_dims

# Example usage
if __name__ == "__main__":
    # Example usage of the TRTEngineManager
    engine_path = "model.engine"  # Replace with your engine file path
    
    # Create and initialize engine manager
    engine_manager = TRTEngineManager(engine_path)
    
    if engine_manager.initialize():
        logger.info("Engine initialized successfully!")
        
        # Get dimensions
        input_dims = engine_manager.get_input_dimensions()
        output_dims = engine_manager.get_output_dimensions()
        
        logger.info(f"Input dimensions: {input_dims}")
        logger.info(f"Output dimensions: {output_dims}")
        
        # Example inference with correct input shape (1, 3, 224, 224)
        input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        
        # Perform inference
        output1, output2 = engine_manager.do_inference(input_data=input_data, batch_size=1)
        
        logger.info(f"Inference completed.")
        logger.info(f"Output 1 shape: {output1.shape}")  # Should be (1, 1024)
        logger.info(f"Output 2 shape: {output2.shape}")  # Should be (1, 1024, 14, 14)
        
    else:
        logger.error("Failed to initialize engine!")