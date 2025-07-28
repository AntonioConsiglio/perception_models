import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).parent.absolute())
sys.path.append(os.path.join(ROOT,'core'))
#import core.vision_encoder.pe as pe
# import core.vision_encoder.transforms as transforms

from PIL import Image
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import onnxruntime as ort
from tensorrt_engine import TRTEngineManager

class ImagePreprocessor:
    def __init__(self, image_size, interpolation=cv2.INTER_LINEAR):
        self.image_size = image_size
        self.interpolation= interpolation
        #self.mean = np.array([0.5,0.5,0.5],dtype=np.float32).reshape(1,1,3)
        #self.std = np.array([0.5,0.5,0.5],dtype=np.float32).reshape(1,1,3)

    def __call__(self,image):
        # This implementation to match the pytorch based implementation
        image = Image.fromarray(image).convert("RGB")
        resized = np.array(image.resize(self.image_size,Image.BILINEAR)) / 255.0
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # resized = cv2.resize(image,
        #                      self.image_size,
        #                      interpolation=self.interpolation) / 255.0
        normalized = (resized - 0.5) / 0.5
        # cv2.imwrite("resized_onnx.png",(normalized*255).astype(np.uint8))
        transposed = np.expand_dims(normalized.transpose(2,0,1),axis=0)
        return transposed

# # class CustomTokenizer(transforms.SimpleTokenizer):
#     def __init__(self,context_length):
#         super().__init__(context_length=context_length)
    
#     def __call__(
#         self, texts: Union[str, List[str]], context_length: Optional[int] = None
#     ) -> np.ndarray:
#         """Returns the tokenized representation of given input string(s)

#         Parameters
#         ----------
#         texts : Union[str, List[str]]
#             An input string or a list of input strings to tokenize
#         context_length : int
#             The context length to use; all CLIP models use 77 as the context length

#         Returns
#         -------
#         A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
#         """
#         if isinstance(texts, str):
#             texts = [texts]

#         context_length = context_length or self.context_length
#         assert context_length, "Please set a valid context length"

#         if self.reduction_fn is not None:
#             # use reduction strategy for tokenize if set, otherwise default to truncation below
#             return self.reduction_fn(
#                 texts,
#                 context_length=context_length,
#                 sot_token_id=self.sot_token_id,
#                 eot_token_id=self.eot_token_id,
#                 encode_fn=self.encode,
#             )

#         all_tokens = [
#             [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
#             for text in texts
#         ]
#         result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

#         for i, tokens in enumerate(all_tokens):
#             if len(tokens) > context_length:
#                 tokens = tokens[:context_length]  # Truncate
#                 tokens[-1] = self.eot_token_id
#             result[i, : len(tokens)] = np.array(tokens)

#         return result
ort.set_default_logger_severity(0)
class ONNXRuntimeSession:
    def __init__(self, model_path: str, mode = "text", providers: Optional[list] = None):
        self.mode = mode
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, inputs: np.ndarray) -> Dict[str, np.ndarray]:
        # Ensure inputs is a numpy array with correct dtype and shape
        if not isinstance(inputs, dict):
            # If a single array is passed, wrap it in a dict with the correct input name
            if self.mode != "text":
                inputs = inputs.astype(np.float32)
            inputs = {self.input_names[0]:inputs}
        else:            
            # If a dict is passed, ensure all values are float32 numpy arrays
            inputs = {k: v.astype(np.float32) for k, v in inputs.items()}
        result = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, result))

def improved_bbox_extraction(
        attention_map: np.ndarray,
        method: str = 'best_region',
        percentile_threshold: float = 80
    ) -> Tuple[int, int, int, int]:

    H, W = attention_map.shape
    threshold = np.percentile(attention_map, percentile_threshold)
    thresholded_map = np.zeros_like(attention_map)
    if method == "best_region":
        from skimage.measure import label, regionprops
        mask = attention_map >= threshold
        thresholded_map[mask] = attention_map[mask]
        # Label connected regions
        labeled = label(mask)

        # Find the best region
        best_region = None
        best_score = -np.inf

        for region in regionprops(labeled, intensity_image=attention_map):
            # Example score: total sum of heatmap values inside region
            score = region.intensity_image[region.image].sum()
            if score > best_score:
                best_score = score
                best_region = region

        if best_region is not None:
            # Bounding box: min_row, min_col, max_row, max_col
            min_row, min_col, max_row, max_col = best_region.bbox
            bbox = (min_col, min_row, max_col, max_row)

            return bbox, thresholded_map

class VideoClasifierONNX:
    def __init__(self,model_name):

        self.video_model = TRTEngineManager("pecore_b16_224_sim.engine")
        #self.video_model = ONNXRuntimeSession("PE-Core-B16-224.onnx",mode="image")
        # self.text_model = ONNXRuntimeSession("text_PE-Core-B16-224.onnx")
        self.img_size = (224,224) 
        self.context_length = 32 
        self.patch_size = (14,14)

        self.frame_bank_size = 8
        self.preprocess = ImagePreprocessor(self.img_size)
        # self.tokenizer = CustomTokenizer(self.context_length)
        self.labels = None
        self.text_label = None
        self.frames_bank = []
        self.patch_bank = []
        self.latest_image_tensor = None

    def encode_labes(self, labels:List[str]):
        self.text_label = labels
        # tokenized_labels = self.tokenizer(labels)
        # encoded_labels = self.text_model(tokenized_labels)["text_features"]
        # self.labels = encoded_labels / np.linalg.norm(encoded_labels, axis=-1, keepdims=True)
        self.labels = np.load("labels.npy").astype(np.float32)
        # del self.text_model
    
    def process_frame(self,frame:np.ndarray):
        # image_tensors = self.preprocess(rgb_image).unsqueeze(0).cpu().numpy()
        image_tensors = self.preprocess(frame)
        # output = self.video_model(image_tensors)
        # frame_features, patch = output["features"], output["patch"]
        frame_features, patch = self.video_model.do_inference(image_tensors)
        self.update_frames_bank(frame_features,patch)
        
    def get_label_propabibility(self):
        # (b, n_frames, emb_dim)
        frames_tensors = np.concatenate(self.frames_bank, axis=0)
        frames_tensors = np.expand_dims(frames_tensors, axis=0).mean(axis=1)
        image_features_norm = frames_tensors / np.linalg.norm(frames_tensors, axis=-1, keepdims=True)

        logits = 100.0 * np.matmul(image_features_norm, self.labels.T)
        exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        probabilities = probabilities[0]
        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[predicted_class])
        return predicted_class, confidence, probabilities
    
    def get_attention_maps_box(self, predicted_class,input_shape):
        w,h = input_shape

        num_frames = len(self.patch_bank)
        patch_features = np.concatenate(self.patch_bank, axis=0)  # shape: (num_frames, 1024)
        patch_features = patch_features.reshape(num_frames, 1024, -1)  # (num_frames, patch_count, 1024)
        patch_features = np.transpose(patch_features, (0, 2, 1))  # (num_frames, patch_count, 1024)
        patch_mean = np.mean(patch_features, axis=0, keepdims=True)  # (1, patch_count, 1024)
        patch_norm = patch_mean / np.linalg.norm(patch_mean, axis=-1, keepdims=True)  # (1, patch_count, 1024)
        
        attention_frame_text = 100.0 * np.matmul(patch_norm, self.labels.T)  # (1, patch_count, text_class)
        exp_x = np.exp(attention_frame_text - np.max(attention_frame_text, axis=-1, keepdims=True))
        attention_frame_text = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        attention_frame_text = np.squeeze(attention_frame_text)  # (patch_count, text_class)
        selected_attention = attention_frame_text[:, predicted_class].reshape(self.patch_size)  # (patch_h, patch_w)
        selected_attention_resized = cv2.resize(
                selected_attention, (w, h), interpolation=cv2.INTER_LINEAR
            )
        
        try:
            bbox, filtered_map = improved_bbox_extraction(
                    selected_attention_resized, 
                    percentile_threshold=80, # Higher threshold for more precise localization
                    method="best_region"
                )
            
            heatmap_colored = cv2.applyColorMap(
                (filtered_map * 255).astype(np.uint8), 
                cv2.COLORMAP_JET  # Better colormap for attention
                )
        except:
            heatmap_colored = bbox = None
        # heatmap_colored = cv2.cvtColor(heatmap_colored,cv2.COLOR_BGR2RGB)

        return heatmap_colored, bbox

    def update_frames_bank(self,new_frames, new_patch):
        
        self.frames_bank.append(new_frames)
        self.patch_bank.append(new_patch)
        # Clean unuseful data
        if len(self.frames_bank) > self.frame_bank_size:
            self.frames_bank.pop(0)
            self.patch_bank.pop(0)
        
    def classify_video(self, frame:np.ndarray):
        H,W,_ = frame.shape
        self.process_frame(frame)
        predicted_class, confidence, all_confidence = self.get_label_propabibility()
        heatmap, bbox = self.get_attention_maps_box(predicted_class,(W,H))
            
        results = {
            'predicted_class': predicted_class,
            'class_name': self.text_label[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_confidence.tolist(),
            'bounding_box': bbox,  # (x1, y1, x2, y2)
            "heatmap": heatmap
        }

        return results

        



