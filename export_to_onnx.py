import os
import sys
from pathlib import Path
import torch
ROOT = str(Path(__file__).parent.absolute())
sys.path.append(os.path.join(ROOT,'core'))
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

model_name = 'PE-Core-B16-224' 

model = pe.CLIP.from_config(model_name, pretrained=True)
# model.forward = model.encode_image
# # Create a dummy input matching the model's expected input shape
# dummy_input = torch.randn(1, 3, 224, 224)

# onnx_path = os.path.join(ROOT, f"{model_name}.onnx")

# torch.onnx.export(
#     model,                                 # model being run
#     dummy_input,                           # model input (or a tuple for multiple inputs)
#     onnx_path,                             # where to save the model
#     export_params=True,                    # store the trained parameter weights inside the model file
#     opset_version=14,                      # the ONNX version to export the model to
#     do_constant_folding=True,              # whether to execute constant folding for optimization
#     input_names=['input'],                 # the model's input names
#     output_names=['features',"patch"],               # the model's output names
#     # dynamic_axes={                         # variable length axes
#     #     'input': {0: 'batch_size'},
#     #     'output': {0: 'batch_size'}
#     # },
#     verbose=True,                          # print a graph of the exported model
#     training=torch.onnx.TrainingMode.EVAL, # export the model in inference mode
#     keep_initializers_as_inputs=False,     # do not keep initializers as inputs
#     custom_opsets=None,                    # custom opset domains
#     enable_onnx_checker=True,              # enable ONNX model checker
#     use_external_data_format=False         # store model weights in a single file
# )

model.forward = model.encode_text
# Create a dummy input matching the model's expected input shape
dummy_input = torch.randint(1,10,(7,32)).to(torch.int32)

onnx_path = os.path.join(ROOT, f"text_{model_name}.onnx")
torch.onnx.export(
    model,                                 # model being run
    dummy_input,                           # model input (or a tuple for multiple inputs)
    onnx_path,                             # where to save the model
    export_params=True,                    # store the trained parameter weights inside the model file
    opset_version=14,                      # the ONNX version to export the model to
    do_constant_folding=True,              # whether to execute constant folding for optimization
    input_names=['input'],                 # the model's input names
    output_names=['text_features'],               # the model's output names
    dynamic_axes={                         # variable length axes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    verbose=True,                          # print a graph of the exported model
    training=torch.onnx.TrainingMode.EVAL, # export the model in inference mode
    keep_initializers_as_inputs=False,     # do not keep initializers as inputs
    custom_opsets=None,                    # custom opset domains
    enable_onnx_checker=True,              # enable ONNX model checker
    use_external_data_format=False         # store model weights in a single file
)