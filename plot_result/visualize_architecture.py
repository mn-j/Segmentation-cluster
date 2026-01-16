

import torch
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights, r2plus1d_18, R2Plus1D_18_Weights, swin3d_b, Swin3D_B_Weights


model = r3d_18(weights=R3D_18_Weights.DEFAULT)

# Define dummy input (the size must match what the model expects)
dummy_input = torch.randn(1, 3, 50, 112, 112)  # (batch_size, channels, height, width)

# Export the model to ONNX format
torch.onnx.export(
    model,             # Model to export
    dummy_input,       # Dummy input to define model structure
    "3Dresnet18.onnx",   # Output file name
    input_names=['input'],  # Name for input layer
    output_names=['output'],  # Name for output layer
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}  # Dynamic axis for batch
)


'''
model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

# Define dummy input (the size must match what the model expects)
dummy_input = torch.randn(1, 3, 50, 112, 112)  # (batch_size, channels, height, width)

# Export the model to ONNX format
torch.onnx.export(
    model,             # Model to export
    dummy_input,       # Dummy input to define model structure
    "r2plus1d_18.onnx",   # Output file name
    input_names=['input'],  # Name for input layer
    output_names=['output'],  # Name for output layer
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}  # Dynamic axis for batch
)
'''
'''
model = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)

# Define dummy input (the size must match what the model expects)
dummy_input = torch.randn(1, 3, 50, 224, 224)  # (batch_size, channels, height, width)

# Export the model to ONNX format
torch.onnx.export(
    model,             # Model to export
    dummy_input,       # Dummy input to define model structure
    "swin3d_b.onnx",   # Output file name
    input_names=['input'],  # Name for input layer
    output_names=['output'],  # Name for output layer
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}  # Dynamic axis for batch
)
'''