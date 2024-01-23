import torch
from torch2trt import torch2trt

# Load your PyTorch model
model_path = "best.pt"
model = torch.load(model_path)
model.eval()  # Make sure to set the model in evaluation mode

# Define the input size expected by your model (replace with actual values)
height = 224
width = 224

# Create a dummy input tensor
input_tensor = torch.ones((1, 3, height, width)).cuda()

# Convert the PyTorch model to TensorRT engine
engine = torch2trt(model, [input_tensor])

# Save the TensorRT engine
with open('new_engine.engine', 'wb') as f:
    f.write(engine.serialize())
