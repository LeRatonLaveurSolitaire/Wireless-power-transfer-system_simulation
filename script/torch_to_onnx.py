import torch
import sys
import os

path_to_model_class = os.path.join(sys.path[0], "..", "src", "parameters_estimators", "NN_estimator")
sys.path.insert(0, path_to_model_class)

from nn_model import NN_model


onnx_model_path = 'model.onnx'

model_path = "src/parameters_estimators/NN_estimator/models/most_accurate_model.pt"




print("Loading model...")

model = NN_model(input_size=30, output_size=2)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
print("Model sucessfully loaded !\n")

model.eval()

sample_input = torch.rand(1,30)
print(sample_input)
y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)