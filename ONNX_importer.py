import sys
import os
import torch
from collections import OrderedDict

# Add the parent directory of TRACER to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TRACER"))

# Import the required modules
from TRACER.model.EfficientNet import EfficientNet
from TRACER.model.TRACER import TRACER
from TRACER.config import getConfig
args = getConfig()

# initialize model
model = TRACER(args)

# load pre-trained weights
state_dict = torch.load("./input_model/TRACER-Efficient-7.pth", map_location=torch.device("cpu"))

# model weights saved from DataParallel model -- need to remove module. prefix so that there is no mismatched keys when loading weights
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Prepare input tensor
dummy_input = torch.randn(1, 3, 256, 256)

# Perform inference
output = model(dummy_input)
print(output)