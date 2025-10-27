import torch
import os, sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
