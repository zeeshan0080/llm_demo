import transformers
import torch
import torchvision

print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("mps available:", torch.backends.mps.is_available())