"""The device to run on: an Nvidia GPU, else Apple's GPU (MPS), else the CPU. torch is imported only when asked, so
importing this file never changes the order in which torch and LightGBM load."""


def torch_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def yolo_device():
    """The same choice in the form Ultralytics takes (0 = the first Nvidia GPU)."""
    d = torch_device()
    return 0 if d == "cuda" else d
