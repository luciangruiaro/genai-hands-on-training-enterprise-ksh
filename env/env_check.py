import platform
import sys
import os
import torch
import psutil
import multiprocessing


def get_virtual_env():
    """Return the name of the active virtual environment, if any."""
    venv = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_DEFAULT_ENV')
    return os.path.basename(venv) if venv else "None"


def print_system_info():
    """Print system and environment details."""
    print("=" * 60)
    print("🖥️  System Information")
    print("=" * 60)
    print(f"🐍 Python version           : {platform.python_version()}")
    print(f"📦 Virtual environment      : {get_virtual_env()}")
    print(f"🧭 OS                       : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"💻 CPU                      : {platform.processor()} ({multiprocessing.cpu_count()} cores)")
    print(f"🧠 RAM                      : {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")
    print()


def print_pytorch_cuda_info():
    """Print PyTorch, CUDA, cuDNN, and GPU configuration."""
    print("=" * 60)
    print("🔍 PyTorch & CUDA Details")
    print("=" * 60)

    print(f"🧠 PyTorch version          : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"⚙️  CUDA available           : {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Number of GPUs            : {gpu_count}")

        for i in range(gpu_count):
            device = torch.cuda.get_device_properties(i)
            print(f"\n   └─ GPU {i}: {device.name}")
            print(f"      • Compute capability   : {device.major}.{device.minor}")
            print(f"      • Memory (total)       : {round(device.total_memory / (1024 ** 3), 2)} GB")
            print(f"      • Multi-processors     : {device.multi_processor_count}")

        print(f"\n📦 cuDNN enabled             : {torch.backends.cudnn.enabled}")
        print(f"🧪 cuDNN benchmark mode     : {torch.backends.cudnn.benchmark}")
        print(f"🔁 cuDNN deterministic      : {torch.backends.cudnn.deterministic}")
    else:
        print("⚠️  CUDA is not available. Running on CPU.")
    print("=" * 60)


if __name__ == "__main__":
    print_system_info()
    print_pytorch_cuda_info()
