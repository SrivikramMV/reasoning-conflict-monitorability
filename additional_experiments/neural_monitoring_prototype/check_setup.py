from __future__ import annotations

import importlib.util
import platform
import sys


PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("plotly", "plotly"),
    ("sklearn", "scikit-learn"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
]


def main():
    print("NMT setup check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

    missing = []
    for module, package in PACKAGES:
        try:
            ok = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            ok = False
        print(f"{package:18s} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(package)

    print()
    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        print(f"Could not inspect torch/CUDA: {exc}")

    print()
    if missing:
        print("Missing packages:")
        print("python -m pip install -r requirements.txt")
    else:
        print("All Python packages required by NMT appear to be installed.")


if __name__ == "__main__":
    main()
