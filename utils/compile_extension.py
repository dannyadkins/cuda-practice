from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline


def compile_extension(file_name, cpp_fn_signature, cpp_fn_name):
    if (Path(file_name).suffix != ".cu"):
        raise Exception("File must be a .cu file")

    cuda_source = Path(file_name).read_text()

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name=file_name.split(".")[0],
        cpp_sources=cpp_fn_signature,
        cuda_sources=cuda_source,
        functions=[cpp_fn_name],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return ext

