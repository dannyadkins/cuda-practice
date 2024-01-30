import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline
# from utils.compile_extension import compile_extension

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


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension(file_name="kernel.cu", cpp_fn_signature="torch::Tensor rgb_to_grayscale(torch::Tensor input);", cpp_fn_name="rgb_to_grayscale")

    x = read_image("Grace_Hopper.jpg").permute(1, 2, 0).cuda()
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    y = ext.rgb_to_grayscale(x)

    print("Output image:", y.shape, y.dtype)
    print("mean", y.float().mean())
    write_png(y.permute(2, 0, 1).cpu(), "output.png")


if __name__ == "__main__":
    main()