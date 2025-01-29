from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="emd4d",
    ext_modules=[CUDAExtension("emd4d", ["4demd.cpp", "4demd_cuda.cu",]),],
    cmdclass={"build_ext": BuildExtension},
)

