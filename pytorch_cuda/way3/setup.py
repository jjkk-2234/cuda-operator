from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = "vector_add",
    version = "0.0.1",
    ext_modules = [CUDAExtension(
            name="vector_add",
            sources=["vector_add.cu"],
            extra_compile_args={"cxx":["-O3"], "nvcc":["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"]
)