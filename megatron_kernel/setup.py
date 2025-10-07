import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = []

ext_modules.append(CUDAExtension(
    name="megatron_ops",
    sources=[
        "megatron_kernel/csrc/moe/topk_softmax_kernels.cu",
        "megatron_kernel/csrc/moe/torch_bindings.cpp",
    ],
    include_dirs=[],
    libraries=["m"],  # 链接数学库（如 -lm）
    extra_compile_args={
        "cxx": [
            "-std=c++17",
            "-O3",
            #"-I./" + third_party_include_dir
        ],         # C++ 编译选项
        "nvcc": [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            #"-I./" + third_party_include_dir
        ],  # NVCC 编译选项
    },
))


def get_requirements() -> list[str]:
    return []

if __name__ == '__main__':
    setup(
        name='megatron_ops',
        version='1.0.0',
        install_requires=get_requirements(),
        packages=['megatron_ops', 'megatron_ops/transformers/moe'],
        package_data={
            'megatron_ops': [
                'csrc/*',
                'csrc/moe/*'
            ]
        },
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": BuildExtension,
        },  # 必须添加以支持混合编译
    )
