# setup.py

import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# ------------------------------------------------------------------------------
# 一、定义 Cython 扩展信息
#    - name: 必须与 import 路径保持一致
#    - sources: .pyx 文件路径
#    - include_dirs: 指定 NumPy 头文件位置
# ------------------------------------------------------------------------------
ext_modules = [
    Extension(
        name="marble.tasks.GTZANBeatTracking.madmom.hmm",
        sources=[os.path.join("marble", "tasks", "GTZANBeatTracking", "madmom", "hmm.pyx")],
        include_dirs=[np.get_include()],
        language="c",  # 若想启用 C++，改成 "c++"
    )
]

# ------------------------------------------------------------------------------
# 二、交给 cythonize 编译 .pyx
#    - language_level="3": 使用 Python 3 语法解析 Cython 文件
# ------------------------------------------------------------------------------
cythonized_extensions = cythonize(
    ext_modules,
    compiler_directives={"language_level": "3"},
    annotate=False,
)

# ------------------------------------------------------------------------------
# 三、调用 setuptools.setup() 填写项目元数据及扩展模块
# ------------------------------------------------------------------------------
setup(
    name="marble",                # 跟 pyproject.toml 中 [project].name 保持一致
    version="0.2.0",              # 跟 pyproject.toml 中 [project].version 保持一致
    description="Marble: a framework for SSL-based audio tasks",
    author="Ruibin Yuan",
    python_requires=">=3.7",      # 这里由 pyproject.toml 标记为 dynamic 字段
    packages=find_packages(where="."),  # 自动查找所有包含 __init__.py 的子包
    install_requires=[
        # 这些是运行时及编译时至少需要的基础包。大依赖（torch, lightning 等）已在 pyproject.toml 中列出
        "numpy>=1.19",
        "cython>=0.29",
        "scipy>=1.5",
    ],
    setup_requires=[
        # 在执行 build_ext 之前，会先安装这两个包
        "cython>=0.29",
        "numpy>=1.19",
    ],
    ext_modules=cythonized_extensions,
    include_package_data=True,
    zip_safe=False,
)
