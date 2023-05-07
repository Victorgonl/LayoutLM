#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="layoutlm",
    version="0.1",
    author="LayoutLM Team",
    packages=find_packages(where="layoutlm"),
    package_dir={'': 'layoutlm'},
    python_requires=">=3.10",
    extras_require={"dev": ["flake8", "isort", "black"]},
)