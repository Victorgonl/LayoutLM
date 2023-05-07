#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="layoutlmft",
    version="0.1",
    author="LayoutLM Team",
    url="https://github.com/microsoft/unilm/tree/master/layoutlmft",
    packages=find_packages(where="layoutlmft"),
    package_dir={'': 'layoutlmft'},
    python_requires=">=3.10",
    extras_require={"dev": ["flake8", "isort", "black"]},
)