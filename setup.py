import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="lora_diffusion",
    py_modules=["lora_diffusion"],
    version="0.1.7",
    description="Low Rank Adaptation for Diffusion Models. Works with Stable Diffusion out-of-the-box.",
    author="Simo Ryu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lora_pti = lora_diffusion.cli_lora_pti:main",
        ],
    },
    
    include_package_data=True,
)
