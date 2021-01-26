#!/usr/bin/env python
"""
Installs keras_ct_seg

> Tyler Ganter, tganter@sandia.gov
"""
import os
from setuptools import setup, find_packages


def get_requirements():
    """Reads the requirements from the `requirements.txt` file in the same dir as
    `setup.py`.

    Returns:
        a list of 3rd-party dependency strings
    """
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as file:
        requirements_list = list(map(str.strip, file.readlines()))
    return requirements_list


setup(
    name="ctseg",
    version="0.1",
    description="CT Segmentation",
    author="Tyler Ganter",
    author_email="tganter@sandia.gov",
    url="https://gitlab.sandia.gov/9323/ami/digital-twins/keras_ct_seg_clean",
    license="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "cpu": ["tensorflow==1.13.1"],
        "gpu": ["tensorflow-gpu==1.10.0"],
    },
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.6",
)
