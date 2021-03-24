# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import os

os.system("pip install -r requirements.txt")

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="FND",
    version="0.0.1",
    description="Fake News Detection AI",
    license="GPL v3.0",
    author="Rifky Bujana Bisri",
    packages=find_packages(),
    install_requires=[],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8.3",
    ]
)
