#!/usr/bin/env python3

from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="DecAF",
    version="1.0.0",
    description="Joint Decoding Answer and Logical Form for Knowledge Base Question Answering through Retrieval",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
        "setuptools>=18.0",
    ],
)