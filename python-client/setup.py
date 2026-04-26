from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vectradb-client",
    version="0.1.0",
    author="VectraDB Team",
    description="Python client library for VectraDB vector database with gRPC interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vectradb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "grpcio>=1.60.0",
        "protobuf>=4.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "grpcio-tools>=1.60.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
)
