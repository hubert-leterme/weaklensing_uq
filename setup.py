from setuptools import setup, find_packages

setup(
    name="wlmmuq",
    version="0.1.0",
    author="Hubert Leterme, Andreas Tersenov",
    author_email="hubert.leterme@ensicaen.fr, atersenov@physics.uoc.gr",
    description="Distribution-free uncertainty quantification for weak lensing mass mapping",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hubert-leterme/weaklensing_uq",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "scikit-image",
        "h5py",
        "matplotlib",
        "astropy",
        "lenspack",
        "deepinv==0.2.2",
        "torch",
        "torchinfo",
        "timm",
        "wandb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
