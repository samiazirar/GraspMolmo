from setuptools import setup, find_namespace_packages

base_deps = [
    "numpy~=1.26.4",
    "pillow~=10.2.0",
    "tqdm~=4.67.1"
]

datagen_deps = [
    "acronym_tools @ git+https://github.com/abhaybd/acronym.git",
    "boto3~=1.36.17",
    "fastapi~=0.115.7",
    "h5py~=3.12.1",
    "open3d~=0.18.0",
    "openai~=1.64.0",
    "pydantic~=2.10.5",
    "pyrender~=0.1.45",
    "requests~=2.32.3",
    "scipy~=1.14.1",
    "trimesh~=4.5.3",
    "types-boto3~=1.36.17",
    "types-boto3-s3~=1.36.15",
    "uvicorn~=0.34.0",
    "scene-synthesizer~=1.13.1",
    "hydra-core~=1.3.2",
    "torch~=2.5.0",
    "einops~=0.8.1",
    "datasets~=3.3.2",
    "shortuuid~=1.0.13"
]

tg_deps = [
    "sam2 @ git+https://github.com/facebookresearch/sam2.git",
    "learning3d"
]

infer_deps = [
    "transformers~=4.52.4",
    "torch~=2.0",
    "torchvision~=0.22.0",
    "tensorflow~=2.0",
    "accelerate~=1.7.0"
]

setup(
    name="graspmolmo",
    version="0.1.0",
    description="Data and eval code for GraspMolmo",
    author="Abhay Deshpande",
    packages=["graspmolmo"] + find_namespace_packages(include=["graspmolmo.*"]),
    python_requires=">=3.8",
    install_requires=base_deps,
    extras_require={
        "datagen": datagen_deps,
        "taskgrasp_image": datagen_deps + tg_deps,
        "infer": infer_deps,
        "all": datagen_deps + tg_deps + infer_deps,
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
