from setuptools import find_packages, setup

requirements = [
    "environs==9.4.0",
    "mmcv-full==1.4.4",
    "mmsegmentation==0.20.2",
    "matplotlib==3.5.1",
    "numpy==1.22.1",
    "pandas==1.3.5",
    "Pillow==8.2.0",
    "seaborn==0.11.2",
    "tqdm==4.59.0",
]

setup(
    name="gcsemseg",
    version="0.1",
    description="Configs, dependencies, env files and custom functions for the multitask network.",
    packages=find_packages(exclude=["dependencies", "docs"]),
    package_data={"": ["*.py", "*.yml", ".env.*", "Dockerfile_cpu", "Dockerfile_gpu"]},
    include_package_data=True,
    install_requires=requirements,
)

