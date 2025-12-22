from setuptools import setup, find_packages

setup(
    name="champs",
    version="0.1.0",
    description="FDTD electromagnetic simulation package",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
