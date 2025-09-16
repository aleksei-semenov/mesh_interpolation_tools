from setuptools import setup, find_packages

setup(
    name="mesh-interpolation-tools",  # ✅ this is the PyPI/distribution name
    version="0.1.0",
    description="Conservative interpolation between triangular meshes and structured grids",
    author="Aleksei Semenov",
    author_email="aleksei.semenov@awi.de",
    url="https://github.com/aleksei-semenov/mesh-interpolation-tools",
    packages=find_packages(include=["mesh_interpolation_tools", "mesh_interpolation_tools.*"]),  # ✅ underscores
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "healpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
# Note: The package name uses underscores, while the distribution name uses hyphens.