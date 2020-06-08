import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='sleepa',
    version='0.0.1',
    author='Emanuel Flores B.',
    author_email='efflores@caltech.edu',
    description='Utilities for unsupervised machine learning applied to whole-brain calcium imaging data.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
