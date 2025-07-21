from setuptools import setup, find_packages

setup(
    name='rfim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0'
    ],
)
