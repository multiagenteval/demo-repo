from setuptools import setup, find_packages

setup(
    name="maee-demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "GitPython>=3.1.0",
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
        ],
        'dev': [
            'jupyter>=1.0.0',
            'black>=22.0.0',
        ],
    }
) 