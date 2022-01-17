from setuptools import setup, find_packages

setup(
    name='reetoolbox',
    version='0.1.0',
    packages=find_packages(include=['reetoolbox', 'reetoolbox.*']),
    install_requires=[
        'numpy==1.19.5',
        'scikit-learn==1.0.2',
        'torch==1.10.0+cu111',
        'torchvision==0.11.1+cu111',
        'matplotlib==3.2.2',
        'seaborn==0.11.2',
        'Pillow==7.1.2'
    ]
)