from setuptools import setup, find_packages

setup(
    name='reetoolbox',
    version='0.1.1',
    packages=find_packages(include=['reetoolbox', 'reetoolbox.*']),
    install_requires=[
        'numpy==1.19.5',
        'scikit-learn==1.0.2',
        'torch==1.10.0',
        'torchvision==0.11.1',
        'matplotlib',
        'seaborn==0.11.2',
        'Pillow'
    ],
    author='Alex Foote',
    author_email='alexjfoote@icloud.com',
    url="https://github.com/alexjfoote/reetoolbox",
)