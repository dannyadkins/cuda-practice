from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()


setup(
    name='cuda-practice',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements()
)