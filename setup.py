#!/usr/bin/env python3
# title: MA721 project 1 (MNIST classifier)

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='mnist_x',
    version='0.1.0',
    description='MNIST classifier (MA721 course project for DL practical '
                'methodologies)',
    long_description=readme,
    license=license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)