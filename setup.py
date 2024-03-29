from setuptools import setup, find_packages

NAME = 'sharp_frame_extractor'

required_packages = find_packages()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=NAME,
    version='1.6.5',
    packages=required_packages,
    url='https://github.com/cansik/sharp-frame-extractor',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Extracts sharp frames from a video by using a time window to detect the sharpest frame.',
    install_requires=required,
)
