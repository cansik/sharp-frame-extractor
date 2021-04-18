from setuptools import setup, find_packages

NAME = 'sharp-frame-extractor'

required_packages = find_packages()
required_packages.append(NAME)

setup(
    app="%s.py" % NAME,
    name=NAME,
    version='1.5.1',
    packages=required_packages,
    url='https://github.com/cansik/sharp-frame-extractor',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Extracts sharp frames from a video by using a time window to detect the sharpest frame.',
    install_requires=['wheel', 'opencv-python', 'numpy', 'psutil'],
)
