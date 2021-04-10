from setuptools import setup

setup(
    name='sharp-frame-extractor',
    version='1.1.0',
    packages=['SFE'],
    url='https://github.com/cansik/sharp-frame-extractor',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Extracts sharp frames from a video by using a time window to detect the sharpest frame.',
    install_requires=['wheel', 'opencv-python', 'numpy'],
)
