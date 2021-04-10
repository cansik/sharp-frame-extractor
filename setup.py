from setuptools import setup

setup(
    name='sharp-frame-extractor',
    version='1.0.0',
    packages=['utils', 'estimator'],
    url='https://github.com/cansik/sharp-frame-extractor',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Extracts sharp frames from a video by using a time window to detect the sharpest frame.',
    install_requires=['opencv-python', 'numpy'],
)
