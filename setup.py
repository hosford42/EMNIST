import os

from setuptools import setup
from io import open


path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(path, 'README.md')) as file:
    long_description = file.read()


setup(
    name='emnist',
    version='0.0',
    description='Extended MNIST - Python Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hosford42/EMNIST',
    author='Aaron Hosford',
    author_email='hosford42@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='MNIST EMNIST image recognition data dataset numpy idx neural network'
             'machine learning',
    packages=['emnist'],
    python_requires='>=3.0',  # TODO: Evaluate for use with Python 2
    install_requires=['numpy', 'requests', 'tqdm'],
    extras_require={
        'inspect': ['matplotlib']
    },
    project_urls={
        'The EMNIST Dataset': 'https://www.nist.gov/itl/iad/image-group/emnist-dataset',
        'The EMNIST Paper': 'https://arxiv.org/abs/1702.05373v1',
    }
)
