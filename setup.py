#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


# Parse the README
with open("README.md", 'r') as fh:
    long_description = fh.read()

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOXENV' in os.environ and 'SETUPPY_CFLAGS' in os.environ:
    os.environ['CFLAGS'] = os.environ['SETUPPY_CFLAGS']

setup(
    name='cuTWED',
    version='0.3.0',
    description='A linear memory CUDA Time Warp Edit Distance algorithm.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Garrett Wright',
    author_email='garrett@gestaltgp.com',
    url='https://github.com/garrettwrong/cuTWED',
    packages=find_packages('cuTWED'),
    package_dir={'': 'cuTWED'},
    py_modules=[splitext(basename(path))[0] for path in glob('cuTWED/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: C',
        'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ],
    tests_require=['pytest'],
    project_urls={
        'Documentation': 'https://cuTWED.readthedocs.io/',
        'Changelog': 'https://cuTWED.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/garrettwrong/cuTWED/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        'cffi>=1.0.0',
        'numpy',
        'pycuda',
    ],
    extras_require={
        'rst': ['docutils>=0.11'],
        'dev': ['tox',
                'pytest',
                'check-manifest',
                'coverage',
                'readme-renderer',
                'isort',
                'recommonmark',
                'sphinx_rtd_theme',
                'twine >= 1.12.0'],
        # ':python_version=="2.6"': ['argparse'],
    },
    # We only require CFFI when compiling.
    # pyproject.toml does not support requirements only for some build actions,
    # but we can do it in setup.py.
    setup_requires=[
        'pytest-runner',
        'cffi>=1.0.0',
    ],
    cffi_modules=[i + ':ffibuilder' for i in glob('cuTWED/_*_build.py')],
)
