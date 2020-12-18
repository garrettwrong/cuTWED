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
with open("README.rst", 'r') as fh:
    long_description = fh.read()

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOXENV' in os.environ and 'SETUPPY_CFLAGS' in os.environ:
    os.environ['CFLAGS'] = os.environ['SETUPPY_CFLAGS']

setup(
    name='cuTWED',
    version='2.0.2',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ],
    tests_require=['pytest',
                   'ipywidgets >= 7.5.1'],
    project_urls={
        'Issue Tracker': 'https://github.com/garrettwrong/cuTWED/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        'cffi>=1.0.0',
        'numpy',
        'pycuda',
        'six',
        'tqdm',
    ],
    extras_require={
        'dev': ['tox',
                'pytest',
                'bumpversion',
                'check-manifest',
                'coverage',
                'docutils>=0.11',
                'matplotlib',
                'readme-renderer',
                'isort',
                'ipywidgets >= 7.5.1',
                'seaborn',
                'sphinx_rtd_theme',
                'twine >= 1.12.0'],
    },
    # We only require CFFI when compiling.
    # pyproject.toml does not support requirements only for some build actions,
    # but we can do it in setup.py.
    setup_requires=[
        'pytest-runner',
        'cffi>=1.0.0',
    ],
    cffi_modules=[i + ':ffibuilder' for i in glob('cuTWED/_*_build.py')],
    headers=['cuTWED/cuTWED.h.i', 'reference_implementation/twed.h'],
)
