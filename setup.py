#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import re
import sys
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


# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
# if 'TOXENV' in os.environ and 'SETUPPY_CFLAGS' in os.environ:
#     os.environ['CFLAGS'] = os.environ['SETUPPY_CFLAGS']

setup(
    name='cuTWED',
    version='0.2.0',
    description='A linear memory CUDA Time Warp Edit Distance algorithm.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.md')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Garrett Wright',
    author_email='garrett@gestaltgp.com',
    url='https://github.com/garrettwrong/cuTWED',
    packages=find_packages('cuTWED'),
    package_dir={'': 'cuTWED'},
    py_modules=[splitext(basename(path))[0] for path in glob('cuTWED/*.py')],
    #py_modules=['cuTWED'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
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
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    # We only require CFFI when compiling.
    # pyproject.toml does not support requirements only for some build actions,
    # but we can do it in setup.py.
    setup_requires=[
        'cffi>=1.0.0'
    ] if any(i.startswith('build') or i.startswith('bdist') for i in sys.argv) else [
    ],
    cffi_modules=[i + ':ffibuilder' for i in glob('cuTWED/_*_build.py')],
)
