#!/bin/bash
set -e -u -x
shopt -s extglob

# Compile wheels
for PYBIN in /opt/python/cp3[6789]*/bin; do
    "${PYBIN}/pip" install auditwheel pytest
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    "${PYBIN}/auditwheel" repair "$whl" --plat "$PLAT" -w /io/wheelhouse/
done


# Install packages and test
for PYBIN in /opt/python/cp3[6789]*/bin/; do
    "${PYBIN}/pip" install cuTWED -f /io/wheelhouse
    "${PYBIN}/python" -m pytest /io/tests
done
