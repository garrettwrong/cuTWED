# Some reminders how I did this for 2.0

docker build -f ci/manylinux2010/Dockerfile-x86_64 -t garrettwrong/cutwed_manylinux2010:2.0.0 .
docker push garrettwrong/cutwed_manylinux2010:2.0.0
docker run --gpus all -it -v `pwd`/wheelhouse:/io/wheelhouse -e PLAT=manylinux2010_x86_64 garrettwrong/cutwed_manylinux2010:2.0.0 ci/build-wheels.sh


# test a wheel
conda create --name cutwed_38_test_wheel python=3.8
conda activate cutwed_38_test_wheel
pip install wheelhouse/cuTWED-2.0.0-cp38-cp38-manylinux2010_x86_64.whl
pip install pytest
pytest tests


# pushing up to pypi
python setup.py sdist # source distro
cp -v wheelhouse/cuTWED-2.0.0-cp3*-manylinux* dist
# https://packaging.python.org/guides/using-testpypi/
twine upload -r testpypi dist/*
pip install -i https://test.pypi.org/simple/ --no-deps cuTWED
# review it, confirm repo up to date, then upload to live index:
#twine upload dist/*
