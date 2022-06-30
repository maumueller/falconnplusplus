import os
import sys

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

# native clang doesn't support openmp
# TODO add better way to check for openmp
use_openmp = sys.platform != 'darwin'
extra_args = ['-std=c++14', '-march=native', '-O3']
extra_link_args = []

if use_openmp:
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    'falconnpp',
    sources=['python/python_wrapper.cpp', 'src/fht.c', 'src/fast_copy.c', 'src/Utilities.cpp', 'src/Header.cpp'],
    extra_compile_args=extra_args,
    extra_link_args=extra_link_args,
    include_dirs=['src', '/usr/include/eigen3','/usr/local/include/eigen3', '/usr/include/pybind11','/usr/local/include/pybind11', 'libs'])

setup(
    name='FALCONNPP',
    version='0.1',
    author='Ninh Pham',
    author_email='',
    url='https://github.com/',
    description=
    'High-Dimenional Similarity search',
    license='MIT',
    keywords=
    'nearest neighbor search similarity lsh locality-sensitive hashing cosine distance',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[module])