import os
import sys
import subprocess
import platform

import pybind11
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

cxx = os.environ.get("CXX")
if cxx == "" or cxx is None:
    raise Exception('Set CXX variable in your environment')

compile_args = []
link_args = []
macros_args = []

is_linux: bool = sys.platform == "linux"
is_macos: bool = sys.platform == "darwin"
is_windows: bool = sys.platform == "win32"

if is_windows:
    raise Exception('Windows not yet implemented')

# TODO: in intel sapphirerapids one have to force LLVM to use 512 registers
if is_linux:
    compile_args.append("-std=c++17")
    compile_args.append("-O3")
    compile_args.append("-march=native")
    compile_args.append("-fPIC")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fdiagnostics-color=always")
    compile_args.append("-Wl,--unresolved-symbols=ignore-in-shared-libs")
    link_args.append("-static-libstdc++")

if is_macos:
    compile_args.append("-std=c++17")
    compile_args.append("-O3")
    compile_args.append("-march=native")
    compile_args.append("-fPIC")
    compile_args.append("-arch")
    compile_args.append("arm64")  # TODO: Currently not supporting non-ARM Macs
    compile_args.append("-Wl,")
    compile_args.append("-Wno-unknown-pragmas")


ext_modules = [
    Pybind11Extension(
        "pdxearch.compiled",
        sources=["python/lib.cpp"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        # define_macros=macros_args,
        language="c++",
    ),
]

include_dirs = [
    "extern",
    "include",
    "python",
]

install_requires = [
    "setuptools>=42",
    "wheel",
    "cmake>=3.22",
    "pybind11",
    "numpy"
]

# Taken from Usearch setup.py (https://github.com/unum-cloud/usearch/blob/main/setup.py)
# With Clang, `setuptools` doesn't properly use the `language="c++"` argument we pass.
# The right thing would be to pass down `-x c++` to the compiler, before specifying the source files.
# This nasty workaround overrides the `CC` environment variable with the `CXX` variable.
cc_compiler_variable = os.environ.get("CC")
cxx_compiler_variable = os.environ.get("CXX")
if cxx_compiler_variable:
    os.environ["CC"] = cxx_compiler_variable

description = "Pruned Vertical Vector Similarity Search"
long_description = ""

setup(
    name="pdxearch",
    version="0.1",
    packages=["pdxearch"],
    package_dir={"pdxearch": "python/pdxearch"},
    description=description,
    author="CWI",
    author_email="lxkr@cwi.nl",
    url="https://github.com/cwida/pdxearch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        # "Operating System :: Microsoft :: Windows",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # cmdclass={"build_ext": build_ext},
    include_dirs=include_dirs,
    ext_modules=ext_modules,
    install_requires=install_requires,
)

# Reset the CC environment variable, that we overrode earlier.
if cxx_compiler_variable and cc_compiler_variable:
    os.environ["CC"] = cc_compiler_variable
