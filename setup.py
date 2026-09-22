import os
import sys
from setuptools import Extension, find_packages, setup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CPP_DIR = os.path.join(BASE_DIR, "highs_turbo", "cpp_engine")

include_dirs = [
    CPP_DIR,
    os.path.join(CPP_DIR, "include"),
    "/opt/homebrew/include",
    "/usr/local/include",
    "/usr/include",
]
include_dirs = [d for d in include_dirs if os.path.isdir(d)]

library_dirs = [
    "/opt/homebrew/lib",
    "/usr/local/lib",
    "/usr/lib",
]
library_dirs = [d for d in library_dirs if os.path.isdir(d)]

sources = [
    os.path.join("highs_turbo", "cpp_engine", "bit_graph.cpp"),
    os.path.join("highs_turbo", "cpp_engine", "rational_verifier.cpp"),
    os.path.join("highs_turbo", "cpp_engine", "cut_engine.cpp"),
    os.path.join("highs_turbo", "cpp_engine", "solver_callback.cpp"),
    os.path.join("highs_turbo", "cpp_engine", "bindings.cpp"),
]

extra_compile_args = [
    "-std=c++20",
    "-O3",
    "-funroll-loops",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]

extra_link_args = []
if sys.platform == "darwin":
    extra_link_args.extend(["-framework", "Security"])


ext_modules = [
    Extension(
        "highs_turbo._compiled_engine",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["gmpxx", "gmp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
        optional=True,
    ),
] if sys.platform == "darwin" else []

setup(
    name="highs-turbo",
    version="0.2.0",
    description="Transparent linear programming acceleration for SciPy and HiGHS",
    long_description=open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(BASE_DIR, "README.md")) else "",
    long_description_content_type="text/markdown",
    author="Neural-Surrogate Team",
    packages=find_packages(include=["highs_turbo", "highs_turbo.*"]),
    ext_modules=ext_modules,
    package_data={"highs_turbo": ["default_weights.pt", "ising_policy.json"]},
    python_requires=">=3.10",
    install_requires=["numpy>=2.0", "scipy>=1.13", "networkx>=3.2", "highspy>=1.11", "dwave-graphs>=1.0"],
    extras_require={"ml": ["torch>=2.0"], "ising": ["dwave-samplers>=1.8"],
                    "test": ["pytest>=7", "torch>=2.0", "dwave-samplers>=1.8"]},
)
