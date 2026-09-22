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
    version="0.3.0",
    description="LP acceleration and checkable Ising, QUBO, and Max-Cut bounds with HiGHS",
    long_description=open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(BASE_DIR, "README.md")) else "",
    long_description_content_type="text/markdown",
    author="Neural-Surrogate Team",
    url="https://github.com/steph4n-gh/libhighs_turbo",
    project_urls={
        "Documentation": "https://github.com/steph4n-gh/libhighs_turbo#readme",
        "API contract": "https://github.com/steph4n-gh/libhighs_turbo/blob/main/API_CONTRACT.md",
        "Source": "https://github.com/steph4n-gh/libhighs_turbo",
        "Issues": "https://github.com/steph4n-gh/libhighs_turbo/issues",
        "Changelog": "https://github.com/steph4n-gh/libhighs_turbo/blob/main/CHANGELOG.md",
        "Releases": "https://github.com/steph4n-gh/libhighs_turbo/releases",
    },
    license="MIT",
    license_files=["LICENSE"],
    keywords=["optimization", "linear-programming", "highs", "scipy", "ising",
              "qubo", "max-cut", "certificates"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(include=["highs_turbo", "highs_turbo.*"]),
    ext_modules=ext_modules,
    package_data={"highs_turbo": ["default_weights.pt", "ising_policy.json"]},
    entry_points={"console_scripts": ["highs-turbo=highs_turbo.__main__:main"]},
    python_requires=">=3.10",
    install_requires=["numpy>=2.0", "scipy>=1.13", "networkx>=3.2", "highspy>=1.11"],
    extras_require={"ml": ["torch>=2.4.1"], "ising": ["dwave-samplers>=1.8", "dwave-graphs>=1.0"],
                    "test": ["pytest>=7", "torch>=2.4.1", "dwave-samplers>=1.8", "dwave-graphs>=1.0"]},
)
