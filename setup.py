import os
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CPP_DIR = os.path.join(BASE_DIR, "highs_turbo", "cpp_engine")

include_dirs = [
    CPP_DIR,
    os.path.join(CPP_DIR, "include"),
    "/opt/homebrew/include",
    "/opt/homebrew/include/highs",
    "/opt/homebrew/Cellar/gmp/6.3.0/include",
    "/usr/local/include",
    "/usr/include",
]
include_dirs.extend([
    p for p in [
        "/opt/homebrew/Cellar/python@3.14/3.14.4_1/Frameworks/Python.framework/Versions/3.14/include/python3.14",
        "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/include/python3.14",
    ] if os.path.isdir(p)
])
include_dirs = [d for d in include_dirs if os.path.isdir(d)]

library_dirs = [
    "/opt/homebrew/lib",
    "/opt/homebrew/Cellar/gmp/6.3.0/lib",
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


class OptionalBuildExt(build_ext):
    """Allows package installation to proceed gracefully with pure-Python fallback if C++ build fails."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            print(f"[highs_turbo] WARNING: Failed to compile C++ extension {ext.name}: {exc}")
            print("[highs_turbo] highs_turbo will continue with pure-Python fallback mode.")


ext_modules = [
    Extension(
        "highs_turbo._compiled_engine",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["gmp", "gmpxx", "highs"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="highs-turbo",
    version="0.1.0",
    description="High-performance Neural-Surrogate Cutting Plane Plugin for HiGHS and SciPy",
    long_description=open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(BASE_DIR, "README.md")) else "",
    long_description_content_type="text/markdown",
    author="Neural-Surrogate Team",
    packages=find_packages(include=["highs_turbo", "highs_turbo.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
    python_requires=">=3.8",
    install_requires=["numpy", "scipy"],
)
