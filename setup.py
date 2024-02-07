from setuptools import setup

setup(
    name="Pypsst",
    version="0.0.1a1",
    author="J. Scheffer",
    description="A Python package for writing error propagation equations",
    py_modules=["uncertainty-propagator"],
    package_dir={"": "src"},
    install_requires=["ipywidgets", "sympy", "uncertainties"],
)
