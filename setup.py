from setuptools import setup

setup(
    name="Uncertainty Propagator",
    version="0.0.1a3",
    author="J. Scheffer",
    description="A Python package for writing error propagation equations",
    py_modules=["uncertainty-propagator"],
    package_dir={"": "src"},
    install_requires=["ipywidgets", "sympy", "uncertainties", "numpy", "tqdm"],
)
