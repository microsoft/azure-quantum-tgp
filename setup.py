# Copyright (c) Microsoft Corporation. All rights reserved.

from setuptools import find_packages, setup


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    from importlib.util import module_from_spec, spec_from_file_location
    import os

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("tgp")


with open("requirements.txt") as f:
    install_requires = f.read().split()


setup(
    name="azure-quantum-tgp",
    author="Microsoft Quantum Hardware",
    author_email="noreply@microsoft.com",
    description="Topogap protocol code",
    license="Proprietary",
    packages=find_packages("."),
    version=version,
    python_requires=">=3.8",
    cmdclass=cmdclass,
    setup_requires=[],
    install_requires=install_requires,
    extras_require={
        "test": [
            "notebook",
            "jupytext",
            "pre-commit",
            "nbmake",
            "coverage",
        ],
    },
    package_data={"tgp": ["tgp/*.yaml"]},
    include_package_data=True,
)
