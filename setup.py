from setuptools import setup

setup(
    name="varbench",
    version="0.1.0",
    install_requires=[
        "latexcompiler",
        ],
    packages=["varbench"],
    package_data={"varbench": ["data/*"]},
)