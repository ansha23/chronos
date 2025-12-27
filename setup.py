from setuptools import setup, find_packages

setup(
    name="chronos",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "chronos=chronos.main:main",
        ],
    },
)

