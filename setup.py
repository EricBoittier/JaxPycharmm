from setuptools import find_packages, setup

setup(
    name="physnetjax",
    version="0.1.0",
    author="Eric Boittier",
    author_email="my.email@example.com",
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/physnet_jax_pycharmm",
    package_dir={"physnetjax": "physnetjax"},
    packages=["physnetjax"],
    install_requires=[
        # "jax",
        # "pycharmm",
        # Add other dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
