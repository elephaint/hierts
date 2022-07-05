import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name="hierts",
	version="1.0",
	description="Hierarchical time series reconciliation",
	author="Olivier Sprangers",
	author_email="o.r.sprangers@uva.nl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elephaint/hierts",
    packages=setuptools.find_packages(where="src"),
    classifiers=[
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent"],
    python_requires='>=3.7',
    install_requires=["pandas>=1.3.5",
                      "numba>=0.53.0"])