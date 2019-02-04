import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="response_logic",
    version="0.0.1",
    author="Torsten Gross",
    author_email="gross.torsten1@gmail.com",
    description="Infer a directed network from perturbation response data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GrossTor/response-logic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
	'numpy',
	'sklearn',
	'networkx',
	'pandas',
    ],
)

