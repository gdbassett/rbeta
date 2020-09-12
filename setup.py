import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rbeta", # Replace with your own username
    version="0.0.7",
    author="Gabriel Bassett, Mayisha Zeb Nakib",
    author_email="gabe@infosecanalytics.com",
    description="Calculate the rBeta correlation between fMRI voxels. Based on https://github.com/remolek/NFC.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TBD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)