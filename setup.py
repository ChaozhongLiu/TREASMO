import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="treasmo",
    version="1.0.6",
    author="Chaozhong Liu",
    author_email="czliubioinfo@gmail.com",
    description="Transcription regulation analysis toolkit for single-cell multi-omics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChaozhongLiu/TREASMO",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ),
)
