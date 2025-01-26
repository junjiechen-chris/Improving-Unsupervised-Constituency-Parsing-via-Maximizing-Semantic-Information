from setuptools import setup, find_packages

setup(
    name="parsing_by_maxseminfo",                 # Package name
    version="0.1.0",                   # Package version
    author="Junjie Chen",                # Author's name
    author_email="chris.jjc@outlook.com",  # Author's email
    description="A code release for Improving Unsupervised Constituency Parsing via Maximizing Semantic Information",
    long_description=open("README.md").read(),  # Detailed description
    long_description_content_type="text/markdown",
    url="https://github.com/junjiechen-chris/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information.git",  # URL of your project
    packages=find_packages(),          # Automatically find packages
    install_requires=[                 # Dependencies
        "numpy", 
        "requests",
        "matplotlib",
        "wandb",
        "transformers==4.48",
        "lightning==2.4.0",
        "nltk",
        "PyStemmer==2.2.0",
        "spacy",
        "prettytable",
        "tensorboard",
        "easydict",
        "triton==3.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',           # Minimum Python version
)