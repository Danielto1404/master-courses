from setuptools import setup, find_packages

setup(
    name="metafora",
    packages=find_packages(
        include=["metafora"],
        exclude=["datasets", "labs", "papers", "slides"]
    ),
    version="0.0.10",
    license="MIT",
    description="This repository contains basic Meta-Learning and Auto ML approaches, also this repo provides "
                "materials from Auto ML course at ITMO University",
    author="Daniil Korolev",
    url="https://github.com/Danielto1404/auto-ml",
    keywords=[
        "Auto Machine Learning",
        "Meta-Learning",
        "Bayesian Optimization",
        "Hyper-Parameters optimization",
        "Machine Learning"
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)
