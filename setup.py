from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="industrial-predictive-maintenance",
    version="1.0.0",
    author="IEEE IES Industrial AI Lab",
    description=(
        "AI pipelines for industrial predictive maintenance: "
        "RUL prediction and fault detection on real sensor data."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IEEE-IES-Industrial-AI-Lab/Industrial-Predictive-Maintenance",
    packages=find_packages(exclude=["notebooks", "benchmarks", "tests"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "predictive maintenance",
        "remaining useful life",
        "fault detection",
        "anomaly detection",
        "industrial AI",
        "IEEE IES",
        "CMAPSS",
        "bearing fault",
    ],
)
