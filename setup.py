from setuptools import setup, find_packages

setup(
    name="nlp-well-report-analyzer",
    version="1.0.0",
    author="Ing. Kelvin Cabrera",
    description="NLP system for analyzing well completion reports and extracting structured information",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "requests>=2.31.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "well-analyzer=app:main",
        ],
    },
)
