"""Setup script for Legal Policy Explainer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().splitlines()
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="legal-policy-explainer",
    version="0.1.0",
    description="LLM-powered assistant for explaining legal policies and regulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Team Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/legal-policy-explainer",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "legal-explainer=app:main",
        ],
    },
)
