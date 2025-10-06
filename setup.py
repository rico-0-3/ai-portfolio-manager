"""
Setup script for AI Portfolio Manager.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

setup(
    name="ai-portfolio-manager",
    version="1.0.0",
    author="AI Portfolio Manager Team",
    author_email="riccardo.deangelis@mail.polimi.it",
    description="Sophisticated AI-powered portfolio management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rico-0-3/ai-portfolio-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "PyPortfolioOpt>=1.5.5",
        "pyyaml>=6.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.30.0",
            "vaderSentiment>=3.3.2",
            "stable-baselines3>=2.0.0",
            "cvxpy>=1.4.0",
            "TA-Lib>=0.4.28",
            "catboost>=1.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-portfolio=src.main:main",
        ],
    },
)
