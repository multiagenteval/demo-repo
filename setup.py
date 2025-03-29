from setuptools import setup, find_packages

setup(
    name="ai-eval-ecosystem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=1.5.0",
        "plotly>=5.13.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "gitpython>=3.1.0",
        "requests>=2.28.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        'console_scripts': [
            'eval-dashboard=ai_eval_ecosystem.dashboard.dashboard:main',
            'eval-analyze=ai_eval_ecosystem.core.agents.commit_analyzer:main',
            'eval-suggest=ai_eval_ecosystem.core.agents.test_suggester:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent evaluation ecosystem for ML model development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-eval-ecosystem",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
) 