"""
V7P3RAI v4.0 Setup Script
Multi-Agent Chess AI Enhancement Layer
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name='v7p3r-chess-ai',
    version='4.0.0',
    description='Multi-Agent Chess AI Enhancement Layer for V7P3R Chess Engine',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    author='V7P3R Development Team',
    author_email='dev@v7p3r.com',
    url='https://github.com/pssnyder/v7p3r-chess-ai',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    python_requires='>=3.11',
    
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'python-chess>=1.999',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'stockfish>=3.28.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'training': [
            'wandb>=0.15.0',
            'tensorboard>=2.13.0',
            'jupyter>=1.0.0',
        ],
        'performance': [
            'numba>=0.57.0',
            'cython>=0.29.0',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Games/Entertainment :: Board Games',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    
    keywords='chess ai machine-learning deep-learning reinforcement-learning multi-agent',
    
    entry_points={
        'console_scripts': [
            'v7p3rai-train=scripts.stage1_train_themes:main',
            'v7p3rai-validate=scripts.validate_agents:main',
        ],
    },
)
