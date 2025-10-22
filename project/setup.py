from setuptools import find_packages, setup

# to install module localy:
# pip install -e <path>  

setup(
    name='automl',
    packages=find_packages(include=['automl']),
    version='0.1.0',
    description='An AutoRL library',
    author='Ricardo Gonçalo',
    install_requires = [
        'numpy',
        'optuna',
        'matplotlib',
        'torch',
        'gymnasium',
        'pandas',
        'wandb',
        'pettingzoo',
        'pygame',
        'scikit-learn',
        'plotly',
        'nbformat'
        
    ]
)