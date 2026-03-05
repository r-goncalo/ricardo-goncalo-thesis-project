from setuptools import find_packages, setup

# to install module localy:
# pip install -e <path>  

setup(
    name='automl',
    packages=find_packages(include=['automl']),

    description='An AutoRL library',
    author='Ricardo Gonçalo',

    use_scm_version={
        "root": "..",              # where the .git folder is
        "relative_to": __file__,   # path reference
        "write_to": "project/automl/_version.py",
    },    

    setup_requires=["setuptools_scm"],

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