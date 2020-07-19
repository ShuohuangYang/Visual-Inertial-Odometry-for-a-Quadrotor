#%%
from distutils.core import setup
from os.path import isdir
from itertools import product

# Gather our flightsim and any projXX packages that happen to exist.
all_packages = ['flightsim']
all_packages.extend([f"proj{a}_{b}" for (a,b) in product(range(10), repeat=2)])
packages = list(filter(isdir, all_packages))

#%%
setup(
    name='aerial_robotics',
    packages=['proj2_2'],
    version='0.1',
    install_requires=[
            'pyyaml',
            'opencv-python',
            'matplotlib',
            'numpy',
            'scipy'])
