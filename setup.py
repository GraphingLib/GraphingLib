from setuptools import setup, find_packages

setup(
    name='GraphingLib',
    version='0.0.1',
    description='An object oriented wrapper combining the functionalities of Matplotlib and Scipy',
    url='https://github.com/yalap13/GraphingLib.git',
    packages=find_packages(),
    install_requires=['numpy','scipy','matplotlib','pyyaml'],
    include_package_data=True
)
