from setuptools import setup, find_packages

setup(
    name='GraphingLib',
    version='0.0.2',
    description='A simpler way to view data in Python',
    url='https://github.com/yalap13/GraphingLib.git',
    packages=find_packages(),
    install_requires=['numpy','scipy','matplotlib','pyyaml'],
    include_package_data=True
)
