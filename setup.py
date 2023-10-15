from setuptools import setup, find_packages

setup(
    name='GraphingLib',
    version='1.2.0',
    description='An object oriented wrapper combining the functionalities of Matplotlib and Scipy',
    author='Gustave Coulombe <magikgus@gmail.com>, Yannick Lapointe <yannicklapointe77@gmail.com>'
    url='https://github.com/yalap13/GraphingLib.git',
    packages=find_packages(),
    install_requires=['numpy','scipy','matplotlib','pyyaml'],
    include_package_data=True
)
