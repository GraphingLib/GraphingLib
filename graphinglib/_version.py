from setuptools_scm import get_version

try:
    version = get_version("..", relative_to=__file__)
except LookupError as e:
    version = "0.0.0.unknown"

__version__ = "1.4.0"
