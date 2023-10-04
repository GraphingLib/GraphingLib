from os import path

import yaml


class FileLoader:
    """
    This class implements the file loader for the default styles files.
    """

    def __init__(self, file_name: str) -> None:
        self._file_name = file_name
        self._file_name = self._resource_path()

    def load(self) -> dict:
        with open(self._file_name, "r") as file:
            info = yaml.safe_load(file)
        return info

    def _resource_path(self) -> str:
        return f"{path.dirname(__file__)}/default_styles/{self._file_name}.yml"


class FileSaver:
    """
    This class implements the file saver for the user styles files.
    """

    def __init__(self, file_name: str, style_prefs: dict) -> None:
        self._file_name = file_name
        self._style_prefs = style_prefs
        self._file_name = self._resource_path()

    def save(self) -> None:
        with open(self._file_name, "w") as file:
            yaml.dump(self._style_prefs, file)

    def _resource_path(self) -> str:
        return f"{path.dirname(__file__)}/default_styles/{self._file_name}.yml"
