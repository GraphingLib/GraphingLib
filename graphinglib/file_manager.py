from os import path

import yaml


class FileLoader:
    def __init__(self, file_name: str) -> None:
        self._file_name = file_name
        self._file_name = self._resource_path()

    def load(self) -> dict:
        with open(self._file_name, "r") as file:
            info = yaml.safe_load(file)
        return info

    def _resource_path(self) -> str:
        return f"{path.dirname(__file__)}/default_styles/{self._file_name}.yml"
