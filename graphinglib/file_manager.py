from os import path

import yaml


class FileLoader:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.filename = self.resource_path()

    def load(self) -> dict:
        with open(self.filename, "r") as file:
            info = yaml.safe_load(file)
        return info

    def resource_path(self) -> str:
        return f"{path.dirname(__file__)}/default_styles/{self.filename}.yml"
