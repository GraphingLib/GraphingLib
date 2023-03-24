import sys
from os import path

import yaml


class FileLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.filename = self.resource_path()
    
    def load(self):
        with open(self.filename, 'r') as file:
            info = yaml.safe_load(file)
        return info
    
    def resource_path(self):
        return f"{path.dirname(__file__)}/Default_styles/{self.filename}.yml"