from os import path, remove

import yaml
from platformdirs import user_config_dir


class FileLoader:
    """
    This class implements the file loader for the default styles files.
    """

    def __init__(self, file_name: str) -> None:
        self._config_dir = user_config_dir(appname="GraphingLib", roaming=True)
        self._file_name = file_name
        self._file_location_defaults = (
            f"{path.dirname(__file__)}/default_styles/{self._file_name}.yml"
        )
        self._file_location_customs = f"{self._config_dir}/{self._file_name}.yml"

    def load(self) -> dict:
        try:
            with open(self._file_location_customs, "r") as file:
                info = yaml.safe_load(file)
        except FileNotFoundError:
            try:
                with open(self._file_location_defaults, "r") as file:
                    info = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find the file {self._file_name}.yml."
                )
        return info


class FileSaver:
    """
    This class implements the file saver for the user styles files.
    """

    def __init__(self, file_name: str, style_prefs: dict) -> None:
        self._config_dir = user_config_dir(appname="GraphingLib", roaming=True)
        self._file_name = file_name
        self._style_prefs = style_prefs
        self._save_location = f"{self._config_dir}/{self._file_name}.yml"

    def save(self) -> None:
        # create the config directory if it doesn't exist
        if not path.exists(self._config_dir):
            from os import mkdir

            mkdir(self._config_dir)
        # save the style to the user's config directory
        with open(self._save_location, "w") as file:
            yaml.dump(self._style_prefs, file)
        print(f"Style saved to {self._save_location}")


class FileDeleter:
    """
    This class implements the file deleter for the user styles files.
    """

    def __init__(self, file_name: str) -> None:
        self._config_dir = user_config_dir(appname="GraphingLib", roaming=True)
        self._file_name = file_name
        self._file_location = f"{self._config_dir}/{self._file_name}.yml"

    def delete(self) -> None:
        # delete the style from the user's config directory

        remove(self._file_location)
        print(f"Style deleted from {self._file_location}")
