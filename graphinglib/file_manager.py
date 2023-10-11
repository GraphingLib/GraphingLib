from os import mkdir, path, remove

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
        if not path.exists(self._config_dir):
            mkdir(self._config_dir)
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
        if not path.exists(self._config_dir):
            mkdir(self._config_dir)
        try:
            remove(self._file_location)
            print(f"Style deleted from {self._file_location}")
        except FileNotFoundError:
            print(f"Could not find the file {self._file_name}.yml.")


class FileUpdater:
    """
    This class implements the file updater for the user styles files (runs when a style is used after a GL update).
    """

    def __init__(self, file_name: str) -> None:
        self._config_dir = user_config_dir(appname="GraphingLib", roaming=True)
        self._file_name = file_name
        self._file_location = f"{self._config_dir}/{self._file_name}.yml"
        self._plain_style_location = (
            f"{path.dirname(__file__)}/default_styles/plain.yml"
        )

    def update(self) -> None:
        """
        Checks every key and subkey in the plain style file. If it doesn't exist in the user's style file, this method adds it to the user's style file with the plain style's value.
        """
        if not path.exists(self._config_dir):
            mkdir(self._config_dir)
        with open(self._file_location, "r") as file:
            user_info = yaml.safe_load(file)
        with open(self._plain_style_location, "r") as file:
            plain_info = yaml.safe_load(file)
        for key in plain_info:
            if key not in user_info:
                user_info[key] = plain_info[key]
            else:
                for subkey in plain_info[key]:
                    if subkey not in user_info[key]:
                        user_info[key][subkey] = plain_info[key][subkey]
        with open(self._file_location, "w") as file:
            yaml.dump(user_info, file)
