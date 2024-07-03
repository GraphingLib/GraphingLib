from os import listdir, mkdir, path, remove
from warnings import warn

import yaml
from matplotlib import pyplot as plt
from platformdirs import user_config_dir

# Force yaml to ignore aliases when dumping
yaml.Dumper.ignore_aliases = lambda *args: True  # type: ignore


class FileLoader:
    """
    This class implements the file loader for the default styles files.
    """

    def __init__(self, file_name: str) -> None:
        self._config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        if "custom_styles" not in listdir(self._config_dir):
            mkdir(f"{self._config_dir}/custom_styles")
        self._file_name = file_name
        self._file_location_defaults = (
            f"{path.dirname(__file__)}/default_styles/{self._file_name}.yml"
        )
        self._file_location_customs = (
            f"{self._config_dir}/custom_styles/{self._file_name}.yml"
        )

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
        try:
            assert info is not None
        except AssertionError:
            raise TypeError(
                f"Could not load the file {self._file_name}.yml. Please check that the file is in the correct format."
            )
        return info


class FileSaver:
    """
    This class implements the file saver for the user styles files.
    """

    def __init__(self, file_name: str, style_prefs: dict) -> None:
        self._config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        if "custom_styles" not in listdir(self._config_dir):
            mkdir(f"{self._config_dir}/custom_styles")
        self._file_name = file_name
        self._style_prefs = style_prefs
        self._save_location = f"{self._config_dir}/custom_styles/{self._file_name}.yml"

    def save(self) -> None:
        with open(self._save_location, "w") as file:
            yaml.dump(self._style_prefs, file)
        print(f"Style saved to {self._save_location}")


class FileDeleter:
    """
    This class implements the file deleter for the user styles files.
    """

    def __init__(self, file_name: str) -> None:
        self._config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        if "custom_styles" not in listdir(self._config_dir):
            mkdir(f"{self._config_dir}/custom_styles")
        self._file_name = file_name
        self._file_location = f"{self._config_dir}/custom_styles/{self._file_name}.yml"

    def delete(self) -> None:
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
        self._config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        if "custom_styles" not in listdir(self._config_dir):
            mkdir(f"{self._config_dir}/custom_styles")
        self._file_name = file_name
        self._file_location = f"{self._config_dir}/custom_styles/{self._file_name}.yml"
        self._plain_style_location = (
            f"{path.dirname(__file__)}/default_styles/plain.yml"
        )

    def update(self) -> None:
        """
        Checks every key and subkey in the plain style file. If it doesn't exist in the user's style file, this method adds it to the user's style file with the plain style's value.
        """
        with open(self._file_location, "r") as file:
            user_info = yaml.safe_load(file)
        with open(self._plain_style_location, "r") as file:
            plain_info = yaml.safe_load(file)
        for key in plain_info:
            if key == "rc_params":
                pass
            elif key not in user_info:
                user_info[key] = plain_info[key]
            else:
                for subkey in plain_info[key]:
                    if subkey not in user_info[key]:
                        user_info[key][subkey] = plain_info[key][subkey]
        with open(self._file_location, "w") as file:
            yaml.dump(user_info, file)


def get_colors(figure_style: str = "plain") -> list[str]:
    """
    Returns a list of colors from the specified style (user created, GL, or Matplotlib style).

    Parameters
    ----------
    figure_style : str
        The name of the style to use.

    Returns
    -------
    list[str]
        A list of colors.
    """
    if figure_style == "matplotlib":
        figure_style = "default"
    try:
        file_loader = FileLoader(figure_style)
        style_info = file_loader.load()
        colors = style_info["rc_params"]["axes.prop_cycle"]
        colors = colors[colors.find("[") + 1 : colors.find("]")].split(", ")
        colors = [color[1:-1] for color in colors]
    except FileNotFoundError:
        if figure_style in plt.style.available or figure_style == "default":
            with plt.style.context(figure_style):
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            raise FileNotFoundError(
                f"Could not find the file {figure_style}.yml or the matplotlib style {figure_style}."
            )
    return colors


def get_color(figure_style: str = "plain", color_number: int = 0) -> str:
    """
    Returns a color from the specified style (user created, GL, or Matplotlib style).

    Parameters
    ----------
    figure_style : str
        The name of the style to use.
    color_number : int
        The color cycle index of the color to return.

    Returns
    -------
    str
        A color.
    """
    colors = get_colors(figure_style)
    try:
        color = colors[color_number]
    except IndexError:
        raise IndexError(
            f"There are only {len(colors)} colors in the {figure_style} style (use index 0 to {len(colors) - 1})."
        )
    return color


def get_styles(
    customs: bool = True,
    gl: bool = True,
    matplotlib: bool = False,
    as_dict: bool = False,
) -> list[str]:
    """
    Returns a list or dict of available styles.

    If as_dict is True, dictionary is returned with keys "customs", "gl", and "matplotlib" (if applicable).

    Parameters
    ----------
    customs : bool
        Whether to include user created styles.
        Default is True.
    gl : bool
        Whether to include GL styles.
        Default is True.
    matplotlib : bool
        Whether to include Matplotlib styles.
        Default is False.

    Returns
    -------
    list[str] or dict[str, str]
        A list or dict of available styles.
    """
    customs_list = []
    gl_list = []
    matplotlib_list = []
    if customs:
        config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        if "custom_styles" in listdir(config_dir):
            customs_list = [
                file.split(".")[0] for file in listdir(f"{config_dir}/custom_styles")
            ]
    if gl:
        gl_list = [
            file.split(".")[0]
            for file in listdir(f"{path.dirname(__file__)}/default_styles")
        ]
    if matplotlib:
        matplotlib_list = plt.style.available
    if as_dict:
        style_dict = {
            "customs": customs_list,
            "gl": gl_list,
            "matplotlib": matplotlib_list,
        }
        # Remove empty lists
        style_dict = {k: v for k, v in style_dict.items() if v}
        return style_dict
    return customs_list + gl_list + matplotlib_list


def get_default_style() -> str:
    """
    Returns the default style.

    Returns
    -------
    str
        The default style.
    """

    # Check if the user has a default style
    config_dir = user_config_dir(
        appname="GraphingLib", roaming=True, ensure_exists=True
    )
    config_file = f"{config_dir}/config.yml"

    # If file exists, load the default style
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        config = config if config is not None else {}
        default_style = config["default_style"]

        # Ensure the style exists
        available_styles = get_styles(matplotlib=True)
        if default_style not in available_styles + ["matplotlib"]:
            warn(
                f"Default style '{default_style}' does not exist. Resetting to 'plain'."
            )
            default_style = "plain"

            # Reset the default style
            config["default_style"] = default_style
            with open(config_file, "w") as file:
                yaml.dump(config, file)

    except KeyError:
        # If file doesn't have the default style key, create it
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        config = config if config is not None else {}
        default_style = "plain"
        config["default_style"] = default_style
        with open(config_file, "w") as file:
            yaml.dump(config, file)

    except FileNotFoundError:
        # If file doesn't exist, create it with the default style
        with open(config_file, "w") as file:
            config = {"default_style": "plain"}
            yaml.dump(config, file)
        default_style = "plain"

    return default_style


def set_default_style(style: str) -> None:
    """
    Sets the default style.

    Parameters
    ----------
    style : str
        The name of the style to set as the default.
    """

    # Ensure the style exists
    available_styles = get_styles(matplotlib=True)
    if style not in available_styles + ["matplotlib"]:
        raise ValueError(f"Style '{style}' does not exist.")

    # Set the default style
    config_dir = user_config_dir(
        appname="GraphingLib", roaming=True, ensure_exists=True
    )
    config_file = f"{config_dir}/config.yml"

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        config = config if config is not None else {}
        old_style = config["default_style"]
        config["default_style"] = style
        with open(config_file, "w") as file:
            yaml.dump(config, file)
        print(f"Default style changed from '{old_style}' to '{style}'.")
    except KeyError:
        # If file doesn't have the default style key, create it
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        config = config if config is not None else {}
        config["default_style"] = style
        with open(config_file, "w") as file:
            yaml.dump(config, file)
        print(f"Default style set to '{style}'.")
    except FileNotFoundError:
        # If file doesn't exist or is empty, create it with the default style
        with open(config_file, "w") as file:
            config = {"default_style": style}
            yaml.dump(config, file)
