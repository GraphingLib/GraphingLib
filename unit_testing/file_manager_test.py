import unittest
from os import path
from unittest.mock import MagicMock, patch

import yaml
from matplotlib import pyplot as plt

from graphinglib.file_manager import (
    FileDeleter,
    FileLoader,
    FileSaver,
    FileUpdater,
    get_color,
    get_colors,
    get_default_style,
    get_styles,
    set_default_style,
)


class TestFileLoader(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        loader = FileLoader(filename)

        expected_path = f"{loader._config_dir}/custom_styles/{filename}.yml"
        self.assertEqual(loader._file_location_customs, expected_path)


class TestFileSaver(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        saver = FileSaver(filename, {"color": "red"})

        expected_path = f"{saver._config_dir}/custom_styles/{filename}.yml"
        self.assertEqual(saver._save_location, expected_path)


class TestFileDeleter(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        deleter = FileDeleter(filename)

        expected_path = f"{deleter._config_dir}/custom_styles/{filename}.yml"
        self.assertEqual(deleter._file_location, expected_path)


class TestFileUpdater(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        updater = FileUpdater(filename)

        expected_path = f"{updater._config_dir}/custom_styles/{filename}.yml"
        self.assertEqual(updater._file_location, expected_path)


class TestGetColors(unittest.TestCase):
    @patch("graphinglib.file_manager.FileLoader")
    def test_get_colors(self, mock_file_loader):
        mock_file_loader_instance = MagicMock()
        mock_file_loader.return_value = mock_file_loader_instance
        mock_file_loader_instance.load.return_value = {
            "rc_params": {
                "axes.prop_cycle": "cycler('cycler', ['red', 'green', 'blue'])"
            }
        }
        colors = get_colors("plain")
        self.assertEqual(colors, ["red", "green", "blue"])
        mock_file_loader.assert_called_once_with("plain")
        mock_file_loader_instance.load.assert_called_once()


class TestGetColor(unittest.TestCase):
    @patch("graphinglib.file_manager.FileLoader")
    def test_get_color(self, mock_file_loader):
        mock_file_loader_instance = MagicMock()
        mock_file_loader.return_value = mock_file_loader_instance
        mock_file_loader_instance.load.return_value = {
            "rc_params": {
                "axes.prop_cycle": "cycler('color', ['red', 'green', 'blue'])"
            }
        }
        color = get_color("plain", 1)
        self.assertEqual(color, "green")
        mock_file_loader.assert_called_once_with("plain")
        mock_file_loader_instance.load.assert_called_once()


class TestGetStyles(unittest.TestCase):
    def test_get_styles(self):
        styles = get_styles(matplotlib=True, gl=False, customs=False)
        self.assertListEqual(styles, plt.style.available)

        # Test with dictionary
        styles = get_styles(matplotlib=True, gl=True, customs=True, as_dict=True)
        self.assertIsInstance(styles, dict)
        self.assertIn("plain", styles["gl"])
        self.assertListEqual(styles["matplotlib"], plt.style.available)


class TestGetDefaultStyle(unittest.TestCase):
    @patch("yaml.load")
    @patch("yaml.dump")
    def test_get_default_style(self, mock_dump, mock_load):
        mock_load.return_value = {"default_style": "plain"}
        style = get_default_style()
        self.assertEqual(style, "plain")
        mock_load.assert_called_once()
        mock_dump.assert_not_called()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_get_default_style_no_file(self, mock_dump, mock_load):
        mock_load.side_effect = FileNotFoundError
        style = get_default_style()
        self.assertEqual(style, "plain")
        mock_load.assert_called_once()
        mock_dump.assert_called_once()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_get_default_style_empty_file(self, mock_dump, mock_load):
        mock_load.return_value = None
        style = get_default_style()
        self.assertEqual(style, "plain")
        self.assertEqual(mock_load.call_count, 2)
        mock_dump.assert_called_once()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_get_default_style_non_existent_style(self, mock_dump, mock_load):
        mock_load.return_value = {"default_style": "non_existent_style"}
        with self.assertWarns(UserWarning):
            style = get_default_style()
        self.assertEqual(style, "plain")
        mock_load.assert_called_once()
        mock_dump.assert_called_once()


class TestSetDefaultStyle(unittest.TestCase):
    @patch("yaml.load")
    @patch("yaml.dump")
    def test_set_default_style(self, mock_dump, mock_load):
        set_default_style("dark")
        mock_load.assert_called_once()
        mock_dump.assert_called_once()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_set_default_style_no_file(self, mock_dump, mock_load):
        mock_load.side_effect = FileNotFoundError
        set_default_style("dark")
        mock_load.assert_called_once()
        mock_dump.assert_called_once()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_set_default_style_empty_file(self, mock_dump, mock_load):
        mock_load.return_value = None
        set_default_style("dark")
        self.assertEqual(mock_load.call_count, 2)
        mock_dump.assert_called_once()

    @patch("yaml.load")
    @patch("yaml.dump")
    def test_set_default_style_non_existent_style(self, mock_dump, mock_load):
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            set_default_style("non_existent_style")
        mock_load.assert_not_called()
        mock_dump.assert_not_called()


class TestStyles(unittest.TestCase):
    def test_styles_are_consistent(self):
        # Start by loading the plain style
        file_plain = f"{path.dirname(__file__)}/../graphinglib/default_styles/plain.yml"
        with open(file_plain, "r") as file:
            plain_dict = yaml.safe_load(file)

        # For each style in built-in styles, check that the style is consistent (same keys and subkeys) with the plain style
        for style in get_styles(gl=True, customs=False, matplotlib=False):
            file_style = (
                f"{path.dirname(__file__)}/../graphinglib/default_styles/{style}.yml"
            )
            with open(file_style, "r") as file:
                style_dict = yaml.safe_load(file)
            res = self._compare_dict_structure(plain_dict, style_dict)
            if not res[0]:
                if res[1] is not None:
                    self.fail(
                        f"Style {style} is inconsistent with plain style: in {res[1]}, check for following keys: {res[2]}"
                    )
                else:
                    self.fail(
                        f"Style {style} is inconsistent with plain style: check for following keys: {res[2]}"
                    )

    def _compare_dict_structure(self, dict1, dict2):
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        if keys1 != keys2:
            return (False, None, keys1.symmetric_difference(keys2))
        for key in keys1:
            if isinstance(dict1[key], dict):
                res = self._compare_dict_structure(dict1[key], dict2[key])
                if not res[0]:
                    return (False, key, res[2])
        return (True, None, None)


if __name__ == "__main__":
    unittest.main()
