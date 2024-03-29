import unittest
from unittest.mock import MagicMock, patch

from matplotlib import pyplot as plt

from graphinglib.file_manager import (
    FileDeleter,
    FileLoader,
    FileSaver,
    FileUpdater,
    get_color,
    get_colors,
    get_styles,
)


class TestFileLoader(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        loader = FileLoader(filename)

        expected_path = f"{loader._config_dir}/{filename}.yml"
        self.assertEqual(loader._file_location_customs, expected_path)


class TestFileSaver(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        saver = FileSaver(filename, {"color": "red"})

        expected_path = f"{saver._config_dir}/{filename}.yml"
        self.assertEqual(saver._save_location, expected_path)


class TestFileDeleter(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        deleter = FileDeleter(filename)

        expected_path = f"{deleter._config_dir}/{filename}.yml"
        self.assertEqual(deleter._file_location, expected_path)


class TestFileUpdater(unittest.TestCase):
    def test_path(self):
        filename = "a_certain_file"
        updater = FileUpdater(filename)

        expected_path = f"{updater._config_dir}/{filename}.yml"
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
