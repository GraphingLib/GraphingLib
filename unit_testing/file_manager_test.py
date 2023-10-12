import unittest
from unittest.mock import mock_open, patch

from graphinglib.file_manager import FileDeleter, FileLoader, FileSaver, FileUpdater


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
