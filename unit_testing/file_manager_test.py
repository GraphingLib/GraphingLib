import unittest

from graphinglib.file_manager import FileLoader


class TestFileLoader(unittest.TestCase):
    def test_resource_path(self):
        filename = "a_certain_file"
        loader = FileLoader(filename)
        self.assertEqual(loader._file_name[-(len(filename) + 5) :], f"/{filename}.yml")
