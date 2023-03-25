import unittest
import sys

sys.path.insert(0,'..')

from File_manager import FileLoader


class TestFileLoader(unittest.TestCase):
    def test_resource_path(self):
        filename = 'a_certain_file'
        loader = FileLoader(filename)
        self.assertEqual(loader.filename[-(len(filename)+5):], f"/{filename}.yml")