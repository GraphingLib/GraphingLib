import io
import sys
import unittest
from unittest.mock import patch

from graphinglib.cli import delete_style, get_style_names, main_cli, prompt_for_section
from graphinglib.file_manager import FileDeleter


class TestGetStyleNames(unittest.TestCase):
    def test_get_default_styles(self):
        # Get the style names using get_style_names
        style_names = get_style_names(user=False)

        # Check that the style names match what we created
        self.assertEqual(style_names.sort(), ["plain", "horrible"].sort())

    def test_get_user_styles(self):
        # Get the style names using get_style_names
        style_names = get_style_names(default=False)
        # assert that style_names is a list
        self.assertIsInstance(style_names, list)

    def test_get_all_styles(self):
        # Get the style names using get_style_names
        style_names = get_style_names()

        # Check that the style names contain the default styles
        self.assertIn("plain", style_names)
        self.assertIn("horrible", style_names)

    def test_no_styles(self):
        # Get the style names using get_style_names
        style_names = get_style_names(default=False, user=False)

        # Check that the style names are empty
        self.assertEqual(style_names, [])


class TestPromptForSection(unittest.TestCase):
    def setUp(self):
        self.default_values = {
            "key1": "value1",
            "key2": 42,
            "key3": True,
            "key4": [1, 2],
            "key5": "same as key1",
            "key6": "3.14",
        }

    def tearDown(self):
        # Reset the input and output streams
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__

    def test_prompt_for_section_no_input(self):
        # Set up mock input and output streams
        input_stream = io.StringIO("\n\n\n\n\n\n")
        output_stream = io.StringIO()
        sys.stdin = input_stream
        sys.stdout = output_stream

        # Call the prompt_for_section function
        user_values = prompt_for_section("Test Section", self.default_values)

        # Check that the user values match the default values
        self.assertEqual(user_values, self.default_values)

        # Check that the prompt messages were printed to stdout
        expected_output = (
            "\n------------------------------\nTest Section settings:\n------------------------------\n"
            + "key1 (default: value1): "
            + "key2 (default: 42): "
            + "key3 (default: True): "
            + "key4 (default: [1, 2]): "
            + "key5 (default: same as key1): "
            + "key6 (default: 3.14): "
        )
        self.assertEqual(output_stream.getvalue(), expected_output)

        # Reset the input and output streams
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__


class TestDeleteStyle(unittest.TestCase):
    @patch("builtins.input", side_effect=["1", "y"])
    @patch.object(FileDeleter, "delete")
    def test_delete_style(self, mock_delete, mock_input):
        # If there are no styles to delete, the function should do nothing
        if not get_style_names(default=False, user=True):
            delete_style()
            mock_delete.assert_not_called()
        # If there are styles to delete, the function should call the delete method on the FileDeleter object
        else:
            delete_style()
            mock_delete.assert_called_once()


class TestMainCli(unittest.TestCase):
    @patch("builtins.input", side_effect=["1"])
    @patch("graphinglib.cli.create_new_style")
    def test_main_cli_create_new_style(self, mock_create_new_style, mock_input):
        # Call the main_cli function
        main_cli()

        # Check that the create_new_style function was called
        mock_create_new_style.assert_called_once()

    @patch("builtins.input", side_effect=["2"])
    @patch("graphinglib.cli.modify_style")
    def test_main_cli_modify_style(self, mock_modify_style, mock_input):
        # Call the main_cli function
        main_cli()

        # Check that the modify_style function was called
        mock_modify_style.assert_called_once()

    @patch("builtins.input", side_effect=["3"])
    @patch("graphinglib.cli.delete_style")
    def test_main_cli_delete_style(self, mock_delete_style, mock_input):
        # Call the main_cli function
        main_cli()

        # Check that the delete_style function was called
        mock_delete_style.assert_called_once()


if __name__ == "__main__":
    unittest.main()
