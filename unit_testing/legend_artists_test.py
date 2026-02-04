import unittest

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from graphinglib.legend_artists import (
    LegendElement,
    LegendLine,
    LegendMarker,
    LegendPatch,
)


class TestLegendElement(unittest.TestCase):
    """Test the LegendElement protocol."""

    def test_legend_element_is_protocol(self):
        """Test that LegendElement is a runtime checkable protocol."""
        # This should raise an error since it's a protocol
        with self.assertRaises(TypeError):
            LegendElement()

    def test_legend_element_requires_handle_property(self):
        """Test that LegendElement requires a handle property."""

        class IncompleteElement:
            def __init__(self):
                self._label = "test"
                self._alpha = 1.0

        # Should not be considered a LegendElement without handle property
        self.assertFalse(isinstance(IncompleteElement(), LegendElement))

    def test_legend_element_with_handle_property(self):
        """Test that a class with handle property is recognized as LegendElement."""

        class CompleteElement(LegendElement):
            def __init__(self):
                self.label = "test"
                self.alpha = 1.0

            @property
            def handle(self) -> Artist:
                return Line2D([], [])

        # Should be considered a LegendElement with handle property
        self.assertTrue(isinstance(CompleteElement(), LegendElement))


class TestLegendLine(unittest.TestCase):
    """Test the LegendLine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.legend_line = LegendLine(
            label="Test Line",
            color="red",
            gap_color="blue",
            line_width=2.5,
            line_style="--",
            alpha=0.8,
        )

    def test_initialization(self):
        """Test proper initialization of LegendLine."""
        self.assertEqual(self.legend_line.label, "Test Line")
        self.assertEqual(self.legend_line.color, "red")
        self.assertEqual(self.legend_line.gap_color, "blue")
        self.assertEqual(self.legend_line.line_width, 2.5)
        self.assertEqual(self.legend_line.line_style, "--")
        self.assertEqual(self.legend_line.alpha, 0.8)

    def test_default_values(self):
        """Test default values for optional parameters."""
        line = LegendLine(label="Test", color="red")
        self.assertIsNone(line.gap_color)
        self.assertEqual(line.line_width, 2.0)
        self.assertEqual(line.line_style, "-")
        self.assertEqual(line.alpha, 1.0)

    def test_is_legend_element(self):
        """Test that LegendLine is recognized as a LegendElement."""
        self.assertTrue(isinstance(self.legend_line, LegendElement))

    def test_handle_property(self):
        """Test the handle property returns a Line2D object."""
        handle = self.legend_line.handle
        self.assertIsInstance(handle, Line2D)
        self.assertEqual(handle.get_label(), "Test Line")
        self.assertEqual(handle.get_color(), "red")
        self.assertEqual(handle.get_gapcolor(), "blue")
        self.assertEqual(handle.get_linewidth(), 2.5)
        self.assertEqual(handle.get_linestyle(), "--")
        self.assertEqual(handle.get_alpha(), 0.8)

    def test_color_validation(self):
        """Test color validation."""
        # Valid colors
        self.legend_line.color = "green"
        self.assertEqual(self.legend_line.color, "green")

        self.legend_line.color = "#FF0000"
        self.assertEqual(self.legend_line.color, "#FF0000")

        self.legend_line.color = (1.0, 0.5, 0.0)
        self.assertEqual(self.legend_line.color, (1.0, 0.5, 0.0))

        # None should be allowed for gap_color
        self.legend_line.gap_color = None
        self.assertIsNone(self.legend_line.gap_color)

        # Invalid color
        with self.assertRaises(ValueError):
            self.legend_line.color = "invalid_color"

    def test_line_width_validation(self):
        """Test line width validation."""
        # Valid values
        self.legend_line.line_width = 5.0
        self.assertEqual(self.legend_line.line_width, 5.0)

        self.legend_line.line_width = 0
        self.assertEqual(self.legend_line.line_width, 0)

        # Invalid values
        with self.assertRaises(TypeError):
            self.legend_line.line_width = "invalid"

        with self.assertRaises(ValueError):
            self.legend_line.line_width = -1

    def test_line_style_validation(self):
        """Test line style validation."""
        # Valid string styles
        for style in ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]:
            self.legend_line.line_style = style
            self.assertEqual(self.legend_line.line_style, style)

        # Valid tuple style
        tuple_style = (0, (5, 10))
        self.legend_line.line_style = tuple_style
        self.assertEqual(self.legend_line.line_style, tuple_style)

        # Invalid string style
        with self.assertRaises(ValueError):
            self.legend_line.line_style = "invalid_style"

        # Invalid tuple style
        with self.assertRaises(ValueError):
            self.legend_line.line_style = (0,)  # Too short

        with self.assertRaises(ValueError):
            self.legend_line.line_style = (0, "invalid")  # Non-numeric sequence

        with self.assertRaises(TypeError):
            self.legend_line.line_style = 123  # Wrong type

    def test_alpha_validation(self):
        """Test alpha validation."""
        # Valid values
        self.legend_line.alpha = 0.5
        self.assertEqual(self.legend_line.alpha, 0.5)

        self.legend_line.alpha = 0
        self.assertEqual(self.legend_line.alpha, 0)

        self.legend_line.alpha = 1
        self.assertEqual(self.legend_line.alpha, 1)

        # Invalid values
        with self.assertRaises(TypeError):
            self.legend_line.alpha = "invalid"

        with self.assertRaises(ValueError):
            self.legend_line.alpha = -0.1

        with self.assertRaises(ValueError):
            self.legend_line.alpha = 1.1


class TestLegendMarker(unittest.TestCase):
    """Test the LegendMarker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.legend_marker = LegendMarker(
            label="Test Marker",
            face_color="red",
            face_color_alt="blue",
            edge_color="black",
            edge_width=2.0,
            marker_size=8.0,
            marker_style="s",
            fill_style="left",
            alpha=0.7,
        )

    def test_initialization(self):
        """Test proper initialization of LegendMarker."""
        self.assertEqual(self.legend_marker.label, "Test Marker")
        self.assertEqual(self.legend_marker.face_color, "red")
        self.assertEqual(self.legend_marker.face_color_alt, "blue")
        self.assertEqual(self.legend_marker.edge_color, "black")
        self.assertEqual(self.legend_marker.edge_width, 2.0)
        self.assertEqual(self.legend_marker.marker_size, 8.0)
        self.assertEqual(self.legend_marker.marker_style, "s")
        self.assertEqual(self.legend_marker.fill_style, "left")
        self.assertEqual(self.legend_marker.alpha, 0.7)

    def test_default_values(self):
        """Test default values for optional parameters."""
        marker = LegendMarker(label="Test")
        self.assertIsNone(marker.face_color)
        self.assertIsNone(marker.face_color_alt)
        self.assertIsNone(marker.edge_color)
        self.assertEqual(marker.edge_width, 1.0)
        self.assertEqual(marker.marker_size, 6.0)
        self.assertEqual(marker.marker_style, "o")
        self.assertEqual(marker.fill_style, "full")
        self.assertEqual(marker.alpha, 1.0)

    def test_is_legend_element(self):
        """Test that LegendMarker is recognized as a LegendElement."""
        self.assertTrue(isinstance(self.legend_marker, LegendElement))

    def test_handle_property(self):
        """Test the handle property returns a Line2D object."""
        handle = self.legend_marker.handle
        self.assertIsInstance(handle, Line2D)
        self.assertEqual(handle.get_label(), "Test Marker")
        self.assertEqual(handle.get_markerfacecolor(), "red")
        self.assertEqual(handle.get_markerfacecoloralt(), "blue")
        self.assertEqual(handle.get_markeredgecolor(), "black")
        self.assertEqual(handle.get_markeredgewidth(), 2.0)
        self.assertEqual(handle.get_markersize(), 8.0)
        self.assertEqual(handle.get_marker(), "s")
        self.assertEqual(handle.get_fillstyle(), "left")
        self.assertEqual(handle.get_alpha(), 0.7)
        self.assertEqual(handle.get_linestyle(), "None")

    def test_color_validation(self):
        """Test color validation for all color properties."""
        # Valid colors
        for color_attr in ["face_color", "face_color_alt", "edge_color"]:
            setattr(self.legend_marker, color_attr, "green")
            self.assertEqual(getattr(self.legend_marker, color_attr), "green")

            setattr(self.legend_marker, color_attr, None)
            self.assertIsNone(getattr(self.legend_marker, color_attr))

        # Invalid color
        with self.assertRaises(ValueError):
            self.legend_marker.face_color = "invalid_color"

    def test_edge_width_validation(self):
        """Test edge width validation."""
        # Valid values
        self.legend_marker.edge_width = 3.5
        self.assertEqual(self.legend_marker.edge_width, 3.5)

        self.legend_marker.edge_width = 0
        self.assertEqual(self.legend_marker.edge_width, 0)

        # Invalid values
        with self.assertRaises(TypeError):
            self.legend_marker.edge_width = "invalid"

        with self.assertRaises(ValueError):
            self.legend_marker.edge_width = -1

    def test_marker_size_validation(self):
        """Test marker size validation."""
        # Valid values
        self.legend_marker.marker_size = 12.0
        self.assertEqual(self.legend_marker.marker_size, 12.0)

        self.legend_marker.marker_size = 0
        self.assertEqual(self.legend_marker.marker_size, 0)

        # Invalid values
        with self.assertRaises(TypeError):
            self.legend_marker.marker_size = "invalid"

        with self.assertRaises(ValueError):
            self.legend_marker.marker_size = -1

    def test_marker_style_validation(self):
        """Test marker style validation."""
        # Valid markers
        valid_markers = [
            ".",
            ",",
            "o",
            "v",
            "^",
            "<",
            ">",
            "s",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "D",
            "d",
        ]
        for marker in valid_markers:
            self.legend_marker.marker_style = marker
            self.assertEqual(self.legend_marker.marker_style, marker)

        # Test with MarkerStyle constructor - valid cases
        self.legend_marker.marker_style = 0  # tickleft
        self.assertEqual(self.legend_marker.marker_style, 0)

        # Invalid marker
        with self.assertRaises(ValueError):
            self.legend_marker.marker_style = "invalid_marker"

    def test_fill_style_validation(self):
        """Test fill style validation."""
        # Valid fill styles
        valid_styles = ["full", "left", "right", "bottom", "top"]
        for style in valid_styles:
            self.legend_marker.fill_style = style
            self.assertEqual(self.legend_marker.fill_style, style)

        # None should be allowed
        self.legend_marker.fill_style = None
        self.assertIsNone(self.legend_marker.fill_style)

        # Invalid fill style
        with self.assertRaises(ValueError):
            self.legend_marker.fill_style = "invalid_style"


class TestLegendPatch(unittest.TestCase):
    """Test the LegendPatch class."""

    def setUp(self):
        """Set up test fixtures."""
        self.legend_patch = LegendPatch(
            label="Test Patch",
            face_color=(1.0, 0, 1.0, 0.7),
            edge_color="red",  # RGBA tuple
            line_width=1.5,
            line_style="-.",
            hatch="//",
            alpha=0.7,
        )

    def test_initialization(self):
        """Test proper initialization of LegendPatch."""
        self.assertEqual(self.legend_patch.label, "Test Patch")
        self.assertEqual(self.legend_patch.face_color, (1.0, 0, 1.0, 0.7))
        self.assertEqual(self.legend_patch.edge_color, "red")
        self.assertEqual(self.legend_patch.line_width, 1.5)
        self.assertEqual(self.legend_patch.line_style, "-.")
        self.assertEqual(self.legend_patch.hatch, "//")
        self.assertEqual(self.legend_patch.alpha, 0.7)

    def test_default_values(self):
        """Test default values for optional parameters."""
        patch = LegendPatch(label="Test")
        self.assertIsNone(patch.face_color)
        self.assertIsNone(patch.edge_color)
        self.assertEqual(patch.line_width, 1.0)
        self.assertEqual(patch.line_style, "-")
        self.assertIsNone(patch.hatch)
        self.assertEqual(patch.alpha, 1.0)

    def test_is_legend_element(self):
        """Test that LegendPatch is recognized as a LegendElement."""
        self.assertTrue(isinstance(self.legend_patch, LegendElement))

    def test_handle_property(self):
        """Test the handle property returns a Patch object."""
        handle = self.legend_patch.handle
        self.assertIsInstance(handle, Patch)
        self.assertEqual(handle.get_label(), "Test Patch")
        self.assertEqual(handle.get_facecolor(), (1.0, 0, 1.0, 0.7))
        self.assertEqual(handle.get_edgecolor(), (1.0, 0, 0, 0.7))
        self.assertEqual(handle.get_linewidth(), 1.5)
        self.assertEqual(handle.get_linestyle(), "-.")
        self.assertEqual(handle.get_hatch(), "//")
        self.assertEqual(handle.get_alpha(), 0.7)

    def test_handle_with_none_colors(self):
        """Test handle property with None colors."""
        patch = LegendPatch(label="Test", face_color=None, edge_color=None)
        handle = patch.handle
        self.assertEqual(handle.get_facecolor(), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(handle.get_edgecolor(), (0, 0, 0, 0))
        self.assertFalse(handle.get_fill())

    def test_color_validation(self):
        """Test color validation for face_color and edge_color."""
        # Valid colors
        for color_attr in ["face_color", "edge_color"]:
            setattr(self.legend_patch, color_attr, "green")
            self.assertEqual(getattr(self.legend_patch, color_attr), "green")

            setattr(self.legend_patch, color_attr, None)
            self.assertIsNone(getattr(self.legend_patch, color_attr))

        # Invalid color
        with self.assertRaises(ValueError):
            self.legend_patch.face_color = "invalid_color"

    def test_line_width_validation(self):
        """Test line width validation."""
        # Valid values
        self.legend_patch.line_width = 3.0
        self.assertEqual(self.legend_patch.line_width, 3.0)

        self.legend_patch.line_width = 0
        self.assertEqual(self.legend_patch.line_width, 0)

        # Invalid values
        with self.assertRaises(TypeError):
            self.legend_patch.line_width = "invalid"

        with self.assertRaises(ValueError):
            self.legend_patch.line_width = -1

    def test_line_style_validation(self):
        """Test line style validation."""
        # Valid string styles
        for style in ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]:
            self.legend_patch.line_style = style
            self.assertEqual(self.legend_patch.line_style, style)

        # Valid tuple style
        tuple_style = (0, (3, 5))
        self.legend_patch.line_style = tuple_style
        self.assertEqual(self.legend_patch.line_style, tuple_style)

        # Invalid string style
        with self.assertRaises(ValueError):
            self.legend_patch.line_style = "invalid_style"

    def test_hatch_validation(self):
        """Test hatch pattern validation."""
        # Valid hatch patterns
        valid_hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        for hatch in valid_hatches:
            self.legend_patch.hatch = hatch
            self.assertEqual(self.legend_patch.hatch, hatch)

        # Valid combinations
        self.legend_patch.hatch = "//"
        self.assertEqual(self.legend_patch.hatch, "//")

        self.legend_patch.hatch = "+-"
        self.assertEqual(self.legend_patch.hatch, "+-")

        # None should be allowed
        self.legend_patch.hatch = None
        self.assertIsNone(self.legend_patch.hatch)

        # Invalid hatch patterns
        with self.assertRaises(ValueError):
            self.legend_patch.hatch = "z"  # Invalid character

        with self.assertRaises(ValueError):
            self.legend_patch.hatch = "/@"  # Mix of valid and invalid


class TestLegendElementIntegration(unittest.TestCase):
    """Integration tests for LegendElement classes."""

    def test_all_classes_implement_protocol(self):
        """Test that all legend element classes implement the LegendElement protocol."""
        classes = [LegendLine, LegendMarker, LegendPatch]

        for cls in classes:
            with self.subTest(cls=cls.__name__):
                if cls == LegendLine:
                    instance = cls(label="Initial", color="red")
                else:
                    instance = cls(label="Initial")
                self.assertTrue(isinstance(instance, LegendElement))
                self.assertTrue(hasattr(instance, "handle"))
                self.assertIsInstance(instance.handle, Artist)

    def test_label_property_consistency(self):
        """Test that label property works consistently across all classes."""
        classes = [LegendLine, LegendMarker, LegendPatch]

        for cls in classes:
            with self.subTest(cls=cls.__name__):
                if cls == LegendLine:
                    instance = cls(label="Initial", color="red")
                else:
                    instance = cls(label="Initial")

                self.assertEqual(instance.label, "Initial")

                instance.label = "Updated"
                self.assertEqual(instance.label, "Updated")

    def test_alpha_property_consistency(self):
        """Test that alpha property works consistently across all classes."""
        classes = [LegendLine, LegendMarker, LegendPatch]

        for cls in classes:
            with self.subTest(cls=cls.__name__):
                if cls == LegendLine:
                    instance = cls(label="Test", color="red", alpha=0.5)
                else:
                    instance = cls(label="Test", alpha=0.5)

                self.assertEqual(instance.alpha, 0.5)

                instance.alpha = 0.8
                self.assertEqual(instance.alpha, 0.8)

                # Test validation
                with self.assertRaises(ValueError):
                    instance.alpha = 1.5


if __name__ == "__main__":
    unittest.main()
