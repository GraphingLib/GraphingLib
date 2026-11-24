import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import CloseEvent
from matplotlib.container import ErrorbarContainer
from numpy.testing import assert_allclose
from PIL import Image, ImageSequence

from graphinglib import Curve, GraphingException, Heatmap, Scatter, SmartFigure, Text


class TestAnimation(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        directory = self.enterContext(TemporaryDirectory())
        self.output = Path(directory) / "animated.gif"

    def test_export_updates_curve_error_artists_and_raster_in_all_panels(self):
        curve = Curve([1.0, 3.0], [1.0, 2.0], color="navy")
        curve.add_errorbars(x_error=0.1, y_error=[0.2, 0.3], cap_width=4)
        heatmap = Heatmap(
            [[0, 1], [0, 0]], color_map_range=(0, 1), show_color_bar=False
        )
        left = SmartFigure(elements=[curve], x_lim=(0, 5), y_lim=(0, 8))
        middle = SmartFigure(elements=[curve], x_lim=(0, 5), y_lim=(0, 8))
        right = SmartFigure(elements=[heatmap])
        fig = SmartFigure(
            1, 3, elements=[left, middle, right], figure_style="plain", size=(9, 3.5)
        )
        source_figure = None
        calls = []

        def update(index):
            nonlocal source_figure
            source_figure = plt.gcf()
            calls.append(index)
            curve.set_data(
                [1, 3 + index / 3],
                [1 + index, 2 + index],
                x_error=[[0.1, 0.2], [0.3, 0.4]],
                y_error=[0.2, 0.4],
            )
            heatmap.image = np.roll([[0, 1], [0, 0]], index).reshape(2, 2)

        self.assertIs(
            fig.save_animation(str(self.output), update, 4, interval=120), fig
        )
        self.assertEqual(calls, [0, 1, 2, 3])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 4)
            frames = []
            for frame in ImageSequence.Iterator(gif):
                self.assertEqual(frame.info["duration"], 120)
                frames.append(np.array(frame.convert("RGB")))
        for start, stop in ((0, 0.32), (0.34, 0.66), (0.7, 1)):
            width = frames[0].shape[1]
            self.assertEqual(
                len(
                    {
                        frame[:, int(width * start) : int(width * stop)].tobytes()
                        for frame in frames
                    }
                ),
                4,
            )

        handles = [
            item
            for axes in source_figure.axes
            for item in axes.containers
            if isinstance(item, ErrorbarContainer)
        ]
        self.assertEqual(len(handles), 2)
        for handle in handles:
            line, caps, bars = handle.lines
            assert_allclose(line.get_xydata(), [[1, 4], [4, 5]])
            assert_allclose(
                bars[0].get_segments(), [[[0.9, 4], [1.3, 4]], [[3.8, 5], [4.4, 5]]]
            )
            assert_allclose(
                bars[1].get_segments(), [[[1, 3.8], [1, 4.2]], [[4, 4.6], [4, 5.4]]]
            )
            self.assertEqual(len(caps), 4)
            for cap, x_values, y_values in zip(
                caps,
                [[0.9, 3.8], [1.3, 4.4], [1, 4], [1, 4]],
                [[4, 5], [4, 5], [3.8, 4.6], [4.2, 5.4]],
                strict=True,
            ):
                assert_allclose(cap.get_xdata(), x_values)
                assert_allclose(cap.get_ydata(), y_values)
        self.assertIsNone(fig._animation)
        self.assertFalse(plt.fignum_exists(source_figure.number))

    def test_curve_data_validation_is_atomic(self):
        curve = Curve([1, 2], [3, 4])
        curve.add_errorbars(y_error=[0.2, 0.3])
        for kwargs in ({}, {"y_error": [-1, 0, 1]}, {"y_error": [[1, 2, 3]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                curve.set_data([1, 2, 3], [3, 4, 5], **kwargs)
            assert_allclose(curve.x_data, [1, 2])
            assert_allclose(curve.y_data, [3, 4])
            assert_allclose(curve.y_error, [0.2, 0.3])
        self.assertIs(curve.set_data([1, 2, 3], [3, 4, 5], y_error=None), curve)
        self.assertIsNone(curve.y_error)

    def test_scatter_errors_and_text_follow_nested_frames(self):
        points = Scatter([1, 2], [2, 3], face_color="teal")
        points.add_errorbars(y_error=0.2, cap_width=4)
        label = Text(0.2, 0.8, "initial")
        panel = SmartFigure(elements=[points, label], x_lim=(0, 5), y_lim=(0, 7))
        caption_panel = SmartFigure(annotations=[label])
        nested = SmartFigure(1, 2, elements=[panel, caption_panel])
        root = SmartFigure(1, 2, elements=[nested], figure_style="plain", size=(10, 5))
        figures = []

        def update(index):
            figures.append(plt.gcf())
            points.set_data([1 + index, 2 + index], [2 + index, 3 + index], y_error=0.4)
            label.x = 0.2 + index / 10
            label.y = 0.8 - index / 10
            label.text = f"frame {index}"

        with patch.object(
            root, "_prepare_figure", wraps=root._prepare_figure
        ) as prepare:
            root.save_animation(str(self.output), update, 3, interval=100)
        prepare.assert_called_once()
        self.assertEqual(len({id(figure) for figure in figures}), 1)
        assert_allclose(points.handle.get_offsets(), [[3, 4], [4, 5]])
        _, caps, bars = points.errorbars_handle.lines
        assert_allclose(
            bars[0].get_segments(), [[[3, 3.6], [3, 4.4]], [[4, 4.6], [4, 5.4]]]
        )
        assert_allclose(caps[0].get_ydata(), [3.6, 4.6])
        assert_allclose(caps[1].get_ydata(), [4.4, 5.4])
        labels = figures[-1].findobj(
            lambda artist: (
                hasattr(artist, "get_text") and artist.get_text() == "frame 2"
            )
        )
        self.assertEqual(len(labels), 2)
        for artist in labels:
            assert_allclose(artist.get_position(), [0.4, 0.6])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_scatter_and_text_roll_back_to_last_successful_frame(self):
        points = Scatter([1, 2], [2, 3])
        label = Text(0.2, 0.8, "initial")
        fig = SmartFigure(elements=[points], annotations=[label], figure_style="plain")
        self.output.write_bytes(b"previous export")
        unrelated = plt.figure()

        def update(index):
            points.y_data = [3 + index, 4 + index]
            label.x = 0.3 + index / 10
            label.text = f"frame {index}"
            if index:
                points.set_data([1], [2, 3])

        with self.assertRaisesRegex(ValueError, "same length"):
            fig.save_animation(str(self.output), update, 3)
        assert_allclose(points.y_data, [3, 4])
        self.assertEqual(label.text, "frame 0")
        self.assertEqual(label.x, 0.3)
        self.assertEqual(self.output.read_bytes(), b"previous export")
        self.assertEqual(plt.get_fignums(), [unrelated.number])
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_keeps_own_window_and_original_error_on_export_failure(self):
        curve = Curve([0, 1], [1, 2])
        fig = SmartFigure(elements=[curve], figure_style="plain")
        self.output.write_bytes(b"previous export")
        owned, other = [], []

        def update(index):
            owned.append(plt.gcf())
            other.append(plt.figure())
            curve.y_data = [5, 6]
            curve.y_data = [7, 8]
            raise RuntimeError("failure in frame zero")

        with plt.rc_context({"axes.facecolor": "gold"}):
            with self.assertRaisesRegex(RuntimeError, "failure in frame zero"):
                fig.save_animation(str(self.output), update, 3)
            self.assertEqual(plt.rcParams["axes.facecolor"], "gold")
        self.assertEqual(self.output.read_bytes(), b"previous export")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        self.assertTrue(plt.fignum_exists(other[0].number))
        self.assertFalse(plt.fignum_exists(owned[0].number))
        self.assertIsNone(fig._animation)
        assert_allclose(curve.y_data, [1, 2])
        fig.save_animation(
            str(self.output), lambda index: setattr(curve, "y_data", [1, 3 + index]), 2
        )
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_rejects_unsupported_live_changes(self):
        curve = Curve([0, 1], [1, 2])
        curve.add_errorbars(y_error=0.2)
        fig = SmartFigure(elements=[curve], figure_style="plain")
        with self.assertRaisesRegex(GraphingException, "errorbar directions"):
            fig.save_animation(
                str(self.output),
                lambda i: curve.set_data([0, 1], [2, 3], y_error=None),
                2,
            )
        assert_allclose(curve.y_data, [1, 2])
        heatmap = Heatmap(
            [[0, 1], [1, 0]],
            x_mesh=[[0, 1], [0, 1]],
            y_mesh=[[0, 0], [1, 1]],
            show_color_bar=False,
        )
        fig = SmartFigure(elements=[heatmap], figure_style="plain")
        with self.assertRaisesRegex(ValueError, "original shape"):
            fig.save_animation(
                str(self.output), lambda i: setattr(heatmap, "image", [[1]]), 2
            )
        assert_allclose(heatmap.image, [[0, 1], [1, 0]])
        self.assertFalse(self.output.exists())

    def test_invalid_coordinates_and_unbound_curves_in_callbacks(self):
        curve = Curve([0, 1], [1, 2])
        scratch = Curve([0, 1], [1, 2])
        fig = SmartFigure(elements=[curve], figure_style="plain")

        def invalid(index):
            scratch.x_data = [0, 1, 2]
            scratch.y_data = [2, 3, 4]
            curve.y_data = [None, object()]

        with self.assertRaisesRegex(ValueError, "real numeric"):
            fig.save_animation(str(self.output), invalid, 2)
        assert_allclose(scratch.y_data, [2, 3, 4])
        assert_allclose(curve.y_data, [1, 2])
        self.assertFalse(self.output.exists())
        fig.save_animation(
            str(self.output), lambda i: setattr(curve, "y_data", [1, 3 + i]), 2
        )
        self.assertTrue(self.output.exists())

    def test_interactive_session_releases_bindings_when_its_figure_closes(self):
        curve = Curve([0, 1], [1, 2])
        fig = SmartFigure(elements=[curve], figure_style="plain")
        unrelated = plt.figure()
        with patch.object(plt, "show"):
            fig.show_animation(lambda i: setattr(curve, "y_data", [2 + i, 3 + i]), 2)
        owned = fig._figure
        animation = fig._animation
        self.assertTrue(animation.event_source.callbacks)
        animation._step()
        assert_allclose(curve.handle.lines[0].get_ydata(), [2, 3])
        owned.canvas.callbacks.process(
            "close_event", CloseEvent("close_event", owned.canvas)
        )
        self.assertIsNone(fig._animation)
        self.assertIsNone(fig._figure)
        self.assertTrue(plt.fignum_exists(unrelated.number))

    def test_validates_parameters_and_writer_before_drawing(self):
        fig = SmartFigure(elements=[Curve([0, 1], [1, 2])], figure_style="plain")
        for kwargs in ({"interval": 0}, {"fps": np.nan}, {"dpi": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fig.save_animation(str(self.output), lambda i: None, 3, **kwargs)
        with self.assertRaises(ValueError):
            fig.save_animation(str(self.output), lambda i: None, 0)
        with self.assertRaisesRegex(GraphingException, "unavailable"):
            fig.save_animation(
                str(self.output), lambda i: None, 3, writer="missing-writer"
            )
        with self.assertRaisesRegex(ValueError, "extension"):
            fig.save_animation(
                str(self.output.with_suffix(".html")), lambda i: None, 3, writer="html"
            )
        self.assertFalse(list(self.output.parent.iterdir()))
        self.assertIsNone(fig._figure)
        self.assertFalse(self.output.exists())


class TestAdditionalAnimationAdapters(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.output = Path(self.enterContext(TemporaryDirectory())) / "demo.gif"

    def test_shares_spanning_layout_and_rolls_back_style_commit(self):
        curve = Curve([0, 1], [1, 2], color="navy")
        points = Scatter([0, 1], [1, 2], marker_size=12)
        label = Text(0.1, 0.8, "initial", font_size=10)
        inner = SmartFigure(
            elements=[curve, points, label], x_lim=(-1, 3), y_lim=(0, 5)
        )
        nested = SmartFigure(1, 2, elements=[inner])
        root = SmartFigure(2, 2, figure_style="plain", share_x=True)
        root[:, 0] = nested
        root[:, 1] = SmartFigure(elements=[curve], x_lim=(-1, 3), y_lim=(0, 5))
        artists = []

        def update(i):
            if not artists:
                for axis in plt.gcf().axes:
                    for line in axis.lines:
                        if line.get_color() == "navy":
                            artists.append(line)
            curve.y_data = [1 + i, 2 + i]
            curve.line_width = 2 + i
            points.marker_size = 30 + i
            label.text = str(i)
            label.font_size = 12 + i
            curve.color = "orange" if i == 0 else "bad color"

        self.output.write_bytes(b"existing")
        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(len(artists), 2)
        for line in artists:
            self.assertEqual(line.get_color(), "orange")
            self.assertEqual(line.get_linewidth(), 2)
            assert_allclose(line.get_ydata(), [1, 2])
            assert_allclose(line.axes.get_xlim(), [-1, 3])
        self.assertEqual(label.text, "0")
        self.assertEqual(label.font_size, 12)
        self.assertEqual(points.marker_size, 30)
        self.assertEqual(curve.color, "orange")
        self.assertIsNone(nested._figure)
        self.assertIsNone(inner._gridspec)

    def test_fixed_style_errorbars_and_changing_fills(self):
        curve = Curve([0, 1, 2], [1, 1, 2], color="green")
        curve.add_error_curves(y_error=0.1, error_curves_fill_between=True)
        curve.get_area_between(0, 2, fill_between=True)
        figure = SmartFigure(elements=[curve], figure_style="plain", y_lim=(0, 6))
        result = []

        def update(frame):
            axes = curve.handle.lines[0].axes
            result.append(
                (id(axes), tuple(map(id, axes.lines)), tuple(map(id, axes.collections)))
            )
            curve.set_data(
                [0, 1, 2], [1, 2 + frame, 3], y_error=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            )

        figure.save_animation(self.output, update, 3)
        self.assertTrue(all(item == result[0] for item in result))
        axes = curve.handle.lines[0].axes
        assert_allclose(axes.lines[1].get_ydata(), [0.9, 3.8, 2.7])
        assert_allclose(axes.lines[2].get_ydata(), [1.4, 4.5, 3.6])
        assert_allclose(max(axes.collections[0].get_paths()[0].vertices[:, 1]), 4.5)
        assert_allclose(max(axes.collections[1].get_paths()[0].vertices[:, 1]), 4.0)

    def test_reference_lines_point_shapes_and_table(self):
        from matplotlib.text import Annotation

        from graphinglib import (
            Arrow,
            Circle,
            Ellipse,
            Hlines,
            Line,
            Point,
            Rectangle,
            Table,
            Vlines,
        )

        horizontal = Hlines([1, 2], x_min=0, x_max=3)
        vertical = Vlines([1, 2], line_widths=2)
        point = Point(0.0, 0.0, label="P")
        point.add_coordinates()
        arrow = Arrow((0, 0), (1, 2))
        line = Line((0, 0), (2, 2), capped_line=True)
        circle = Circle(0, 0, 1)
        ellipse = Ellipse(1, 1, 0.5, 0.4)
        rectangle = Rectangle(0, 0, 1, 1)
        table = Table([["old"]], row_labels=["row"], col_labels=["header"])
        panel = SmartFigure(
            elements=[
                horizontal,
                vertical,
                point,
                arrow,
                line,
                circle,
                ellipse,
                rectangle,
            ],
            x_lim=(-1, 7),
            y_lim=(-1, 7),
        )
        root = SmartFigure(1, 2, elements=[panel, table], figure_style="plain")
        captures = []
        axes = None

        def update(i):
            nonlocal axes
            axes = next(
                axis
                for axis in plt.gcf().axes
                if any(
                    isinstance(artist, Annotation) and "P" in artist.get_text()
                    for artist in axis.texts
                )
            )
            captures.append(
                tuple(
                    map(
                        id, [*axes.lines, *axes.collections, *axes.patches, *axes.texts]
                    )
                )
            )
            horizontal.y = [1 + i, 2 + i]
            horizontal.x_max = 3 + i
            vertical.x = [2 + i, 3 + i]
            point.coordinates = i + 1.0, i + 2.0
            arrow.pointB = 2 + i, 4
            arrow.color = "red"
            line.pointA = 3, 2 + i
            circle.x_center = i
            ellipse.angle = 10 * i
            ellipse.x_radius = 0.75 + 0.25 * i
            rectangle.translate(0.5, 0.25)
            table.cell_text = [[str(i)]]
            table.text_color = "purple"

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(capture == captures[0] for capture in captures))
        assert_allclose(
            axes.collections[0].get_segments(), [[[0, 3], [5, 3]], [[0, 4], [5, 4]]]
        )
        assert_allclose(axes.lines[0].get_xdata(), [4, 4])
        assert_allclose(axes.collections[-1].get_offsets(), [[3, 4]])
        self.assertIn(
            "(3.000, 4.000)",
            next(text for text in axes.texts if "P" in text.get_text()).get_text(),
        )
        arrows = [
            text
            for text in axes.texts
            if isinstance(text, Annotation) and text.arrow_patch is not None
        ]
        self.assertEqual([artist.xy for artist in arrows], [(4, 4), (3, 4)])
        self.assertAlmostEqual(circle.x_center, 2)
        self.assertEqual(ellipse.angle, 20)
        assert_allclose(rectangle.center, (2, 1.25))
        self.assertEqual(table.handle[1, 0].get_text().get_text(), "2")
        self.assertEqual(table.handle[0, 0].get_text().get_text(), "header")
        self.assertEqual(table.handle[1, 0].get_text().get_color(), "purple")

    def test_mesh_quiver_and_contour_keep_their_artists(self):
        from matplotlib.collections import QuadMesh
        from matplotlib.contour import QuadContourSet
        from matplotlib.quiver import Quiver

        from graphinglib import Contour, VectorField

        if not hasattr(QuadContourSet, "set_paths"):
            self.skipTest("Contour paths require Matplotlib 3.8.")
        grid = np.arange(4)
        x, y = np.meshgrid(grid, grid)
        heatmap = Heatmap(x + y, x_mesh=x, y_mesh=y, show_color_bar=True)
        quiver = VectorField(x, y, x, y, scale=1)
        contour = Contour(
            x + y,
            x_mesh=x,
            y_mesh=y,
            levels=[0, 2, 4, 6],
            show_color_bar=False,
            filled=False,
        )
        root = SmartFigure(
            1, 3, elements=[heatmap, quiver, contour], figure_style="plain"
        )
        captures = []
        geometry = []

        def update(i):
            artists = [
                artist
                for axis in plt.gcf().axes
                for artist in axis.collections
                if isinstance(artist, (QuadMesh, Quiver, QuadContourSet))
            ]
            (mesh,) = [
                artist
                for artist in artists
                if isinstance(artist, QuadMesh) and artist.get_array().size == 16
            ]
            (field,) = [artist for artist in artists if isinstance(artist, Quiver)]
            (contours,) = [
                artist for artist in artists if isinstance(artist, QuadContourSet)
            ]
            captures.append((mesh, field, contours))
            geometry.append(contours.get_paths()[1].vertices.copy())
            heatmap.image = x * i + y
            heatmap.color_map_range = (10 * i, 6 + 10 * i)
            heatmap.color_map = "plasma"
            quiver.u_data = x + i
            quiver.v_data = y + i
            contour.z_data = x + y + i * 0.5
            contour.color_map = "magma"

        root.save_animation(self.output, update, 3)
        self.assertTrue(
            all(
                all(a is b for a, b in zip(capture, captures[0]))
                for capture in captures
            )
        )
        self.assertFalse(np.array_equal(geometry[0], geometry[-1]))
        mesh, field, contours = captures[-1]
        assert_allclose(mesh.get_array().reshape((4, 4)), 2 * x + y)
        self.assertEqual(mesh.get_clim(), (20, 26))
        self.assertEqual(mesh.get_cmap().name, "plasma")
        assert_allclose(field.U, (x + 2).ravel())
        assert_allclose(field.V, (y + 2).ravel())
        self.assertEqual(contours.get_cmap().name, "magma")

    def test_histogram_bars_and_steps_follow_new_data(self):
        from matplotlib.container import BarContainer
        from matplotlib.patches import Polygon

        from graphinglib import Histogram

        bar = Histogram(
            [0, 1, 2], bins=2, normalize=False, show_params=False, hist_type="bar"
        )
        step = Histogram(
            [0, 1, 2],
            bins=2,
            normalize=False,
            show_params=False,
            hist_type="stepfilled",
            orientation="horizontal",
        )
        root = SmartFigure(1, 2, elements=[bar, step], figure_style="plain")
        captures = []

        def update(i):
            (bars,) = [
                item
                for axis in plt.gcf().axes
                for item in axis.containers
                if isinstance(item, BarContainer)
            ]
            (steps,) = [
                item
                for axis in plt.gcf().axes
                for item in axis.patches
                if isinstance(item, Polygon)
            ]
            captures.append((bars, steps))
            bar.data = step.data = [0, i + 1, i + 2, i + 2]

        root.save_animation(self.output, update, 3)
        self.assertTrue(
            all(
                all(a is b for a, b in zip(capture, captures[0]))
                for capture in captures
            )
        )
        bars, steps = captures[-1]
        assert_allclose([patch.get_x() for patch in bars], [0, 2])
        assert_allclose([patch.get_width() for patch in bars], [2, 2])
        assert_allclose([patch.get_height() for patch in bars], [1, 3])
        assert_allclose(steps.get_xy()[:, 0].max(), 3)
        assert_allclose(steps.get_xy()[:, 1].max(), 4)

    def test_rejects_layout_mutation_and_preserves_spanning_plot(self):
        original = Curve([0, 1], [0, 1])
        leaf = SmartFigure(elements=[original])
        root = SmartFigure(2, 2, figure_style="plain")
        root[0, :] = leaf
        self.output.write_bytes(b"existing")
        replacement = Curve([0, 1], [1, 0])

        def change_layout(i):
            original.y_data = [1, 1]
            root[0, :] = replacement

        for update in (
            change_layout,
            lambda i: root.add_elements(replacement),
            lambda i: leaf.__iadd__(replacement),
        ):
            with self.assertRaisesRegex(
                GraphingException, "layout and axis properties"
            ):
                root.save_animation(self.output, update, 2)
            self.assertIs(root[0, 1].elements[0], original)
            assert_allclose(original.y_data, [0, 1])
            self.assertEqual(self.output.read_bytes(), b"existing")

    def test_stream_changes_line_and_arrow_styles(self):
        from matplotlib.colors import to_rgba
        from matplotlib.patches import FancyArrowPatch

        from graphinglib import Stream

        grid = np.linspace(0, 1, 7)
        x, y = np.meshgrid(grid, grid)
        stream = Stream(grid, grid, np.ones_like(x), 0.2 * np.ones_like(y), density=0.5)
        root = SmartFigure(elements=[stream], figure_style="plain")
        captures = []

        def update(i):
            patches = tuple(
                artist
                for axis in plt.gcf().axes
                for artist in axis.patches
                if isinstance(artist, FancyArrowPatch)
            )
            (lines,) = [
                artist for axis in plt.gcf().axes for artist in axis.collections
            ]
            captures.append((lines, patches))
            stream.color = ["navy", "red"][i]
            stream.line_width = 1 + i

        root.save_animation(self.output, update, 2)
        self.assertEqual(captures[0], captures[1])
        lines, arrows = captures[-1]
        self.assertTrue(arrows)
        assert_allclose(lines.get_colors()[0], to_rgba("red"))
        assert_allclose(lines.get_linewidth()[0], 2)
        for arrow in arrows:
            assert_allclose(arrow.get_facecolor(), to_rgba("red"))
            assert_allclose(arrow.get_linewidth(), 2)

    def test_shared_axis_limits_and_repeated_artists(self):
        curve = Curve([0, 1], [0, 1], color="navy")
        nested = SmartFigure(elements=[Text(0.4, 0.5, "nested")])
        root = SmartFigure(3, 2, figure_style="plain", share_x=True, x_lim=(-2, 3))
        root[:, 0] = nested
        root[:2, 1] = [curve]
        root[2, 1] = [curve]
        seen = []

        def update(i):
            one, two = [
                line
                for ax in plt.gcf().axes
                for line in ax.lines
                if line.get_color() == "navy"
            ]
            seen.append((one, two))
            self.assertTrue(one.axes.get_shared_x_axes().joined(one.axes, two.axes))
            curve.x_data = [i, i + 1]
            curve.y_data = [i, 2 * i + 1]

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(artists == seen[0] for artists in seen))
        for artist in seen[-1]:
            assert_allclose(artist.get_xdata(), [2, 3])
            assert_allclose(artist.get_ydata(), [2, 5])
            assert_allclose(artist.axes.get_xlim(), [-2, 3])

    def test_failed_initial_prepare_restores_nested_layout(self):
        curve = Curve([0, 1], [1, 3], color="invalid-color")
        nested = SmartFigure(elements=[curve])
        root = SmartFigure(2, 2, figure_style="plain", subtitles=["Temporary"])
        root[:, :] = nested
        unrelated = plt.figure()
        self.output.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            root.save_animation(
                self.output, lambda i: self.fail("No frame should run."), 2
            )
        self.assertEqual(plt.get_fignums(), [unrelated.number])
        self.assertIsNone(nested.title)
        self.assertIsNone(nested._figure)
        self.assertIsNone(root._figure)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_resizes_scatter_and_keeps_error_artists_through_empty_frames(self):
        scatter = Scatter([0, 1], [2, 3], face_color="navy")
        scatter.add_errorbars(y_error=[0.1, 0.2], cap_width=4)
        figure = SmartFigure(
            elements=[scatter], figure_style="plain", x_lim=(-1, 4), y_lim=(-1, 5)
        )
        snapshots = []

        def update(frame):
            bars = scatter.errorbars_handle.lines
            snapshots.append((scatter.handle, bars, len(scatter.handle.get_offsets())))
            x = np.arange([0, 3, 1, 2][frame])
            scatter.set_data(x, x + 1, y_error=np.full(x.size, 0.25))

        figure.save_animation(self.output, update, 4)
        self.assertEqual([item[2] for item in snapshots], [2, 0, 3, 1])
        collection, bars, _ = snapshots[0]
        self.assertTrue(
            all(item[0] is collection and item[1] is bars for item in snapshots)
        )
        assert_allclose(collection.get_offsets(), [[0, 1], [1, 2]])
        assert_allclose(
            bars[2][0].get_segments(), [[[0, 0.75], [0, 1.25]], [[1, 1.75], [1, 2.25]]]
        )
        assert_allclose(bars[1][0].get_ydata(), [0.75, 1.75])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 4)

    def test_shared_fills_update_from_both_curves_including_an_unplotted_boundary(self):
        first = Curve([0, 1, 2], [0, 1, 0])
        second = Curve([0, 2], [3, 3])
        first.get_area_between(0.5, 1.5, other_curve=second, fill_between=True)
        figure = SmartFigure(
            1, 2, elements=[[first], [first]], figure_style="plain", y_lim=(-1, 6)
        )
        seen = []

        def update(frame):
            artists = tuple(
                axis.collections[0] for axis in plt.gcf().axes if axis.collections
            )
            seen.append(artists)
            if frame == 0:
                second.y_data = [4, 4]
            if frame == 1:
                first.y_data = [0, 2, 0]
                second.y_data = [5, 5]
            if frame == 2:
                first.set_data([1, 2], [0, 1])

        self.output.write_bytes(b"previous")
        with self.assertRaisesRegex(ValueError, "interpolation range"):
            figure.save_animation(self.output, update, 3)
        self.assertEqual(self.output.read_bytes(), b"previous")
        self.assertTrue(all(items == seen[0] for items in seen))
        self.assertEqual(len(seen[0]), 2)
        for artist in seen[0]:
            vertices = artist.get_paths()[0].vertices
            assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()], [1, 5])
        assert_allclose(first.x_data, [0, 1, 2])
        assert_allclose(second.y_data, [5, 5])

    def test_resamples_raster_at_the_same_physical_extent(self):
        heat = Heatmap([[0, 1], [1, 0]], show_color_bar=False, color_map_range=(0, 1))
        figure = SmartFigure(elements=[heat], figure_style="plain")
        seen = []

        def update(frame):
            (artist,) = [image for axes in plt.gcf().axes for image in axes.images]
            seen.append((artist, tuple(artist.get_extent()), artist.get_array().shape))
            heat.image = np.eye(frame + 3)

        figure.save_animation(self.output, update, 3)
        self.assertTrue(
            all(
                artist is seen[0][0] and extent == seen[0][1]
                for artist, extent, _ in seen
            )
        )
        self.assertEqual([shape for _, _, shape in seen], [(2, 2), (3, 3), (4, 4)])
        self.assertEqual(seen[0][0].get_array().shape, (5, 5))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_rejects_resizing_scatter_with_per_point_colors(self):
        scatter = Scatter([0, 1], [2, 3], face_color=["red", "blue"])
        figure = SmartFigure(elements=[scatter], figure_style="plain")
        with self.assertRaisesRegex(ValueError, "per-point colors"):
            figure.save_animation(self.output, lambda _: scatter.set_data([0], [2]), 1)
        assert_allclose(scatter.handle.get_offsets(), [[0, 2], [1, 3]])
        assert_allclose(scatter.x_data, [0, 1])
        self.assertFalse(self.output.exists())

    def test_scatter_point_colors_resize_with_collection_and_colorbar(self):
        scatter = Scatter(
            [0, 1],
            [1, 2],
            face_color=[0.0, 1.0],
            show_color_bar=True,
            color_map_range=(0, 1),
        )
        root = SmartFigure(
            1,
            2,
            elements=[[scatter], [scatter]],
            figure_style="plain",
            x_lim=(-1, 4),
            y_lim=(0, 4),
        )
        seen = []

        def update(i):
            existing = tuple(
                artist
                for axis in plt.gcf().axes
                if axis.get_label() != "<colorbar>"
                for artist in axis.collections
            )
            scales = tuple(
                axis for axis in plt.gcf().axes if axis.get_label() == "<colorbar>"
            )
            seen.append((existing, scales))
            if i == 0:
                scatter.face_color = [10.0, 15.0, 20.0]
                scatter.set_data([0, 1, 2], [1, 2, 3])
            else:
                scatter.set_data([0, 1, 2, 3], [1, 2, 3, 1])
                scatter.face_color = [10.0, 12.0, 15.0, 20.0]
            scatter.color_map = "plasma"
            scatter.color_map_range = (10, 20)
            scatter.marker_style = "s"

        root.save_animation(self.output, update, 2)
        self.assertEqual(seen[0], seen[-1])
        artists, scales = seen[-1]
        self.assertEqual(len(artists), 2)
        self.assertEqual(len(scales), 2)
        for artist in artists:
            self.assertEqual(len(artist.get_paths()[0].vertices), 5)
            self.assertEqual(len(artist.get_offsets()), 4)
            assert_allclose(
                artist.get_facecolor(), plt.get_cmap("plasma")([0.0, 0.2, 0.5, 1.0])
            )
        for axis in scales:
            assert_allclose(axis.get_ylim(), [10, 20])
            self.assertEqual(axis._colorbar.mappable.get_cmap().name, "plasma")
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_scatter_color_and_marker_updates_roll_back_with_point_data(self):
        scatter = Scatter([0, 1], [1, 2], face_color=["red", "blue"])
        scatter.add_errorbars(y_error=0.2)
        root = SmartFigure(elements=[scatter], figure_style="plain")
        self.output.write_bytes(b"existing")

        def update(i):
            if i == 0:
                scatter.face_color = (0.0, 0.5, 0.0)
                scatter.marker_style = "s"
            else:
                scatter.face_color = ["red", "blue"]
                scatter.set_data([0, 1, 2], [1, 2, 3])
                scatter.marker_style = "x"

        with self.assertRaisesRegex(ValueError, "final point count"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(scatter.marker_style, "s")
        self.assertEqual(scatter.face_color, (0.0, 0.5, 0.0))
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(len(scatter.handle.get_paths()[0].vertices), 5)
        assert_allclose(scatter.x_data, [0, 1])
        assert_allclose(scatter.handle.get_offsets(), [[0, 1], [1, 2]])
        assert_allclose(scatter.handle.get_facecolor(), [[0.0, 0.5, 0.0, 1.0]])
        assert_allclose(
            scatter.errorbars_handle.lines[2][0].get_color(), [[0.0, 0.5, 0.0, 1.0]]
        )

    def test_animates_limits_on_spanning_shared_and_nested_axes(self):
        first = Curve([0, 6], [0, 6], label="first")
        second = Curve([0, 6], [1, 5], label="second")
        nested_curve = Curve([0, 6], [2, 5], label="nested")
        nested = SmartFigure(elements=[nested_curve], x_lim=(0, 6), y_lim=(0, 6))
        root = SmartFigure(
            2,
            3,
            figure_style="plain",
            share_x=True,
            x_lim=(0, 6),
            y_lim=(0, 6),
            invert_y_axis=True,
        )
        root[:, 0] = nested
        root[0, 1:3] = [first]
        root[1, 1:3] = [second]
        seen = []

        def update(frame):
            axes = tuple(
                artist.handle.lines[0].axes for artist in (first, second, nested_curve)
            )
            seen.append(
                (axes, [ax.get_xlim() for ax in axes], [ax.get_ylim() for ax in axes])
            )
            self.assertTrue(axes[0].get_shared_x_axes().joined(*axes[:2]))
            if frame == 0:
                root.x_lim = [None, (1, 5), None]
                root.y_lim = [None, (1, 4), (2, 5)]
                nested.x_lim = (2, 6)
            elif frame == 1:
                root.x_lim = (None, 4)
                root.y_lim = None
                nested.x_lim = (None, 5)
            else:
                root.x_lim = None

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(axes == seen[0][0] for axes, _, _ in seen))
        assert_allclose(seen[1][1], [(1, 5), (1, 5), (2, 6)])
        assert_allclose(seen[1][2], [(4, 1), (5, 2), (0, 6)])
        assert_allclose(seen[2][1], [(1, 4), (1, 4), (2, 5)])
        assert_allclose(seen[2][2], [(6, 0), (6, 0), (0, 6)])
        assert_allclose([ax.get_xlim() for ax in seen[-1][0]], [(0, 6), (0, 6), (2, 5)])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_shared_axis_conflict_restores_live_limits_and_models(self):
        first = Curve([0, 6], [0, 6])
        second = Curve([0, 6], [1, 5])
        root = SmartFigure(
            1,
            2,
            elements=[first, second],
            figure_style="plain",
            share_x=True,
            x_lim=(0, 6),
            y_lim=(0, 6),
        )
        self.output.write_bytes(b"existing")

        def update(frame):
            root.y_lim = (frame, 5)
            root.x_lim = (1, 5) if frame == 0 else [(0, 4), (1, 4)]

        with self.assertRaisesRegex(ValueError, "Shared axes"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(root.x_lim, (1, 5))
        self.assertEqual(root.y_lim, (0, 5))
        for curve in first, second:
            assert_allclose(curve.handle.lines[0].axes.get_xlim(), (1, 5))
            assert_allclose(curve.handle.lines[0].axes.get_ylim(), (0, 5))

    def test_updates_twin_log_limits_and_leaves_shared_primary_dimension_intact(self):
        from graphinglib import SmartTwinAxis

        main = Curve([1, 5], [1, 6])
        secondary = Curve([1, 5], [10, 100])
        twin = SmartTwinAxis(
            elements=[secondary], log_scale=True, axis_lim=(1, 1000), invert_axis=True
        )
        root = SmartFigure(
            elements=[main],
            figure_style="plain",
            twin_y_axis=twin,
            x_lim=(1, 5),
            y_lim=(0, 6),
        )
        self.output.write_bytes(b"existing")

        def update(frame):
            root.x_lim = (1, 4 - frame)
            twin.axis_lim = (10, 100) if frame == 0 else (0, 200)

        with self.assertRaisesRegex(
            ValueError, "logarithmic axis limits must be positive"
        ):
            root.save_animation(self.output, update, 2)
        axes = main.handle.lines[0].axes
        twin_axes = secondary.handle.lines[0].axes
        assert_allclose(axes.get_xlim(), (1, 4))
        assert_allclose(twin_axes.get_xlim(), (1, 4))
        assert_allclose(axes.get_ylim(), (0, 6))
        assert_allclose(twin_axes.get_ylim(), (100, 10))
        self.assertEqual(twin.axis_lim, (10, 100))
        self.assertEqual(root.x_lim, (1, 4))
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_animates_figure_axis_and_twin_text_without_replacing_artists(self):
        from graphinglib import SmartTwinAxis

        main = Curve([0, 1], [0, 1])
        second = Curve([0, 1], [1, 0])
        twin = SmartTwinAxis(label="Right scale", elements=[second])
        nested = SmartFigure(elements=[main], title="Initial", twin_y_axis=twin)
        root = SmartFigure(2, 2, figure_style="plain", title="Root", x_label="Root x")
        root[:, :] = nested
        identities = []
        root_axes = []

        def update(frame):
            root_axes[:] = plt.gcf().axes
            axes = main.handle.lines[0].axes
            secondary = second.handle.lines[0].axes
            identities.append(
                (axes.title, axes.xaxis.label, axes.yaxis.label, secondary.yaxis.label)
            )
            nested.title = f"Section {frame}"
            nested.x_label = f"Position {frame}"
            nested.y_label = "Value" if frame == 0 else None
            twin.label = f"Right scale {frame}"
            root.title = f"Overview {frame}"

        root.save_animation(self.output, update, 2)
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(
            [artist.get_text() for artist in identities[0]],
            ["Section 1", "Position 1", "", "Right scale 1"],
        )
        self.assertTrue(any(axis.get_title() == "Overview 1" for axis in root_axes))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_artist_commit_failure_restores_shared_viewports_and_labels(self):
        moving = Curve([0, 6], [0, 6], color="navy")
        fixed = Curve([0, 6], [1, 5])
        root = SmartFigure(
            1,
            2,
            elements=[moving, fixed],
            figure_style="plain",
            share_x=True,
            x_lim=(0, 6),
        )
        self.output.write_bytes(b"existing")
        titles = []

        def update(frame):
            titles[:] = [axis.title for axis in plt.gcf().axes]
            root.x_lim = (1, 5) if frame == 0 else (2, 4)
            root.title = "valid" if frame == 0 else "failed"
            moving.color = "navy" if frame == 0 else "invalid-color"

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(root.x_lim, (1, 5))
        self.assertEqual(root.title, "valid")
        self.assertIn("valid", [artist.get_text() for artist in titles])
        self.assertNotIn("failed", [artist.get_text() for artist in titles])
        for curve in (moving, fixed):
            assert_allclose(curve.handle.lines[0].axes.get_xlim(), (1, 5))

    def test_refits_every_builtin_model_on_existing_artists(self):
        from graphinglib import (
            FitFromExponential,
            FitFromFOTF,
            FitFromFunction,
            FitFromGaussian,
            FitFromLog,
            FitFromPolynomial,
            FitFromSine,
            FitFromSquareRoot,
        )

        def affine(x, slope, intercept):
            return slope * x + intercept

        cases = [
            (lambda samples: FitFromPolynomial(samples, 2), lambda x: 1 + 2 * x + x**2),
            (
                lambda samples: FitFromSine(samples, guesses=[2, 1, 0.3, 0.1]),
                lambda x: 2 * np.sin(x + 0.3) + 0.1,
            ),
            (
                lambda samples: FitFromExponential(samples, guesses=[2, 0.4, 0.2]),
                lambda x: 2 * np.exp(0.4 * x + 0.2),
            ),
            (
                lambda samples: FitFromGaussian(samples, guesses=[2, 2, 0.6]),
                lambda x: 2 * np.exp(-0.5 * ((x - 2) / 0.6) ** 2),
            ),
            (
                lambda samples: FitFromSquareRoot(samples, guesses=[2, 1, 0.2]),
                lambda x: 2 * np.sqrt(x + 1) + 0.2,
            ),
            (
                lambda samples: FitFromLog(samples, log_base=10, guesses=[2, 1, 0.2]),
                lambda x: 2 * np.log10(x + 1) + 0.2,
            ),
            (
                lambda samples: FitFromFunction(affine, samples, guesses=[2, 1]),
                lambda x: 2 * x + 1,
            ),
            (
                lambda samples: FitFromFOTF(samples, guesses=[2, 1]),
                lambda x: 2 * (1 - np.exp(-x)),
            ),
        ]
        x = np.linspace(0.5, 5, 40)
        for create, model in cases:
            source = Scatter(x, model(x))
            fit = create(source)
            with self.subTest(fit=type(fit).__name__):
                previous_func = fit.function
                previous_values = previous_func(x)
                initial_results = fit.parameters.copy()
                expected = create(Scatter(x, 1.2 * model(x)))
                panel = SmartFigure(
                    elements=[fit],
                    figure_style="plain",
                    show_legend=False,
                    y_lim=(0, max(1.3 * model(x))),
                )
                handles = []

                def update(i, fit=fit, source=source, model=model, handles=handles):
                    handles.append(fit.handle)
                    source.set_data(x, (1 + 0.2 * i) * model(x))
                    self.assertIs(fit.refit(), fit)

                panel.save_animation(self.output, update, 2)
                self.assertIs(handles[0], handles[1])
                self.assertFalse(np.allclose(initial_results, fit.parameters))
                assert_allclose(fit.parameters, expected.parameters)
                assert_allclose(fit.cov_matrix, expected.cov_matrix)
                assert_allclose(fit.handle.get_ydata(), expected.y_data)
                assert_allclose(fit.function(x), expected.function(x))
                assert_allclose(previous_func(x), previous_values)
                with Image.open(self.output) as gif:
                    self.assertEqual(gif.n_frames, 2)

    def test_refit_updates_residuals_fills_and_rolls_back_after_artist_failure(self):
        from graphinglib import FitFromPolynomial

        x = np.linspace(0, 4, 8)
        disturbance = np.sin(3 * x)
        source = Scatter(x, 1 + x + disturbance)
        fit = FitFromPolynomial(source, degree=1, label="Trend")
        fit.show_residual_curves(sigma_multiplier=2)
        other = FitFromPolynomial(Curve(x, 8 + 0.5 * x), degree=1)
        fit.get_area_between(1, 3, fill_between=True, other_curve=other)
        trigger = Curve(x, x, color="navy")
        root = SmartFigure(
            1, 2, elements=[[fit, source, trigger], [fit]], figure_style="plain"
        )
        seen = []
        last_valid = []
        self.output.write_bytes(b"existing")

        def update(i):
            fit_axes = [
                axis
                for axis in plt.gcf().axes
                if any(line.get_label().startswith("Trend") for line in axis.lines)
            ]
            seen.append(
                tuple((tuple(ax.lines[:3]), ax.collections[0]) for ax in fit_axes)
            )
            source.set_data(x, (i + 2) + (i + 1) * x + (i + 2) * disturbance)
            fit.refit()
            other.refit(Curve(x, 10 + i + 0.5 * x))
            if not i:
                last_valid.append(
                    (
                        fit.parameters.copy(),
                        fit.label,
                        fit.y_data.copy(),
                        np.std(fit.get_residuals()) * 2,
                    )
                )
            trigger.color = "navy" if not i else "invalid-color"

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(seen[0], seen[-1])
        self.assertEqual(len(seen[0]), 2)
        coeffs, label, fitted, spread = last_valid[0]
        self.assertEqual(fit.label, label)
        assert_allclose(fit.parameters, coeffs)
        assert_allclose(source.y_data, 2 + x + 2 * disturbance)
        assert_allclose(other.parameters, [10, 0.5])
        for (line, lower, upper), fill in seen[0]:
            self.assertEqual(line.get_label(), label)
            assert_allclose(line.get_ydata(), fitted)
            assert_allclose(lower.get_ydata(), fitted - spread)
            assert_allclose(upper.get_ydata(), fitted + spread)
            self.assertAlmostEqual(fill.get_paths()[0].vertices[:, 1].max(), 11.5)

    def test_shared_fit_equations_and_twin_labels_update_figure_legend(self):
        from graphinglib import FitFromPolynomial, LegendLine, SmartTwinAxis

        x = np.arange(4)
        fit = FitFromPolynomial(Curve(x, x + 1), 1, label="Model")
        auxiliary = Scatter(x, x + 1, label="Auxiliary")
        twin = SmartTwinAxis(elements=[auxiliary])
        leaf = SmartFigure(elements=[fit], twin_y_axis=twin)
        root = SmartFigure(
            1,
            2,
            elements=[leaf, [fit]],
            general_legend=True,
            show_legend=True,
            figure_style="plain",
            y_lim=(0, 10),
        )
        root.set_custom_legend([LegendLine(label="Guide", color="black")])
        seen = []

        def update(i):
            (legend,) = plt.gcf().legends
            seen.append((legend, tuple(legend.get_texts())))
            fit.refit(Curve(x, (i + 1) * x + 3))
            fit.color = "red"
            auxiliary.label = f"Auxiliary {i}"

        root.save_animation(self.output, update, 2)
        self.assertEqual(seen[0], seen[1])
        legend, texts = seen[-1]
        self.assertEqual(
            [text.get_text() for text in texts],
            [fit.label, "Auxiliary 1", fit.label, "Guide"],
        )
        for i in (0, 2):
            self.assertEqual(legend.legend_handles[i].get_color(), "red")
        self.assertEqual(fit.handle.get_label(), fit.label)
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_subplot_legend_follows_scatter_styles_and_can_clear_text(self):
        from matplotlib.colors import to_rgba

        scatter = Scatter([0, 1], [0, 1], face_color="blue", label="Samples")
        curve = Curve([0, 1], [1, 0], label="Reference", color="navy")
        root = SmartFigure(
            elements=[scatter, curve], show_legend=True, figure_style="plain"
        )
        captures = []

        def update(i):
            legend = scatter.handle.axes.get_legend()
            captures.append(
                (legend, tuple(legend.get_texts()), tuple(legend.legend_handles))
            )
            # Alpha schedules a legend update before the colors schedule a plot update.
            scatter.alpha = 0.6
            scatter.face_color = "green"
            scatter.marker_style = "s"
            scatter.marker_size = 45
            scatter.label = f"Samples {i}"
            curve.label = "Reference" if i == 0 else None
            curve.color = "orange"

        root.save_animation(self.output, update, 2)
        self.assertEqual(captures[0], captures[1])
        _, texts, (scatter_key, line_key) = captures[-1]
        self.assertEqual([item.get_text() for item in texts], ["Samples 1", ""])
        self.assertEqual(len(scatter_key.get_paths()[0].vertices), 5)
        assert_allclose(scatter_key.get_facecolor(), [to_rgba("green", 0.6)])
        assert_allclose(scatter_key.get_sizes(), [45])
        self.assertEqual(line_key.get_color(), "orange")
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_histogram_density_and_legend_follow_statistics_after_zero_spread(self):
        from graphinglib import Histogram

        hist = Histogram(
            [1, 2, 3], bins=3, normalize=False, show_params=True, label="Samples"
        )
        hist.add_pdf(show_mean=True, show_std=True)
        root = SmartFigure(
            elements=[hist],
            show_legend=True,
            x_lim=(0, 12),
            y_lim=(0, 5),
            figure_style="plain",
        )
        state = []

        def update(i):
            axes = next(
                axis for axis in plt.gcf().axes if axis.get_legend() is not None
            )
            items = (axes.lines[0], axes.get_legend().get_texts()[0], *axes.collections)
            state.append((items, len(items[0].get_xdata()), items[1].get_text()))
            hist.data = [[4, 4, 4], [5, 7, 9], [6, 8, 10]][i]
            hist.label = f"Bin samples {i}"

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(items == state[0][0] for items, _, _ in state))
        self.assertEqual([n for _, n, _ in state], [500, 0, 500])
        self.assertIn(r"\mu$ = 4.000", state[1][2])
        pdf, label, std, mean = state[-1][0]
        self.assertIn(r"\mu$ = 8.000", label.get_text())
        self.assertIn("Bin samples 2", label.get_text())
        assert_allclose(pdf.get_xdata()[[0, -1]], [6, 10])
        assert_allclose(mean.get_segments()[0][:, 0], [8, 8])
        assert_allclose(
            [line[0, 0] for line in std.get_segments()],
            [8 - hist.standard_deviation, 8 + hist.standard_deviation],
        )
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)
