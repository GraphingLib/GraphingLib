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
