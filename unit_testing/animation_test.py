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
        with self.assertRaisesRegex(ValueError, "mesh must match"):
            fig.save_animation(
                str(self.output),
                lambda i: setattr(heatmap, "image", [[1], [2], [3]]),
                2,
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

    def test_errorbar_styles_reuse_caps_and_legend_in_nested_plots(self):
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D

        curve = Curve([0, 1], [0, 1], label="Curve", color="navy")
        curve.add_errorbars(
            x_error=0.1,
            y_error=0.2,
            cap_width=0,
            errorbars_color="same as curve",
            errorbars_line_width="same as curve",
            cap_thickness="same as curve",
        )
        scatter = Scatter([0, 1], [1, 0], face_color="blue", label="Points")
        scatter.add_errorbars(
            y_error=0.1, cap_width=0, errorbars_color="same as scatter"
        )
        panel = SmartFigure(elements=[curve, scatter], show_legend=True)
        root = SmartFigure(1, 2, elements=[panel, curve], figure_style="plain")
        captures = []
        handle_groups = []

        def update(i):
            root_figure = plt.gcf()
            handles = tuple(
                tuple(
                    item
                    for item in axes.containers
                    if isinstance(item, ErrorbarContainer)
                )
                for axes in root_figure.axes
            )
            handle_groups.append(handles)
            self.assertEqual(handles, handle_groups[0])
            if i == 2:
                segment = scatter.errorbars_handle.lines[2][0]
                assert_allclose(segment.get_color(), [to_rgba("green")])
            if i == 0:
                self.assertTrue(
                    all(
                        not cap.get_visible()
                        for group in handles
                        for handle in group
                        for cap in handle.lines[1]
                    )
                )
            curve.errorbars_color = ("black", "same as curve", "same as curve")[i]
            curve.color = ("navy", "green", "purple")[i]
            curve.line_width = 3 + i
            curve.cap_width = (0, 3, 1)[i]
            scatter.errorbars_color = ("orange", "same as scatter", "red")[i]
            scatter.face_color = "green" if i else "blue"
            scatter.errorbars_line_width = 5
            scatter.cap_thickness = 2
            scatter.cap_width = (0, 4, 0)[i]
            scatter.set_data([0, 1], [1 + i, 0], y_error=0.2 + i / 10)
            if i:
                captures.append(
                    tuple(
                        (cap.get_visible(), cap.get_markersize())
                        for cap in scatter.errorbars_handle.lines[1]
                    )
                )

        root.save_animation(self.output, update, 3)
        self.assertIn(((True, 8), (True, 8)), captures)
        for group in handle_groups[-1]:
            for handle in group:
                _, caps, segments = handle.lines
                is_curve = handle.has_xerr
                expected = "purple" if is_curve else "red"
                self.assertEqual(len(caps), 4 if is_curve else 2)
                for segment in segments:
                    assert_allclose(segment.get_color(), [to_rgba(expected)])
                    assert_allclose(segment.get_linewidth(), [5])
                for cap in caps:
                    assert_allclose(
                        to_rgba(cap.get_markeredgecolor()), to_rgba(expected)
                    )
                    self.assertEqual(cap.get_visible(), is_curve)
                    self.assertEqual(cap.get_markersize(), 2 if is_curve else 0)
                    self.assertEqual(cap.get_markeredgewidth(), 5 if is_curve else 2)

        legend = scatter.handle.axes.get_legend()
        assert_allclose(legend.legend_handles[0].get_color(), [to_rgba("purple")])
        cap_keys = [
            line for line in legend.findobj(Line2D) if line.get_marker() in ("|", "_")
        ]
        self.assertEqual(len(cap_keys), 4)
        for line in cap_keys:
            self.assertTrue(line.get_visible())
            self.assertEqual(line.get_markersize(), 2)
            self.assertEqual(line.get_markeredgewidth(), 5)
            self.assertEqual(line.get_markeredgecolor(), "purple")
        assert_allclose(
            scatter.errorbars_handle.lines[2][0].get_segments()[0], [[0, 2.6], [0, 3.4]]
        )
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_hidden_errorbars_reuse_artists_geometry_and_legend(self):
        curve = Curve([0, 1], [0, 1], label="Curve", color="navy")
        curve.add_errorbars(x_error=10, y_error=12, cap_width=3)
        curve.show_errorbars = False
        scatter = Scatter([0, 1], [1, 0], label="Points")
        scatter.add_errorbars(y_error=12, cap_width=2)
        scatter.show_errorbars = False
        figure = SmartFigure(
            elements=[curve, scatter], show_legend=True, figure_style="plain"
        )
        captured = []
        states = {}

        def update(i):
            ax = plt.gcf().axes[0]
            legend = ax.get_legend()
            handles = (curve.handle, scatter.errorbars_handle, legend.legend_handles[0])
            captured.append(handles)
            states[i] = (
                (ax.get_xlim(), ax.get_ylim()),
                tuple(
                    segment.get_visible()
                    for handle in handles[:2]
                    for segment in handle.lines[2]
                ),
                tuple(
                    cap.get_visible()
                    for handle in handles[:2]
                    for cap in handle.lines[1]
                ),
                handles[-1].get_visible(),
            )
            curve.show_errorbars = scatter.show_errorbars = i % 2 == 1
            curve.cap_width = (2, 3, 0, 4)[i]
            scatter.cap_width = (1, 0, 2, 3)[i]
            if i == 0:
                curve.set_data([0, 0.5, 1], [0, 0.5, 1], x_error=0.1, y_error=0.15)
                scatter.set_data([0, 0.5, 1], [1, 0.5, 0], y_error=0.1)
            if i == 2:
                curve.set_data([0, 1], [0, 1], x_error=0.1, y_error=0.1)
                scatter.set_data([0, 1], [1, 0], y_error=0.1)

        figure.save_animation(self.output, update, 4)
        self.assertTrue(all(items == captured[0] for items in captured))
        for index in (0, 1, 3):
            self.assertFalse(any(states[index][1]))
            self.assertFalse(any(states[index][2]))
            self.assertFalse(states[index][3])
        self.assertTrue(all(states[2][1]))
        self.assertEqual(states[2][2], (True, True, True, True, False, False))
        self.assertTrue(states[2][3])
        # The large configured errors were hidden at setup and did not autoscale.
        for limits, *_ in states.values():
            assert_allclose(limits, ((-0.05, 1.05), (-0.05, 1.05)))
        for handle in captured[-1][:2]:
            self.assertTrue(all(segment.get_visible() for segment in handle.lines[2]))
            self.assertTrue(all(cap.get_visible() for cap in handle.lines[1]))
            self.assertTrue(
                all(len(segment.get_segments()) == 2 for segment in handle.lines[2])
            )
        self.assertTrue(captured[-1][-1].get_visible())

    def test_errorbar_visibility_rolls_back_after_later_artist_failure(self):
        curve = Curve([0, 1], [0, 1], label="Curve")
        curve.add_errorbars(y_error=0.2, cap_width=3)
        scatter = Scatter([0, 1], [1, 0])
        figure = SmartFigure(
            elements=[curve, scatter], show_legend=True, figure_style="plain"
        )
        captured = []

        def update(i):
            captured[:] = [
                curve.handle,
                plt.gcf().axes[0].get_legend().legend_handles[0],
            ]
            curve.show_errorbars = bool(i)
            curve.cap_width = 5 if i else 2
            scatter.face_color = "red" if i else "blue"
            if i:
                original = scatter.handle.set_facecolor

                def fail_after_color(values):
                    original(values)
                    if np.allclose(values, [[1, 0, 0, 1]]):
                        self.assertTrue(captured[0].lines[2][0].get_visible())
                        raise ValueError("failed to apply colors")

                self.enterContext(
                    patch.object(scatter.handle, "set_facecolor", fail_after_color)
                )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        handle, swatch = captured
        self.assertFalse(curve.show_errorbars)
        self.assertEqual(curve.cap_width, 2)
        self.assertFalse(swatch.get_visible())
        self.assertTrue(all(not cap.get_visible() for cap in handle.lines[1]))
        self.assertTrue(all(cap.get_markersize() == 4 for cap in handle.lines[1]))
        self.assertFalse(handle.lines[2][0].get_visible())
        self.assertFalse(self.output.exists())

    def test_curve_uncertainty_bands_reuse_hidden_lines_and_fill(self):
        from matplotlib.colors import to_rgba

        curve = Curve([0, 1], [1, 2], color="navy")
        curve.add_error_curves(y_error=20, error_curves_fill_between=False)
        curve.show_error_curves = False
        nested = SmartFigure(elements=[curve])
        figure = SmartFigure(1, 2, elements=[nested, curve], figure_style="plain")
        captures = []
        states = {}

        def update(i):
            artists = [
                (ax, *ax.lines, *ax.collections) for ax in plt.gcf().axes if ax.lines
            ]
            captures.append(artists)
            states[i] = []
            for axes, _, low, high, fill in artists:
                states[i].append(
                    (
                        axes.get_ylim(),
                        low.get_visible(),
                        high.get_visible(),
                        fill.get_visible(),
                    )
                )
            curve.show_error_curves = i in (0, 1, 3)
            curve.error_curves_fill_between = i != 0
            curve.color = "green"
            if i == 2:
                curve.set_data(
                    [0, 1, 2], [1, 2, 3], y_error=[[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]
                )
            elif i == 0:
                curve.set_data([0, 1], [1, 2], y_error=0.1)

        figure.save_animation(self.output, update, 4)
        self.assertTrue(all(capture == captures[0] for capture in captures))
        for i, expected in (
            (0, (False, False, False)),
            (1, (True, True, False)),
            (2, (True, True, True)),
            (3, (False, False, False)),
        ):
            for limits, *visible in states[i]:
                assert_allclose(limits, [0.95, 2.05])
                self.assertEqual(tuple(visible), expected)
        for _, _, low, high, fill in captures[-1]:
            self.assertTrue(all(artist.get_visible() for artist in (low, high, fill)))
            assert_allclose(low.get_ydata(), [0.9, 1.8, 2.7])
            assert_allclose(high.get_ydata(), [1.3, 2.4, 3.5])
            assert_allclose(fill.get_paths()[0].vertices[:, 1].max(), 3.5)
            assert_allclose(fill.get_facecolor(), [to_rgba("green", 0.2)])

    def test_curve_decorations_restore_last_frame_after_invalid_style(self):
        from matplotlib.colors import to_rgba

        curve = Curve([0, 1, 2], [1, 2, 1], color="blue")
        curve.add_errorbars(y_error=0.3, cap_width=0)
        curve.add_error_curves(y_error=0.3, error_curves_fill_between=True)
        curve.get_area_between(0, 2, fill_between=True, fill_color="same as curve")
        figure = SmartFigure(elements=[curve], figure_style="plain")

        def update(i):
            curve.color = "orange" if i else "purple"
            curve.line_style = "--" if i else ":"
            curve.error_curves_color = "invalid color" if i else "same as curve"
            curve.error_curves_line_style = "same as curve"
            curve.error_curves_line_width = 4
            curve.fill_between_color = "green" if i else "same as curve"
            curve.cap_width = 3 if i else 2

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        self.assertEqual(curve.color, "purple")
        self.assertEqual(curve.line_style, ":")
        self.assertEqual(curve.error_curves_color, "same as curve")
        self.assertEqual(curve.cap_width, 2)
        main, caps, _ = curve.handle.lines
        self.assertEqual(main.get_color(), "purple")
        self.assertTrue(all(cap.get_markersize() == 4 for cap in caps))
        for artist in main.axes.lines[-2:]:
            self.assertEqual(artist.get_color(), "purple")
            self.assertEqual(artist.get_linestyle(), ":")
            self.assertEqual(artist.get_linewidth(), 4)
        for fill in main.axes.collections[-2:]:
            assert_allclose(fill.get_facecolor(), [to_rgba("purple", 0.2)])

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

    def test_point_marker_and_caption_styles_update_in_every_occurrence(self):
        from matplotlib.colors import to_rgba
        from matplotlib.text import Annotation

        from graphinglib import Point

        point = Point(1, 1)
        nested = SmartFigure(elements=[point], x_lim=(0, 4), y_lim=(0, 4))
        root = SmartFigure(2, 2, figure_style="plain", x_lim=(0, 4), y_lim=(0, 4))
        root[0, 0] = nested
        root[1, :] = [point]
        bindings = []
        states = {}

        def update(i):
            pairs = [
                (axis.collections[0], axis.texts[0])
                for axis in plt.gcf().axes
                if len(axis.collections)
                and len(axis.texts)
                and isinstance(axis.texts[0], Annotation)
            ]
            self.assertEqual(len(pairs), 2)
            if bindings:
                self.assertEqual(pairs, bindings)
            else:
                bindings.extend(pairs)
            states[i] = [
                (
                    note.get_text(),
                    note.get_color(),
                    note.get_fontsize(),
                    marker.get_edgecolor().copy(),
                )
                for marker, note in pairs
            ]
            point.coordinates = (i + 1, i + 1)
            point.marker_style = ("s", "x", "^")[i]
            point.face_color = ("blue", "orange", None)[i]
            point.edge_color = ("red", "blue", "green")[i]
            point.label = "sample" if i < 2 else None
            point.show_coordinates = i != 0
            point.h_align = "right"
            point.v_align = "top"
            point.font_size = 16 if i == 0 else "same as figure"
            point.marker_size = 50
            point.edge_width = 2.5
            point.alpha = 0.75

        root.save_animation(self.output, update, 3)
        for text, color, size, stroke in states[1]:
            self.assertEqual((text, color, size), ("sample ", "red", 16))
            assert_allclose(stroke, [to_rgba("red", 0.75)])
        for text, color, _, stroke in states[2]:
            self.assertEqual((text, color), ("sample : (2.000, 2.000) ", "blue"))
            assert_allclose(stroke, [to_rgba("orange", 0.75)])
        for (marker, note), (_, _, starting_size, _) in zip(
            bindings, states[0], strict=True
        ):
            self.assertEqual(note.get_text(), "(3.000, 3.000) ")
            self.assertEqual(note.get_color(), "green")
            self.assertEqual(note.get_fontsize(), starting_size)
            self.assertEqual(note.get_horizontalalignment(), "right")
            self.assertEqual(note.get_verticalalignment(), "top")
            self.assertEqual(note.xy, (3, 3))
            assert_allclose(marker.get_offsets(), [[3, 3]])
            assert_allclose(marker.get_sizes(), [50])
            assert_allclose(marker.get_linewidth(), [2.5])
            assert_allclose(marker.get_edgecolor(), [to_rgba("green", 0.75)])
            self.assertEqual(len(marker.get_facecolor()), 0)
            self.assertEqual(len(marker.get_paths()[0].vertices), 4)
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_text_highlight_and_arrow_can_appear_and_clear_during_playback(self):
        from matplotlib.colors import to_rgba
        from matplotlib.text import Annotation

        text = Text(1, 2, "start", color="black", font_size=14)
        heading = Text(0.5, 0.96, "heading")
        root = SmartFigure(
            elements=[text],
            annotations=[heading],
            figure_style="plain",
            x_lim=(0, 6),
            y_lim=(0, 6),
        )
        artists = []
        states = {}

        def update(i):
            fig = plt.gcf()
            (body,) = fig.axes[0].texts
            self.assertIsInstance(body, Annotation)
            arrow = body.arrow_patch
            panel = fig.texts[0]
            box, panel_box = body.get_bbox_patch(), panel.get_bbox_patch()
            if artists:
                self.assertEqual(artists, [body, arrow, box, panel, panel_box])
            else:
                artists.extend([body, arrow, box, panel, panel_box])
            states[i] = (
                body.xy,
                arrow.get_visible(),
                box.get_visible(),
                box.get_facecolor(),
                box.get_boxstyle().pad,
                arrow.get_facecolor(),
                panel_box.get_visible(),
            )
            text.x = 1 + i / 4
            text.text = f"sample {i}"
            text.color = "purple"
            text.font_size = 15
            text.h_align = "center"
            text.v_align = "top"
            text.alpha = 0.7
            text.rotation = 15
            text.highlight_color = None if i == 1 else "gold"
            text.highlight_alpha = 0.4
            text.highlight_padding = 0.3
            heading.highlight_color = "cyan"
            heading.highlight_padding = 0.2
            if i == 0:
                text.add_arrow(
                    (4, 4), width=1, head_width=6, head_length=5, shrink=0.15, alpha=0.6
                )
            elif i == 1:
                text.arrow_pointing_to = None
            elif i == 2:
                text.arrow_pointing_to = (8, 3)
            else:
                text.arrow_pointing_to = (5, 3)

        root.save_animation(self.output, update, 4)
        self.assertEqual(states[0][1:3], (False, False))
        self.assertFalse(states[0][-1])
        self.assertEqual(states[1][:3], ((4, 4), True, True))
        assert_allclose(states[1][3], to_rgba("gold", 0.4))
        self.assertEqual(states[1][4], 0.3)
        assert_allclose(states[1][5], to_rgba("purple", 0.6))
        self.assertEqual(states[2][1:3], (False, False))
        self.assertEqual(states[3][:3], ((8, 3), False, True))
        body, arrow, box, _, panel_box = artists
        self.assertEqual(body.xy, (5, 3))
        self.assertEqual(
            body.arrowprops,
            {"width": 1, "headwidth": 6, "headlength": 5, "shrink": 0.15, "alpha": 0.6},
        )
        self.assertTrue(arrow.get_visible())
        self.assertEqual(body.get_text(), "sample 3")
        self.assertEqual(body.get_position(), (1.75, 2))
        self.assertEqual(body.get_fontsize(), 15)
        self.assertEqual(body.get_rotation(), 15)
        self.assertEqual(body.get_horizontalalignment(), "center")
        self.assertEqual(body.get_verticalalignment(), "top")
        self.assertAlmostEqual(body.get_alpha(), 0.7)
        assert_allclose(arrow.get_facecolor(), to_rgba("purple", 0.6))
        self.assertTrue(box.get_visible())
        self.assertTrue(panel_box.get_visible())
        self.assertEqual(panel_box.get_boxstyle().pad, 0.2)

    def test_failed_point_and_text_styles_restore_previous_artists(self):
        from matplotlib.colors import to_rgba

        from graphinglib import Point

        point = Point(0, 0, label="base", face_color="blue")
        text = Text(1, 1, "start")
        figure = SmartFigure(
            elements=[point, text], figure_style="plain", x_lim=(0, 4), y_lim=(0, 4)
        )
        artists = []

        def update(i):
            axis = plt.gcf().axes[0]
            artists[:] = [axis.collections[0], *axis.texts]
            text.highlight_color = "pink" if i else "gold"
            text.highlight_padding = 0.4 if i else 0.2
            text.add_arrow((3, 3) if i else (2, 2), width=3 if i else 2)
            point.coordinates = (i + 1, i + 1)
            point.label = f"point {i}"
            point.marker_style = "D" if i else "s"
            point.face_color = "invalid color" if i else "purple"

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        marker, caption, label = artists
        self.assertEqual(point.coordinates, (1, 1))
        self.assertEqual(point.marker_style, "s")
        self.assertEqual(point.label, "point 0")
        self.assertEqual(caption.get_text(), " point 0")
        self.assertEqual(caption.xy, (1, 1))
        assert_allclose(marker.get_offsets(), [[1, 1]])
        assert_allclose(marker.get_facecolor(), [to_rgba("purple")])
        self.assertEqual(len(marker.get_paths()[0].vertices), 5)
        self.assertEqual(text.highlight_color, "gold")
        self.assertEqual(text.arrow_pointing_to, (2, 2))
        self.assertEqual(label.xy, (2, 2))
        self.assertEqual(label.arrowprops["width"], 2)
        assert_allclose(label.get_bbox_patch().get_facecolor(), to_rgba("gold"))
        self.assertEqual(label.get_bbox_patch().get_boxstyle().pad, 0.2)

    def test_arrowhead_modes_and_line_caps_reuse_patches(self):
        from warnings import catch_warnings, simplefilter

        from matplotlib.text import Annotation

        from graphinglib import Arrow, Line

        arrow = Arrow((0, 0), (4, 1), style="->")
        line = Line((0, 2), (4, 2), capped_line=False)
        nested = SmartFigure(elements=[arrow, line], x_lim=(-1, 5), y_lim=(-1, 5))
        root = SmartFigure(2, 2, figure_style="plain", x_lim=(-1, 5), y_lim=(-1, 5))
        root[0, 0] = nested
        root[1, :] = [arrow, line]
        artists = []
        observations = {}

        def update(i):
            annotations = [
                item
                for axes in plt.gcf().axes
                for item in axes.texts
                if isinstance(item, Annotation) and item.arrow_patch is not None
            ]
            captured = [(item, item.arrow_patch) for item in annotations]
            self.assertEqual(len(captured), 4)
            if artists:
                self.assertEqual(captured, artists)
            else:
                artists.extend(captured)
            observations[i] = [
                (
                    item.arrowprops["arrowstyle"],
                    item.xy,
                    item.get_position(),
                    patch.get_visible(),
                )
                for item, patch in captured
            ]
            arrow.style = ("-[", "-|>", "fancy", "->")[i]
            arrow.two_sided = i < 2
            arrow.head_size = (2, 3, 0, 0.75)[i]
            arrow.shrink = (0.125, 0.5, 0, 0)[i]
            arrow.pointB = (4, 1 + i / 2)
            line.capped_line = True
            line.cap_width = (4, 0, 2, 6)[i]
            line.pointA = (0, 2 + i / 4)

        with catch_warnings():
            simplefilter("error", RuntimeWarning)
            root.save_animation(self.output, update, 4)
        for i in (0, 2):
            self.assertIn("]-[", observations[1][i][0])
            self.assertIn("widthA=0.4", observations[1][i][0])
            self.assertIn("widthB=0.4", observations[1][i][0])
            self.assertEqual(observations[1][i][1:3], ((3.5, 0.875), (0.5, 0.125)))
            self.assertIn("<|-|>", observations[2][i][0])
            self.assertEqual(observations[2][i][1], observations[2][i][2])
            self.assertFalse(observations[2][i][3])
            self.assertEqual(observations[3][i][0], "-")
            self.assertTrue(observations[3][i][3])
            arrow_style = artists[i][1].get_arrowstyle()
            self.assertAlmostEqual(arrow_style.head_width, 0.15)
            self.assertAlmostEqual(arrow_style.head_length, 0.3)
            self.assertIn("|-|", observations[1][i + 1][0])
            self.assertEqual(observations[2][i + 1][0], "-")
            line_style = artists[i + 1][1].get_arrowstyle()
            self.assertEqual(line_style.widthA, 3)
            self.assertEqual(line_style.widthB, 3)
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 4)

    def test_polygon_layers_toggle_and_round_shapes_resample_in_place(self):
        from matplotlib.colors import to_rgba

        from graphinglib import Circle, Ellipse, Polygon

        polygon = Polygon([], fill=False, edge_color=None)
        circle = Circle(-2, -1, 1, number_of_points=16)
        ellipse = Ellipse(2, 2, 2, 1, angle=25, number_of_points=20)
        ellipse.scale(1.5, 0.8)
        ellipse.skew(10, 0)
        figure = SmartFigure(
            elements=[polygon, circle, ellipse],
            figure_style="plain",
            x_lim=(-4, 7),
            y_lim=(-4, 6),
        )
        before_circle = circle.vertices.copy()
        before_ellipse = ellipse.vertices.copy()
        artists = []
        layer_states = {}

        def update(i):
            patches = list(plt.gcf().axes[0].patches)
            if artists:
                self.assertEqual(artists, patches)
            else:
                self.assertEqual(len(patches), 6)
                artists.extend(patches)
            layer_states[i] = [
                (
                    patch.get_visible(),
                    patch.get_facecolor(),
                    patch.get_edgecolor(),
                    len(patch.get_xy()),
                )
                for patch in patches[:2]
            ]
            polygon.vertices = [] if i == 1 else [(0, 0), (0, 1 + i), (1, 0)]
            polygon.fill = i != 1
            polygon.fill_color = "orange"
            polygon.fill_alpha = 0.4
            polygon.edge_color = None if i == 1 else "green"
            polygon.line_width = 4
            polygon.line_style = ":"
            circle.number_of_points = (32, 4, 16)[i]
            ellipse.number_of_points = (40, 4, 20)[i]
            circle.x_center = -2 if i == 2 else -1
            ellipse.y_center = 2 if i == 2 else 3

        figure.save_animation(self.output, update, 3)
        self.assertEqual([row[0] for row in layer_states[0]], [False, False])
        self.assertEqual([row[3] for row in layer_states[0]], [0, 0])
        self.assertEqual([row[0] for row in layer_states[1]], [True, True])
        self.assertEqual([row[3] for row in layer_states[1]], [4, 4])
        self.assertEqual([row[0] for row in layer_states[2]], [False, False])
        assert_allclose(artists[0].get_facecolor(), to_rgba("orange", 0.4))
        assert_allclose(artists[1].get_edgecolor(), to_rgba("green"))
        self.assertEqual(artists[1].get_linewidth(), 4)
        self.assertEqual(artists[1].get_linestyle(), ":")
        assert_allclose(circle.vertices, before_circle, atol=1e-12)
        assert_allclose(ellipse.vertices, before_ellipse, atol=1e-12)
        for artist in artists[2:4]:
            assert_allclose(artist.get_xy(), circle.vertices)
        for artist in artists[4:]:
            assert_allclose(artist.get_xy(), ellipse.vertices)

    def test_failed_shape_update_restores_style_visibility_and_resolution(self):
        from graphinglib import Arrow, Circle

        arrow = Arrow((0, 0), (2, 2), style="->")
        circle = Circle(1, 1, 1, fill=False, edge_color=None, number_of_points=16)
        figure = SmartFigure(
            elements=[arrow, circle], figure_style="plain", x_lim=(-1, 4), y_lim=(-1, 4)
        )
        artists = []

        def update(i):
            axis = plt.gcf().axes[0]
            artists[:] = [axis.texts[0], *axis.patches]
            arrow.two_sided = not i
            arrow.head_size = i + 1
            circle.number_of_points = (32, 4)[i]
            circle.fill = bool(i)
            circle.edge_color = "invalid color" if i else "purple"

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        self.assertTrue(arrow.two_sided)
        self.assertEqual(arrow.head_size, 1)
        self.assertTrue(artists[0].arrowprops["arrowstyle"].startswith("<->"))
        self.assertEqual(circle.number_of_points, 32)
        self.assertEqual(len(circle.vertices), 33)
        self.assertFalse(circle.fill)
        self.assertFalse(artists[1].get_visible())
        self.assertTrue(artists[2].get_visible())
        assert_allclose(artists[2].get_xy(), circle.vertices)

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

    def test_heatmap_mesh_updates_centers_corners_and_colorbars(self):
        from matplotlib.collections import QuadMesh

        mesh = Heatmap(
            [[0, 1], [1, 0]], x_mesh=[0, 1], y_mesh=[0, 1], show_color_bar=True
        )
        nested = SmartFigure(elements=[mesh], x_lim=(-2, 8), y_lim=(-1, 6))
        root = SmartFigure(2, 2, figure_style="plain", x_lim=(-2, 8), y_lim=(-1, 6))
        root[0, 0] = nested
        root[1, :] = [mesh]
        captures = []
        states = {}

        def update(i):
            items = [
                (item, item.colorbar, item.colorbar.ax)
                for axis in plt.gcf().axes
                for item in axis.collections
                if isinstance(item, QuadMesh) and item.colorbar is not None
            ]
            self.assertEqual(len(items), 2)
            captures.append(items)
            states[i] = [
                (item.get_clim(), bar_axes.get_visible(), item.get_coordinates().copy())
                for item, _, bar_axes in items
            ]
            if i == 0:
                mesh.set_data(
                    np.arange(6).reshape(2, 3), x_mesh=[0, 2, 4], y_mesh=[0, 2]
                )
                mesh.show_color_bar = False
                mesh.color_map_range = (0, 8)
            else:
                data = np.ma.array(
                    [[4, 5], [6, 7], [8, 9]], mask=[[0, 1], [0, 0], [0, 0]]
                )
                mesh.x_mesh = [1, 2, 5]
                mesh.image = data
                mesh.y_mesh = [0, 1, 2, 4]
                mesh.color_map_range = None
                mesh.show_color_bar = True
                mesh.aspect_ratio = "equal"
                mesh.color_map = "plasma"

        root.save_animation(self.output, update, 2)
        self.assertTrue(all(items == captures[0] for items in captures))
        for clim, visible, coords in states[1]:
            self.assertEqual(clim, (0, 8))
            self.assertFalse(visible)
            assert_allclose(coords[0, :, 0], [-1, 1, 3, 5])
            assert_allclose(coords[:, 0, 1], [-1, 1, 3])
        for artist, bar, bar_axes in captures[-1]:
            assert_allclose(artist.get_coordinates()[0, :, 0], [1, 2, 5])
            assert_allclose(artist.get_coordinates()[:, 0, 1], [0, 1, 2, 4])
            self.assertEqual(len(artist.get_paths()), 6)
            self.assertTrue(artist.get_array().mask[0, 1])
            self.assertEqual(artist.get_clim(), (0, 1))
            self.assertEqual(bar.cmap.name, "plasma")
            self.assertTrue(bar_axes.get_visible())
            self.assertEqual(artist.axes.get_aspect(), 1)
            assert_allclose(artist.axes.get_xlim(), (-2, 8))
            assert_allclose(artist.axes.get_ylim(), (-1, 6))

    def test_failed_rgb_heatmap_resize_restores_mesh_coordinates_and_paths(self):
        from matplotlib.collections import QuadMesh

        mesh = Heatmap(
            np.ones((2, 2, 3)), x_mesh=[0, 1], y_mesh=[0, 1], show_color_bar=False
        )
        root = SmartFigure(elements=[mesh], figure_style="plain")
        artist = None
        previous = None

        def update(i):
            nonlocal artist, previous
            (artist,) = [
                item
                for item in plt.gcf().axes[0].collections
                if isinstance(item, QuadMesh)
            ]
            count = i + 3
            data = np.zeros((2, count, 4))
            data[..., 3] = 1
            mesh.set_data(data, x_mesh=np.arange(count), y_mesh=[0, 1])
            if i:
                previous = artist.get_paths()
                setter = artist.set_array

                def fail_after_set(value):
                    setter(value)
                    if value.shape[1] == 4:
                        raise ValueError("could not update pixel colors")

                self.enterContext(patch.object(artist, "set_array", fail_after_set))

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        self.assertIs(artist.get_paths(), previous)
        self.assertEqual(artist.get_array().shape, (2, 3, 4))
        self.assertEqual(mesh.image.shape, (2, 3, 4))
        assert_allclose(mesh.x_mesh, [0, 1, 2])
        assert_allclose(artist.get_coordinates()[0, :, 0], [-0.5, 0.5, 1.5, 2.5])

    def test_filled_contour_updates_mesh_levels_and_existing_colorbars(self):
        from matplotlib.contour import QuadContourSet

        from graphinglib import Contour

        xx, yy = np.meshgrid(np.arange(3), np.arange(3))
        contour = Contour(xx + yy, levels=[0, 2, 4], filled=True, show_color_bar=True)
        contour.set_color_bar_params(label="Field", ticks=[1, 4], format="%.1f")
        nested = SmartFigure(elements=[contour], x_lim=(0, 8), y_lim=(0, 6))
        root = SmartFigure(2, 2, figure_style="plain", x_lim=(0, 8), y_lim=(0, 6))
        root[0, 0] = nested
        root[1, :] = [contour]
        captures = []

        def update(i):
            artists = tuple(
                item
                for axis in plt.gcf().axes
                for item in axis.collections
                if isinstance(item, QuadContourSet)
            )
            self.assertEqual(len(artists), 2)
            captures.append(
                tuple((item, item.colorbar, item.colorbar.ax) for item in artists)
            )
            x, y = np.arange(3 + i) + i, np.arange(3) * (i + 1)
            contour.set_data(x + y[:, None], x_mesh=x, y_mesh=y)
            contour.levels = [1, 2, 5] if i == 0 else [3, 4, 6, 8]
            contour.color_map_range = (0, 10) if i == 0 else None
            contour.color_map = "magma"
            contour.alpha = 0.45 if i else 0.75

        root.save_animation(self.output, update, 2)
        self.assertTrue(all(sample == captures[0] for sample in captures))
        for artist, colorbar, axes in captures[-1]:
            self.assertEqual(len(artist.get_paths()), 3)
            assert_allclose(artist.levels, [3, 4, 6, 8])
            assert_allclose(artist.get_array(), [3.5, 5, 7])
            self.assertEqual(artist.get_clim(), (3, 8))
            self.assertEqual(artist.get_alpha(), 0.45)
            self.assertEqual(colorbar.cmap.name, "magma")
            assert_allclose(colorbar.boundaries, [3, 4, 6, 8])
            assert_allclose(colorbar.solids.get_array().ravel(), [3.5, 5, 7])
            assert_allclose(colorbar.get_ticks(), [1, 4])
            self.assertEqual(axes.get_ylabel(), "Field")
            self.assertEqual(colorbar.formatter(4), "4.0")
            assert_allclose(artist.axes.get_xlim(), (0, 8))
            assert_allclose(artist.axes.get_ylim(), (0, 6))
            x, y = np.arange(4) + 1, np.arange(3) * 2
            reference_figure, reference_axes = plt.subplots()
            reference = reference_axes.contourf(
                x, y, x + y[:, None], levels=[3, 4, 6, 8]
            )
            for path, expected in zip(
                artist.get_paths(), reference.get_paths(), strict=True
            ):
                assert_allclose(path.vertices, expected.vertices)
                np.testing.assert_array_equal(path.codes, expected.codes)
            plt.close(reference_figure)
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_line_contour_moves_from_single_level_through_masked_frames(self):
        from warnings import catch_warnings, simplefilter

        from matplotlib.contour import QuadContourSet

        from graphinglib import Contour

        x, y = np.meshgrid(np.arange(4), np.arange(3))
        contour = Contour(x + y, levels=[1], filled=False, show_color_bar=True)
        contour.set_color_bar_params(position="bottom")
        figure = SmartFigure(elements=[contour], figure_style="plain")
        samples = {}
        artists = []

        def update(i):
            (artist,) = [
                c
                for ax in plt.gcf().axes
                for c in ax.collections
                if isinstance(c, QuadContourSet)
            ]
            artists.append(artist)
            samples[i] = (
                np.array(artist.levels),
                tuple(len(path.vertices) for path in artist.get_paths()),
                artist.get_clim(),
                tuple(artist.colorbar.get_ticks()),
            )
            if i == 0:
                return
            if i == 1:
                contour.set_data(2 * (x + y))
                contour.levels = 4
                contour.line_widths = [1, 2]
            elif i == 2:
                contour.z_data = np.ma.masked_all((5, 3))
            elif i == 3:
                contour.set_data(100 + x + y)
                contour.levels = 4
            else:
                contour.set_data(2 * (x + y))
                contour.levels = [3]
                contour.color_map_range = (0, 5)
                contour.alpha = 0.4

        with catch_warnings():
            simplefilter("error", RuntimeWarning)
            figure.save_animation(self.output, update, 5)
        self.assertTrue(all(item is artists[0] for item in artists))
        self.assertEqual(samples[0][3], (1,))
        assert_allclose(samples[3][0], samples[2][0])
        self.assertTrue(all(not n for n in samples[3][1]))
        assert_allclose(samples[3][2], samples[2][2])
        self.assertGreater(samples[4][0][0], samples[3][0][-1])
        item = artists[0]
        assert_allclose(item.levels, [3])
        self.assertEqual(len(item.get_paths()), 1)
        self.assertEqual(item.get_clim(), (0, 5))
        self.assertEqual(item.get_alpha(), 0.4)
        bar = item.colorbar
        self.assertEqual(bar.orientation, "horizontal")
        assert_allclose(bar.get_ticks(), [3])
        assert_allclose(bar.lines[0].get_segments(), [[[3, 0], [3, 1]]])
        assert_allclose(bar.lines[0].get_linewidth(), [1])
        assert_allclose(bar.lines[0].get_colors()[:, -1], [0.4])

    def test_contour_with_empty_start_uses_first_finite_range(self):
        from matplotlib.contour import QuadContourSet

        from graphinglib import Contour

        contour = Contour(np.ma.masked_all((3, 3)), levels=4, show_color_bar=True)
        figure = SmartFigure(elements=[contour], figure_style="plain")
        states = {}

        def update(i):
            (artist,) = [
                c
                for ax in plt.gcf().axes
                for c in ax.collections
                if isinstance(c, QuadContourSet)
            ]
            states[i] = artist.levels.copy(), artist.colorbar.solids
            if i == 1:
                contour.set_data(100 + np.arange(9).reshape((3, 3)))
            elif i == 2:
                contour.set_data(105 + np.arange(9).reshape((3, 3)))

        figure.save_animation(self.output, update, 4)
        self.assertLess(states[1][0][-1], 2)
        self.assertGreaterEqual(states[2][0][0], 95)
        assert_allclose(states[3][0], states[2][0])
        self.assertIs(states[3][1], states[2][1])

    def test_contour_can_switch_modes_and_hide_its_existing_colorbar(self):
        from matplotlib.contour import QuadContourSet

        from graphinglib import Contour

        z = np.arange(9).reshape((3, 3))
        contour = Contour(z, levels=[0, 4, 8], filled=True, show_color_bar=True)
        figure = SmartFigure(elements=[contour], figure_style="plain")
        captures = {}

        def update(i):
            (artist,) = [
                c
                for ax in plt.gcf().axes
                for c in ax.collections
                if isinstance(c, QuadContourSet)
            ]
            bar = artist.colorbar
            captures[i] = (
                artist,
                bar,
                bar.ax,
                artist.filled,
                len(artist.get_paths()),
                bar.ax.get_visible(),
                bar.solids,
                tuple(bar.lines),
                artist.get_facecolor().copy(),
                artist.get_edgecolor().copy(),
            )
            contour.filled = bool(i % 2)
            contour.show_color_bar = i > 0
            contour.line_widths = [1, 2]

        figure.save_animation(self.output, update, 3)
        self.assertTrue(
            all(sample[:3] == captures[0][:3] for sample in captures.values())
        )
        _, bar, axes, filled, count, visible, solids, lines, faces, edges = captures[1]
        self.assertEqual((filled, count, visible, solids), (False, 3, False, None))
        self.assertEqual(len(lines), 1)
        assert_allclose(lines[0].get_linewidth(), [1, 2, 1])
        self.assertEqual(len(faces), 0)
        self.assertEqual(len(edges), 3)

        _, _, _, filled, count, visible, solids, lines, faces, edges = captures[2]
        self.assertEqual((filled, count, visible, lines), (True, 2, True, ()))
        self.assertIsNotNone(solids)
        self.assertEqual(len(faces), 2)
        self.assertEqual(len(edges), 0)
        self.assertTrue(axes.get_visible())
        self.assertEqual(len(bar.lines), 1)
        self.assertIsNone(bar.solids)
        self.assertEqual(
            len(axes.collections), 2
        )  # dividers and current line collection

    def test_failed_contour_update_restores_paths_levels_and_colorbar(self):
        from matplotlib.contour import QuadContourSet

        from graphinglib import Contour

        z = np.arange(9).reshape((3, 3))
        contour = Contour(z, levels=[0, 4, 8], filled=True, show_color_bar=True)
        figure = SmartFigure(elements=[contour], figure_style="plain")
        artist = None
        previous_paths = None

        def update(i):
            nonlocal artist, previous_paths
            (artist,) = [
                c
                for ax in plt.gcf().axes
                for c in ax.collections
                if isinstance(c, QuadContourSet)
            ]
            contour.set_data(z + i)
            contour.levels = [2, 5, 7, 9] if i else [0, 2, 8]
            contour.color_map = "magma" if i else "plasma"
            contour.filled = i == 0
            if i:
                previous_paths = tuple(artist.get_paths())
                original = artist.changed

                def fail_after_refresh():
                    original()
                    if artist.get_cmap().name == "magma":
                        raise ValueError("could not update contour colors")

                self.enterContext(patch.object(artist, "changed", fail_after_refresh))

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        assert_allclose(contour.z_data, z)
        assert_allclose(contour.levels, [0, 2, 8])
        self.assertEqual(contour.color_map, "plasma")
        self.assertTrue(contour.filled)
        self.assertTrue(artist.filled)
        self.assertEqual(tuple(artist.get_paths()), previous_paths)
        assert_allclose(artist.levels, [0, 2, 8])
        self.assertEqual(artist.get_cmap().name, "plasma")
        assert_allclose(artist.colorbar.boundaries, [0, 2, 8])
        assert_allclose(artist.colorbar.solids.get_array().ravel(), [1, 5])
        assert_allclose(artist.colorbar.get_ticks(), [0, 2, 8])

    def test_vector_field_geometry_updates_in_nested_spanning_plots(self):
        from matplotlib.colors import to_rgba
        from matplotlib.quiver import Quiver

        from graphinglib import VectorField

        field = VectorField([0], [0], [1], [0], scale=1)
        nested = SmartFigure(elements=[field], x_lim=(-1, 5), y_lim=(-1, 3))
        root = SmartFigure(
            2, 2, figure_style="plain", x_lim=[None, (-1, 5)], y_lim=[None, (-1, 3)]
        )
        root[0, 0] = nested
        root[1, :] = [field]
        observations = []
        artists = []

        def update(i):
            current = [
                item
                for ax in plt.gcf().axes
                for item in ax.collections
                if isinstance(item, Quiver)
            ]
            self.assertEqual(len(current), 2)
            if artists:
                self.assertTrue(
                    all(old is new for old, new in zip(artists, current, strict=True))
                )
            else:
                artists.extend(current)
            if i:
                observations.append((i, [len(item.get_paths()) for item in current]))
            if i == 1:
                self.assertTrue(all(item.headwidth == 0 for item in current))
            if i == 0:
                field.set_data([1, 0], [0, 1], x_data=[1, 2], y_data=[0, 1])
            elif i == 1:
                field.set_data([], [], x_data=[], y_data=[])
            elif i == 2:
                field.set_data(
                    0.5,
                    np.ma.array(
                        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], mask=[[0, 1, 0], [0, 0, 0]]
                    ),
                    x_data=[0, 2, 4],
                    y_data=[0, 2],
                )
            else:
                field.set_data(
                    np.ma.array([1, 0, 3, 1, 2, 3], mask=[0, 1, 0, 0, 0, 0]), 0.75
                )
            field.arrow_width = min(i + 1, 3)
            field.arrow_head_size = 0 if i == 0 else 2
            field.scale = 2
            field.make_angles_axes_independent = i >= 2
            field.color = "red" if i >= 2 else "blue"

        root.save_animation(self.output, update, 4)
        self.assertEqual(observations, [(1, [2, 2]), (2, [0, 0]), (3, [6, 6])])
        coords = [[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2]]
        for item in artists:
            assert_allclose(item.get_offsets(), coords)
            assert_allclose(item.XY, coords)
            assert_allclose(item.V, np.full(6, 0.75))
            np.testing.assert_array_equal(item.Umask, [0, 1, 0, 0, 0, 0])
            self.assertEqual(item.N, 6)
            self.assertEqual(len(item.get_paths()), 6)
            self.assertTrue(np.isnan(item.get_paths()[1].vertices).all())
            self.assertAlmostEqual(item.width, 0.015)
            for head in (item.headwidth, item.headlength, item.headaxislength):
                self.assertAlmostEqual(head, 8 / 3)
            self.assertEqual(item.angles, "uv")
            self.assertEqual(item.scale, 0.5)
            assert_allclose(item.get_facecolor(), [to_rgba("red")])
            assert_allclose(item.axes.get_xlim(), (-1, 5))
            assert_allclose(item.axes.get_ylim(), (-1, 3))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 4)

    def test_vector_field_auto_scale_waits_for_motion_and_can_reset(self):
        from warnings import catch_warnings, simplefilter

        from matplotlib.quiver import Quiver

        from graphinglib import VectorField

        field = VectorField([], [], [], [], make_angles_axes_independent=True)
        root = SmartFigure(
            elements=[field], figure_style="plain", x_lim=(-2, 2), y_lim=(-2, 2)
        )
        states = {}
        quiver = None

        def update(i):
            nonlocal quiver
            (quiver,) = [
                item
                for item in plt.gcf().axes[0].collections
                if isinstance(item, Quiver)
            ]
            states[i] = quiver.scale, len(quiver.get_paths())
            if i == 0:
                field.set_data(0, 0, x_data=[0], y_data=[0])
            elif i == 1:
                field.set_data(1, 0.5)
            elif i == 2:
                field.set_data(2, 1)
            elif i == 3:
                field.scale = None
                field.set_data(4, 2)
            else:
                field.set_data(np.ma.masked, np.nan)

        with catch_warnings(), np.errstate(divide="raise", invalid="raise"):
            simplefilter("error", RuntimeWarning)
            root.save_animation(self.output, update, 5)

        self.assertEqual(states[0], (None, 0))
        self.assertEqual(states[1], (None, 1))
        first_nonzero = states[2][0]
        self.assertGreater(first_nonzero, 0)
        self.assertEqual(states[3][0], first_nonzero)
        self.assertAlmostEqual(states[4][0], 4 * first_nonzero)
        self.assertEqual(quiver.scale, states[4][0])
        self.assertTrue(np.isnan(quiver.get_paths()[0].vertices).all())

    def test_vector_field_failed_frame_restores_model_and_artist(self):
        from matplotlib.quiver import Quiver

        from graphinglib import VectorField

        field = VectorField([0], [0], [1], [0], arrow_width=1, scale=1)
        root = SmartFigure(elements=[field], figure_style="plain")
        quiver = None

        def update(i):
            nonlocal quiver
            (quiver,) = [
                item
                for item in plt.gcf().axes[0].collections
                if isinstance(item, Quiver)
            ]
            count = i + 2
            field.set_data(1, 0, x_data=np.arange(count), y_data=np.zeros(count))
            field.arrow_width = count
            field.scale = count
            if i:
                original = quiver.set_UVC

                def fail_after_edit(u, v):
                    original(u, v)
                    if len(u) == 3:
                        raise ValueError("could not update vectors")

                self.enterContext(patch.object(quiver, "set_UVC", fail_after_edit))

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertFalse(self.output.exists())
        assert_allclose(field.x_data, [0, 1])
        assert_allclose(quiver.get_offsets(), [[0, 0], [1, 0]])
        assert_allclose(quiver.XY, [[0, 0], [1, 0]])
        assert_allclose(quiver.U, [1, 1])
        self.assertEqual(field.arrow_width, 2)
        self.assertEqual(field.scale, 2)
        self.assertAlmostEqual(quiver.width, 0.01)
        self.assertEqual(quiver.scale, 0.5)

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
                if isinstance(item, Polygon) and item.get_visible()
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

    def test_histogram_reuses_bars_and_updates_orientation_density_and_styles(self):
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Rectangle

        from graphinglib import Histogram

        data = [0, 1, 1, 2, 3, 5]
        hist = Histogram(
            data, bins=2, label="Values", show_params=True, hist_type="bar"
        )
        hist.add_pdf(show_mean=True, show_std=True)
        nested = SmartFigure(
            elements=[hist], show_legend=True, x_lim=(0, 7), y_lim=(0, 7)
        )
        root = SmartFigure(
            1,
            2,
            elements=[nested, hist],
            figure_style="plain",
            x_lim=(0, 7),
            y_lim=(0, 7),
        )
        items = []

        def update(i):
            axes = [ax for ax in plt.gcf().axes if ax.patches]
            items.append(
                [
                    [bar for bar in ax.patches if isinstance(bar, Rectangle)]
                    for ax in axes
                ]
            )
            hist.bins = (5, 1, [0, 1, 3, 6])[i]
            hist.normalize = i == 2
            hist.orientation = "horizontal" if i == 2 else "vertical"
            hist.show_params = i != 2
            hist.face_color = "orange"
            hist.edge_color = "purple"
            hist.alpha = 0.4
            hist.line_width = 3

        root.save_animation(self.output, update, 3)
        self.assertEqual([len(group) for group in items[0]], [2, 2])
        self.assertEqual([len(group) for group in items[1]], [5, 5])
        self.assertEqual(items[1], items[2])
        expected = [1 / 6, 1 / 4, 1 / 9]
        for initial, final in zip(items[0], items[-1], strict=True):
            self.assertEqual(initial, final[:2])
            self.assertEqual(
                [patch.get_visible() for patch in final],
                [True, True, True, False, False],
            )
            assert_allclose([patch.get_width() for patch in final[:3]], expected)
            assert_allclose([patch.get_y() for patch in final[:3]], [0, 1, 3])
            assert_allclose([patch.get_height() for patch in final[:3]], [1, 2, 3])
            for bar in final[:3]:
                assert_allclose(bar.get_facecolor(), to_rgba("orange", 0.4))
                assert_allclose(bar.get_edgecolor(), to_rgba("purple"))
                self.assertEqual(bar.get_linewidth(), 3)
            axes = final[0].axes
            assert_allclose(axes.lines[0].get_ydata()[[0, -1]], [0, 6])
            mean = axes.collections[1].get_segments()[0]
            assert_allclose(mean[:, 1], [hist.mean, hist.mean])
            if axes.get_legend() is not None:
                legend = axes.get_legend()
                self.assertEqual(legend.get_texts()[0].get_text(), "Values")
                assert_allclose(
                    legend.legend_handles[0].get_facecolor(), to_rgba("orange", 0.4)
                )

    def test_histogram_failed_growth_rolls_back_new_bars_and_normalized_state(self):
        from matplotlib.patches import Rectangle

        from graphinglib import Histogram

        hist = Histogram([0, 1, 2, 3], bins=2, normalize=False, hist_type="bar")
        root = SmartFigure(elements=[hist], figure_style="plain")
        observed = []

        def update(i):
            axis = plt.gcf().axes[0]
            observed[:] = [axis, *axis.patches]
            hist.bins = 5 if i else 2
            hist.normalize = bool(i)
            hist.data = [0, 1, 2, 3, 4] if i else [0, 1, 2, 3]
            if i:
                original = axis.add_artist

                def fail_after_attach(artist):
                    original(artist)
                    raise ValueError("failed to add bar")

                self.enterContext(patch.object(axis, "add_artist", fail_after_attach))

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        axis, *bars = observed
        self.assertEqual(list(axis.patches), bars)
        self.assertEqual(hist.bins, 2)
        self.assertFalse(hist.normalize)
        assert_allclose(hist.data, [0, 1, 2, 3])
        assert_allclose(hist.bin_heights, [2, 2])
        assert_allclose(
            [item.get_height() for item in bars if isinstance(item, Rectangle)], [2, 2]
        )
        self.assertFalse(self.output.exists())

    def test_histogram_modes_and_pdf_visibility_reuse_artists(self):
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Polygon, Rectangle

        from graphinglib import Histogram

        hist = Histogram([0, 1, 2, 3], bins=2, normalize=False, hist_type="stepfilled")
        root = SmartFigure(
            elements=[hist], figure_style="plain", x_lim=(0, 5), y_lim=(0, 5)
        )
        captures = []
        states = {}

        def update(i):
            ax = plt.gcf().axes[0]
            (step,) = [artist for artist in ax.patches if isinstance(artist, Polygon)]
            (pdf,) = ax.lines
            std, mean = ax.collections
            captures.append((step, pdf, std, mean))
            bars = tuple(item for item in ax.patches if isinstance(item, Rectangle))
            states[i] = (
                step.get_visible(),
                step.get_fill(),
                step.get_closed(),
                tuple(bar for bar in bars if bar.get_visible()),
                pdf.get_visible(),
                std.get_visible(),
                mean.get_visible(),
            )
            hist.hist_type = ("bar", "step", "stepfilled", "barstacked")[i]
            if i == 0:
                hist.add_pdf(
                    show_std=False,
                    show_mean=True,
                    curve_color="purple",
                    mean_color="green",
                )
            elif i == 1:
                hist.remove_pdf()
            else:
                hist.add_pdf(
                    show_std=True,
                    show_mean=False,
                    std_color="orange",
                    curve_color="navy",
                )

        root.save_animation(self.output, update, 4)
        self.assertTrue(all(items == captures[0] for items in captures))
        self.assertEqual(states[0][:3], (True, True, True))
        self.assertEqual(states[0][4:], (False, False, False))
        self.assertFalse(states[1][0])
        self.assertEqual(len(states[1][3]), 2)
        self.assertEqual(states[1][4:], (True, False, True))
        self.assertEqual(states[2], (True, False, False, (), False, False, False))
        self.assertEqual(states[3], (True, True, True, (), True, True, False))
        step, pdf, std, mean = captures[0]
        self.assertFalse(step.get_visible())
        self.assertTrue(pdf.get_visible())
        self.assertEqual(pdf.get_color(), "navy")
        self.assertTrue(std.get_visible())
        self.assertFalse(mean.get_visible())
        assert_allclose(std.get_colors(), [to_rgba("orange")])
        self.assertEqual(
            tuple(
                artist
                for artist in pdf.axes.patches
                if isinstance(artist, Rectangle) and artist.get_visible()
            ),
            states[1][3],
        )

    def test_failed_histogram_mode_and_pdf_change_restore_visible_artists(self):
        from matplotlib.patches import Polygon

        from graphinglib import Histogram

        hist = Histogram([0, 1, 2], bins=2, hist_type="bar")
        root = SmartFigure(elements=[hist], figure_style="plain")
        seen = []

        def update(i):
            ax = plt.gcf().axes[0]
            (step,) = [artist for artist in ax.patches if isinstance(artist, Polygon)]
            seen[:] = [step, *ax.lines, *ax.collections]
            hist.hist_type = "step" if i else "stepfilled"
            hist.add_pdf(curve_color="bad color" if i else "green", show_mean=False)
            hist.bins = 4 if i else 3

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        step, pdf, std, mean = seen
        self.assertEqual(hist.hist_type, "stepfilled")
        self.assertEqual(hist.bins, 3)
        self.assertEqual(hist._pdf_curve_color, "green")
        self.assertTrue(step.get_visible())
        self.assertTrue(step.get_fill())
        self.assertTrue(step.get_closed())
        self.assertEqual(pdf.get_color(), "green")
        self.assertTrue(std.get_visible())
        self.assertFalse(mean.get_visible())
        self.assertFalse(self.output.exists())

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

        with patch.object(stream, "_draw_stream", wraps=stream._draw_stream) as draw:
            root.save_animation(self.output, update, 2)
        draw.assert_called_once()
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
        with self.assertRaisesRegex(ValueError, "cover the animated fill interval"):
            figure.save_animation(self.output, update, 3)
        self.assertEqual(self.output.read_bytes(), b"previous")
        self.assertTrue(all(items == seen[0] for items in seen))
        self.assertEqual(len(seen[0]), 2)
        for artist in seen[0]:
            vertices = artist.get_paths()[0].vertices
            assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()], [1, 5])
        assert_allclose(first.x_data, [0, 1, 2])
        assert_allclose(second.y_data, [5, 5])

    def test_curve_fill_intervals_hide_and_follow_both_curves(self):
        x = [0, 1, 2, 3]
        zero = Curve(x, [2, 3, 4, 5])
        shared = Curve(x, [1, 2, 3, 4])
        boundary = Curve([0, 3], [2, 2])
        zero.get_area_between(0, 3, fill_between=True)
        shared.get_area_between(0, 3, fill_between=True, other_curve=boundary)
        shared.fill_between_bounds = None
        root = SmartFigure(1, 2, elements=[[zero], [shared]], figure_style="plain")
        artists = []
        before = {}

        def update(i):
            current = [axis.collections[0] for axis in plt.gcf().axes if axis.lines]
            if artists:
                self.assertEqual(artists, current)
            else:
                artists.extend(current)
            before[i] = [artist.get_visible() for artist in current]
            if i == 0:
                zero.fill_between_bounds = (1, 3)
                shared.fill_between_bounds = (0.5, 2.5)
                boundary.y_data = [6, 6]
            elif i == 1:
                zero.fill_between_bounds = (2, 2)
                shared.fill_between_bounds = None
                boundary.set_data([1, 4], [7, 7])
            else:
                shared.fill_between_bounds = (1.25, 2.75)
                shared.y_data = [3, 4, 5, 6]
                boundary.y_data = [8, 8]
                zero.fill_between_bounds = (1, 2.5)

        root.save_animation(self.output, update, 3)
        self.assertEqual(before, {0: [True, False], 1: [True, True], 2: [False, False]})
        self.assertTrue(all(artist.get_visible() for artist in artists))
        zero_xy, shared_xy = (artist.get_paths()[0].vertices for artist in artists)
        assert_allclose([zero_xy[:, 0].min(), zero_xy[:, 0].max()], [1, 2])
        assert_allclose([shared_xy[:, 0].min(), shared_xy[:, 0].max()], [1.25, 2.75])
        assert_allclose([shared_xy[:, 1].min(), shared_xy[:, 1].max()], [4.25, 8])

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

    def test_scatter_colorbars_hide_and_restore_independent_scales(self):
        from matplotlib.colors import Normalize

        expected_intensity = Normalize(1.8, 2.2)([2.0, 2.0])
        scatter = Scatter(
            [0, 1],
            [1, 0],
            face_color=[2.0, 2.0],
            edge_color=[10.0, 20.0],
            color_map="viridis",
            show_color_bar=True,
        )
        nested = SmartFigure(elements=[scatter], x_lim=(0, 1), y_lim=(0, 1))
        root = SmartFigure(1, 2, elements=[nested, scatter], figure_style="plain")
        states = {}
        identities = []

        def update(i):
            bars = [
                axis._colorbar
                for axis in plt.gcf().axes
                if axis.get_label() == "<colorbar>"
            ]
            self.assertEqual(len(bars), 4)
            identities.append([(bar, bar.ax) for bar in bars])
            states[i] = (
                [
                    (
                        bar.ax.get_visible(),
                        bar.mappable.get_clim(),
                        bar.cmap.name,
                        bar.solids,
                    )
                    for bar in bars
                ],
                scatter.handle.get_facecolor().copy(),
            )
            if i == 0:
                scatter.show_color_bar = False
                scatter.marker_style = "s"
            elif i == 1:
                scatter.color_map = "plasma"
                scatter.color_map_range = (0, 25)
            elif i == 2:
                scatter.show_color_bar = True
            else:
                scatter.color_map_range = None

        root.save_animation(self.output, update, 4)
        self.assertTrue(all(group == identities[0] for group in identities))
        assert_allclose(states[0][1], states[1][1])
        assert_allclose(states[0][1], plt.get_cmap("viridis")(expected_intensity))
        self.assertTrue(all(not state[0] for state in states[1][0]))
        self.assertTrue(
            all(
                state[3] is old[3]
                for state, old in zip(states[1][0], states[0][0], strict=True)
            )
        )
        self.assertEqual(
            [state[:3] for state in states[2][0]], [(False, (0, 25), "plasma")] * 4
        )
        self.assertEqual(
            [state[:3] for state in states[3][0]], [(True, (0, 25), "plasma")] * 4
        )
        for bar, axis in identities[-1]:
            self.assertTrue(axis.get_visible())
            self.assertEqual(bar.cmap.name, "plasma")
        limits = [bar.mappable.get_clim() for bar, _ in identities[-1]]
        assert_allclose(limits, [(1.8, 2.2), (10, 20)] * 2)
        assert_allclose(
            scatter.handle.get_facecolor(), plt.get_cmap("plasma")(expected_intensity)
        )

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

    def test_axis_inversion_combines_with_shared_spanning_and_twin_limits(self):
        from graphinglib import SmartTwinAxis

        a = Curve([0, 6], [0, 6])
        b = Curve([0, 6], [1, 2])
        c = Curve([0, 6], [2, 3])
        right = Curve([0, 6], [10, 30])
        twin = SmartTwinAxis(elements=[right], axis_lim=(5, 50))
        nested = SmartFigure(elements=[c], x_lim=(0, 6), y_lim=(0, 6), twin_y_axis=twin)
        root = SmartFigure(
            2, 3, share_x=True, x_lim=(0, 6), y_lim=(0, 6), figure_style="plain"
        )
        root[:, 0] = nested
        root[0, 1:3] = [a]
        root[1, 1:3] = [b]
        views = []

        def update(i):
            axes = tuple(curve.handle.lines[0].axes for curve in (a, b, c, right))
            views.append(
                (
                    axes,
                    tuple(ax.get_xlim() for ax in axes),
                    tuple(ax.get_ylim() for ax in axes),
                )
            )
            if i == 0:
                root.x_lim = [None, (1, 5), None]
                root.invert_x_axis = [None, None, True]
                nested.invert_y_axis = True
                nested.y_lim = (2, None)
                twin.invert_axis = True
                twin.axis_lim = (10, 40)
            elif i == 1:
                root.x_lim = (None, 4)
                twin.axis_lim = (20, 30)
                nested.y_lim = None
                nested.invert_y_axis = False
            else:
                root.invert_x_axis = False
                root.x_lim = None
                twin.invert_axis = False

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(record[0] == views[0][0] for record in views))
        self.assertEqual(views[1][1], ((5, 1), (5, 1), (0, 6), (0, 6)))
        self.assertEqual(views[1][2], ((0, 6), (0, 6), (6, 2), (40, 10)))
        self.assertEqual(views[2][1][:2], ((4, 1), (4, 1)))
        self.assertEqual(views[2][2][2:], ((0, 6), (30, 20)))
        a_axes, b_axes, nested_axes, twin_axes = views[-1][0]
        self.assertEqual(a_axes.get_xlim(), (0, 6))
        self.assertEqual(b_axes.get_xlim(), (0, 6))
        self.assertEqual(nested_axes.get_ylim(), (0, 6))
        self.assertEqual(twin_axes.get_ylim(), (20, 30))

    def test_conflicting_inversion_of_shared_axes_restores_previous_state(self):
        a = Curve([0, 6], [0, 6])
        b = Curve([0, 6], [1, 5])
        root = SmartFigure(
            1, 2, elements=[a, b], share_x=True, x_lim=(0, 6), figure_style="plain"
        )
        self.output.write_bytes(b"existing")

        def update(i):
            root.invert_x_axis = True if i == 0 else [True, False]
            root.x_lim = (1, 4) if i == 0 else (0, 6)

        with self.assertRaisesRegex(ValueError, "matching inversion settings"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertIs(root.invert_x_axis, True)
        self.assertEqual(root.x_lim, (1, 4))
        for curve in (a, b):
            self.assertEqual(curve.handle.lines[0].axes.get_xlim(), (4, 1))

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

    def test_live_axis_visibility_is_local_to_nested_spanning_and_twin_axes(self):
        from graphinglib import SmartTwinAxis

        upper, lower, inner, secondary = (Curve([1, 10], [1, 10]) for _ in range(4))
        twin = SmartTwinAxis(elements=[secondary], remove_axes=True, remove_ticks=True)
        nested = SmartFigure(elements=[inner], twin_y_axis=twin, y_label="Nested")
        root = SmartFigure(
            2,
            3,
            share_x=True,
            remove_x_ticks=True,
            x_lim=(1, 10),
            figure_style="plain",
        )
        root[:, 0] = nested
        root[0, 1:3] = [upper]
        root[1, 1:3] = [lower]
        root.set_ticks(
            x_ticks=[1, 10], x_tick_labels=["one", "ten"], minor_x_ticks=[5]
        )
        root.set_tick_params("x", "minor", draw_bottom_ticks=False, draw_top_ticks=True)
        snapshots = []

        def update(i):
            artists = tuple(
                curve.handle.lines[0] for curve in (upper, lower, inner, secondary)
            )
            a, b, c, d = (artist.axes for artist in artists)
            snapshots.append(
                (
                    artists,
                    (a.axison, b.axison, c.axison, d.axison),
                    tuple(
                        (
                            tick.tick1line.get_visible(),
                            tick.tick2line.get_visible(),
                            tick.label1.get_visible(),
                            tick.label2.get_visible(),
                        )
                        for tick in (
                            a.xaxis.get_major_ticks(1)[0],
                            b.xaxis.get_major_ticks(1)[0],
                            c.yaxis.get_major_ticks(1)[0],
                            d.yaxis.get_major_ticks(1)[0],
                        )
                    ),
                    c.get_ylabel(),
                    b.get_xlim(),
                    b.get_xscale(),
                    b.xaxis.get_major_locator(),
                    tuple(
                        ax.xaxis.get_minor_ticks(1)[0].tick2line.get_visible()
                        for ax in (a, b)
                    ),
                )
            )
            if i == 0:
                root.remove_x_ticks = False
                root.remove_axes = [None, True, None]
                root.log_scale_x = True
                root.x_lim = (1, 100)
                nested.remove_y_ticks = True
                twin.remove_axes = False
                twin.remove_ticks = False
            elif i == 1:
                root.remove_x_ticks = [None, None, True]
                root.remove_axes = False
                nested.remove_y_ticks = False
                twin.remove_ticks = True

        root.save_animation(self.output, update, 3, interval=100)
        off = (False, False, False, False)
        self.assertEqual(snapshots[0][1], (True, True, True, False))
        self.assertEqual(snapshots[0][2], (off, off, (True, False, True, False), off))
        self.assertEqual(snapshots[1][1], (False, True, True, True))
        self.assertEqual(
            snapshots[1][2],
            (
                (True, False, False, False),
                (True, False, True, False),
                off,
                (False, True, False, True),
            ),
        )
        self.assertEqual(snapshots[2][1], (True,) * 4)
        self.assertEqual(
            snapshots[2][2],
            ((True, False, False, False), off, (True, False, True, False), off),
        )
        self.assertEqual(snapshots[1][3:6], ("Nested", (1, 100), "log"))
        self.assertEqual(
            [snapshot[7] for snapshot in snapshots],
            [(False, False), (True, True), (True, False)],
        )
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)
            self.assertEqual(gif.info["duration"], 100)
            gif.seek(1)
            self.assertEqual(gif.info["duration"], 200)
        for snapshot in snapshots:
            self.assertEqual(snapshot[0], snapshots[0][0])
            self.assertIs(snapshot[6], snapshots[0][6])

    def test_failed_frame_restores_axis_and_tick_visibility(self):
        scatter = Scatter([1, 10], [1, 10], face_color="navy")
        root = SmartFigure(
            elements=[scatter], remove_x_ticks=True, figure_style="plain"
        )

        def update(i):
            if not i:
                return
            axes = scatter.handle.axes
            root.remove_axes = True
            root.remove_x_ticks = False
            scatter.face_color = "red"
            original = scatter.handle.set_facecolor

            def fail_after_change(colors):
                original(colors)
                if np.allclose(colors, [[1, 0, 0, 1]]):
                    self.assertFalse(axes.axison)
                    self.assertTrue(
                        axes.xaxis.get_major_ticks(1)[0].label1.get_visible()
                    )
                    raise ValueError("failed after changing visibility")

            self.enterContext(
                patch.object(scatter.handle, "set_facecolor", fail_after_change)
            )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        axes = scatter.handle.axes
        self.assertTrue(axes.axison)
        self.assertFalse(axes.xaxis.get_major_ticks(1)[0].label1.get_visible())
        self.assertFalse(root.remove_axes)
        self.assertTrue(root.remove_x_ticks)

    def test_live_toggle_wcs_tick_marks_and_labels(self):
        from graphinglib.smart_figure import HAS_ASTROPY, SmartFigureWCS

        if not HAS_ASTROPY:
            self.skipTest("Astropy is required for WCSAxes")
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        curve = Curve([1, 10], [1, 10])
        figure = SmartFigureWCS(
            projection=wcs, elements=[curve], remove_x_ticks=True, figure_style="plain"
        )
        snapshots = []

        def update(i):
            x, y = curve.handle.lines[0].axes.coords
            snapshots.append(
                tuple((axis.get_ticks_visible(), axis.get_ticklabel_visible()) for axis in (x, y))
            )
            if i == 0:
                figure.remove_x_ticks = False
                figure.remove_y_ticks = True

        figure.save_animation(self.output, update, 2)
        self.assertEqual(snapshots, [((False, False), (True, True)), ((True, True), (False, False))])

    def test_live_grid_visibility_is_local_and_keeps_configured_axes_and_style(self):
        upper, lower, inner = (Curve([1, 10], [1, 10]) for _ in range(3))
        nested = SmartFigure(
            elements=[inner],
            x_lim=(1, 10),
            y_lim=(1, 10),
        )
        root = SmartFigure(2, 3, share_x=True, x_lim=(1, 10), figure_style="plain")
        root[:, 0] = nested
        root[0, 1:3] = [upper]
        root[1, 1:3] = [lower]
        root.set_ticks(minor_x_ticks=[5])
        root.set_grid(
            visible_y=False, which_x="minor", color="tab:purple", line_width=2
        )
        root.show_grid = False
        captures = []

        def update(frame):
            axes = [curve.handle.lines[0].axes for curve in (upper, lower, inner)]
            captures.append(
                [
                    (
                        ax.xaxis.get_tick_params(which="major")["gridOn"],
                        ax.xaxis.get_tick_params(which="minor")["gridOn"],
                        ax.yaxis.get_tick_params(which="major")["gridOn"],
                    )
                    for ax in axes
                ]
            )
            gridline = axes[-2].xaxis.get_minor_ticks(1)[0].gridline
            self.assertEqual(gridline.get_color(), "tab:purple")
            self.assertEqual(gridline.get_linewidth(), 2)
            if frame == 0:
                root.show_grid = [None, True, True]
                nested.show_grid = True
            elif frame == 1:
                root.show_grid = [None, False, None]
                root.x_lim = (1, 100)
                root.log_scale_x = True
                nested.show_grid = False
            self.assertTrue(axes[0].get_shared_x_axes().joined(*axes[:2]))

        root.save_animation(self.output, update, 3, interval=100)
        off = (False, False, False)
        minor = (False, True, False)
        self.assertEqual(
            captures,
            [
                [off, off, off],
                [minor, minor, (True, False, True)],
                [off, minor, off],
            ],
        )
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)
            self.assertEqual(gif.info["duration"], 100)
            gif.seek(1)
            self.assertEqual(gif.info["duration"], 200)

    def test_grid_visibility_mask_respects_initial_style_and_rolls_back(self):
        scatter = Scatter([1, 10], [1, 10], face_color="navy")
        figure = SmartFigure(elements=[scatter], figure_style="plain")
        figure.set_rc_params({"axes.grid": True, "axes.grid.axis": "y"})
        gridlines = []

        def update(frame):
            axes = scatter.handle.axes
            y_grid = axes.yaxis.get_gridlines()[0]
            gridlines.append(y_grid)
            self.assertTrue(y_grid.get_visible())
            self.assertFalse(axes.xaxis.get_gridlines()[0].get_visible())
            if frame:
                figure.show_grid = False
                scatter.face_color = "red"
                before = scatter.handle.set_facecolor

                def fail_after_change(color):
                    before(color)
                    if np.allclose(color, [[1, 0, 0, 1]]):
                        self.assertFalse(y_grid.get_visible())
                        raise ValueError("failed after changing grid visibility")

                self.enterContext(
                    patch.object(scatter.handle, "set_facecolor", fail_after_change)
                )
            else:
                figure.show_grid = True

        self.output.write_bytes(b"existing")
        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            figure.save_animation(self.output, update, 2)
        self.assertTrue(figure.show_grid)
        self.assertIs(gridlines[0], gridlines[1])
        self.assertTrue(gridlines[0].get_visible())
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_static_grid_toggle_can_hide_style_default_and_restore_it(self):
        curve = Curve([1, 10], [1, 10])
        figure = SmartFigure(elements=[curve], figure_style="plain")
        figure.set_rc_params({"axes.grid": True, "axes.grid.axis": "y"})

        for show_grid, expected in ((None, True), (False, False), (True, True)):
            if show_grid is not None:
                figure.show_grid = show_grid
            figure.save(str(self.output.with_suffix(".png")))
            axes = curve.handle.lines[0].axes
            self.assertFalse(axes.xaxis.get_gridlines()[0].get_visible())
            self.assertIs(axes.yaxis.get_gridlines()[0].get_visible(), expected)

    def test_live_wcs_grid_visibility_restores_the_configured_coordinate(self):
        from graphinglib.smart_figure import HAS_ASTROPY, SmartFigureWCS

        if not HAS_ASTROPY:
            self.skipTest("Astropy is required for WCSAxes")
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        curve = Curve([1, 10], [1, 10])
        figure = SmartFigureWCS(wcs, elements=[curve], figure_style="plain")
        figure.set_grid(visible_y=False, color="red")
        figure.show_grid = False
        states = []

        def update(frame):
            x, y = curve.handle.lines[0].axes.coords
            states.append(
                tuple(coord._grid_lines_kwargs["visible"] for coord in (x, y))
            )
            figure.show_grid = frame == 0

        figure.save_animation(self.output, update, 3)
        self.assertEqual(states, [(False, False), (True, False), (False, False)])

    def test_live_set_grid_changes_groups_and_styles_and_restores_style_defaults(self):
        from matplotlib.colors import to_rgba

        from graphinglib.file_manager import FileLoader

        left, upper, lower = (Curve([1, 10], [1, 10]) for _ in range(3))
        nested = SmartFigure(elements=[left])
        root = SmartFigure(2, 3, share_x=True, figure_style="plain")
        root[:, 0] = nested
        root[0, 1:3] = [upper]
        root[1, 1:3] = [lower]
        root.set_ticks(minor_x_ticks=[3, 5, 7])
        root.set_grid(visible_y=False, which_x="major", color="red")
        nested.set_grid(visible_y=False, color="orange")
        seen = []

        def state(ax):
            x_major = ax.xaxis.get_major_ticks(1)[0].gridline
            x_minor = ax.xaxis.get_minor_ticks(1)[0].gridline
            y_major = ax.yaxis.get_major_ticks(1)[0].gridline
            return (
                (x_major.get_visible(), x_minor.get_visible(), y_major.get_visible()),
                to_rgba(x_major.get_color()),
                x_major.get_alpha(),
                x_major.get_linestyle(),
                x_major.get_linewidth(),
            )

        def update(frame):
            a, b, c = (obj.handle.lines[0].axes for obj in (left, upper, lower))
            seen.append(
                (
                    tuple(state(ax) for ax in (a, b, c)),
                    tuple(a.lines + b.lines + c.lines),
                )
            )
            if frame == 0:
                root.set_grid(
                    visible_y=False,
                    which_x="minor",
                    color="blue",
                    alpha=0.35,
                    line_style="--",
                    line_width=3,
                )
                root.show_grid = [None, True, False]
                nested.set_grid(reset=True, visible_y=False)
            elif frame == 1:
                root.set_grid(reset=True, visible_x=False, which_y="major")
                root.x_lim = (1, 100)
                root.log_scale_x = True

        root.save_animation(self.output, update, 3)
        self.assertEqual(seen[0][0][0][1], to_rgba("orange"))
        self.assertEqual(seen[1][0][0][1], to_rgba("red"))
        self.assertEqual(
            seen[1][0][1], ((False, True, False), to_rgba("blue"), 0.35, "--", 3)
        )
        self.assertEqual(seen[1][0][2][0], (False, False, False))
        defaults = FileLoader("plain").load()["rc_params"]
        for states, handles in seen:
            self.assertEqual(handles, seen[0][1])
        for flags, color, alpha, _, width in seen[2][0][1:]:
            self.assertEqual(flags, (False, False, True))
            self.assertEqual(color, to_rgba(defaults["grid.color"]))
            self.assertEqual(alpha, defaults["grid.alpha"])
            self.assertEqual(width, defaults["grid.linewidth"])

        root.save(str(self.output.with_suffix(".png")))
        self.assertEqual(state(upper.handle.lines[0].axes), seen[2][0][1])

    def test_failed_live_set_grid_restores_style_and_model(self):
        curve = Curve([1, 10], [1, 10])
        root = SmartFigure(elements=[curve], figure_style="plain")
        root.set_grid(visible_y=False, color="orange")
        dash_pattern = (1, (2, 5))
        root.set_rc_params({"grid.linestyle": dash_pattern})
        self.output.write_bytes(b"existing")
        applied = []

        def update(frame):
            if not frame:
                return
            ax = curve.handle.lines[0].axes
            original = ax.yaxis.set_tick_params
            root.set_grid(visible_x=False, color="red", line_style="-")
            root.show_grid = False

            def fail_after_apply(*args, **kwargs):
                original(*args, **kwargs)
                if kwargs.get("grid_color") == "red":
                    applied.append(ax.xaxis.get_gridlines()[0].get_color())
                    raise ValueError("failed after setting grid style")

            self.enterContext(
                patch.object(ax.yaxis, "set_tick_params", fail_after_apply)
            )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(applied, ["red"])
        self.assertTrue(root.show_grid)
        self.assertEqual(root._user_rc_dict["grid.color"], "orange")
        self.assertTrue(root._grid["visible_x"])
        self.assertFalse(root._grid["visible_y"])
        ax = curve.handle.lines[0].axes
        self.assertTrue(ax.xaxis.get_gridlines()[0].get_visible())
        self.assertEqual(ax.xaxis.get_gridlines()[0].get_color(), "orange")
        self.assertEqual(ax.xaxis.get_gridlines()[0]._unscaled_dash_pattern, dash_pattern)
        self.assertFalse(ax.yaxis.get_gridlines()[0].get_visible())
        self.assertEqual(ax.yaxis.get_gridlines()[0].get_color(), "orange")
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_live_set_grid_changes_wcs_coordinate_and_style(self):
        from graphinglib.smart_figure import HAS_ASTROPY, SmartFigureWCS

        if not HAS_ASTROPY:
            self.skipTest("Astropy is required for WCSAxes")
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        curve = Curve([1, 10], [1, 10])
        figure = SmartFigureWCS(wcs, elements=[curve], figure_style="plain")
        figure.set_grid(visible_y=False, color="red")
        states = []

        def update(frame):
            x, y = curve.handle.lines[0].axes.coords
            states.append(
                [
                    (
                        p["visible"],
                        p["edgecolor"],
                        p["alpha"],
                        p["linestyle"],
                        p["linewidth"],
                    )
                    for p in (x._grid_lines_kwargs, y._grid_lines_kwargs)
                ]
            )
            if frame == 0:
                figure.set_grid(
                    visible_x=False,
                    color="purple",
                    alpha=0.5,
                    line_style=":",
                    line_width=2,
                )
            elif frame == 1:
                figure.show_grid = False

        figure.save_animation(self.output, update, 3)
        self.assertEqual(
            [pair[:2] for pair in states[0]], [(True, "red"), (False, "red")]
        )
        self.assertEqual(
            states[1], [(False, "purple", 0.5, ":", 2), (True, "purple", 0.5, ":", 2)]
        )
        self.assertEqual(states[2], [(False, "purple", 0.5, ":", 2)] * 2)

    def test_live_log_scale_combines_final_limits_and_preserves_manual_ticks(self):
        from warnings import catch_warnings, simplefilter

        from graphinglib import SmartTwinAxis

        first = Curve([1, 10, 100], [1, 2, 3])
        second = Curve([1, 10, 100], [3, 2, 1])
        inner = Curve([1, 10, 100], [0, 1, 2])
        data = Curve([1, 10, 100], [10, 50, 90])
        twin = SmartTwinAxis(elements=[data], axis_lim=(-1, 100))
        nested = SmartFigure(elements=[inner], twin_y_axis=twin, x_lim=(-1, 120))
        root = SmartFigure(2, 3, share_x=True, x_lim=(-1, 120), figure_style="plain")
        root[:, 0] = nested
        root[0, 1:3] = [first]
        root[1, 1:3] = [second]
        root.set_ticks(x_ticks=[1, 10, 100], x_tick_labels=["one", "ten", "hundred"])
        states = {}

        def update(i):
            a, b, c, d = (
                curve.handle.lines[0].axes for curve in (first, second, inner, data)
            )
            states[i] = (
                (a, b, c, d),
                (a.get_xscale(), b.get_xscale(), c.get_xscale(), d.get_yscale()),
                (a.get_xlim(), b.get_xlim(), c.get_xlim(), d.get_ylim()),
                (a.xaxis.get_major_locator(), a.xaxis.get_major_formatter()),
            )
            if i == 0:
                root.log_scale_x = [None, None, True]
                root.x_lim = [None, (1, 100), None]
                root.invert_x_axis = True
                twin.axis_lim = (1, 100)
                twin.log_scale = True
            elif i == 1:
                root.x_lim = (-2, 120)
                root.log_scale_x = False
                root.invert_x_axis = False
                twin.log_scale = False
                twin.axis_lim = (-5, 100)

        with catch_warnings():
            simplefilter("error", UserWarning)
            root.save_animation(self.output, update, 3)
        self.assertEqual(states[0][1], ("linear", "linear", "linear", "linear"))
        self.assertEqual(states[1][1], ("log", "log", "linear", "log"))
        self.assertEqual(states[1][2], ((100, 1), (100, 1), (-1, 120), (1, 100)))
        self.assertEqual(states[2][1], ("linear",) * 4)
        self.assertEqual(states[2][2], ((-2, 120), (-2, 120), (-1, 120), (-5, 100)))
        self.assertTrue(
            all(
                state[0] == states[0][0] and state[3] == states[0][3]
                for state in states.values()
            )
        )

    def test_live_log_scale_validates_shared_requests_and_final_limits(self):
        a = Curve([1, 10], [1, 10])
        b = Curve([1, 10], [10, 1])
        for conflict, message in (
            (False, "logarithmic axis limits must be positive"),
            (True, "matching scale settings"),
        ):
            with self.subTest(conflict=conflict):
                root = SmartFigure(
                    1,
                    2,
                    elements=[a, b],
                    share_x=True,
                    x_lim=(-1, 10),
                    figure_style="plain",
                )
                self.output.write_bytes(b"existing")

                def update(i, root=root, conflict=conflict):
                    root.log_scale_x = (
                        False if i == 0 else ([True, False] if conflict else True)
                    )

                with self.assertRaisesRegex(ValueError, message):
                    root.save_animation(self.output, update, 2)
                self.assertFalse(root.log_scale_x)
                for curve in (a, b):
                    axes = curve.handle.lines[0].axes
                    self.assertEqual(axes.get_xscale(), "linear")
                    self.assertEqual(axes.get_xlim(), (-1, 10))
                self.assertEqual(self.output.read_bytes(), b"existing")

    def test_failed_frame_restores_log_scale_and_tick_objects(self):
        scatter = Scatter([1, 10], [1, 10], face_color="navy")
        root = SmartFigure(elements=[scatter], x_lim=(-2, 10), figure_style="plain")
        captures = []

        def update(i):
            axes = scatter.handle.axes
            axis = axes.xaxis
            captures[:] = [axes, axis.get_major_locator(), axis.get_major_formatter()]
            if i:
                root.log_scale_x = True
                root.x_lim = (1, 10)
                scatter.face_color = "red"
                original = scatter.handle.set_facecolor

                def fail_after_change(colors):
                    original(colors)
                    if np.allclose(colors, [[1, 0, 0, 1]]):
                        self.assertEqual(axes.get_xscale(), "log")
                        self.assertEqual(axes.get_xlim(), (1, 10))
                        raise ValueError("failed after changing scale")

                self.enterContext(
                    patch.object(scatter.handle, "set_facecolor", fail_after_change)
                )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        axes, locator, formatter = captures
        self.assertFalse(root.log_scale_x)
        self.assertEqual(root.x_lim, (-2, 10))
        self.assertEqual(axes.get_xscale(), "linear")
        self.assertEqual(axes.get_xlim(), (-2, 10))
        self.assertIs(axes.xaxis.get_major_locator(), locator)
        self.assertIs(axes.xaxis.get_major_formatter(), formatter)
        self.assertFalse(self.output.exists())

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

    def test_live_subplot_text_handles_spans_and_nested_fallback(self):
        first = Curve([0, 1], [0, 1])
        second = Curve([0, 1], [0, 1])
        inner = Curve([0, 1], [0, 1])
        nested = SmartFigure(elements=[inner], x_label="Child x")
        root = SmartFigure(
            2,
            3,
            figure_style="plain",
            title="Overview",
            subtitles=("Inherited", "Upper", "Lower"),
            sub_x_labels=["Parent x", "Upper x"],
        )
        root[:, 0] = nested
        root[0, 1:3] = [first]
        root[1, 1:3] = [second]
        root.set_text_padding_params(sub_x_labels_pad=[0, 14, 20])
        captures = []

        def update(frame):
            axes = [curve.handle.lines[0].axes for curve in (inner, first, second)]
            artists = tuple((ax.title, ax.xaxis.label, ax.yaxis.label) for ax in axes)
            captures.append(
                (
                    artists,
                    tuple(
                        tuple(text.get_text() for text in group) for group in artists
                    ),
                )
            )
            if frame == 0:
                root.subtitles = ("Live parent", "Changing", None)
                root.sub_x_labels = (label for label in ("Changed x", None, "Lower x"))
                root.sub_y_labels = ["Nested y", "Upper y", "Lower y"]
                nested.title = "Child title"
                nested.x_label = None
            elif frame == 1:
                nested.title = None
                root.subtitles = ["Fallback title"]
                root.sub_x_labels = None
                nested.sub_x_labels = ["Child subplot x"]
                root.sub_y_labels = [None, "Upper y"]
            self.assertEqual(axes[-1].xaxis.labelpad, 20)
            self.assertTrue(any(ax.get_title() == "Overview" for ax in plt.gcf().axes))

        root.save_animation(self.output, update, 3)
        self.assertEqual(
            [capture[1] for capture in captures],
            [
                (
                    ("Inherited", "Child x", ""),
                    ("Upper", "Upper x", ""),
                    ("Lower", "", ""),
                ),
                (
                    ("Child title", "Changed x", "Nested y"),
                    ("Changing", "", "Upper y"),
                    ("", "Lower x", "Lower y"),
                ),
                (
                    ("Fallback title", "Child subplot x", ""),
                    ("", "", "Upper y"),
                    ("", "", ""),
                ),
            ],
        )
        self.assertTrue(all(capture[0] == captures[0][0] for capture in captures))

    def test_live_subplot_text_uses_wcs_coordinate_labels(self):
        from graphinglib.smart_figure import HAS_ASTROPY, SmartFigureWCS

        if not HAS_ASTROPY:
            self.skipTest("Astropy is required for WCSAxes")
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        first, second = (Curve([0, 1], [0, 1]) for _ in range(2))
        root = SmartFigureWCS(
            projection=wcs,
            num_rows=1,
            num_cols=2,
            figure_style="plain",
            elements=[first, second],
        )
        labels = []

        def update(frame):
            labels.append(
                [
                    (axes.get_xlabel(), axes.get_ylabel())
                    for axes in (
                        curve.handle.lines[0].axes for curve in (first, second)
                    )
                ]
            )
            if frame == 0:
                root.sub_x_labels = ["RA one", "RA two"]
                root.sub_y_labels = ["Dec one", "Dec two"]

        root.save_animation(self.output, update, 2)
        self.assertEqual(labels[1], [("RA one", "Dec one"), ("RA two", "Dec two")])

    def test_failed_subplot_text_commit_restores_other_titles(self):
        from matplotlib.text import Text as MatplotlibText

        first, second = (Curve([0, 1], [0, 1]) for _ in range(2))
        root = SmartFigure(
            1,
            2,
            elements=[first, second],
            subtitles=["One", "Two"],
            figure_style="plain",
        )
        before = MatplotlibText.set_text
        self.output.write_bytes(b"existing")
        did_apply = []

        def fail_on_title(artist, text):
            before(artist, text)
            if text == "fail-on-write":
                did_apply.append(first.handle.lines[0].axes.title.get_text())
                raise ValueError("failed writing subplot title")

        def update(frame):
            if frame == 0:
                root.subtitles = ["First", "Second"]
            else:
                root.subtitles = ["Changed first", "fail-on-write"]

        with (
            patch.object(MatplotlibText, "set_text", fail_on_title),
            self.assertRaisesRegex(GraphingException, "animation frame 1"),
        ):
            root.save_animation(self.output, update, 2)
        self.assertEqual(did_apply, ["Changed first"])
        self.assertEqual(root.subtitles, ["First", "Second"])
        self.assertEqual(
            [curve.handle.lines[0].axes.title.get_text() for curve in (first, second)],
            ["First", "Second"],
        )
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_artist_commit_failure_restores_shared_viewports_and_labels(self):
        moving = Scatter([0, 6], [0, 6], face_color="navy")
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
            if frame:
                root.invert_x_axis = True
                moving.face_color = "red"
                original = moving.handle.set_facecolor

                def fail_after_edit(color):
                    original(color)
                    if np.allclose(color, [[1, 0, 0, 1]]):
                        self.assertEqual(moving.handle.axes.get_xlim(), (4, 2))
                        raise ValueError("could not set marker color")

                self.enterContext(
                    patch.object(moving.handle, "set_facecolor", fail_after_edit)
                )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(root.x_lim, (1, 5))
        self.assertEqual(root.title, "valid")
        self.assertIn("valid", [artist.get_text() for artist in titles])
        self.assertNotIn("failed", [artist.get_text() for artist in titles])
        self.assertFalse(root.invert_x_axis)
        for axes in (moving.handle.axes, fixed.handle.lines[0].axes):
            assert_allclose(axes.get_xlim(), (1, 5))

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

    def test_stream_fields_change_trajectories_and_reuse_arrowheads(self):
        from graphinglib import Stream

        grid = np.linspace(0, 1, 7)
        uniform = np.ones((7, 7))
        stream = Stream(grid, grid, uniform, 0 * uniform, density=0.5, color="navy")
        figure = SmartFigure(
            elements=[stream], figure_style="plain", x_lim=(0, 1), y_lim=(0, 1)
        )
        history = []

        def update(i):
            axis = next(axis for axis in plt.gcf().axes if axis.collections)
            (lines,) = axis.collections
            history.append(
                (
                    lines,
                    lines.get_segments(),
                    tuple(axis.patches),
                    sum(patch.get_visible() for patch in axis.patches),
                )
            )
            if i == 0:
                stream.set_data(0 * uniform, 0 * uniform)
            elif i == 1:
                field = np.ones((9, 9))
                coords = np.linspace(0, 1, 9)
                stream.set_data(0 * field, field, x_data=coords, y_data=coords)
                stream.density = 1.2
                stream.color = "red"
                stream.arrow_size = 2
            else:
                stream.density = 0.4

        figure.save_animation(self.output, update, 3)
        line, horizontal, old_arrows, initial_count = history[0]
        self.assertTrue(all(item[0] is line for item in history))
        self.assertGreater(initial_count, 0)
        self.assertEqual(len(history[1][1]), 0)
        self.assertEqual(history[1][-1], 0)
        self.assertEqual(history[1][2], old_arrows)
        self.assertGreater(history[2][-1], initial_count)
        self.assertEqual(history[2][2][:initial_count], old_arrows)
        for segment in horizontal:
            self.assertAlmostEqual(np.ptp(segment[:, 1]), 0)
        for segment in line.get_segments():
            self.assertAlmostEqual(np.ptp(segment[:, 0]), 0)
        self.assertLess(sum(p.get_visible() for p in line.axes.patches), history[2][-1])
        self.assertTrue(all(p.get_mutation_scale() == 20 for p in line.axes.patches))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_stream_mapped_styles_follow_field_and_keep_initial_color_range(self):
        from graphinglib import Stream

        coords = np.linspace(0, 1, 7)
        x, y = np.meshgrid(coords, coords)
        stream = Stream(
            coords,
            coords,
            np.ones_like(x),
            0.2 + 0 * x,
            color=x + y,
            line_width=1 + x,
            density=0.5,
            color_map="viridis",
        )
        figure = SmartFigure(elements=[stream], figure_style="plain")
        items = []

        def update(i):
            axis = next(axis for axis in plt.gcf().axes if axis.collections)
            items.append((axis.collections[0], axis.collections[0].get_clim()))
            stream.u_data = 0.2 + 0 * x
            stream.v_data = np.ones_like(x)
            stream.color = 10 * i + x + y
            stream.line_width = 2 + y
            stream.color_map = "magma"

        figure.save_animation(self.output, update, 2)
        self.assertIs(items[0][0], items[1][0])
        (line, scale) = items[0]
        self.assertEqual(scale, (0, 2))
        self.assertEqual(line.get_clim(), (0, 2))
        self.assertEqual(line.get_cmap().name, "magma")
        self.assertGreater(min(line.get_array()), 9)
        self.assertGreater(max(line.get_linewidth()), 2)
        self.assertTrue(all(p.get_mutation_scale() > 0 for p in line.axes.patches))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_failed_stream_artist_commit_restores_arrow_pool_and_field(self):
        from graphinglib import Stream

        grid = np.linspace(0, 1, 7)
        ones = np.ones((7, 7))
        stream = Stream(grid, grid, ones, 0 * ones, density=0.5)
        figure = SmartFigure(elements=[stream], figure_style="plain")
        previous = []
        attempts = []
        self.output.write_bytes(b"existing")

        def update(i):
            axis = next(axis for axis in plt.gcf().axes if axis.collections)
            if i == 0:
                stream.color = "green"
                return

            (lines,) = axis.collections
            previous.append(
                (
                    axis,
                    lines,
                    lines.get_segments(),
                    tuple(axis.patches),
                    axis.dataLim.get_points().copy(),
                )
            )
            add_artist = axis.add_artist

            def fail_on_second_arrow(artist):
                attempts.append(artist)
                if len(attempts) == 2:
                    raise ValueError("failed to attach arrow")
                return add_artist(artist)

            self.enterContext(
                patch.object(axis, "add_artist", side_effect=fail_on_second_arrow)
            )
            stream.set_data(0 * ones, ones)
            stream.density = 2
            stream.color = "red"

        with self.assertRaisesRegex(GraphingException, "failed to attach arrow"):
            figure.save_animation(self.output, update, 2)
        self.assertEqual(len(attempts), 2)
        axis, lines, segments, old_arrows, data_limits = previous[0]
        self.assertEqual(tuple(axis.patches), old_arrows)
        self.assertIsNone(attempts[0].axes)
        self.assertTrue(all(arrow.get_visible() for arrow in old_arrows))
        for current, old in zip(lines.get_segments(), segments, strict=True):
            assert_allclose(current, old)
        assert_allclose(axis.dataLim.get_points(), data_limits)
        assert_allclose(stream.u_data, ones)
        assert_allclose(stream.v_data, 0 * ones)
        self.assertEqual(stream.density, 0.5)
        self.assertEqual(stream.color, "green")
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_table_edits_headers_cell_colors_dimensions_and_placement(self):
        from matplotlib.colors import to_rgba

        from graphinglib import Table

        table = Table(
            [["1", "2"], ["3", "4"]],
            row_labels=["a", "b"],
            col_labels=["X", "Y"],
            col_widths=[0.2, 0.3],
            location="upper left",
        )
        root = SmartFigure(elements=[table], figure_style="plain")
        seen = []

        def update(i):
            cells = table.handle.get_celld()
            seen.append((table.handle, tuple(cells.values()), cells[1, 0].get_height()))
            table.cell_text = [[f"Body {i}", "right"], ["bottom", "corner"]]
            table.cell_colors = [["pink", "orange"], ["cyan", "yellow"]]
            table.col_labels = [f"X {i}", "Y"]
            table.col_colors = "red"
            table.row_labels = ["Row A", f"Row {i}"]
            table.row_colors = ["green", "blue"]
            table.cell_align = "left"
            table.row_align = "center"
            table.col_align = "left"
            table.scaling = (2.0, 3.0)
            table.col_widths = [0.1, 0.25]
            table.location = "lower right"

        root.save_animation(self.output, update, 3)
        self.assertTrue(all(observed[:2] == seen[0][:2] for observed in seen))
        self.assertAlmostEqual(seen[1][2], 2 * seen[0][2])
        self.assertEqual(seen[1][2], seen[2][2])
        cells = table.handle.get_celld()
        self.assertAlmostEqual(cells[1, 0].get_width(), 0.2)
        self.assertAlmostEqual(cells[1, 1].get_width(), 0.5)
        self.assertEqual(cells[1, 0].get_text().get_text(), "Body 2")
        self.assertEqual(cells[2, -1].get_text().get_text(), "Row 2")
        self.assertEqual(cells[0, 0].get_text().get_text(), "X 2")
        assert_allclose(cells[1, 1].get_facecolor(), to_rgba("orange"))
        assert_allclose(cells[0, 0].get_facecolor(), to_rgba("red"))
        assert_allclose(cells[2, -1].get_facecolor(), to_rgba("blue"))
        self.assertEqual(cells[1, 0].get_text().get_ha(), "left")
        self.assertEqual(cells[0, 0].get_text().get_ha(), "left")
        self.assertEqual(cells[1, -1].get_text().get_ha(), "center")
        self.assertEqual(table.handle._loc, table.handle.codes["lower right"])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 3)

    def test_table_resizes_and_reuses_body_and_headers_by_logical_position(self):
        from graphinglib import Table

        table = Table([["old", "second"]])
        nested = SmartFigure(elements=[table])
        root = SmartFigure(2, 2, figure_style="plain")
        root[0, 0] = nested
        root[1, :] = [table]
        snapshots = {}
        widths = np.array([0.2, 0.3, 0.4])

        def update(i):
            handles = [item for ax in plt.gcf().axes for item in ax.tables]
            self.assertEqual(len(handles), 2)
            snapshots[i] = [(item, dict(item.get_celld())) for item in handles]
            if i == 1:
                table.cell_text = [["only"]]
                table.col_labels = table.row_labels = None
                table.col_colors = table.row_colors = None
                table.col_widths = None
            else:
                table.col_labels = ["A", "B", "C"]
                table.row_labels = ["first", "second"]
                table.col_colors = "orange"
                table.row_colors = "gray"
                table.cell_text = [[f"frame {i}"] * 3, ["past", "row", "end"]]
                table.col_widths = widths
                table.scaling = (1.5, 2)

        root.save_animation(self.output, update, 3)
        assert_allclose(widths, [0.2, 0.3, 0.4])
        for original, expanded, contracted in zip(
            snapshots[0], snapshots[1], snapshots[2], strict=True
        ):
            handle, start = original
            _, larger = expanded
            _, smaller = contracted
            self.assertIs(handle, expanded[0])
            self.assertIs(handle, contracted[0])
            self.assertEqual(smaller.keys(), {(0, 0)})
            final = handle.get_celld()
            self.assertEqual(final.keys(), larger.keys())
            self.assertIs(start[1, 0], larger[1, 0])
            self.assertIs(larger[1, 0], smaller[0, 0])
            self.assertTrue(all(final[key] is larger[key] for key in larger))
            self.assertEqual(final[1, 0].get_text().get_text(), "frame 2")
            self.assertEqual(final[0, 2].get_text().get_text(), "C")
            self.assertEqual(final[2, -1].get_text().get_text(), "second")
            assert_allclose([final[1, c].get_width() for c in range(3)], widths * 1.5)

    def test_failed_table_frame_restores_cell_map_and_dimensions(self):
        from graphinglib import Table

        table = Table([["start"]], col_colors=None, row_colors=None)
        scatter = Scatter([0, 1], [1, 0])
        root = SmartFigure(elements=[table, scatter], figure_style="plain")
        observed = []

        def update(i):
            observed[:] = [table.handle.get_celld()[0, 0]]
            table.cell_text = [["retained"]] if i == 0 else [["expanded", "new"]]
            if i:
                table.col_labels = ["new header", "another"]
                scatter.face_color = "red"
                original = scatter.handle.set_facecolor

                def fail_after_colors(value):
                    original(value)
                    if np.allclose(value, [[1, 0, 0, 1]]):
                        self.assertEqual(len(table.handle.get_celld()), 4)
                        raise ValueError("could not set scatter colors")

                self.enterContext(
                    patch.object(scatter.handle, "set_facecolor", fail_after_colors)
                )

        with self.assertRaisesRegex(GraphingException, "animation frame 1"):
            root.save_animation(self.output, update, 2)
        self.assertEqual(table.cell_text, [["retained"]])
        self.assertIsNone(table.col_labels)
        self.assertEqual(table.handle.get_celld(), {(0, 0): observed[0]})
        self.assertEqual(observed[0].get_text().get_text(), "retained")
        self.assertFalse(self.output.exists())

    def test_reference_line_styles_and_bounded_counts_animate(self):
        from matplotlib.colors import to_rgba

        from graphinglib import Hlines, Vlines

        horizontal = Hlines([1, 2], x_min=0, x_max=2, colors="black")
        vertical = Vlines([1, 2], colors="black", line_widths=2.0)
        root = SmartFigure(
            elements=[horizontal, vertical],
            figure_style="plain",
            x_lim=(0, 4),
            y_lim=(0, 4),
        )
        seen = []

        def update(i):
            axes = next(axes for axes in plt.gcf().axes if axes.collections)
            seen.append((axes.collections[0], tuple(axes.lines)))
            horizontal.y = [1, 2, 3]
            horizontal.x_max = 3
            horizontal.colors = ["red", "green", "blue"]
            horizontal.line_widths = [1.0, 2.0, 3.0]
            horizontal.line_styles = ["--", "-", ":"]
            vertical.x = [1 + 0.2 * i, 2 + 0.3 * i]
            vertical.colors = ["navy", "orange"]
            vertical.line_widths = [1.0, 2.0]
            vertical.line_styles = ["-.", ":"]
            vertical.alpha = 0.6

        root.save_animation(self.output, update, 2)
        self.assertEqual(seen[0], seen[1])
        collection, lines = seen[-1]
        self.assertEqual(len(collection.get_segments()), 3)
        assert_allclose(
            collection.get_colors(), [to_rgba(c) for c in ("red", "green", "blue")]
        )
        assert_allclose(collection.get_linewidths(), [1, 2, 3])
        self.assertEqual([line.get_color() for line in lines], ["navy", "orange"])
        self.assertEqual([line.get_linestyle() for line in lines], ["-.", ":"])
        self.assertEqual([line.get_alpha() for line in lines], [0.6, 0.6])
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)

    def test_raster_extent_and_origin_update_without_panning_view(self):
        heat = Heatmap([[0, 1], [0, 0]], origin_position="upper", show_color_bar=False)
        root = SmartFigure(
            elements=[heat], figure_style="plain", x_lim=(-2, 6), y_lim=(-3, 6)
        )
        captures = []

        def update(i):
            (artist,) = [image for axes in plt.gcf().axes for image in axes.images]
            captures.append(
                (
                    artist,
                    tuple(artist.get_extent()),
                    artist.origin,
                    artist.axes.get_xlim(),
                )
            )
            heat.x_axis_range = (1, 4) if i == 0 else None
            heat.y_axis_range = (2, 5)
            heat.origin_position = "lower" if i == 0 else "upper"
            heat.image = np.eye(3 + i)

        root.save_animation(self.output, update, 2)
        self.assertIs(captures[0][0], captures[1][0])
        self.assertEqual(captures[1][1:], ((1, 4, 2, 5), "lower", (-2, 6)))
        artist = captures[-1][0]
        self.assertEqual(tuple(artist.get_extent()), (*captures[0][1][:2], 2, 5))
        self.assertEqual(artist.origin, "upper")
        self.assertEqual(artist.axes.get_xlim(), (-2, 6))
        with Image.open(self.output) as gif:
            self.assertEqual(gif.n_frames, 2)
