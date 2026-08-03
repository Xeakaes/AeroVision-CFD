"""
Post-Processing UI for AeroVision CFD — PyQt6 + PyVista prototype

Features:
- Left `TreeView` for simulation items
- Interactive scalar cut planes (X/Y/Z) with sliders
- Streamlines toggle (attempts to use velocity vectors)
- Color legend (LUT) with Red=Max, Blue=Min
- Dark/light theme toggle

Run: `python postprocess_ui.py` (loads `cfd_output_results.vtk` if present)
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTreeWidget,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QDockWidget,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PyQt6.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor

DEFAULT_VTK = Path(__file__).resolve().parent / "cfd_output_results.vtk"


class PostProcessWindow(QMainWindow):
    def __init__(self, vtk_path=None):
        super().__init__()
        self.setWindowTitle("AeroVision — Post-Processing")
        self.resize(1400, 900)

        self.vtk_path = Path(vtk_path) if vtk_path else (DEFAULT_VTK if DEFAULT_VTK.exists() else None)

        self._build_ui()
        self.mesh = None
        if self.vtk_path:
            self.load_mesh(self.vtk_path)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Tree view
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setFixedWidth(280)
        layout.addWidget(self.tree)

        # Center: PyVista 3D view
        self.plotter_widget = QtInteractor(self)
        layout.addWidget(self.plotter_widget.interactor)
        self.plotter = self.plotter_widget

        # Right dock: controls
        dock = QDockWidget("Controls", self)
        dock_widget = QWidget()
        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        vbox = QVBoxLayout(dock_widget)

        # File loader
        btn_load = QPushButton("Load VTK / Mesh")
        btn_load.clicked.connect(self._on_load)
        vbox.addWidget(btn_load)

        # Scalar selection & LUT
        lut_group = QGroupBox("Scalar / LUT")
        lut_layout = QVBoxLayout(lut_group)
        lut_layout.addWidget(QLabel("Scalar array:"))
        self.scalar_combo = QComboBox()
        lut_layout.addWidget(self.scalar_combo)
        # min/max display
        minmax_row = QWidget()
        mm_layout = QHBoxLayout(minmax_row)
        mm_layout.setContentsMargins(0, 0, 0, 0)
        mm_layout.addWidget(QLabel("Min:"))
        self.lbl_min = QLabel("-")
        mm_layout.addWidget(self.lbl_min)
        mm_layout.addWidget(QLabel("Max:"))
        self.lbl_max = QLabel("-")
        mm_layout.addStretch()
        lut_layout.addWidget(minmax_row)
        vbox.addWidget(lut_group)

        # Cut plane controls
        slice_group = QGroupBox("Cut Planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.addWidget(QLabel("Cut plane X:"))
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setMinimum(0)
        self.slider_x.setMaximum(100)
        self.slider_x.valueChanged.connect(lambda v: self.update_cut_plane("x", v))
        slice_layout.addWidget(self.slider_x)

        slice_layout.addWidget(QLabel("Cut plane Y:"))
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setMinimum(0)
        self.slider_y.setMaximum(100)
        self.slider_y.valueChanged.connect(lambda v: self.update_cut_plane("y", v))
        slice_layout.addWidget(self.slider_y)

        slice_layout.addWidget(QLabel("Cut plane Z:"))
        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.setMinimum(0)
        self.slider_z.setMaximum(100)
        self.slider_z.valueChanged.connect(lambda v: self.update_cut_plane("z", v))
        slice_layout.addWidget(self.slider_z)

        # slice appearance
        slice_layout.addWidget(QLabel("Slice opacity:"))
        self.slice_opacity = QDoubleSpinBox()
        self.slice_opacity.setRange(0.0, 1.0)
        self.slice_opacity.setSingleStep(0.05)
        self.slice_opacity.setValue(0.9)
        self.slice_opacity.valueChanged.connect(lambda _: self.redraw_slice())
        slice_layout.addWidget(self.slice_opacity)

        vbox.addWidget(slice_group)

        # Streamlines group
        stream_group = QGroupBox("Streamlines")
        stream_layout = QVBoxLayout(stream_group)
        self.btn_stream = QPushButton("Toggle Streamlines")
        self.btn_stream.setCheckable(True)
        self.btn_stream.clicked.connect(self.toggle_streamlines)
        stream_layout.addWidget(self.btn_stream)

        self.chk_tubes = QCheckBox("Draw as tubes")
        self.chk_tubes.setChecked(True)
        stream_layout.addWidget(self.chk_tubes)

        # stream params
        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(QLabel("Seeds:"))
        self.spin_seeds = QSpinBox()
        self.spin_seeds.setRange(1, 10000)
        self.spin_seeds.setValue(200)
        rlay.addWidget(self.spin_seeds)
        rlay.addWidget(QLabel("Max time:"))
        self.spin_maxt = QDoubleSpinBox()
        self.spin_maxt.setRange(0.01, 100.0)
        self.spin_maxt.setValue(1.0)
        rlay.addWidget(self.spin_maxt)
        stream_layout.addWidget(row)

        # tube radius
        row2 = QWidget()
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(QLabel("Tube radius:"))
        self.spin_tuber = QDoubleSpinBox()
        self.spin_tuber.setRange(0.0001, 1.0)
        self.spin_tuber.setValue(0.01)
        r2.addWidget(self.spin_tuber)
        stream_layout.addWidget(row2)

        vbox.addWidget(stream_group)

        # Color legend hint
        vbox.addWidget(QLabel("Color Legend: Red = Max, Blue = Min"))

        # Theme toggle
        self.btn_theme = QPushButton("Toggle Theme")
        self.btn_theme.clicked.connect(self.toggle_theme)
        vbox.addWidget(self.btn_theme)

        self.status_label = QLabel("Durum: Hazır.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9db3c4;")
        vbox.addWidget(self.status_label)

        vbox.addStretch()

        # internal state
        self.current_slice_actor = None
        self.stream_actor = None
        self.is_dark = True
        self.apply_theme()

        # populate tree with placeholder
        self._populate_tree()

    def _populate_tree(self):
        self.tree.clear()
        self.tree.expandAll()

    def apply_theme(self):
        if self.is_dark:
            self.setStyleSheet("QWidget{background:#1e1e1e;color:#ddd}")
            self.plotter.set_background("#222222")
        else:
            self.setStyleSheet("QWidget{background:#f6f6f6;color:#111}")
            self.plotter.set_background("#ffffff")

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.apply_theme()

    def set_status(self, message, is_error=False):
        self.status_label.setText(f"Durum: {message}")
        self.status_label.setStyleSheet("color: #d49a61;" if is_error else "color: #9db3c4;")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open mesh", str(Path.cwd()), "VTK Files (*.vtk);;All Files (*)"
        )
        if path:
            self.load_mesh(Path(path))

    def load_mesh(self, path: Path):
        try:
            self.mesh = pv.read(str(path))
        except Exception as e:
            self.set_status(f"Mesh yüklenemedi: {e}", is_error=True)
            return

        # populate scalar combo
        arrays = list(self.mesh.array_names)
        if not arrays:
            arrays = ["scalars"]
        self.scalar_combo.clear()
        self.scalar_combo.addItems(arrays)
        self.scalar_combo.currentTextChanged.connect(lambda _: self.redraw_slice())

        # initial view: add mesh surface with a clean, low-contrast material
        self.plotter.clear()
        self.surface_actor = self.plotter.add_mesh(self.mesh, color="#9aa5b1", opacity=0.25, lighting=True)
        # update scalar range and UI
        self.update_scalar_info()
        self.redraw_slice()
        self.plotter.reset_camera()

    def update_scalar_info(self):
        """Update scalar array list and min/max display based on current mesh."""
        if self.mesh is None:
            return
        arrays = list(self.mesh.array_names)
        if arrays:
            # ensure current list
            cur = self.scalar_combo.currentText()
            self.scalar_combo.blockSignals(True)
            self.scalar_combo.clear()
            self.scalar_combo.addItems(arrays)
            if cur in arrays:
                self.scalar_combo.setCurrentText(cur)
            self.scalar_combo.blockSignals(False)
            # update min/max for the selected scalar
            name = self.get_scalar_name()
            if name and name in self.mesh.array_names:
                try:
                    arr = self.mesh.get_array(name)
                    vmin = float(arr.min())
                    vmax = float(arr.max())
                    self.lbl_min.setText(f"{vmin:.4g}")
                    self.lbl_max.setText(f"{vmax:.4g}")
                except Exception:
                    self.lbl_min.setText("-")
                    self.lbl_max.setText("-")
        else:
            self.scalar_combo.clear()
            self.lbl_min.setText("-")
            self.lbl_max.setText("-")

    def get_scalar_name(self):
        return self.scalar_combo.currentText() if self.scalar_combo.count() else None

    def update_cut_plane(self, axis, value_percent):
        if self.mesh is None:
            return
        bounds = self.mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
        if axis == "x":
            origin = (bounds[0] + (bounds[1] - bounds[0]) * value_percent / 100.0, 0, 0)
            normal = (1, 0, 0)
        elif axis == "y":
            origin = (0, bounds[2] + (bounds[3] - bounds[2]) * value_percent / 100.0, 0)
            normal = (0, 1, 0)
        else:
            origin = (0, 0, bounds[4] + (bounds[5] - bounds[4]) * value_percent / 100.0)
            normal = (0, 0, 1)

        # build slice using robust slicing and preserve interpolation
        try:
            slice_mesh = self.mesh.slice(normal=normal, origin=origin)
            # attempt to resample onto a structured grid for smoother shading if possible
            try:
                grid = slice_mesh.delaunay_2d()
                if grid.n_points > 0:
                    slice_mesh = grid
            except Exception:
                pass
        except Exception:
            slice_mesh = None

        # remove previous slice actor
        if self.current_slice_actor:
            try:
                self.plotter.remove_actor(self.current_slice_actor)
            except Exception:
                pass
            self.current_slice_actor = None

        scalar = self.get_scalar_name()
        if slice_mesh is not None and getattr(slice_mesh, "n_points", 0) > 0:
            cmap = "RdYlBu_r"
            opacity = float(self.slice_opacity.value()) if hasattr(self, "slice_opacity") else 0.9
            if scalar and scalar in getattr(slice_mesh, "array_names", []):
                # use scalar coloring with explicit clim from mesh
                try:
                    arr = self.mesh.get_array(scalar)
                    clim = (float(arr.min()), float(arr.max()))
                except Exception:
                    clim = None
                try:
                    self.current_slice_actor = self.plotter.add_mesh(
                        slice_mesh, scalars=scalar, cmap=cmap, opacity=opacity, clim=clim
                    )
                except Exception as e:
                    self.set_status(f"Slice çizimi başarısız: {e}", is_error=True)
                    self.current_slice_actor = None
                # update scalar bar text labels
                try:
                    self.plotter.add_scalar_bar(title=scalar, n_labels=5, fmt="%0.3f", vertical=True)
                except Exception:
                    pass
                # update min/max display
                try:
                    if clim is not None:
                        self.lbl_min.setText(f"{clim[0]:.4g}")
                        self.lbl_max.setText(f"{clim[1]:.4g}")
                except Exception:
                    pass
            else:
                try:
                    self.current_slice_actor = self.plotter.add_mesh(
                        slice_mesh, color="orange", opacity=opacity
                    )
                except Exception as e:
                    self.set_status(f"Slice çizimi başarısız: {e}", is_error=True)
                    self.current_slice_actor = None

    def toggle_streamlines(self):
        if self.mesh is None:
            return
        if self.btn_stream.isChecked():
            # attempt to find vector field
            vec_names = [
                name
                for name in self.mesh.array_names
                if "vel" in name.lower() or "vector" in name.lower() or "velocity" in name.lower()
            ]
            if not vec_names:
                self.set_status("Streamline için uygun vektör alanı bulunamadı.", is_error=True)
                self.btn_stream.setChecked(False)
                return
            vec = vec_names[0]
            # build seed box (centered plane) using number of seeds
            nseeds = int(self.spin_seeds.value()) if hasattr(self, "spin_seeds") else 200
            maxt = float(self.spin_maxt.value()) if hasattr(self, "spin_maxt") else 1.0
            radius = float(self.spin_tuber.value()) if hasattr(self, "spin_tuber") else 0.01
            try:
                # make a regular grid of seeds on a plane at mid-height
                b = self.mesh.bounds
                x0, x1 = b[0], b[1]
                y0, y1 = b[2], b[3]
                zmid = (b[4] + b[5]) / 2.0
                # distribute seeds in a grid
                nx = max(2, int((nseeds) ** (1 / 2)))
                ny = max(2, int((nseeds) // nx))
                xs = [x0 + (x1 - x0) * (i / (nx - 1)) for i in range(nx)]
                ys = [y0 + (y1 - y0) * (j / (ny - 1)) for j in range(ny)]
                pts = []
                for xi in xs:
                    for yj in ys:
                        pts.append((xi, yj, zmid))
                seeds = pv.PolyData(pts)
            except Exception:
                seeds = None

            try:
                # Use DataSetFilters helper to avoid internal argument forwarding issues
                stream = pv.DataSetFilters.streamlines_from_source(
                    self.mesh, seeds, vectors=vec, max_length=maxt, integration_direction="both"
                )
            except TypeError:
                # fallback to method call with explicit keywords
                try:
                    stream = self.mesh.streamlines(
                        vectors=vec, source=seeds, max_length=maxt, integration_direction="both"
                    )
                except Exception as e:
                    self.set_status(f"Streamlines oluşturulamadı: {e}", is_error=True)
                    self.btn_stream.setChecked(False)
                    return
            except Exception as e:
                self.set_status(f"Streamlines oluşturulamadı: {e}", is_error=True)
                self.btn_stream.setChecked(False)
                return

            if stream is None or stream.n_points == 0:
                self.set_status("Streamlines boş çıktı verdi.", is_error=True)
                self.btn_stream.setChecked(False)
                return

            if self.chk_tubes.isChecked():
                try:
                    tubed = stream.tube(radius=radius)
                    self.stream_actor = self.plotter.add_mesh(tubed, scalars=None)
                except Exception:
                    self.stream_actor = self.plotter.add_mesh(stream, scalars=None, color="cyan")
            else:
                self.stream_actor = self.plotter.add_mesh(stream, scalars=None, color="cyan")
            self.plotter.render()
        else:
            if self.stream_actor:
                try:
                    self.plotter.remove_actor(self.stream_actor)
                except Exception:
                    pass
                self.stream_actor = None
                self.plotter.render()


def main():
    app = QApplication(sys.argv)
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
    win = PostProcessWindow(path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
