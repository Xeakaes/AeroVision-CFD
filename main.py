"""
AeroVision CFD - Main Application
Professional CFD simulation interface with turbulence modeling.
Run: python main.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import get_context
from queue import Empty

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QFileDialog,
    QStatusBar,
    QMessageBox,
    QComboBox,
    QSplitter,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QShortcut, QKeySequence, QAction

import pyvista as pv
from pyvistaqt import QtInteractor

from crash_logger import register_logger

register_logger()

from solver import (
    create_voxel_mask,
    solve_potential_flow,
    solve_navier_stokes_full,
    compute_pressure_field,
    interpolate_cp_to_vertices,
    rotate_vertices,
    apply_offset,
    compute_force_coefficients,
    GPU_DEVICE_NAME,
    _to_numpy,
)
from ui_components import build_left_panel, RES_MAP, get_resolution_key, get_turbulence_model

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

TUNNEL_DIMS = (10.0, 4.0, 4.0)
CP_CMAP, WIRE_COLOR = "RdYlBu_r", "#00ffcc"

FLUIDS = {
    "air": ("Hava (20°C)", 1.225, 1.5e-5),
    "water": ("Su (20°C)", 998.2, 1.004e-6),
    "co2": ("CO2 (20°C)", 1.87, 1.47e-5),
}

HIDDEN_CONFIG_FILE = Path(__file__).resolve().parent / ".aerovision_cfd_config.json"
DEFAULT_HIDDEN_SETTINGS = {
    "timeout_base": 600,
    "timeout_resolution_factor": 120,
    "timeout_iter_factor": 2,
    "timeout_spalart_multiplier": 1.25,
    "timeout_min": 600,
    "timeout_max": 7200,
}


def load_hidden_settings():
    if HIDDEN_CONFIG_FILE.exists():
        try:
            data = json.loads(HIDDEN_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**DEFAULT_HIDDEN_SETTINGS, **data}
        except Exception:
            pass
    HIDDEN_CONFIG_FILE.write_text(json.dumps(DEFAULT_HIDDEN_SETTINGS, indent=2), encoding="utf-8")
    return DEFAULT_HIDDEN_SETTINGS.copy()


HIDDEN_SETTINGS = load_hidden_settings()


def estimate_timeout(nx, ny, nz, iterations, turbulence_model):
    """Çözünürlük ve iterasyon sayısına göre dinamik timeout hesaplar."""
    cells = int(nx) * int(ny) * int(nz)
    base = HIDDEN_SETTINGS["timeout_base"]
    resolution_factor = HIDDEN_SETTINGS["timeout_resolution_factor"]
    timeout = int(
        base + cells / 100000.0 * resolution_factor + iterations * HIDDEN_SETTINGS["timeout_iter_factor"]
    )
    if str(turbulence_model).lower() in (
        "spalart_allmaras",
        "spalart-allmaras",
        "k_epsilon",
        "k-epsilon",
        "k_omega_sst",
        "k-omega-sst",
    ):
        timeout = int(timeout * HIDDEN_SETTINGS["timeout_spalart_multiplier"])
    return min(max(timeout, HIDDEN_SETTINGS["timeout_min"]), HIDDEN_SETTINGS["timeout_max"])


def compute_cfd_process(
    result_queue,
    verts,
    i_idx,
    j_idx,
    k_idx,
    yaw,
    pitch,
    roll,
    ox,
    oy,
    oz,
    wind_speed,
    nx,
    ny,
    nz,
    iterations,
    turbulence_model,
    use_wall_functions,
    gpu_selection="auto",
    rho=1.225,
    nu_mol=1.5e-5,
):
    def child_progress(message):
        result_queue.put(("PROGRESS", message))

    try:
        # GPU secimini uygula
        from solver import _detect_gpu

        _detect_gpu(gpu_selection)

        child_progress("Geometri islaniyor...")
        v = rotate_vertices(verts, yaw, pitch, roll)
        v = apply_offset(v, ox, oy, oz)
        child_progress("Geometri hazırlandı.")

        child_progress("Maske hesaplanıyor...")
        mask, solid_fraction, success = create_voxel_mask(v, i_idx, j_idx, k_idx, nx, ny, nz, TUNNEL_DIMS)
        child_progress("Maske tamamlandı.")

        child_progress("Potansiyel akış çözümü başlatılıyor...")
        u, v_arr, w = solve_potential_flow(
            mask, wind_speed, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=TUNNEL_DIMS
        )
        child_progress("Potansiyel akış tamamlandı.")

        child_progress("Navier-Stokes baslatiliyor...")

        def _ns_progress(step, residual, dt):
            result_queue.put(("CONV", step, residual))

        u, v_arr, w, p_field = solve_navier_stokes_full(
            u,
            v_arr,
            w,
            mask,
            wind_speed,
            nx,
            ny,
            nz,
            solid_fraction=solid_fraction,
            iterations=iterations,
            turbulence_model=turbulence_model,
            use_wall_functions=use_wall_functions,
            rho=rho,
            nu_mol=nu_mol,
            progress_callback=_ns_progress,
        )
        child_progress("Navier-Stokes tamamlandi.")

        cp_field = compute_pressure_field(u, v_arr, w, wind_speed)
        vertex_cp = interpolate_cp_to_vertices(v, cp_field, mask, TUNNEL_DIMS, nx, ny, nz)

        result_queue.put((v, u, v_arr, w, vertex_cp, cp_field, mask, success, None, p_field))
    except Exception as e:
        result_queue.put((None, None, None, None, None, None, None, False, str(e), None))


class CFDWorker(QThread):
    finished = pyqtSignal(object, object, object, object, object, object, object, bool, str, object)
    progress = pyqtSignal(str)
    progress_pct = pyqtSignal(int)
    convergence = pyqtSignal(float, float)  # step, residual

    def __init__(
        self,
        verts,
        i_idx,
        j_idx,
        k_idx,
        yaw,
        pitch,
        roll,
        ox,
        oy,
        oz,
        wind_speed,
        nx,
        ny,
        nz,
        iterations=180,
        turbulence_model="spalart_allmaras",
        use_wall_functions=False,
        gpu_selection="auto",
        rho=1.225,
        nu_mol=1.5e-5,
    ):
        super().__init__()
        self.verts, self.i_idx, self.j_idx, self.k_idx = verts, i_idx, j_idx, k_idx
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.ox, self.oy, self.oz = ox, oy, oz
        self.wind_speed = wind_speed
        self.nx, self.ny, self.nz = nx, ny, nz
        self.iterations = iterations
        self.turbulence_model = turbulence_model
        self.use_wall_functions = use_wall_functions
        self.gpu_selection = gpu_selection
        self.rho = rho
        self.nu_mol = nu_mol
        self.process = None

    def run(self):
        ctx = get_context("spawn")
        queue = ctx.Queue()
        result_data = None
        try:
            self.progress.emit("Geometri döndürülüyor...")
            self.progress_pct.emit(5)
            self.progress.emit("Geometri taraması gerçekleştiriliyor...")
            self.process = ctx.Process(
                target=compute_cfd_process,
                args=(
                    queue,
                    self.verts,
                    self.i_idx,
                    self.j_idx,
                    self.k_idx,
                    self.yaw,
                    self.pitch,
                    self.roll,
                    self.ox,
                    self.oy,
                    self.oz,
                    self.wind_speed,
                    self.nx,
                    self.ny,
                    self.nz,
                    int(self.iterations),
                    self.turbulence_model,
                    self.use_wall_functions,
                    self.gpu_selection,
                    self.rho,
                    self.nu_mol,
                ),
            )
            self.process.start()
            timeout_seconds = estimate_timeout(
                self.nx, self.ny, self.nz, self.iterations, self.turbulence_model
            )
            self.progress.emit(f"Zaman aşımı limiti: {timeout_seconds} saniye")
            deadline = time.time() + timeout_seconds
            step_count = 0
            total_steps = 10  # Geometri, Maske, Potansiyel, NS, Q-Crit, VTK = ~10 adım
            while self.process.is_alive():
                try:
                    item = queue.get(timeout=0.25)
                except Empty:
                    item = None

                if item is not None:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "PROGRESS":
                        self.progress.emit(item[1])
                        step_count = min(step_count + 1, total_steps)
                        pct = 10 + int(80 * step_count / total_steps)
                        self.progress_pct.emit(pct)
                    elif isinstance(item, tuple) and len(item) == 2 and item[0] == "PROGRESS_PCT":
                        self.progress_pct.emit(int(item[1]))
                    elif isinstance(item, tuple) and len(item) == 3 and item[0] == "CONV":
                        self.convergence.emit(float(item[1]), float(item[2]))
                    else:
                        result_data = item
                        self.progress_pct.emit(90)

                if time.time() > deadline:
                    self.progress.emit("Zaman aşımı - process sonlandırılıyor...")
                    self.process.terminate()
                    self.process.join(timeout=5)
                    raise RuntimeError("CFD hesaplaması zaman aşımına uğradı veya asıldı.")

            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except Empty:
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "PROGRESS":
                    self.progress.emit(item[1])
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "PROGRESS_PCT":
                    self.progress_pct.emit(int(item[1]))
                elif isinstance(item, tuple) and len(item) == 3 and item[0] == "CONV":
                    self.convergence.emit(float(item[1]), float(item[2]))
                else:
                    result_data = item

            if result_data is None:
                raise RuntimeError("Process sonuç kuyruğunda beklenen veri bulunamadı.")

            if isinstance(result_data, tuple) and len(result_data) == 10:
                v, u, v_arr, w, vertex_cp, cp_field, mask, success, error, p_field = result_data
            else:
                raise RuntimeError("Process beklenmeyen sonuç döndürdü.")

            if not success:
                raise RuntimeError(error or "CFD işlemi başarısız oldu")

            self.progress.emit("Q-Criterion hesaplaniyor...")
            self.progress_pct.emit(92)
            import numpy as np
            from solver import compute_q_criterion, GPU_AVAILABLE as _gpu_avail

            nx_p = int(self.nx.item()) if hasattr(self.nx, "item") else int(self.nx)
            ny_p = int(self.ny.item()) if hasattr(self.ny, "item") else int(self.ny)
            nz_p = int(self.nz.item()) if hasattr(self.nz, "item") else int(self.nz)

            tx_p = float(TUNNEL_DIMS[0].item()) if hasattr(TUNNEL_DIMS[0], "item") else float(TUNNEL_DIMS[0])
            ty_p = float(TUNNEL_DIMS[1].item()) if hasattr(TUNNEL_DIMS[1], "item") else float(TUNNEL_DIMS[1])
            tz_p = float(TUNNEL_DIMS[2].item()) if hasattr(TUNNEL_DIMS[2], "item") else float(TUNNEL_DIMS[2])

            dx, dy, dz = tx_p / (nx_p - 1), ty_p / (ny_p - 1), tz_p / (nz_p - 1)
            q_crit = compute_q_criterion(u, v_arr, w, dx, dy, dz, GPU=_gpu_avail)

            self.progress.emit("ParaView .vtk dosyasi diske yaziliyor...")
            self.progress_pct.emit(95)

            u_cpu = np.ascontiguousarray(_to_numpy(u))
            v_cpu = np.ascontiguousarray(_to_numpy(v_arr))
            w_cpu = np.ascontiguousarray(_to_numpy(w))
            p_cpu = np.ascontiguousarray(_to_numpy(cp_field))
            q_cpu = np.ascontiguousarray(_to_numpy(q_crit))

            grid_para = pv.ImageData()
            grid_para.dimensions = (nx_p, ny_p, nz_p)
            grid_para.spacing = (tx_p / (nx_p - 1), ty_p / (ny_p - 1), tz_p / (nz_p - 1))
            grid_para.origin = (-tx_p / 2.0, -ty_p / 2.0, -tz_p / 2.0)

            num_points = nx_p * ny_p * nz_p
            vector_data = np.column_stack(
                [u_cpu.ravel(order="F"), v_cpu.ravel(order="F"), w_cpu.ravel(order="F")]
            )
            if vector_data.shape[0] != num_points:
                raise RuntimeError(
                    f"VTK veri boyutu eşleşmiyor: beklenen {num_points}, ancak velocity dizisi {vector_data.shape[0]} satır içeriyor."
                )
            grid_para.point_data["Velocity"] = vector_data
            grid_para.point_data["Pressure_Cp"] = p_cpu.ravel(order="F")
            grid_para.point_data["Q-Criterion"] = q_cpu.ravel(order="F")

            output_path = Path(__file__).resolve().parent / "cfd_output_results.vtk"
            try:
                grid_para.save(str(output_path))
            except Exception as save_error:
                fallback_msg = ""
                if hasattr(pv, "save_meshio"):
                    try:
                        pv.save_meshio(str(output_path), grid_para)
                        fallback_msg = " (yedek kaydetme yöntemiyle başarıyla tamamlandı)"
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"VTK kaydı başarısız: {save_error} ve yedek yöntem de başarısız oldu: {fallback_error}"
                        ) from fallback_error
                else:
                    raise RuntimeError(f"VTK kaydı başarısız: {save_error}") from save_error
                self.progress.emit(f"VTK kaydı aracı yapılandırıldı{fallback_msg}")

            self.progress_pct.emit(100)
            self.progress.emit(f"ParaView çıktısı oluşturuldu: {output_path}")

            self.finished.emit(v, u, v_arr, w, vertex_cp, cp_field, mask, success, None, p_field)

        except Exception as e:
            self.progress.emit(f"Hata: {e}")
            self.finished.emit(None, None, None, None, None, None, None, False, str(e), None)
        finally:
            if self.process is not None and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)


class AeroVisionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroVision CFD v1.0")
        self.resize(1500, 900)
        self.verts_base = None
        self.i_idx = self.j_idx = self.k_idx = None
        self.cfd_worker = None
        self.sim_history = []
        self.recent_files = []
        self._build_ui()
        self._init_scene()
        self._setup_shortcuts()
        self._load_recent_files()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_stl)
        QShortcut(QKeySequence("Ctrl+R"), self, self.run_cfd)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_screenshot)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_results_csv)
        QShortcut(QKeySequence("Ctrl+Shift+E"), self, self.export_results_vtk)
        QShortcut(QKeySequence("Ctrl+P"), self, self.save_preset)
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, self.load_preset)
        QShortcut(QKeySequence("Ctrl+B"), self, self.run_batch)
        QShortcut(QKeySequence("F5"), self, self.run_cfd)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def _load_recent_files(self):
        try:
            recent_path = Path(__file__).parent / ".recent_files.json"
            if recent_path.exists():
                self.recent_files = json.loads(recent_path.read_text(encoding="utf-8"))
        except Exception:
            self.recent_files = []

    def _save_recent_files(self):
        try:
            recent_path = Path(__file__).parent / ".recent_files.json"
            recent_path.write_text(json.dumps(self.recent_files[:10], ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _create_menu_bar(self):
        """Menu cubugu olustur - dinamik dil destegi ile."""
        menu_bar = self.menuBar()
        self.menu_refs = {}

        # Menu metinleri sozlugu
        self.menu_texts = {
            "tr": {
                "file": "Dosya",
                "load_stl": "STL Yukle",
                "save_img": "Gorsel Kaydet",
                "export": "Disa Aktar",
                "recent": "Son Dosyalar",
                "quit": "Cikis",
                "edit": "Duzenle",
                "preset_save": "Preset Kaydet",
                "preset_load": "Preset Yukle",
                "sim": "Simulasyon",
                "run": "Simulasyonu Baslat",
                "batch": "Toplu Simulasyon",
                "compare": "Karsilastirma Tablosu",
                "animate": "3D Animasyon Kaydet",
                "view": "Gorunum",
                "preview": "Onizlemeyi Guncelle",
                "help": "Yardim",
                "about": "Hakkinda",
            },
            "en": {
                "file": "File",
                "load_stl": "Load STL",
                "save_img": "Save Image",
                "export": "Export",
                "recent": "Recent Files",
                "quit": "Exit",
                "edit": "Edit",
                "preset_save": "Save Preset",
                "preset_load": "Load Preset",
                "sim": "Simulate",
                "run": "Start Simulation",
                "batch": "Batch Simulation",
                "compare": "Comparison Table",
                "animate": "Record 3D Animation",
                "view": "View",
                "preview": "Update Preview",
                "help": "Help",
                "about": "About",
            },
        }

        t = self.menu_texts["tr"]  # Varsayilan Turkce

        # File menu
        self.menu_refs["file"] = menu_bar.addMenu(t["file"])

        self.menu_refs["load_stl"] = QAction(t["load_stl"], self)
        self.menu_refs["load_stl"].setShortcut("Ctrl+O")
        self.menu_refs["load_stl"].triggered.connect(self.load_stl)
        self.menu_refs["file"].addAction(self.menu_refs["load_stl"])

        self.menu_refs["file"].addSeparator()

        self.menu_refs["save_img"] = QAction(t["save_img"], self)
        self.menu_refs["save_img"].setShortcut("Ctrl+S")
        self.menu_refs["save_img"].triggered.connect(self.save_screenshot)
        self.menu_refs["file"].addAction(self.menu_refs["save_img"])

        self.menu_refs["file"].addSeparator()

        self.menu_refs["export_menu"] = self.menu_refs["file"].addMenu(t["export"])

        export_csv = QAction("CSV", self)
        export_csv.setShortcut("Ctrl+E")
        export_csv.triggered.connect(self.export_results_csv)
        self.menu_refs["export_menu"].addAction(export_csv)

        export_json = QAction("JSON", self)
        export_json.triggered.connect(self.export_results_json)
        self.menu_refs["export_menu"].addAction(export_json)

        export_vtk = QAction("VTK", self)
        export_vtk.setShortcut("Ctrl+Shift+E")
        export_vtk.triggered.connect(self.export_results_vtk)
        self.menu_refs["export_menu"].addAction(export_vtk)

        self.menu_refs["file"].addSeparator()

        self.menu_refs["recent_menu"] = self.menu_refs["file"].addMenu(t["recent"])
        self._update_recent_menu()

        self.menu_refs["file"].addSeparator()

        self.menu_refs["quit"] = QAction(t["quit"], self)
        self.menu_refs["quit"].setShortcut("Ctrl+Q")
        self.menu_refs["quit"].triggered.connect(self.close)
        self.menu_refs["file"].addAction(self.menu_refs["quit"])

        # Edit menu
        self.menu_refs["edit"] = menu_bar.addMenu(t["edit"])

        self.menu_refs["preset_save"] = QAction(t["preset_save"], self)
        self.menu_refs["preset_save"].setShortcut("Ctrl+P")
        self.menu_refs["preset_save"].triggered.connect(self.save_preset)
        self.menu_refs["edit"].addAction(self.menu_refs["preset_save"])

        self.menu_refs["preset_load"] = QAction(t["preset_load"], self)
        self.menu_refs["preset_load"].setShortcut("Ctrl+Shift+P")
        self.menu_refs["preset_load"].triggered.connect(self.load_preset)
        self.menu_refs["edit"].addAction(self.menu_refs["preset_load"])

        # Simulate menu
        self.menu_refs["sim"] = menu_bar.addMenu(t["sim"])

        self.menu_refs["run"] = QAction(t["run"], self)
        self.menu_refs["run"].setShortcut("F5")
        self.menu_refs["run"].triggered.connect(self.run_cfd)
        self.menu_refs["sim"].addAction(self.menu_refs["run"])

        self.menu_refs["sim"].addSeparator()

        self.menu_refs["batch"] = QAction(t["batch"], self)
        self.menu_refs["batch"].setShortcut("Ctrl+B")
        self.menu_refs["batch"].triggered.connect(self.run_batch)
        self.menu_refs["sim"].addAction(self.menu_refs["batch"])

        self.menu_refs["compare"] = QAction(t["compare"], self)
        self.menu_refs["compare"].triggered.connect(self.show_comparison_table)
        self.menu_refs["sim"].addAction(self.menu_refs["compare"])

        self.menu_refs["sim"].addSeparator()

        self.menu_refs["animate"] = QAction(t["animate"], self)
        self.menu_refs["animate"].triggered.connect(self.record_animation)
        self.menu_refs["sim"].addAction(self.menu_refs["animate"])

        # View menu
        self.menu_refs["view"] = menu_bar.addMenu(t["view"])

        self.menu_refs["preview"] = QAction(t["preview"], self)
        self.menu_refs["preview"].setShortcut("Ctrl+R")
        self.menu_refs["preview"].triggered.connect(self.update_preview)
        self.menu_refs["view"].addAction(self.menu_refs["preview"])

        # Help menu
        self.menu_refs["help"] = menu_bar.addMenu(t["help"])

        self.menu_refs["about"] = QAction(t["about"], self)
        self.menu_refs["about"].triggered.connect(self._show_about)
        self.menu_refs["help"].addAction(self.menu_refs["about"])

    def _update_menu_language(self, lang="tr"):
        """Menu cubugu dilini guncelle."""
        t = self.menu_texts.get(lang, self.menu_texts["tr"])
        self.menu_refs["file"].setTitle(t["file"])
        self.menu_refs["edit"].setTitle(t["edit"])
        self.menu_refs["sim"].setTitle(t["sim"])
        self.menu_refs["view"].setTitle(t["view"])
        self.menu_refs["help"].setTitle(t["help"])
        self.menu_refs["load_stl"].setText(t["load_stl"])
        self.menu_refs["save_img"].setText(t["save_img"])
        self.menu_refs["export_menu"].setTitle(t["export"])
        self.menu_refs["recent_menu"].setTitle(t["recent"])
        self.menu_refs["quit"].setText(t["quit"])
        self.menu_refs["preset_save"].setText(t["preset_save"])
        self.menu_refs["preset_load"].setText(t["preset_load"])
        self.menu_refs["run"].setText(t["run"])
        self.menu_refs["batch"].setText(t["batch"])
        self.menu_refs["compare"].setText(t["compare"])
        self.menu_refs["animate"].setText(t["animate"])
        self.menu_refs["preview"].setText(t["preview"])
        self.menu_refs["about"].setText(t["about"])

    def _update_recent_menu(self):
        """Son dosyalar menusunu guncelle."""
        if "recent_menu" not in self.menu_refs:
            return
        self.menu_refs["recent_menu"].clear()
        for file_path in self.recent_files[:5]:
            action = QAction(Path(file_path).name, self)
            action.setData(file_path)
            action.triggered.connect(lambda checked, p=file_path: self._load_recent_file(p))
            self.menu_refs["recent_menu"].addAction(action)

    def _load_recent_file(self, file_path):
        """Son dosyadan yukle."""
        if Path(file_path).exists():
            self.load_stl_file(file_path)
        else:
            QMessageBox.warning(self, "Uyari", f"Dosya bulunamadi: {file_path}")

    def _show_about(self):
        """Hakkinda penceresi."""
        QMessageBox.about(
            self,
            "AeroVision CFD v1.0",
            "AeroVision CFD v1.0\n\n"
            "Professional CFD Simulation Interface\n"
            "with Turbulence Modeling and 3D Visualization\n\n"
            "Features:\n"
            "- Navier-Stokes Solver\n"
            "- Multiple Turbulence Models\n"
            "- 3D Visualization\n"
            "- Batch Processing\n"
            "- GPU Acceleration (NVIDIA)",
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Splitter ile panel ve plotter arasinda surukle-birak ile boyutlandirma
        splitter = QSplitter(Qt.Orientation.Horizontal)

        panel, self.w = build_left_panel()
        splitter.addWidget(panel)

        # Menu cubugu olustur
        self._create_menu_bar()

        # Buton baglantilari
        self.w["btn_load"].clicked.connect(self.load_stl)
        self.w["btn_preview"].clicked.connect(self.update_preview)
        self.w["btn_run"].clicked.connect(self.run_cfd)
        self.w["btn_screenshot"].clicked.connect(self.save_screenshot)
        self.w["btn_export_csv"].clicked.connect(self.export_results_csv)
        self.w["btn_export_json"].clicked.connect(self.export_results_json)
        self.w["btn_export_vtk"].clicked.connect(self.export_results_vtk)
        self.w["btn_preset_save"].clicked.connect(self.save_preset)
        self.w["btn_preset_load"].clicked.connect(self.load_preset)
        self.w["btn_batch"].clicked.connect(self.run_batch)
        self.w["btn_compare"].clicked.connect(self.show_comparison_table)
        self.w["btn_animate"].clicked.connect(self.record_animation)

        # Dil degisiminde menu cubugunu da guncelle
        from ui_components import set_language_callback

        set_language_callback(self._update_menu_language)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0d0d0d")
        self.plotter.setAcceptDrops(True)
        self.plotter.dragEnterEvent = self._drag_enter_event
        self.plotter.dropEvent = self._drop_event
        splitter.addWidget(self.plotter.interactor)

        # Splitter ayarlari: baslangicta panel 300px, geri kalan plotter
        splitter.setSizes([300, 700])
        splitter.setStretchFactor(0, 0)  # Panel esnemez
        splitter.setStretchFactor(1, 1)  # Plotter esner

        # Layout'a splitter'i ekle
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        gpu_info = (
            f"GPU: {GPU_DEVICE_NAME}" if GPU_DEVICE_NAME and GPU_DEVICE_NAME != "Yok" else "GPU: Yok (CPU)"
        )
        self.status_bar.showMessage(f"Hazir. {gpu_info} | v1.0")

        # İlerleme çubuğu percentage mode
        self.w["progress_bar"].setRange(0, 100)
        self.w["progress_bar"].setValue(0)

        # Karşılaştırma combo box
        from PyQt6.QtWidgets import QLabel as QLbl

        self.comparison_combo = QComboBox()
        self.comparison_combo.currentIndexChanged.connect(self._on_comparison_select)
        self.comparison_combo.setVisible(False)
        self.lbl_comparison = QLbl("Karşılaştırma:")
        self.lbl_comparison.setVisible(False)

        # Rapor grubuna ekle
        report_layout = self.w["report_group"].layout()
        if report_layout:
            report_layout.addWidget(self.lbl_comparison)
            report_layout.addWidget(self.comparison_combo)

        # Yakınsama grafiği (matplotlib)
        self._convergence_times = []
        self._convergence_values = []
        if HAS_MATPLOTLIB:
            self.conv_figure = Figure(figsize=(3, 2), dpi=100)
            self.conv_figure.set_facecolor("#1e1e1e")
            self.conv_canvas = FigureCanvas(self.conv_figure)
            self.conv_canvas.setFixedHeight(180)
            self.conv_ax = self.conv_figure.add_subplot(111)
            self.conv_ax.set_facecolor("#1e1e1e")
            self.conv_ax.set_title("Yakınsama", color="white", fontsize=9)
            self.conv_ax.set_xlabel("Süre (s)", color="white", fontsize=7)
            self.conv_ax.set_ylabel("Cd", color="white", fontsize=7)
            self.conv_ax.tick_params(colors="white", labelsize=6)
            for spine in self.conv_ax.spines.values():
                spine.set_color("#444")
            self.conv_figure.tight_layout()
            report_layout.addWidget(self.conv_canvas)

    def _ensure_tunnel(self):
        tx, ty, tz = TUNNEL_DIMS
        box = pv.Box(bounds=(-tx / 2, tx / 2, -ty / 2, ty / 2, -tz / 2, tz / 2))
        self.plotter.add_mesh(box, style="wireframe", color=WIRE_COLOR, line_width=1.5, name="tunnel")

    def _init_scene(self):
        self._ensure_tunnel()
        self.plotter.add_axes()
        self.plotter.view_xz()
        self.plotter.camera.roll = 0
        self.plotter.enable_cell_picking(callback=self._on_cell_picked, show=False)
        self.plotter.reset_camera()

    def _resolve_wind_speed(self):
        if "wind_speed" in self.w:
            value = self.w["wind_speed"]
            if hasattr(value, "value"):
                return float(value.value())
            return float(value)
        return float(self.w["spin_wind"].value())

    def _resolve_gpu(self):
        """Secilen GPU/CPu birimini dondur."""
        if "combo_gpu" in self.w:
            return self.w["combo_gpu"].currentData()
        return "auto"

    def _resolve_fluid(self):
        """Secilen akiskanin (rho, nu_mol) degerlerini dondur."""
        key = self.w["combo_fluid"].currentData() if "combo_fluid" in self.w else "air"
        _, rho, nu_mol = FLUIDS.get(key, FLUIDS["air"])
        return float(rho), float(nu_mol)

    def _on_cell_picked(self, cell_mesh):
        if cell_mesh is None or "Cp" not in cell_mesh.cell_data:
            return
        picked_idx = self.plotter.picker.GetCellId()
        if picked_idx < 0:
            return
        cp_val = cell_mesh.cell_data["Cp"][picked_idx]
        rho, v_inf = 1.225, self._resolve_wind_speed()
        local_force = max(0.0, float(cp_val * (0.5 * rho * (v_inf**2)) * 0.002))
        self.plotter.add_text(
            f"Secilen Nokta Indeksi: {picked_idx}\nLokal Basinc (Cp): {cp_val:.3f}\nAnlik Ruzgar Direnci: {local_force:.2f} N",
            position="upper_left",
            font_size=11,
            color="white",
            name="probe_text",
        )

    def _drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".stl"):
                    event.acceptProposedAction()
                    return

    def _drop_event(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".stl"):
                self.load_stl_file(file_path)
                break

    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "STL Dosyasi Sec", "", "STL Files (*.stl)")
        if not path:
            return
        from stl import mesh as stl_mesh

        try:
            m = stl_mesh.Mesh.from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"STL dosyasi okunamadi:\n{e}")
            return

        verts = m.vectors.reshape(-1, 3).astype(np.float32)
        if len(verts) < 3:
            QMessageBox.warning(self, "Uyari", "STL dosyasi yeterli ucgen icermiyor.")
            return

        if not np.isfinite(verts).all():
            QMessageBox.critical(self, "Hata", "STL dosyasi gecersiz (sonlu olmayan) koordinatlar iceriyor.")
            return

        verts -= verts.mean(axis=0)
        scale = TUNNEL_DIMS[1] * 0.35 / (np.max(np.abs(verts)) + 1e-12)
        verts *= scale

        n_tri = len(verts) // 3
        self.verts_base = verts
        self.i_idx = np.arange(0, n_tri * 3, 3, dtype=np.int32)
        self.j_idx = self.i_idx + 1
        self.k_idx = self.i_idx + 2

        self.stl_normals = m.normals.copy().astype(np.float64)
        norm_lens = np.linalg.norm(self.stl_normals, axis=1, keepdims=True)
        self.stl_normals = self.stl_normals / (norm_lens + 1e-12)

        self.w["_loaded_file_text"] = f"{Path(path).name}\n{n_tri:,} triangle"
        self.w["lbl_file"].setText(self.w["_loaded_file_text"])
        self.w["btn_preview"].setEnabled(True)
        self.w["btn_run"].setEnabled(True)
        self.w["btn_screenshot"].setEnabled(True)
        self.status_bar.showMessage(f"Loaded: {Path(path).name} ({n_tri:,} triangles)")
        self.update_preview()

        # Son dosya listesini güncelle
        if path not in self.recent_files:
            self.recent_files.insert(0, path)
            self.recent_files = self.recent_files[:10]
            self._save_recent_files()

    def load_stl_file(self, path):
        """Drag and drop icin STL dosyasi yukleme."""
        from stl import mesh as stl_mesh

        try:
            m = stl_mesh.Mesh.from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"STL dosyasi okunamadi:\n{e}")
            return

        verts = m.vectors.reshape(-1, 3).astype(np.float32)
        if len(verts) < 3:
            QMessageBox.warning(self, "Uyari", "STL dosyasi yeterli ucgen icermiyor.")
            return

        if not np.isfinite(verts).all():
            QMessageBox.critical(self, "Hata", "STL dosyasi gecersiz (sonlu olmayan) koordinatlar iceriyor.")
            return

        verts -= verts.mean(axis=0)
        scale = TUNNEL_DIMS[1] * 0.35 / (np.max(np.abs(verts)) + 1e-12)
        verts *= scale

        n_tri = len(verts) // 3
        self.verts_base = verts
        self.i_idx = np.arange(0, n_tri * 3, 3, dtype=np.int32)
        self.j_idx = self.i_idx + 1
        self.k_idx = self.i_idx + 2

        self.stl_normals = m.normals.copy().astype(np.float64)
        norm_lens = np.linalg.norm(self.stl_normals, axis=1, keepdims=True)
        self.stl_normals = self.stl_normals / (norm_lens + 1e-12)

        self.w["_loaded_file_text"] = f"{Path(path).name}\n{n_tri:,} triangle"
        self.w["lbl_file"].setText(self.w["_loaded_file_text"])
        self.w["btn_preview"].setEnabled(True)
        self.w["btn_run"].setEnabled(True)
        self.w["btn_screenshot"].setEnabled(True)
        self.status_bar.showMessage(f"Loaded: {Path(path).name} ({n_tri:,} triangles)")
        self.update_preview()

        # Son dosya listesini güncelle
        if path not in self.recent_files:
            self.recent_files.insert(0, path)
            self.recent_files = self.recent_files[:10]
            self._save_recent_files()

    def update_preview(self):
        if self.verts_base is None:
            return
        self._ensure_tunnel()
        v = rotate_vertices(
            self.verts_base,
            self.w["spin_yaw"].value(),
            self.w["spin_pitch"].value(),
            self.w["spin_roll"].value(),
        )
        v = apply_offset(v, self.w["spin_ox"].value(), self.w["spin_oy"].value(), self.w["spin_oz"].value())
        n_tri = len(self.i_idx)
        faces_pv = np.hstack(
            [
                np.full((n_tri, 1), 3, dtype=np.int32),
                self.i_idx.reshape(-1, 1),
                self.j_idx.reshape(-1, 1),
                self.k_idx.reshape(-1, 1),
            ]
        ).ravel()
        mesh_pv = pv.PolyData(v, faces_pv)
        self.plotter.remove_actor("model_preview")
        self.plotter.remove_actor("model_cfd")
        self.plotter.add_mesh(
            mesh_pv, color="#7da2be", opacity=0.7, smooth_shading=True, name="model_preview"
        )

    def run_cfd(self):
        if self.verts_base is None:
            return
        import time as _time

        # ── Girdi doğrulama: geometri tünel sınırları içinde mi? ─────────
        v_check = rotate_vertices(
            self.verts_base,
            self.w["spin_yaw"].value(),
            self.w["spin_pitch"].value(),
            self.w["spin_roll"].value(),
        )
        v_check = apply_offset(
            v_check, self.w["spin_ox"].value(), self.w["spin_oy"].value(), self.w["spin_oz"].value()
        )
        margin = 0.05 * TUNNEL_DIMS[0]
        bounds = (TUNNEL_DIMS[0] / 2 - margin, TUNNEL_DIMS[1] / 2 - margin, TUNNEL_DIMS[2] / 2 - margin)
        outside = (np.abs(v_check) > np.array(bounds)).any()
        if outside:
            reply = QMessageBox.question(
                self,
                "Geometri Tünel Dışında",
                "Model, hesaplama tünelinin sınırları dışına taşıyor. Sonuçlar hatalı olabilir.\n\nYine de devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._cfd_start_time = _time.time()
        self._last_res_key = get_resolution_key(self.w)
        nx, ny, nz = RES_MAP[self._last_res_key]
        self._last_wind_speed = self._resolve_wind_speed()
        self._last_turbulence_model = get_turbulence_model(self.w)
        self._last_rho, self._last_nu_mol = self._resolve_fluid()
        wind_speed = self._last_wind_speed
        turbulence_model = self._last_turbulence_model
        use_wall_functions = bool(
            self.w.get("chk_wall_functions") and self.w["chk_wall_functions"].isChecked()
        )
        self.w["btn_run"].setEnabled(False)
        self.w["btn_preview"].setEnabled(False)
        self.w["progress_bar"].setVisible(True)
        self.w["progress_bar"].setRange(0, 100)
        self.w["progress_bar"].setValue(0)
        self._residual_steps, self._residual_values = [], []
        self.cfd_worker = CFDWorker(
            self.verts_base,
            self.i_idx,
            self.j_idx,
            self.k_idx,
            self.w["spin_yaw"].value(),
            self.w["spin_pitch"].value(),
            self.w["spin_roll"].value(),
            self.w["spin_ox"].value(),
            self.w["spin_oy"].value(),
            self.w["spin_oz"].value(),
            wind_speed,
            nx,
            ny,
            nz,
            iterations=180 if turbulence_model in ("spalart_allmaras", "k_epsilon", "k_omega_sst") else 120,
            turbulence_model=turbulence_model,
            use_wall_functions=use_wall_functions,
            gpu_selection=self._resolve_gpu(),
            rho=self._last_rho,
            nu_mol=self._last_nu_mol,
        )
        self.cfd_worker.progress.connect(self._on_progress)
        self.cfd_worker.progress_pct.connect(self._on_progress_pct)
        self.cfd_worker.convergence.connect(self._on_convergence)
        self.cfd_worker.finished.connect(self._on_cfd_done)
        self.cfd_worker.start()

    def _on_progress(self, msg):
        self.w["lbl_status"].setText(msg)
        self.status_bar.showMessage(msg)

    def _on_progress_pct(self, pct):
        self.w["progress_bar"].setValue(pct)

    def _on_convergence(self, step, residual):
        """Canli yakinsama (residual) grafigi — NS her 10 adimda bildirir."""
        if not HAS_MATPLOTLIB or not hasattr(self, "conv_ax"):
            return
        if not hasattr(self, "_residual_steps"):
            self._residual_steps, self._residual_values = [], []
        self._residual_steps.append(step)
        self._residual_values.append(residual)
        ax = self.conv_ax
        ax.clear()
        ax.set_facecolor("#1e1e1e")
        ax.semilogy(self._residual_steps, self._residual_values, color="#4fc3f7", linewidth=1.2)
        ax.set_title("Yakınsama (Residual)", color="white", fontsize=9)
        ax.set_xlabel("NS Adımı", color="white", fontsize=7)
        ax.set_ylabel("Residual", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#444")
        self.conv_figure.tight_layout()
        self.conv_canvas.draw_idle()

    def _on_cfd_done(
        self,
        final_verts,
        u,
        v_arr,
        w,
        vertex_cp,
        cp_field,
        mask,
        voxel_success,
        error_message=None,
        p_field=None,
    ):
        self.w["progress_bar"].setVisible(False)
        self.w["btn_run"].setEnabled(True)
        self.w["btn_preview"].setEnabled(True)
        if final_verts is None:
            error_text = f"❌ Hesaplama başarısız: {error_message or 'Bilinmeyen hata.'}"
            self.w["lbl_status"].setText(error_text)
            QMessageBox.critical(self, "Hata", error_text)
            return

        self.plotter.clear_actors()
        self._ensure_tunnel()

        if mask is not None and not mask.any():
            QMessageBox.warning(
                self, "Uyari", "Geometri tünel ızgarasında tespit edilemedi. Sonuçlar güvenilir olmayabilir."
            )

        use_wall_functions = bool(
            self.w.get("chk_wall_functions") and self.w["chk_wall_functions"].isChecked()
        )
        stl_normals = getattr(self, "stl_normals", None)
        if stl_normals is not None:
            # final_verts yaw/pitch/roll ile döndürülmüş olduğundan normaler
            # de aynı rotasyonla döndürülmeli (aksi halde kuvvet yönleri bozuk).
            from solver import rotate_vertices

            stl_normals = rotate_vertices(
                stl_normals,
                self.w["spin_yaw"].value(),
                self.w["spin_pitch"].value(),
                self.w["spin_roll"].value(),
            )
            norm_lens = np.linalg.norm(stl_normals, axis=1, keepdims=True)
            stl_normals = stl_normals / (norm_lens + 1e-12)
        forces = compute_force_coefficients(
            final_verts,
            self.i_idx,
            self.j_idx,
            self.k_idx,
            cp_field,
            u,
            v_arr,
            w,
            TUNNEL_DIMS,
            u.shape[0],
            u.shape[1],
            u.shape[2],
            self._resolve_wind_speed(),
            mask=mask,
            use_wall_functions=use_wall_functions,
            stl_normals=stl_normals,
            p_field=p_field,
            rho=getattr(self, "_last_rho", None),
            nu_mol=getattr(self, "_last_nu_mol", None),
        )
        face_cp = (vertex_cp[self.i_idx] + vertex_cp[self.j_idx] + vertex_cp[self.k_idx]) / 3.0

        self.report_group = self.w.get("report_group")
        if self.report_group:
            self.report_group.setVisible(True)
        drag_value = "N/A" if not forces.get("valid", True) else f"{forces['drag_N']:.1f} N"
        if "lbl_drag" in self.w:
            self.w["lbl_drag"].setText(
                f"Sürükleme Kuvveti (Fd): {drag_value}\nDikey Karakteristik: {forces['status']}"
            )
        if "lbl_efficiency" in self.w:
            self.w["lbl_efficiency"].setText(forces["label"])

        n_tri = len(self.i_idx)
        faces_pv = np.hstack(
            [
                np.full((n_tri, 1), 3, dtype=np.int32),
                self.i_idx.reshape(-1, 1),
                self.j_idx.reshape(-1, 1),
                self.k_idx.reshape(-1, 1),
            ]
        ).ravel()
        mesh_pv = pv.PolyData(final_verts, faces_pv)
        mesh_pv.cell_data["Cp"] = face_cp

        self.plotter.add_mesh(
            mesh_pv,
            scalars="Cp",
            cmap=CP_CMAP,
            clim=(-3.0, 1.0),
            smooth_shading=True,
            scalar_bar_args={
                "title": "Cp",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.15,
                "width": 0.04,
                "height": 0.7,
                "color": "white",
            },
            name="model_cfd",
        )

        nx_p, ny_p, nz_p = int(u.shape[0]), int(u.shape[1]), int(u.shape[2])
        tx_p, ty_p, tz_p = 10.0, 4.0, 4.0

        grid = pv.ImageData()
        grid.dimensions = (nx_p, ny_p, nz_p)
        grid.spacing = (tx_p / (nx_p - 1), ty_p / (ny_p - 1), tz_p / (nz_p - 1))
        grid.origin = (-tx_p / 2.0, -ty_p / 2.0, -tz_p / 2.0)

        velocities = np.stack([u.flatten("F"), v_arr.flatten("F"), w.flatten("F")], axis=1)
        grid.point_data["velocity"] = velocities
        grid.point_data["Hiz_Buyuklugu"] = np.linalg.norm(velocities, axis=1)

        # ── Çizim pürüzsüzleştirmesi: Alanda daha temiz bir segment görselleştirmesi sağlanır ──
        slice_y = grid.slice(normal="y", origin=(0, 0, 0))
        self.plotter.add_mesh(
            slice_y,
            scalars="Hiz_Buyuklugu",
            cmap="turbo_r",
            opacity=0.7,
            interpolate_before_map=True,
            smooth_shading=True,  # Akış görselleştirme kalitesini artırır
            scalar_bar_args={
                "title": "Ruzgar Hizi (m/s)",
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.05,
                "width": 0.5,
                "height": 0.05,
                "color": "white",
            },
        )

        def safe_add_streams(source_plane, label):
            try:
                s = grid.streamlines_from_source(
                    source_plane,
                    vectors="velocity",
                    max_length=tx_p * 3,
                    integration_direction="forward",
                    max_steps=5000,
                    terminal_speed=0.01,
                )
                if s.n_points > 10:
                    tube = s.tube(radius=0.008)
                    if tube.n_points > 0:
                        self.plotter.add_mesh(tube, cmap="turbo_r", show_scalar_bar=False)
            except Exception as e:
                print(f"[Streamline {label}] {e}")

        safe_add_streams(
            pv.Plane(
                center=(-tx_p / 2 + 0.1, -ty_p / 4, 0),
                direction=(1, 0, 0),
                i_size=tz_p * 0.8,
                j_size=tx_p * 0.3,
                i_resolution=6,
                j_resolution=15,
            ),
            "sol",
        )
        safe_add_streams(
            pv.Plane(
                center=(-tx_p / 2 + 0.1, ty_p / 4, 0),
                direction=(1, 0, 0),
                i_size=tz_p * 0.8,
                j_size=tx_p * 0.3,
                i_resolution=6,
                j_resolution=15,
            ),
            "sag",
        )
        safe_add_streams(
            pv.Plane(
                center=(-tx_p / 2 + 0.1, 0, tz_p * 0.05),
                direction=(1, 0, 0),
                i_size=tz_p * 0.6,
                j_size=ty_p * 0.5,
                i_resolution=10,
                j_resolution=10,
            ),
            "orta",
        )

        self.plotter.camera_position = [(-tx_p * 0.6, -ty_p * 1.5, tz_p * 0.8), (0, 0, 0), (0, 0, 1)]
        self.plotter.reset_camera()
        self.status_bar.showMessage("Simülasyon tamamlandı. Kuvvet analizi ve akış çizgileri yüklendi.")
        if "lbl_status" in self.w:
            self.w["lbl_status"].setText("Tamamlandı.")

        # Karşılaştırma modu: sonucu kaydet
        self.w["btn_export_csv"].setEnabled(True)
        self.w["btn_export_csv"].setEnabled(True)
        self.w["btn_export_json"].setEnabled(True)
        self.w["btn_export_vtk"].setEnabled(True)
        self.w["btn_batch"].setEnabled(True)
        self.w["btn_compare"].setEnabled(len(self.sim_history) > 1)
        self.w["btn_animate"].setEnabled(True)

        self.sim_history.append(
            {
                "name": f"Sim #{len(self.sim_history) + 1}",
                "forces": forces,
                "params": {
                    "wind": self._last_wind_speed,
                    "resolution": self._last_res_key,
                    "turbulence": self._last_turbulence_model,
                    "wall_fn": use_wall_functions,
                },
            }
        )
        self.comparison_combo.clear()
        for s in self.sim_history:
            self.comparison_combo.addItem(f"{s['name']} — Cd={s['forces']['Cd']:.4f}")
        self.comparison_combo.setVisible(len(self.sim_history) > 1)
        self.lbl_comparison.setVisible(len(self.sim_history) > 1)

        # Batch processing: sonraki simulasyonu baslat
        if hasattr(self, "batch_queue") and self.batch_queue:
            self.batch_results.append(
                {
                    "name": f"Wind {self._current_batch_wind} m/s",
                    "forces": forces,
                    "params": {"wind": self._current_batch_wind},
                }
            )
            self._run_next_batch()

        # Yakınsama grafiğini güncelle
        if HAS_MATPLOTLIB and forces.get("valid", True) and np.isfinite(forces.get("Cd", float("nan"))):
            import time as _time

            elapsed = getattr(self, "_cfd_start_time", _time.time())
            elapsed = _time.time() - elapsed
            self._convergence_times.append(elapsed)
            self._convergence_values.append(forces["Cd"])
            self.conv_ax.clear()
            self.conv_ax.set_facecolor("#1e1e1e")
            self.conv_ax.plot(self._convergence_times, self._convergence_values, "cyan", linewidth=1.5)
            self.conv_ax.set_title("Yakınsama (Cd)", color="white", fontsize=9)
            self.conv_ax.set_xlabel("Süre (s)", color="white", fontsize=7)
            self.conv_ax.set_ylabel("Cd", color="white", fontsize=7)
            self.conv_ax.tick_params(colors="white", labelsize=6)
            for spine in self.conv_ax.spines.values():
                spine.set_color("#444")
            self.conv_figure.tight_layout()
            self.conv_canvas.draw()

    def _on_comparison_select(self, index):
        if index < 0 or index >= len(self.sim_history):
            return
        s = self.sim_history[index]
        f = s["forces"]
        drag_value = "N/A" if not f.get("valid", True) else f"{f['drag_N']:.1f} N"
        if "lbl_drag" in self.w:
            self.w["lbl_drag"].setText(
                f"Sürükleme Kuvveti (Fd): {drag_value}\nDikey Karakteristik: {f['status']}"
            )
        if "lbl_efficiency" in self.w:
            self.w["lbl_efficiency"].setText(f"{f['label']}\nCd={f['Cd']:.4f} | Cl={f['Cl']:.4f}")

    def export_results_csv(self):
        if not self.sim_history:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sonuçları CSV Olarak Kaydet", "aerovision_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Simulasyon",
                    "Ruzgar Hizi",
                    "Cozunurluk",
                    "Turbulans",
                    "Cd",
                    "Cl",
                    "Surukleme (N)",
                    "Kaldirma (N)",
                    "A_ref",
                    "Durum",
                ]
            )
            for s in self.sim_history:
                frc = s["forces"]
                p = s["params"]
                writer.writerow(
                    [
                        s["name"],
                        p["wind"],
                        p["resolution"],
                        p["turbulence"],
                        f"{frc['Cd']:.6f}",
                        f"{frc['Cl']:.6f}",
                        f"{frc['drag_N']:.2f}",
                        f"{frc['lift_N']:.2f}",
                        f"{frc['A_ref']:.4f}",
                        frc["status"],
                    ]
                )
        self.status_bar.showMessage(f"CSV kaydedildi: {path}")

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Preset Kaydet", "aerovision_preset.json", "JSON (*.json)"
        )
        if not path:
            return
        preset = {
            "wind_speed": self.w["spin_wind"].value(),
            "yaw": self.w["spin_yaw"].value(),
            "pitch": self.w["spin_pitch"].value(),
            "roll": self.w["spin_roll"].value(),
            "ox": self.w["spin_ox"].value(),
            "oy": self.w["spin_oy"].value(),
            "oz": self.w["spin_oz"].value(),
            "resolution_index": self.w["combo_res"].currentIndex(),
            "turbulence_index": self.w["combo_turb"].currentIndex(),
            "wall_functions": bool(
                self.w.get("chk_wall_functions") and self.w["chk_wall_functions"].isChecked()
            ),
        }
        Path(path).write_text(json.dumps(preset, indent=2), encoding="utf-8")
        self.status_bar.showMessage(f"Preset kaydedildi: {path}")

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Preset Yükle", "", "JSON (*.json)")
        if not path:
            return
        try:
            preset = json.loads(Path(path).read_text(encoding="utf-8"))
            self.w["spin_wind"].setValue(preset.get("wind_speed", 30))
            self.w["spin_yaw"].setValue(preset.get("yaw", 0))
            self.w["spin_pitch"].setValue(preset.get("pitch", 0))
            self.w["spin_roll"].setValue(preset.get("roll", 0))
            self.w["spin_ox"].setValue(preset.get("ox", 0))
            self.w["spin_oy"].setValue(preset.get("oy", 0))
            self.w["spin_oz"].setValue(preset.get("oz", 0))
            if "resolution_index" in preset and preset["resolution_index"] < self.w["combo_res"].count():
                self.w["combo_res"].setCurrentIndex(preset["resolution_index"])
            if "turbulence_index" in preset and preset["turbulence_index"] < self.w["combo_turb"].count():
                self.w["combo_turb"].setCurrentIndex(preset["turbulence_index"])
            if "wall_functions" in preset and self.w.get("chk_wall_functions"):
                self.w["chk_wall_functions"].setChecked(preset["wall_functions"])
            self.update_preview()
            self.status_bar.showMessage(f"Preset yüklendi: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Preset yüklenemedi: {e}")

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Rapor Goruntusu Kaydet", "AeroVision_CFD_Report.png", "PNG (*.png)"
        )
        if path:
            self.plotter.screenshot(path)
            self.status_bar.showMessage(f"Screenshot saved: {path}")

    def export_results_vtk(self):
        """Mevcut CFD sonuclarini VTK olarak disa aktar."""
        vtk_path = Path(__file__).resolve().parent / "cfd_output_results.vtk"
        if vtk_path.exists():
            path, _ = QFileDialog.getSaveFileName(self, "VTK Disa Aktar", str(vtk_path), "VTK Files (*.vtk)")
            if path:
                import shutil

                shutil.copy2(str(vtk_path), path)
                self.status_bar.showMessage(f"VTK exported: {path}")
        else:
            QMessageBox.information(self, "Bilgi", "Henuz bir simulasyon sonucu bulunmuyor.")

    def export_results_json(self):
        """Sonuclari JSON olarak disa aktar."""
        if not self.sim_history:
            QMessageBox.information(self, "Bilgi", "Aktarilacak sonuc bulunmuyor.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "JSON Disa Aktar", "aerovision_results.json", "JSON (*.json)"
        )
        if not path:
            return
        export_data = []
        for s in self.sim_history:
            entry = {
                "name": s["name"],
                "params": s["params"],
                "forces": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                    for k, v in s["forces"].items()
                },
            }
            export_data.append(entry)
        Path(path).write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_bar.showMessage(f"JSON exported: {path}")

    def run_batch(self):
        """Toplu simulasyon calistirma - farkli parametre kombinasyonlari."""
        if self.verts_base is None:
            QMessageBox.warning(self, "Uyari", "Once bir STL dosyasi yukleyin.")
            return

        # Basit batch: farkli ruzgar hizlari
        wind_speeds = [15, 25, 35, 45]
        reply = QMessageBox.question(
            self,
            "Toplu Simulasyon",
            f"Farkli ruzgar hizlari ile {len(wind_speeds)} simulasyon calistirilacak:\n{wind_speeds}\n\nDevam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.batch_queue = list(wind_speeds)
        self.batch_results = []
        self._run_next_batch()

    def _run_next_batch(self):
        if not self.batch_queue:
            self._show_batch_results()
            return
        wind_speed = self.batch_queue.pop(0)
        self.w["spin_wind"].setValue(wind_speed)
        self.status_bar.showMessage(f"Toplu: {wind_speed} m/s simülasyonu baslatiliyor...")
        self._current_batch_wind = wind_speed

        # Mevcut run_cfd'yi cagir, bitince batch devam edecek
        self.run_cfd()

    def _show_batch_results(self):
        if not self.batch_results:
            QMessageBox.information(self, "Bilgi", "Batch sonucu bulunmuyor.")
            return
        msg = "Toplu Simulasyon Sonuclari:\n\n"
        for r in self.batch_results:
            msg += f"{r['name']}: Cd={r['forces']['Cd']:.4f}, Fd={r['forces']['drag_N']:.1f}N\n"
        QMessageBox.information(self, "Batch Sonuclari", msg)

    def show_comparison_table(self):
        """Karsilastirma tablosu goster."""
        if len(self.sim_history) < 2:
            QMessageBox.information(self, "Bilgi", "Karsilastirma icin en az 2 simulasyon gerekli.")
            return

        # Basit tablo penceresi
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem

        dialog = QDialog(self)
        dialog.setWindowTitle("Simulasyon Karsilastirmasi")
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)

        table = QTableWidget(len(self.sim_history), 6)
        table.setHorizontalHeaderLabels(["Simulasyon", "Cd", "Cl", "Fd (N)", "Fl (N)", "Ruzgar (m/s)"])

        for i, s in enumerate(self.sim_history):
            f = s["forces"]
            table.setItem(i, 0, QTableWidgetItem(s["name"]))
            table.setItem(i, 1, QTableWidgetItem(f"{f['Cd']:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{f['Cl']:.4f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{f['drag_N']:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{f['lift_N']:.2f}"))
            table.setItem(i, 5, QTableWidgetItem(str(s["params"]["wind"])))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        dialog.exec()

    def record_animation(self):
        """Akis cizgileri animasyonu kaydet."""
        self.status_bar.showMessage("Animasyon kaydi baslatildi...")
        # Basit screenshot serisi
        path, _ = QFileDialog.getSaveFileName(self, "Animasyon Kaydet", "animation_frame_", "PNG (*.png)")
        if not path:
            return

        # Mevcut goruntuyu kaydet
        self.plotter.screenshot(path)
        self.status_bar.showMessage(f"Animasyon karesi kaydedildi: {path}")

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    window = AeroVisionWindow()
    window.show()
    sys.exit(app.exec())
