"""
AeroVision CFD — UI Panel Factory
Professional-grade control panel for CFD simulation parameters.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QProgressBar,
    QCheckBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt

LANGUAGE_OPTIONS = [
    ("Türkçe", "tr"),
    ("English", "en"),
]

RES_MAP = {
    "low": (50, 18, 18),
    "medium": (60, 22, 22),
    "high": (75, 26, 26),
    "ultra": (120, 45, 45),
    "extreme": (180, 60, 60),
}

RES_LABELS = {
    "tr": {
        "low": "Düşük     (50×18×18)",
        "medium": "Orta      (60×22×22)",
        "high": "Yüksek    (75×26×26)",
        "ultra": "Ultra     (120×45×45)",
        "extreme": "Maksimum  (180×60×60)",
    },
    "en": {
        "low": "Low       (50×18×18)",
        "medium": "Medium    (60×22×22)",
        "high": "High      (75×26×26)",
        "ultra": "Ultra     (120×45×45)",
        "extreme": "Maximum   (180×60×60)",
    },
}

TURBULENCE_OPTIONS = {
    "tr": [
        ("Smagorinsky", "smagorinsky"),
        ("Spalart-Allmaras", "spalart_allmaras"),
        ("k-epsilon", "k_epsilon"),
        ("k-omega SST", "k_omega_sst"),
    ],
    "en": [
        ("Smagorinsky", "smagorinsky"),
        ("Spalart-Allmaras", "spalart_allmaras"),
        ("k-epsilon", "k_epsilon"),
        ("k-omega SST", "k_omega_sst"),
    ],
}

FLUID_OPTIONS = {
    "tr": [
        ("Hava (20°C)", "air"),
        ("Su (20°C)", "water"),
        ("CO2 (20°C)", "co2"),
    ],
    "en": [
        ("Air (20°C)", "air"),
        ("Water (20°C)", "water"),
        ("CO2 (20°C)", "co2"),
    ],
}

UI_TEXT = {
    "tr": {
        "model_group": "Model Yönetimi",
        "load_button": "STL Dosyası Yükle",
        "file_none": "Dosya seçilmedi",
        "orientation_group": "Model Yönü (Rotasyon)",
        "offset_group": "Konum Ofseti (Metre)",
        "simulation_group": "Akışkan ve Izgara Ayarı",
        "wind_speed": "Rüzgar Hızı:",
        "resolution": "Çözünürlük:",
        "turbulence_label": "Turbulans Modeli:",
        "fluid_label": "Akışkan:",
        "report_group": "CFD Analiz Özeti",
        "drag_label": "Sürükleme Kuvveti (Fd): -- N",
        "efficiency_label": "Model Durumu: Analiz Bekleniyor",
        "update_preview": "Önizlemeyi Güncelle",
        "run_simulation": "Simülasyonu Başlat",
        "save_report": "Rapor Görseli Kaydet",
        "wall_function_label": "Duvar Fonksiyonu Kullan",
        "language_group": "Dil Seçimi",
        "language_label": "Arayüz Dili:",
        "export_csv": "CSV Disa Aktar",
        "export_json": "JSON Disa Aktar",
        "export_vtk": "VTK Disa Aktar",
        "preset_save": "Preset Kaydet",
        "preset_load": "Preset Yukle",
        "batch_sim": "Toplu Simulasyon (Ctrl+B)",
        "compare_table": "Karsilastirma Tablosu",
        "animate_3d": "3D Animasyon Kaydet",
        "gpu_label": "Hesaplama Birimi:",
        "gpu_auto": "Otomatik",
        "gpu_cpu": "CPU",
        "gpu_nvidia": "NVIDIA GPU",
        "gpu_amd": "AMD GPU",
        "gpu_intel": "Intel GPU",
    },
    "en": {
        "model_group": "Model Management",
        "load_button": "Load STL File",
        "file_none": "No file selected",
        "orientation_group": "Model Orientation (Rotation)",
        "offset_group": "Position Offset (Meters)",
        "simulation_group": "Fluid and Grid Settings",
        "wind_speed": "Wind Speed:",
        "resolution": "Resolution:",
        "turbulence_label": "Turbulence Model:",
        "fluid_label": "Fluid:",
        "report_group": "CFD Analysis Summary",
        "drag_label": "Drag Force (Fd): -- N",
        "efficiency_label": "Model Status: Awaiting Analysis",
        "update_preview": "Update Preview",
        "run_simulation": "Start Simulation",
        "save_report": "Save Report Image",
        "wall_function_label": "Use Wall Functions",
        "language_group": "Language Selection",
        "language_label": "Interface Language:",
        "export_csv": "Export CSV",
        "export_json": "Export JSON",
        "export_vtk": "Export VTK",
        "preset_save": "Save Preset",
        "preset_load": "Load Preset",
        "batch_sim": "Batch Simulation (Ctrl+B)",
        "compare_table": "Comparison Table",
        "animate_3d": "Record 3D Animation",
        "gpu_label": "Compute Unit:",
        "gpu_auto": "Auto",
        "gpu_cpu": "CPU",
        "gpu_nvidia": "NVIDIA GPU",
        "gpu_amd": "AMD GPU",
        "gpu_intel": "Intel GPU",
    },
}


def _detect_available_gpus():
    """Mevcut GPU'lari algila ve (vendor, label) listesi dondur."""
    gpus = []

    # 1. cupy ile NVIDIA kontrol
    try:
        import cupy as cp
        import os

        if "CUDA_PATH" not in os.environ:
            os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
        cp.cuda.Device(0).use()
        _ = cp.array([1.0]) * 2.0
        gpus.append(("nvidia", "NVIDIA GPU"))
    except Exception:
        pass

    # 2. OpenCL ile Intel/AMD kontrol
    try:
        import pyopencl as cl

        for platform in cl.get_platforms():
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            except cl.RuntimeError:
                continue
            for dev in devices:
                if "NVIDIA" in dev.vendor:
                    if "nvidia" not in [g[0] for g in gpus]:
                        gpus.append(("nvidia", f"NVIDIA GPU ({dev.name})"))
                elif "Intel" in dev.vendor:
                    gpus.append(("intel", f"Intel GPU ({dev.name})"))
                elif "AMD" in dev.vendor or "Radeon" in dev.name:
                    gpus.append(("amd", f"AMD GPU ({dev.name})"))
    except Exception:
        pass

    return gpus


_language_callback = None


def set_language_callback(callback):
    """Dil degisiminde cagrilacak callback fonksiyonunu ayarla."""
    global _language_callback
    _language_callback = callback


def set_ui_language(refs, lang="tr"):
    texts = UI_TEXT.get(lang, UI_TEXT["tr"])
    refs["group_model"].setTitle(texts["model_group"])
    refs["btn_load"].setText(texts["load_button"])
    # Yüklü dosya metni korunur; yüklenmemişse varsayılan metin gösterilir
    refs["lbl_file"].setText(refs.get("_loaded_file_text") or texts["file_none"])
    refs["group_orientation"].setTitle(texts["orientation_group"])
    refs["group_offset"].setTitle(texts["offset_group"])
    refs["group_simulation"].setTitle(texts["simulation_group"])
    refs["label_wind"].setText(texts["wind_speed"])
    refs["label_res"].setText(texts["resolution"])
    refs["label_turb"].setText(texts["turbulence_label"])
    refs["group_report"].setTitle(texts["report_group"])
    refs["lbl_drag"].setText(texts["drag_label"])
    refs["lbl_efficiency"].setText(texts["efficiency_label"])
    refs["btn_preview"].setText(texts["update_preview"])
    refs["btn_run"].setText(texts["run_simulation"])
    refs["btn_screenshot"].setText(texts["save_report"])
    if "chk_wall_functions" in refs:
        refs["chk_wall_functions"].setText(texts["wall_function_label"])
    refs["group_language"].setTitle(texts["language_group"])
    refs["label_language"].setText(texts["language_label"])

    # Yeni butonlar icin dil degisimi
    if "btn_export_csv" in refs:
        refs["btn_export_csv"].setText(texts["export_csv"])
    if "btn_export_json" in refs:
        refs["btn_export_json"].setText(texts["export_json"])
    if "btn_export_vtk" in refs:
        refs["btn_export_vtk"].setText(texts["export_vtk"])
    if "btn_preset_save" in refs:
        refs["btn_preset_save"].setText(texts["preset_save"])
    if "btn_preset_load" in refs:
        refs["btn_preset_load"].setText(texts["preset_load"])
    if "btn_batch" in refs:
        refs["btn_batch"].setText(texts["batch_sim"])
    if "btn_compare" in refs:
        refs["btn_compare"].setText(texts["compare_table"])
    if "btn_animate" in refs:
        refs["btn_animate"].setText(texts["animate_3d"])
    if "label_gpu" in refs:
        refs["label_gpu"].setText(texts["gpu_label"])
    if "label_fluid" in refs:
        refs["label_fluid"].setText(texts["fluid_label"])

    current_res = refs["combo_res"].currentIndex()
    current_turb = refs["combo_turb"].currentIndex()
    current_fluid = refs["combo_fluid"].currentData() if "combo_fluid" in refs else None

    refs["combo_res"].clear()
    for key in RES_MAP:
        refs["combo_res"].addItem(RES_LABELS[lang][key])
    if 0 <= current_res < refs["combo_res"].count():
        refs["combo_res"].setCurrentIndex(current_res)

    refs["combo_turb"].clear()
    for label, _ in TURBULENCE_OPTIONS[lang]:
        refs["combo_turb"].addItem(label)
    if 0 <= current_turb < refs["combo_turb"].count():
        refs["combo_turb"].setCurrentIndex(current_turb)

    if "combo_fluid" in refs:
        refs["combo_fluid"].clear()
        for label, key in FLUID_OPTIONS[lang]:
            refs["combo_fluid"].addItem(label, key)
        if current_fluid is not None:
            idx = refs["combo_fluid"].findData(current_fluid)
            if idx >= 0:
                refs["combo_fluid"].setCurrentIndex(idx)
    refs["current_language"] = lang
    if _language_callback is not None:
        _language_callback(lang)


def get_resolution_key(refs):
    index = refs["combo_res"].currentIndex()
    keys = list(RES_MAP.keys())
    return keys[index] if 0 <= index < len(keys) else "balanced"


def get_turbulence_model(refs):
    options = TURBULENCE_OPTIONS.get(refs.get("current_language", "tr"), TURBULENCE_OPTIONS["tr"])
    index = refs["combo_turb"].currentIndex()
    if 0 <= index < len(options):
        return options[index][1]
    return options[0][1]


def make_angle_spinbox():
    sb = QDoubleSpinBox()
    # ── 🌟 KİLİT KIRILDI: Minimum ve maksimum açıkça setlendi, 3 hane serbest! ──
    sb.setMinimum(-180.0)
    sb.setMaximum(180.0)
    sb.setValue(0.0)
    sb.setSuffix("°")
    sb.setSingleStep(5.0)
    sb.setDecimals(1)
    return sb


def make_offset_spinbox():
    sb = QDoubleSpinBox()
    sb.setRange(-5, 5)
    sb.setValue(0)
    sb.setSuffix(" m")
    sb.setSingleStep(0.1)
    return sb


def make_wind_spinbox():
    sb = QDoubleSpinBox()
    sb.setRange(5, 150)
    sb.setValue(30)
    sb.setSuffix(" m/s")
    return sb


def build_left_panel():
    """
    Sol kontrol panelini izole insa eder ve main launcher'a referans sozlugu (refs) dondurur.
    Kaydirilabilir ve yeniden boyutlandirilabilir panel.
    """

    # Kaydirilabilir panel
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_area.setMinimumWidth(240)
    scroll_area.setMaximumWidth(450)

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setSpacing(8)
    refs = {}

    # Dosya grubu - kompakt
    fg = QGroupBox("Model")
    fl = QVBoxLayout(fg)
    fl.setSpacing(4)
    refs["btn_load"] = QPushButton("STL Yukle")
    refs["lbl_file"] = QLabel("Dosya secilmedi")
    refs["lbl_file"].setWordWrap(True)
    refs["lbl_file"].setStyleSheet("font-size: 10px;")
    fl.addWidget(refs["btn_load"])
    fl.addWidget(refs["lbl_file"])
    layout.addWidget(fg)

    # Rotasyon grubu - kompakt
    rg = QGroupBox("Yon (Derece)")
    rl = QGridLayout(rg)
    rl.setSpacing(4)
    refs["spin_yaw"] = make_angle_spinbox()
    refs["spin_pitch"] = make_angle_spinbox()
    refs["spin_roll"] = make_angle_spinbox()
    for row, (label, key) in enumerate(
        [
            ("Yaw:", "spin_yaw"),
            ("Pitch:", "spin_pitch"),
            ("Roll:", "spin_roll"),
        ]
    ):
        rl.addWidget(QLabel(label), row, 0)
        rl.addWidget(refs[key], row, 1)
    layout.addWidget(rg)

    # Ofset grubu - kompakt
    og = QGroupBox("Ofset (m)")
    ol = QGridLayout(og)
    ol.setSpacing(4)
    refs["spin_ox"] = make_offset_spinbox()
    refs["spin_oy"] = make_offset_spinbox()
    refs["spin_oz"] = make_offset_spinbox()
    for row, (label, key) in enumerate(
        [
            ("X:", "spin_ox"),
            ("Y:", "spin_oy"),
            ("Z:", "spin_oz"),
        ]
    ):
        ol.addWidget(QLabel(label), row, 0)
        ol.addWidget(refs[key], row, 1)
    layout.addWidget(og)

    # Simulasyon grubu - kompakt
    sg = QGroupBox("Simulasyon")
    sl = QGridLayout(sg)
    sl.setSpacing(4)
    refs["spin_wind"] = make_wind_spinbox()
    refs["combo_res"] = QComboBox()
    refs["label_wind"] = QLabel("Ruzgar:")
    refs["label_res"] = QLabel("Cozunurluk:")
    refs["label_turb"] = QLabel("Turbulans:")
    sl.addWidget(refs["label_wind"], 0, 0)
    sl.addWidget(refs["spin_wind"], 0, 1)
    sl.addWidget(refs["label_res"], 1, 0)
    sl.addWidget(refs["combo_res"], 1, 1)
    refs["combo_turb"] = QComboBox()
    sl.addWidget(refs["label_turb"], 2, 0)
    sl.addWidget(refs["combo_turb"], 2, 1)
    refs["chk_wall_functions"] = QCheckBox("Wall Functions")
    refs["chk_wall_functions"].setChecked(False)
    sl.addWidget(refs["chk_wall_functions"], 3, 0, 1, 2)

    # GPU secimi — sadece algilanan cihazlari goster
    refs["label_gpu"] = QLabel("Hesaplama:")
    refs["combo_gpu"] = QComboBox()
    refs["combo_gpu"].addItem("Otomatik", "auto")
    refs["combo_gpu"].addItem("CPU", "cpu")
    # Mevcut GPU'lari algila
    _detected_gpus = _detect_available_gpus()
    for vendor, label in _detected_gpus:
        refs["combo_gpu"].addItem(label, vendor)
    sl.addWidget(refs["label_gpu"], 4, 0)
    sl.addWidget(refs["combo_gpu"], 4, 1)

    # Akışkan secimi
    refs["label_fluid"] = QLabel("Akışkan:")
    refs["combo_fluid"] = QComboBox()
    for label, key in FLUID_OPTIONS["tr"]:
        refs["combo_fluid"].addItem(label, key)
    sl.addWidget(refs["label_fluid"], 5, 0)
    sl.addWidget(refs["combo_fluid"], 5, 1)
    layout.addWidget(sg)

    # Dil secimi - kompakt
    lg = QGroupBox("Dil")
    ll = QGridLayout(lg)
    ll.setSpacing(4)
    refs["combo_language"] = QComboBox()
    for label, _ in LANGUAGE_OPTIONS:
        refs["combo_language"].addItem(label)
    refs["label_language"] = QLabel("Arayuz:")
    ll.addWidget(refs["label_language"], 0, 0)
    ll.addWidget(refs["combo_language"], 0, 1)
    layout.addWidget(lg)

    # Rapor grubu
    rpg = QGroupBox("Sonuclar")
    rpl = QVBoxLayout(rpg)
    rpl.setSpacing(4)
    refs["lbl_drag"] = QLabel("Surukleme: -- N")
    refs["lbl_efficiency"] = QLabel("Durum: Bekleniyor")
    rpl.addWidget(refs["lbl_drag"])
    rpl.addWidget(refs["lbl_efficiency"])
    layout.addWidget(rpg)
    refs["report_group"] = rpg
    rpg.setVisible(False)

    # Ana butonlar - kompakt
    refs["btn_preview"] = QPushButton("Onizleme Guncelle")
    refs["btn_preview"].setEnabled(False)
    layout.addWidget(refs["btn_preview"])

    refs["btn_run"] = QPushButton("Simulasyonu Baslat")
    refs["btn_run"].setEnabled(False)
    refs["btn_run"].setFixedHeight(32)
    refs["btn_run"].setStyleSheet(
        "QPushButton{background:#1a6b3c;color:white;font-weight:bold;padding:4px;border-radius:4px;}"
        "QPushButton:hover{background:#25a05a;}"
        "QPushButton:disabled{background:#444;color:#888;}"
    )
    layout.addWidget(refs["btn_run"])

    refs["btn_screenshot"] = QPushButton("Gorsel Kaydet")
    refs["btn_screenshot"].setEnabled(False)
    layout.addWidget(refs["btn_screenshot"])

    # Disa aktarma - tek satir
    export_layout = QGridLayout()
    export_layout.setSpacing(4)
    refs["btn_export_csv"] = QPushButton("CSV")
    refs["btn_export_csv"].setEnabled(False)
    refs["btn_export_json"] = QPushButton("JSON")
    refs["btn_export_json"].setEnabled(False)
    refs["btn_export_vtk"] = QPushButton("VTK")
    refs["btn_export_vtk"].setEnabled(False)
    export_layout.addWidget(refs["btn_export_csv"], 0, 0)
    export_layout.addWidget(refs["btn_export_json"], 0, 1)
    export_layout.addWidget(refs["btn_export_vtk"], 0, 2)
    layout.addLayout(export_layout)

    # Preset - tek satir
    preset_layout = QGridLayout()
    preset_layout.setSpacing(4)
    refs["btn_preset_save"] = QPushButton("Kaydet")
    refs["btn_preset_load"] = QPushButton("Yukle")
    preset_layout.addWidget(refs["btn_preset_save"], 0, 0)
    preset_layout.addWidget(refs["btn_preset_load"], 0, 1)
    layout.addLayout(preset_layout)

    # Alt butonlar - tek satir
    refs["btn_batch"] = QPushButton("Toplu")
    refs["btn_batch"].setEnabled(False)
    refs["btn_batch"].setStyleSheet("background:#8B4513;color:white;")

    refs["btn_compare"] = QPushButton("Karsilastir")
    refs["btn_compare"].setEnabled(False)

    refs["btn_animate"] = QPushButton("Animasyon")
    refs["btn_animate"].setEnabled(False)
    refs["btn_animate"].setStyleSheet("background:#4a0080;color:white;")

    bottom_layout = QGridLayout()
    bottom_layout.setSpacing(4)
    bottom_layout.addWidget(refs["btn_batch"], 0, 0)
    bottom_layout.addWidget(refs["btn_compare"], 0, 1)
    bottom_layout.addWidget(refs["btn_animate"], 0, 2)
    layout.addLayout(bottom_layout)

    refs["progress_bar"] = QProgressBar()
    refs["progress_bar"].setRange(0, 0)
    refs["progress_bar"].setVisible(False)
    refs["progress_bar"].setFixedHeight(16)
    layout.addWidget(refs["progress_bar"])

    refs["lbl_status"] = QLabel("")
    refs["lbl_status"].setWordWrap(True)
    refs["lbl_status"].setStyleSheet("font-size: 10px;")
    layout.addWidget(refs["lbl_status"])

    refs["group_model"] = fg
    refs["group_orientation"] = rg
    refs["group_offset"] = og
    refs["group_simulation"] = sg
    refs["group_language"] = lg
    refs["group_report"] = rpg
    refs["_loaded_file_text"] = None

    refs["combo_language"].currentIndexChanged.connect(
        lambda index: set_ui_language(refs, LANGUAGE_OPTIONS[index][1])
    )
    set_ui_language(refs, "tr")

    layout.addStretch()

    scroll_area.setWidget(panel)
    return scroll_area, refs
