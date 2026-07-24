"""
AeroVision CFD — UI Panel Factory
Professional-grade control panel for CFD simulation parameters.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox,
    QProgressBar, QCheckBox
)

LANGUAGE_OPTIONS = [
    ("Türkçe", "tr"),
    ("English", "en"),
]

RES_MAP = {
    "low":      (50, 18, 18),
    "medium":   (60, 22, 22),
    "high":     (75, 26, 26),
    "ultra":    (120, 45, 45),
    "extreme":  (180, 60, 60),
}

RES_LABELS = {
    "tr": {
        "low":     "Düşük     (50×18×18)",
        "medium":  "Orta      (60×22×22)",
        "high":    "Yüksek    (75×26×26)",
        "ultra":   "Ultra     (120×45×45)",
        "extreme": "Maksimum  (180×60×60)",
    },
    "en": {
        "low":     "Low       (50×18×18)",
        "medium":  "Medium    (60×22×22)",
        "high":    "High      (75×26×26)",
        "ultra":   "Ultra     (120×45×45)",
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
    },
}


def set_ui_language(refs, lang="tr"):
    texts = UI_TEXT.get(lang, UI_TEXT["tr"])
    refs['group_model'].setTitle(texts['model_group'])
    refs['btn_load'].setText(texts['load_button'])
    refs['lbl_file'].setText(texts['file_none'] if refs['lbl_file'].text() == UI_TEXT['tr']['file_none'] or refs['lbl_file'].text() == UI_TEXT['en']['file_none'] else refs['lbl_file'].text())
    refs['group_orientation'].setTitle(texts['orientation_group'])
    refs['group_offset'].setTitle(texts['offset_group'])
    refs['group_simulation'].setTitle(texts['simulation_group'])
    refs['label_wind'].setText(texts['wind_speed'])
    refs['label_res'].setText(texts['resolution'])
    refs['label_turb'].setText(texts['turbulence_label'])
    refs['group_report'].setTitle(texts['report_group'])
    refs['lbl_drag'].setText(texts['drag_label'])
    refs['lbl_efficiency'].setText(texts['efficiency_label'])
    refs['btn_preview'].setText(texts['update_preview'])
    refs['btn_run'].setText(texts['run_simulation'])
    refs['btn_screenshot'].setText(texts['save_report'])
    if 'chk_wall_functions' in refs:
        refs['chk_wall_functions'].setText(texts['wall_function_label'])
    refs['group_language'].setTitle(texts['language_group'])
    refs['label_language'].setText(texts['language_label'])
    
    # Yeni butonlar icin dil degisimi
    if 'btn_export_csv' in refs:
        refs['btn_export_csv'].setText(texts['export_csv'])
    if 'btn_export_json' in refs:
        refs['btn_export_json'].setText(texts['export_json'])
    if 'btn_export_vtk' in refs:
        refs['btn_export_vtk'].setText(texts['export_vtk'])
    if 'btn_preset_save' in refs:
        refs['btn_preset_save'].setText(texts['preset_save'])
    if 'btn_preset_load' in refs:
        refs['btn_preset_load'].setText(texts['preset_load'])
    if 'btn_batch' in refs:
        refs['btn_batch'].setText(texts['batch_sim'])
    if 'btn_compare' in refs:
        refs['btn_compare'].setText(texts['compare_table'])
    if 'btn_animate' in refs:
        refs['btn_animate'].setText(texts['animate_3d'])

    current_res = refs['combo_res'].currentIndex()
    current_turb = refs['combo_turb'].currentIndex()

    refs['combo_res'].clear()
    for key in RES_MAP:
        refs['combo_res'].addItem(RES_LABELS[lang][key])
    if 0 <= current_res < refs['combo_res'].count():
        refs['combo_res'].setCurrentIndex(current_res)

    refs['combo_turb'].clear()
    for label, _ in TURBULENCE_OPTIONS[lang]:
        refs['combo_turb'].addItem(label)
    if 0 <= current_turb < refs['combo_turb'].count():
        refs['combo_turb'].setCurrentIndex(current_turb)
    refs['current_language'] = lang


def get_resolution_key(refs):
    index = refs['combo_res'].currentIndex()
    keys = list(RES_MAP.keys())
    return keys[index] if 0 <= index < len(keys) else 'balanced'


def get_turbulence_model(refs):
    options = TURBULENCE_OPTIONS.get(refs.get('current_language', 'tr'), TURBULENCE_OPTIONS['tr'])
    index = refs['combo_turb'].currentIndex()
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
    Sol kontrol panelini izole inşa eder ve main launcher'a referans sözlüğü (refs) döndürür.
    """
    panel = QWidget()
    panel.setFixedWidth(300)
    layout = QVBoxLayout(panel)
    layout.setSpacing(10)
    refs = {}

    # Dosya grubu
    fg = QGroupBox("Model Yönetimi")
    fl = QVBoxLayout(fg)
    refs['btn_load'] = QPushButton("STL Dosyası Yükle")
    refs['lbl_file'] = QLabel("Dosya seçilmedi")
    refs['lbl_file'].setWordWrap(True)
    fl.addWidget(refs['btn_load'])
    fl.addWidget(refs['lbl_file'])
    layout.addWidget(fg)

    # Rotasyon grubu
    rg = QGroupBox("Model Yönü (Rotasyon)")
    rl = QGridLayout(rg)
    refs['spin_yaw']   = make_angle_spinbox()
    refs['spin_pitch'] = make_angle_spinbox()
    refs['spin_roll']  = make_angle_spinbox()
    for row, (label, key) in enumerate([
        ("Yaw   (Z°):", 'spin_yaw'),
        ("Pitch (Y°):", 'spin_pitch'),
        ("Roll  (X°):", 'spin_roll'),
    ]):
        rl.addWidget(QLabel(label), row, 0)
        rl.addWidget(refs[key], row, 1)
    layout.addWidget(rg)

    # Ofset grubu
    og = QGroupBox("Konum Ofseti (Metre)")
    ol = QGridLayout(og)
    refs['spin_ox'] = make_offset_spinbox()
    refs['spin_oy'] = make_offset_spinbox()
    refs['spin_oz'] = make_offset_spinbox()
    for row, (label, key) in enumerate([
        ("X (İleri-Geri):", 'spin_ox'),
        ("Y (Sağ-Sol):",    'spin_oy'),
        ("Z (Aşağı-Yukarı):", 'spin_oz'),
    ]):
        ol.addWidget(QLabel(label), row, 0)
        ol.addWidget(refs[key], row, 1)
    layout.addWidget(og)

    # Simülasyon grubu
    sg = QGroupBox("Akışkan ve Izgara Ayarı")
    sl = QGridLayout(sg)
    refs['spin_wind'] = make_wind_spinbox()
    refs['combo_res'] = QComboBox()
    refs['label_wind'] = QLabel("Rüzgar Hızı:")
    refs['label_res'] = QLabel("Çözünürlük:")
    refs['label_turb'] = QLabel("Turbulence Model:")
    sl.addWidget(refs['label_wind'], 0, 0); sl.addWidget(refs['spin_wind'], 0, 1)
    sl.addWidget(refs['label_res'],  1, 0); sl.addWidget(refs['combo_res'], 1, 1)
    refs['combo_turb'] = QComboBox()
    sl.addWidget(refs['label_turb'], 2, 0); sl.addWidget(refs['combo_turb'], 2, 1)
    refs['chk_wall_functions'] = QCheckBox("Use Wall Functions")
    refs['chk_wall_functions'].setChecked(False)
    sl.addWidget(refs['chk_wall_functions'], 3, 0, 1, 2)
    layout.addWidget(sg)

    # Dil seçimi grubu
    lg = QGroupBox("Dil Seçimi")
    ll = QGridLayout(lg)
    refs['combo_language'] = QComboBox()
    for label, _ in LANGUAGE_OPTIONS:
        refs['combo_language'].addItem(label)
    refs['label_language'] = QLabel("Arayüz Dili:")
    ll.addWidget(refs['label_language'], 0, 0)
    ll.addWidget(refs['combo_language'], 0, 1)
    layout.addWidget(lg)

    # Rapor grubu
    rpg = QGroupBox("CFD Analiz Özeti")
    rpl = QVBoxLayout(rpg)
    refs['lbl_drag']       = QLabel("Sürükleme Kuvveti (Fd): -- N")
    refs['lbl_efficiency'] = QLabel("Model Durumu: Analiz Bekleniyor")
    rpl.addWidget(refs['lbl_drag'])
    rpl.addWidget(refs['lbl_efficiency'])
    layout.addWidget(rpg)
    refs['report_group'] = rpg
    rpg.setVisible(False)

    # İşlem Butonları
    refs['btn_preview'] = QPushButton("Önizlemeyi Güncelle")
    refs['btn_preview'].setEnabled(False)
    layout.addWidget(refs['btn_preview'])

    refs['btn_run'] = QPushButton("Simülasyonu Başlat")
    refs['btn_run'].setEnabled(False)
    refs['btn_run'].setStyleSheet(
        "QPushButton{background:#1a6b3c;color:white;font-weight:bold;padding:8px;border-radius:4px;}"
        "QPushButton:hover{background:#25a05a;}"
        "QPushButton:disabled{background:#444;color:#888;}"
    )
    layout.addWidget(refs['btn_run'])

    refs['btn_screenshot'] = QPushButton("Rapor Görseli Kaydet")
    refs['btn_screenshot'].setEnabled(False)
    layout.addWidget(refs['btn_screenshot'])

    # Disa aktarma butonlari
    refs['btn_export_csv'] = QPushButton("CSV Disa Aktar")
    refs['btn_export_csv'].setEnabled(False)
    layout.addWidget(refs['btn_export_csv'])

    refs['btn_export_json'] = QPushButton("JSON Disa Aktar")
    refs['btn_export_json'].setEnabled(False)
    layout.addWidget(refs['btn_export_json'])

    refs['btn_export_vtk'] = QPushButton("VTK Disa Aktar")
    refs['btn_export_vtk'].setEnabled(False)
    layout.addWidget(refs['btn_export_vtk'])

    # Preset butonlari
    refs['btn_preset_save'] = QPushButton("Preset Kaydet")
    layout.addWidget(refs['btn_preset_save'])

    refs['btn_preset_load'] = QPushButton("Preset Yukle")
    layout.addWidget(refs['btn_preset_load'])

    # Toplu islem butonu
    refs['btn_batch'] = QPushButton("Toplu Simulasyon (Ctrl+B)")
    refs['btn_batch'].setEnabled(False)
    refs['btn_batch'].setStyleSheet(
        "QPushButton{background:#8B4513;color:white;font-weight:bold;padding:6px;border-radius:4px;}"
        "QPushButton:hover{background:#A0522D;}"
        "QPushButton:disabled{background:#444;color:#888;}"
    )
    layout.addWidget(refs['btn_batch'])

    # Karsilastirma butonu
    refs['btn_compare'] = QPushButton("Karsilastirma Tablosu")
    refs['btn_compare'].setEnabled(False)
    layout.addWidget(refs['btn_compare'])

    # Animasyon butonu
    refs['btn_animate'] = QPushButton("3D Animasyon Kaydet")
    refs['btn_animate'].setEnabled(False)
    refs['btn_animate'].setStyleSheet(
        "QPushButton{background:#4a0080;color:white;font-weight:bold;padding:6px;border-radius:4px;}"
        "QPushButton:hover{background:#6a2db8;}"
        "QPushButton:disabled{background:#444;color:#888;}"
    )
    layout.addWidget(refs['btn_animate'])

    refs['progress_bar'] = QProgressBar()
    refs['progress_bar'].setRange(0, 0)
    refs['progress_bar'].setVisible(False)
    refs['lbl_status'] = QLabel("")
    refs['lbl_status'].setWordWrap(True)
    layout.addWidget(refs['progress_bar'])
    layout.addWidget(refs['lbl_status'])

    refs['group_model'] = fg
    refs['group_orientation'] = rg
    refs['group_offset'] = og
    refs['group_simulation'] = sg
    refs['group_language'] = lg
    refs['group_report'] = rpg

    refs['combo_language'].currentIndexChanged.connect(
        lambda index: set_ui_language(refs, LANGUAGE_OPTIONS[index][1])
    )
    set_ui_language(refs, "tr")

    layout.addStretch()
    return panel, refs