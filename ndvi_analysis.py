"""
Forest NDVI Monitor — мониторинг лесного покрова по картам NDVI (GeoTIFF), PyQt6.

Два режима:
  1. Состояние леса по одному снимку: здоровый лес, ослабленный древостой,
     молодняк, вырубки / гари / открытые участки.
  2. Динамика по двум датам («сравнить с более ранним»): потеря лесного покрова
     (вырубки, гари, ветровал) и лесовосстановление.

Установка:   pip install PyQt6 matplotlib rasterio numpy
Запуск:      python forest_ndvi_gui.py
"""
import os
import sys

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# ======================================================================
# НАСТРОЙКИ (пороги эвристические — калибруйте под регион, сезон и породы)
# ======================================================================
PIXEL_AREA_HA = 0.01          # запасное значение: Sentinel-2, пиксель 10×10 м = 0.01 га
SOURCE_LABEL = "Sentinel-2"   # подпись источника в итоговом тексте

# Классы состояния лесного покрова (по одному снимку)
T_OPEN = 0.30    # ниже — вырубки, гари, открытые участки
T_YOUNG = 0.50   # ниже — молодняк, кустарник, редины
T_WEAK = 0.65    # ниже — ослабленный древостой, выше — здоровый сомкнутый лес
FOREST_MIN = T_YOUNG   # NDVI ≥ этого значения считаем «покрытой лесом» площадью

# Классы динамики (Δ NDVI = поздний − ранний)
D_STRONG = 0.30    # |Δ| больше — резкое изменение (вырубка / гарь / восстановление)
D_MODERATE = 0.10  # |Δ| больше — заметное изменение
RECOVERY_GAIN = 0.20  # рост NDVI на ранее нелесной площади = вероятное лесовосстановление

# Цвета интерфейса
BG = "#0b111a"
CARD = "#131c2b"
CARD_ALT = "#0f1623"
BORDER = "#22304a"
TEXT = "#e6edf7"
MUTED = "#8b9bb4"
ACCENT = "#34d399"
ACCENT_HOVER = "#6ee7b7"
WARN = "#f59e0b"
RED = "#f87171"

# Цвета классов (порядок совпадает с get_state_classes / get_change_classes)
STATE_COLORS = ["#3b82f6", "#b45309", "#86efac", "#f59e0b", "#15803d"]
CHANGE_COLORS = ["#dc2626", "#f97316", "#475569", "#86efac", "#15803d"]


# ======================================================================
# ЯДРО: загрузка и анализ данных (без интерфейса)
# ======================================================================
def load_ndvi(path):
    """Читает первый канал GeoTIFF. При проблеме бросает ValueError с понятным текстом."""
    if not os.path.isfile(path):
        raise ValueError("Файл не найден.")
    if not path.lower().endswith((".tif", ".tiff")):
        raise ValueError("Нужен файл формата GeoTIFF (.tif или .tiff).")

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("В файле нет геопривязки (проекции), это не GeoTIFF.")
            ndvi = src.read(1, masked=True).astype("float32").filled(np.nan)
            info = {"shape": src.shape, "crs": src.crs, "res": src.res,
                    "transform": src.transform}
    except RasterioIOError as e:
        raise ValueError(f"Не удалось открыть файл как растровое изображение:\n{e}")

    if np.isnan(ndvi).all():
        raise ValueError("В файле нет валидных пикселей (всё nodata).")
    return ndvi, info


def normalize_ndvi(ndvi):
    """Если значения упакованы в 0–255, переводит их в шкалу −1…+1."""
    raw_min, raw_max = np.nanmin(ndvi), np.nanmax(ndvi)
    was_packed = raw_max > 2
    if was_packed:
        ndvi = (ndvi / 255.0) * 2.0 - 1.0
    return ndvi, raw_min, raw_max, was_packed


def pixel_area_ha(crs, res):
    """
    Площадь пикселя в гектарах. Считаем по разрешению файла, если проекция метрическая
    (например, UTM). Для градусов и Web Mercator (искажает площади) берём запасное значение.
    """
    try:
        metric = (crs.is_projected
                  and str(crs.linear_units).lower() in ("metre", "meter")
                  and crs.to_epsg() != 3857)
    except Exception:
        metric = False
    if metric:
        return abs(res[0] * res[1]) / 10_000.0, "по разрешению файла"
    return PIXEL_AREA_HA, "константа PIXEL_AREA_HA"


def compute_stats(ndvi):
    return {
        "min": np.nanmin(ndvi),
        "max": np.nanmax(ndvi),
        "mean": np.nanmean(ndvi),
        "median": np.nanmedian(ndvi),
        "std": np.nanstd(ndvi),
        "valid": int(np.sum(~np.isnan(ndvi))),
    }


def describe_heterogeneity(std):
    if std > 0.25:
        return "лесной покров ОЧЕНЬ неоднородный (много нарушенных и нелесных участков)"
    if std > 0.15:
        return "лесной покров УМЕРЕННО неоднородный"
    return "лесной покров ДОВОЛЬНО ОДНОРОДНЫЙ"


def get_state_classes(ndvi):
    """Классы состояния лесного покрова: (название, маска, пояснение)."""
    return [
        ("Вода / снег / облака",           ndvi < 0,                          "холодная или тёмная поверхность"),
        ("Вырубки, гари, открытые участки", (ndvi >= 0) & (ndvi < T_OPEN),    "нет сомкнутого полога"),
        ("Молодняк, кустарник, редины",    (ndvi >= T_OPEN) & (ndvi < T_YOUNG), "возобновление, редкостойный лес"),
        ("Ослабленный древостой",          (ndvi >= T_YOUNG) & (ndvi < T_WEAK), "усыхание, вредители, засуха?"),
        ("Здоровый сомкнутый лес",         ndvi >= T_WEAK,                    "максимум зелёной биомассы"),
    ]


def get_change_classes(d):
    """Классы динамики Δ NDVI: (название, маска, пояснение)."""
    return [
        ("Резкое снижение",   d < -D_STRONG,                       "вырубка, гарь, ветровал"),
        ("Умеренное снижение", (d >= -D_STRONG) & (d < -D_MODERATE), "ослабление, повреждение, усыхание"),
        ("Без изменений",     (d >= -D_MODERATE) & (d <= D_MODERATE), "стабильное состояние"),
        ("Умеренный рост",    (d > D_MODERATE) & (d <= D_STRONG),   "улучшение состояния, зарастание"),
        ("Сильный рост",      d > D_STRONG,                        "лесовосстановление, активный рост"),
    ]


def compute_class_areas(classes, total_valid, px_ha):
    results = []
    for name, mask, explanation in classes:
        count = int(np.sum(mask))
        results.append({
            "name": name,
            "explanation": explanation,
            "area": count * px_ha,
            "pct": count / total_valid * 100 if total_valid > 0 else 0,
        })
    return results


def align_to(ref, other):
    """
    Приводит NDVI другого снимка к сетке опорного (проекция, размер, привязка).
    Возвращает (массив, был_ли_пересчёт).
    """
    ri, oi = ref["info"], other["info"]
    same = (ri["shape"] == oi["shape"] and ri["transform"] == oi["transform"]
            and ri["crs"] == oi["crs"])
    if same:
        return other["ndvi"], False

    dst = np.full(ref["ndvi"].shape, np.nan, dtype="float32")
    reproject(
        source=other["ndvi"], destination=dst,
        src_transform=oi["transform"], src_crs=oi["crs"],
        dst_transform=ri["transform"], dst_crs=ri["crs"],
        resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst, True


def analyze(path):
    """Полный анализ одного снимка."""
    ndvi, info = load_ndvi(path)
    ndvi, raw_min, raw_max, was_packed = normalize_ndvi(ndvi)
    stats = compute_stats(ndvi)
    px_ha, px_source = pixel_area_ha(info["crs"], info["res"])
    total_area = stats["valid"] * px_ha
    classes = compute_class_areas(get_state_classes(ndvi), stats["valid"], px_ha)

    # индексы классов: 0 вода, 1 вырубки/гари, 2 молодняк, 3 ослабленный, 4 здоровый
    forest = {
        "forest_area": classes[3]["area"] + classes[4]["area"],
        "forest_pct": classes[3]["pct"] + classes[4]["pct"],
        "weak_area": classes[3]["area"],
        "open_area": classes[1]["area"],
    }
    return {
        "path": path, "name": os.path.basename(path), "info": info, "ndvi": ndvi,
        "raw_min": raw_min, "raw_max": raw_max, "was_packed": was_packed,
        "stats": stats, "px_ha": px_ha, "px_source": px_source,
        "total_area": total_area, "class_areas": classes, "forest": forest,
    }


def analyze_change(cur, prev):
    """Динамика: cur — поздний снимок (опорная сетка), prev — более ранний."""
    prev_ndvi, reprojected = align_to(cur, prev)
    d = cur["ndvi"] - prev_ndvi
    valid = int(np.sum(~np.isnan(d)))
    if valid == 0:
        raise ValueError("Снимки не пересекаются: в общей области нет валидных пикселей.")

    px_ha = cur["px_ha"]
    loss = (prev_ndvi >= FOREST_MIN) & (d < -D_STRONG)         # был лес → резкое падение
    recovery = (prev_ndvi < FOREST_MIN) & (d > RECOVERY_GAIN)  # не лес → заметный рост
    return {
        "prev_name": prev["name"],
        "d": d,
        "reprojected": reprojected,
        "valid": valid,
        "overlap_pct": valid / cur["stats"]["valid"] * 100,
        "mean": float(np.nanmean(d)),
        "classes": compute_class_areas(get_change_classes(d), valid, px_ha),
        "loss_area": float(np.sum(loss)) * px_ha,
        "recovery_area": float(np.sum(recovery)) * px_ha,
    }


def build_summary(r, ch=None):
    st, fr = r["stats"], r["forest"]
    ca = r["class_areas"]
    text = (
        f"По данным NDVI ({SOURCE_LABEL}):\n\n"
        f"• Проанализировано {r['total_area']:,.0f} га территории.\n"
        f"• Средний NDVI = {st['mean']:.2f}, медиана = {st['median']:.2f}.\n"
        f"• {ca[4]['pct']:.1f}% — здоровый сомкнутый лес.\n"
        f"• {ca[3]['pct']:.1f}% ({ca[3]['area']:,.1f} га) — ослабленный древостой, "
        f"приоритет лесопатологического обследования.\n"
        f"• {ca[1]['pct']:.1f}% ({ca[1]['area']:,.1f} га) — вырубки, гари и открытые "
        f"участки, требующие проверки на местности."
    )
    if ch is not None:
        text += (
            f"\n\nДинамика относительно более раннего снимка:\n"
            f"• {ch['loss_area']:,.1f} га — вероятная потеря лесного покрова "
            f"(вырубки, гари, ветровал);\n"
            f"• {ch['recovery_area']:,.1f} га — вероятное лесовосстановление;\n"
            f"• средняя Δ NDVI = {ch['mean']:+.3f}."
        )
    text += (
        "\n\nРекомендация: регулярный NDVI-мониторинг по Sentinel-2 (каждые 5 дней "
        "в вегетационный сезон) для раннего выявления вырубок, очагов усыхания "
        "и контроля лесовосстановления."
    )
    return text


# ======================================================================
# КАРТЫ (matplotlib в тёмной теме)
# ======================================================================
def style_axes(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Пиксели по X", color=MUTED, fontsize=9)
    ax.set_ylabel("Пиксели по Y", color=MUTED, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def draw_colorbar(fig, ax, im, label):
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label, color=TEXT)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)


def build_classified_map(data, classes):
    """0 = нет данных, 1…N = номер класса."""
    out = np.zeros_like(data)
    for i, (_, mask, _) in enumerate(classes, start=1):
        out[mask] = i
    return out


def draw_classified(fig, data, classes, colors, title):
    ax = fig.add_subplot(111)
    cmap = ListedColormap([CARD] + colors)
    ax.imshow(build_classified_map(data, classes), cmap=cmap,
              vmin=0, vmax=len(colors), interpolation="nearest")
    style_axes(ax, title)
    handles = [Patch(facecolor=c, label=name) for c, (name, _, _) in zip(colors, classes)]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              facecolor=CARD_ALT, edgecolor=BORDER, labelcolor=TEXT)


def draw_ndvi_map(fig, ndvi):
    ax = fig.add_subplot(111)
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    style_axes(ax, "Карта NDVI (красный = вода/вырубки, зелёный = здоровый лес)")
    draw_colorbar(fig, ax, im, "NDVI")


def draw_state_map(fig, ndvi):
    draw_classified(fig, ndvi, get_state_classes(ndvi), STATE_COLORS,
                    "Состояние лесного покрова по NDVI")


def draw_delta_map(fig, d):
    ax = fig.add_subplot(111)
    im = ax.imshow(d, cmap="RdYlGn", vmin=-0.6, vmax=0.6)
    style_axes(ax, "Изменение NDVI (красный = потеря, зелёный = рост)")
    draw_colorbar(fig, ax, im, "Δ NDVI")


def draw_change_map(fig, d):
    draw_classified(fig, d, get_change_classes(d), CHANGE_COLORS,
                    "Классы изменений (вырубки, гари, восстановление)")


_open_windows = []  # держим ссылки, иначе окна удалит сборщик мусора


class MapWindow(QMainWindow):
    """Карта в отдельном окне: масштаб, перемещение, сохранение картинки."""

    def __init__(self, title, draw_fn, data):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1000, 760)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        fig = Figure(figsize=(8, 6.5), layout="constrained", facecolor=CARD)
        draw_fn(fig, data)
        canvas = FigureCanvasQTAgg(fig)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        lay.addWidget(NavigationToolbar2QT(canvas, self))
        lay.addWidget(canvas, 1)
        self.setCentralWidget(central)

        _open_windows.append(self)

    def closeEvent(self, event):
        if self in _open_windows:
            _open_windows.remove(self)
        super().closeEvent(event)


# ======================================================================
# ВИДЖЕТЫ-КАРТОЧКИ
# ======================================================================
def make_card(title=None):
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(10)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        lay.addWidget(label)
    return card, lay


def primary_button(text):
    btn = QPushButton(text)
    btn.setObjectName("primary")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def ghost_button(text):
    btn = QPushButton(text)
    btn.setObjectName("ghost")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def note_label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("muted")
    return label


class StatBox(QFrame):
    """Маленькая плитка: название, крупное значение, подсказка."""

    def __init__(self, title, value, hint, color=TEXT, value_px=24):
        super().__init__()
        self.setObjectName("stat")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)

        t = QLabel(title)
        t.setObjectName("muted")
        t.setWordWrap(True)
        v = QLabel(value)
        v.setStyleSheet(f"font-size: {value_px}px; font-weight: 700; color: {color};")
        h = QLabel(hint)
        h.setObjectName("statHint")
        h.setWordWrap(True)

        for w in (t, v, h):
            lay.addWidget(w)
        lay.addStretch()


class ClassRow(QWidget):
    """Строка класса: цветная точка, название, доля, полоса, площадь."""

    def __init__(self, name, explanation, area, pct, color):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet("background: @C@; border-radius: 5px;".replace("@C@", color))
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: 600;")
        pct_label = QLabel(f"{pct:.1f}%")
        pct_label.setStyleSheet("font-weight: 700;")
        top.addWidget(dot)
        top.addWidget(name_label)
        top.addStretch()
        top.addWidget(pct_label)
        lay.addLayout(top)

        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(pct * 10))
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet(
            "QProgressBar { background: @BG@; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background: @C@; border-radius: 4px; }"
            .replace("@BG@", BG).replace("@C@", color)
        )
        lay.addWidget(bar)

        bottom = QHBoxLayout()
        expl = QLabel(explanation)
        expl.setObjectName("statHint")
        area_label = QLabel(f"{area:,.1f} га")
        area_label.setObjectName("statHint")
        bottom.addWidget(expl)
        bottom.addStretch()
        bottom.addWidget(area_label)
        lay.addLayout(bottom)


class MapCard(QFrame):
    """Карточка с картой и кнопкой «Открыть в отдельном окне»."""

    def __init__(self, title, draw_fn, data, file_name):
        super().__init__()
        self.setObjectName("card")
        self._title, self._draw_fn, self._data, self._file = title, draw_fn, data, file_name

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(8)

        head = QHBoxLayout()
        label = QLabel(title)
        label.setStyleSheet("font-size: 15px; font-weight: 700;")
        btn = ghost_button("⤢  Открыть в отдельном окне")
        btn.setToolTip("В отдельном окне доступны масштаб, перемещение и сохранение картинки")
        btn.clicked.connect(self.open_window)
        head.addWidget(label)
        head.addStretch()
        head.addWidget(btn)
        lay.addLayout(head)

        fig = Figure(figsize=(6, 5), layout="constrained", facecolor=CARD)
        draw_fn(fig, data)
        lay.addWidget(FigureCanvasQTAgg(fig), 1)

    def open_window(self):
        MapWindow(f"{self._title} — {self._file}", self._draw_fn, self._data).show()


# ======================================================================
# СТРАНИЦА РЕЗУЛЬТАТОВ
# ======================================================================
def kv_grid(rows):
    grid = QGridLayout()
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(8)
    grid.setColumnStretch(1, 1)
    for i, (key, value) in enumerate(rows):
        shown = value if len(value) <= 80 else value[:77] + "…"
        k = QLabel(key)
        k.setObjectName("muted")
        v = QLabel(shown)
        v.setWordWrap(True)
        v.setToolTip(value)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(k, i, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(v, i, 1)
    return grid


class ResultsPage(QWidget):
    def __init__(self, r, ch=None):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # ---------- левая колонка: данные ----------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(500)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(0, 0, 10, 0)
        col.setSpacing(14)
        col.addWidget(self._info_card(r))
        col.addWidget(self._stats_card(r))
        col.addWidget(self._classes_card(r))
        col.addWidget(self._forest_card(r))
        if ch is not None:
            col.addWidget(self._change_card(r, ch))
        col.addWidget(self._summary_card(r, ch))
        col.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)

        # ---------- правая колонка: карты ----------
        tabs = QTabWidget()
        name = r["name"]
        tabs.addTab(MapCard("Карта NDVI", draw_ndvi_map, r["ndvi"], name), "Карта NDVI")
        tabs.addTab(MapCard("Состояние леса", draw_state_map, r["ndvi"], name), "Состояние леса")
        if ch is not None:
            tabs.addTab(MapCard("Изменение NDVI", draw_delta_map, ch["d"], name), "Динамика NDVI")
            tabs.addTab(MapCard("Классы изменений", draw_change_map, ch["d"], name),
                        "Вырубки и восстановление")
        root.addWidget(tabs, 1)

    # --- карточки ---
    def _info_card(self, r):
        card, lay = make_card("Файл")
        info = r["info"]
        scale = ("0–255 → пересчитано в −1…+1" if r["was_packed"]
                 else "−1…+1 (пересчёт не нужен)")
        lay.addLayout(kv_grid([
            ("Файл", r["name"]),
            ("Размер", f"{info['shape'][1]} × {info['shape'][0]} пикселей"),
            ("Проекция", str(info["crs"])),
            ("Пиксель", f"{info['res'][0]:g} × {info['res'][1]:g} (ед. проекции)"),
            ("Площадь пикселя", f"{r['px_ha']:.4f} га ({r['px_source']})"),
            ("Исходный диапазон", f"{r['raw_min']:.2f} … {r['raw_max']:.2f}"),
            ("Шкала NDVI", scale),
        ]))
        return card

    def _stats_card(self, r):
        card, lay = make_card("Общая статистика")
        s = r["stats"]
        boxes = [
            StatBox("Минимум", f"{s['min']:.2f}", "вода, гари, тени"),
            StatBox("Максимум", f"{s['max']:.2f}", "сомкнутый лес"),
            StatBox("Среднее", f"{s['mean']:.2f}", "средний уровень", color=ACCENT),
            StatBox("Медиана", f"{s['median']:.2f}", "типичное значение"),
            StatBox("Ст. откл.", f"{s['std']:.2f}", "разброс значений"),
            StatBox("Площадь", f"{r['total_area']:,.0f} га", "проанализировано", value_px=17),
        ]
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, box in enumerate(boxes):
            grid.addWidget(box, i // 3, i % 3)
        lay.addLayout(grid)

        verdict = QLabel(f"<b>Вывод:</b> {describe_heterogeneity(s['std'])}")
        verdict.setWordWrap(True)
        verdict.setObjectName("muted")
        lay.addWidget(verdict)
        return card

    def _classes_card(self, r):
        card, lay = make_card("Состояние лесного покрова")
        for c, color in zip(r["class_areas"], STATE_COLORS):
            lay.addWidget(ClassRow(c["name"], c["explanation"], c["area"], c["pct"], color))
        return card

    def _forest_card(self, r):
        card, lay = make_card("Лес и зоны проверки")
        fr = r["forest"]
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(StatBox("Покрыто лесом", f"{fr['forest_area']:,.1f} га",
                              f"{fr['forest_pct']:.1f}% · NDVI ≥ {FOREST_MIN:g}",
                              color=ACCENT, value_px=15))
        row.addWidget(StatBox("Ослабленный лес", f"{fr['weak_area']:,.1f} га",
                              "лесопатологическое обследование", color=WARN, value_px=15))
        row.addWidget(StatBox("Вырубки, гари", f"{fr['open_area']:,.1f} га",
                              "проверка рубок и следов пожара", color=RED, value_px=15))
        lay.addLayout(row)
        lay.addWidget(note_label(
            "Возможные причины ослабления: короед и другие вредители, болезни, засуха, "
            "ветровал. По одному NDVI лес не отличить от полей и лугов — для точных "
            "площадей примените маску лесных земель (границы лесничеств)."
        ))
        return card

    def _change_card(self, r, ch):
        card, lay = make_card("Динамика: вырубки, гари, восстановление")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(StatBox("Потеря леса", f"{ch['loss_area']:,.1f} га",
                              "лес → резкое падение NDVI", color=RED, value_px=15))
        row.addWidget(StatBox("Восстановление", f"{ch['recovery_area']:,.1f} га",
                              "нелесная площадь → рост NDVI", color=ACCENT, value_px=15))
        row.addWidget(StatBox("Средняя Δ NDVI", f"{ch['mean']:+.3f}",
                              "по общей области", value_px=15))
        lay.addLayout(row)

        for c, color in zip(ch["classes"], CHANGE_COLORS):
            lay.addWidget(ClassRow(c["name"], c["explanation"], c["area"], c["pct"], color))

        text = (f"Ранний снимок: {ch['prev_name']}. "
                "Сравнивайте снимки одного сезона (один и тот же период вегетации), "
                "иначе сезонная смена NDVI будет принята за вырубку или рост.")
        if ch["reprojected"]:
            text += " Ранний снимок приведён к сетке текущего."
        if ch["overlap_pct"] < 95:
            text += f" Снимки перекрываются лишь на {ch['overlap_pct']:.0f}% территории."
        lay.addWidget(note_label(text))
        return card

    def _summary_card(self, r, ch):
        card, lay = make_card("Итог для кейса")
        summary = build_summary(r, ch)
        text = QPlainTextEdit(summary)
        text.setReadOnly(True)
        text.setFixedHeight(360 if ch is not None else 270)
        lay.addWidget(text)

        btn = ghost_button("Скопировать текст")

        def copy():
            QApplication.clipboard().setText(summary)
            btn.setText("Скопировано ✓")
            QTimer.singleShot(1500, lambda: btn.setText("Скопировать текст"))

        btn.clicked.connect(copy)
        lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)
        return card


# ======================================================================
# ГЛАВНОЕ ОКНО
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forest NDVI Monitor")
        self.resize(1440, 900)
        self.setMinimumSize(1200, 720)
        self.setAcceptDrops(True)
        self._last_dir = ""
        self._current = None   # результат анализа текущего (позднего) снимка
        self._change = None    # результат сравнения с ранним снимком
        self._results = None   # виджет страницы результатов

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(18)
        self.setCentralWidget(central)

        # ---------- шапка ----------
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        h1 = QLabel("🌲  Forest NDVI Monitor")
        h1.setObjectName("h1")
        sub = QLabel("Мониторинг лесного покрова · вырубки · гари · состояние древостоя · лесовосстановление")
        sub.setObjectName("muted")
        titles.addWidget(h1)
        titles.addWidget(sub)
        header.addLayout(titles)
        header.addStretch()

        self.chip = QLabel()
        self.chip.setObjectName("chip")
        self.chip.hide()
        header.addWidget(self.chip)
        header.addSpacing(12)

        self.cmp_btn = ghost_button("⇄  Сравнить с более ранним")
        self.cmp_btn.setToolTip("Загрузить снимок более ранней даты (того же сезона) и найти изменения")
        self.cmp_btn.setEnabled(False)
        self.cmp_btn.clicked.connect(self.choose_previous)
        header.addWidget(self.cmp_btn)
        header.addSpacing(8)

        open_btn = primary_button("📂  Открыть снимок")
        open_btn.clicked.connect(self.choose_file)
        header.addWidget(open_btn)
        root.addLayout(header)

        # ---------- страницы ----------
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_empty_page())
        root.addWidget(self.stack, 1)

    def _build_empty_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card, lay = make_card()
        card.setFixedWidth(540)
        lay.setContentsMargins(32, 36, 32, 36)
        lay.setSpacing(12)

        icon = QLabel("🛰️")
        icon.setStyleSheet("font-size: 56px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Откройте карту NDVI")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel("Перетащите файл GeoTIFF (.tif) в окно или нажмите кнопку.\n"
                      "Затем можно сравнить с более ранним снимком и найти\n"
                      "вырубки, гари и участки лесовосстановления.")
        text.setObjectName("muted")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn = primary_button("Выбрать файл")
        btn.clicked.connect(self.choose_file)

        for w in (icon, title, text):
            lay.addWidget(w)
        lay.addSpacing(8)
        lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    # ---------- выбор и загрузка файлов ----------
    def choose_file(self):
        self._pick_and_run("Выберите GeoTIFF (NDVI) — текущий снимок", self._load_current)

    def choose_previous(self):
        self._pick_and_run("Выберите более ранний GeoTIFF (NDVI) для сравнения",
                           self._load_previous)

    def _pick_and_run(self, dialog_title, action):
        """Диалог выбора файла + действие; при ошибке предлагает выбрать другой файл."""
        while True:
            path, _ = QFileDialog.getOpenFileName(
                self, dialog_title, self._last_dir, "GeoTIFF (*.tif *.tiff)"
            )
            if not path:  # «Отмена»
                return
            if self._try(lambda: action(path)) != "retry":
                return

    def _try(self, fn):
        """Выполняет fn. Возвращает "ok", "cancel" или "retry"."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fn()
        except ValueError as e:
            title, msg = "Ошибка", str(e)
        except Exception as e:
            title, msg = "Непредвиденная ошибка", f"{type(e).__name__}: {e}"
        else:
            return "ok"
        finally:
            QApplication.restoreOverrideCursor()
        return "retry" if self._ask_retry(title, msg) else "cancel"

    def _load_current(self, path):
        result = analyze(path)
        self._current, self._change = result, None
        self._last_dir = os.path.dirname(path)
        self.show_result()

    def _load_previous(self, path):
        prev = analyze(path)
        change = analyze_change(self._current, prev)  # может бросить ValueError
        self._change = change
        self._last_dir = os.path.dirname(path)
        self.show_result()

    def _ask_retry(self, title, msg):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(msg)
        box.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Retry).setText("Выбрать другой файл")
        box.button(QMessageBox.StandardButton.Cancel).setText("Отмена")
        return box.exec() == QMessageBox.StandardButton.Retry

    def show_result(self):
        if self._results is not None:
            self.stack.removeWidget(self._results)
            self._results.deleteLater()
        self._results = ResultsPage(self._current, self._change)
        self.stack.addWidget(self._results)
        self.stack.setCurrentWidget(self._results)

        name = self._current["name"]
        chip = f"📄 {name}"
        if self._change is not None:
            chip = f"📄 {self._change['prev_name']}  →  {name}"
        self.chip.setText(chip)
        self.chip.show()
        self.cmp_btn.setEnabled(True)
        self.setWindowTitle(f"Forest NDVI Monitor — {name}")

    # ---------- drag & drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if self._try(lambda: self._load_current(path)) == "retry":
            self.choose_file()


# ======================================================================
# ТЕМА И ЗАПУСК
# ======================================================================
STYLESHEET = """
QMainWindow { background: @BG@; }
QLabel { color: @TEXT@; background: transparent; }
QLabel#h1 { font-size: 24px; font-weight: 700; }
QLabel#muted { color: @MUTED@; }
QLabel#cardTitle { color: @MUTED@; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#statHint { color: @MUTED@; font-size: 11px; }
QLabel#chip { background: @CARD@; border: 1px solid @BORDER@; border-radius: 14px;
              padding: 6px 14px; color: @MUTED@; }

QFrame#card { background: @CARD@; border: 1px solid @BORDER@; border-radius: 14px; }
QFrame#stat { background: @CARD_ALT@; border: 1px solid @BORDER@; border-radius: 10px; }

QPushButton#primary { background: @ACCENT@; color: #04130c; border: none; border-radius: 10px;
                      padding: 10px 22px; font-size: 14px; font-weight: 700; }
QPushButton#primary:hover { background: @ACCENT_HOVER@; }
QPushButton#ghost { background: transparent; color: @TEXT@; border: 1px solid @BORDER@;
                    border-radius: 8px; padding: 6px 14px; }
QPushButton#ghost:hover { border-color: @ACCENT@; color: @ACCENT@; }
QPushButton#ghost:disabled { color: #4b5b75; border-color: #1a2438; }

QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: @MUTED@; padding: 8px 18px;
               border-bottom: 2px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: @TEXT@; border-bottom: 2px solid @ACCENT@; }
QTabBar::tab:hover { color: @TEXT@; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: @BORDER@; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QPlainTextEdit { background: @CARD_ALT@; border: 1px solid @BORDER@; border-radius: 8px;
                 color: @TEXT@; padding: 8px; }
QToolTip { background: @CARD_ALT@; color: @TEXT@; border: 1px solid @BORDER@; }
"""


def apply_theme(app):
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: CARD,
        QPalette.ColorRole.AlternateBase: BG,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: CARD,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.ToolTipBase: CARD_ALT,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#04130c",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    app.setPalette(pal)

    tokens = {"BG": BG, "CARD": CARD, "CARD_ALT": CARD_ALT, "BORDER": BORDER, "TEXT": TEXT,
              "MUTED": MUTED, "ACCENT": ACCENT, "ACCENT_HOVER": ACCENT_HOVER}
    qss = STYLESHEET
    for key, value in tokens.items():
        qss = qss.replace(f"@{key}@", value)
    app.setStyleSheet(qss)


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()