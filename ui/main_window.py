import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QComboBox,
    QVBoxLayout, QMessageBox, QFileDialog,
    QCheckBox, QSlider, QToolBar, QAction,
    QStyle, QDockWidget
)
from PyQt5.QtCore import Qt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import ClimateDataLoader
from visualization import VisualizationManager
from aggregation import MonthlyAggregation, HourlyAggregation
from indicators import TemperatureIndicator, PrecipitationIndicator
from netCDF4 import Dataset


DARK_STYLE = """
QWidget {
    background-color: #121212;
    color: #E0E0E0;
}
QComboBox, QCheckBox {
    background-color: #1E1E1E;
    color: #E0E0E0;
}
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Climate Indicator Comparison System")
        self.resize(1200, 850)

        self.loader = ClimateDataLoader()
        self.visualization = VisualizationManager()

        self.indicators = {
            "Temperature": TemperatureIndicator(),
            "Precipitation": PrecipitationIndicator()
        }

        self.era5_files = {"Temperature": None, "Precipitation": None}
        self.csv_file = None

        self.variable_map = {
            "Temperature": "t2m",
            "Precipitation": "tp"
        }

        self.init_toolbar()
        self.init_sidebar()
        self.init_central()

        self.statusBar().showMessage("Ready")

    # ==================================================
    # TOOLBAR
    # ==================================================
    def init_toolbar(self):
        style = self.style()
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(icon, text, fn, shortcut=None):
            a = QAction(style.standardIcon(icon), text, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(fn)
            tb.addAction(a)

        act(QStyle.SP_DialogOpenButton, "Load ERA5", self.load_era5, "Ctrl+E")
        act(QStyle.SP_FileIcon, "Load CSV", self.load_csv, "Ctrl+O")
        tb.addSeparator()

        act(QStyle.SP_DriveFDIcon, "Export PDF", self.export_pdf, "Ctrl+P")
        act(QStyle.SP_DriveHDIcon, "Export CSV", self.export_csv)
        act(QStyle.SP_DialogSaveButton, "Export GIF", self.export_gif)
        tb.addSeparator()

        act(QStyle.SP_DialogSaveButton, "Save Session", self.save_session, "Ctrl+S")
        act(QStyle.SP_DialogOpenButton, "Load Session", self.load_session, "Ctrl+L")
        tb.addSeparator()

        act(QStyle.SP_MediaPlay, "Compare", self.compare, "Ctrl+Return")
        act(QStyle.SP_BrowserReload, "Reset", self.visualization.reset, "Ctrl+R")

    # ==================================================
    # SIDEBAR
    # ==================================================
    def init_sidebar(self):
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        panel = QWidget()
        layout = QVBoxLayout()

        def label(text):
            layout.addWidget(QLabel(text))

        label("Indicator 1")
        self.combo1 = QComboBox()
        self.combo1.addItems(self.indicators.keys())
        layout.addWidget(self.combo1)

        label("Indicator 2")
        self.combo2 = QComboBox()
        self.combo2.addItems(self.indicators.keys())
        layout.addWidget(self.combo2)

        label("Data Source")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["ERA5 NetCDF", "CSV File"])
        layout.addWidget(self.source_combo)

        label("Aggregation")
        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems([
            "Monthly Average (Jan–Dec)",
            "Yearly Hourly Time Series"
        ])
        layout.addWidget(self.aggregation_combo)

        label("Graph Type")
        self.graph_combo = QComboBox()
        self.graph_combo.addItems([
            "Line", "Scatter", "Bar", "Area", "Step",
            "Histogram", "Boxplot", "Heatmap",
            "Spatial Heatmap"
        ])
        self.graph_combo.currentTextChanged.connect(self._sync_layout_rules)
        layout.addWidget(self.graph_combo)

        label("Layout")
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Single", "Dual"])
        layout.addWidget(self.layout_combo)

        self.predict_cb = QCheckBox("Prediction")
        self.predict_cb.setChecked(True)
        layout.addWidget(self.predict_cb)

        self.animate_cb = QCheckBox("Animate")
        layout.addWidget(self.animate_cb)

        self.normalize_cb = QCheckBox("Normalize (Z-score)")
        self.normalize_cb.stateChanged.connect(self._sync_layout_rules)
        layout.addWidget(self.normalize_cb)

        self.dark_cb = QCheckBox("Dark Mode")
        self.dark_cb.stateChanged.connect(self.toggle_dark)
        layout.addWidget(self.dark_cb)

        label("Rolling Average")
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setRange(1, 24)
        self.roll_slider.setValue(1)
        layout.addWidget(self.roll_slider)

        label("CSV Temperature Column")
        self.csv_temp = QComboBox()
        layout.addWidget(self.csv_temp)

        label("CSV Precipitation Column")
        self.csv_precip = QComboBox()
        layout.addWidget(self.csv_precip)

        panel.setLayout(layout)
        dock.setWidget(panel)

    # ==================================================
    # CENTRAL
    # ==================================================
    def init_central(self):
        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.visualization.canvas, stretch=1)
        central.setLayout(layout)
        self.setCentralWidget(central)

    # ==================================================
    # UI RULES
    # ==================================================
    def _sync_layout_rules(self):
        graph = self.graph_combo.currentText()
        if graph in ("Histogram", "Boxplot", "Heatmap", "Spatial Heatmap") or self.normalize_cb.isChecked():
            self.layout_combo.setCurrentText("Single")
            self.layout_combo.setEnabled(False)
        else:
            self.layout_combo.setEnabled(True)

    # ==================================================
    # UI ACTIONS
    # ==================================================
    def toggle_dark(self):
        if self.dark_cb.isChecked():
            self.setStyleSheet(DARK_STYLE)
            plt.style.use("dark_background")
        else:
            self.setStyleSheet("")
            plt.style.use("default")

    def load_era5(self):
        path, _ = QFileDialog.getOpenFileName(self, "ERA5 NetCDF", "", "*.nc")
        if not path:
            return

        ds = Dataset(path)
        vars_ = ds.variables.keys()
        ds.close()

        if "t2m" in vars_:
            self.era5_files["Temperature"] = path
        if "tp" in vars_:
            self.era5_files["Precipitation"] = path

        self.statusBar().showMessage("ERA5 loaded")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV File", "", "*.csv")
        if not path:
            return

        self.csv_file = path
        df = pd.read_csv(path)

        self.csv_temp.clear()
        self.csv_precip.clear()
        self.csv_temp.addItems(df.columns)
        self.csv_precip.addItems(df.columns)

        self.statusBar().showMessage("CSV loaded")

    # ==================================================
    # MAIN LOGIC (FULLY FIXED)
    # ==================================================
    def compare(self):
        graph = self.graph_combo.currentText()
        ind1 = self.combo1.currentText()

        # ---------- SPATIAL HEATMAP ----------
        if graph == "Spatial Heatmap":
            if self.source_combo.currentText() != "ERA5 NetCDF":
                QMessageBox.warning(self, "Invalid Source",
                                    "Spatial heatmap requires ERA5 NetCDF data.")
                return

            path = self.era5_files.get(ind1)
            if path is None:
                QMessageBox.warning(self, "Missing ERA5 File",
                                    f"Please load an ERA5 file for {ind1}.")
                return

            lons, lats, field = self.loader.load_variable_2d(
                path, self.variable_map[ind1]
            )

            self.visualization.plot_spatial_heatmap(
                lons, lats, field,
                title=f"{ind1} Spatial Distribution",
                unit=self.indicators[ind1].unit
            )
            return

        # ---------- TIME SERIES ----------
        ind2 = self.combo2.currentText()
        strategy = (
            MonthlyAggregation()
            if self.aggregation_combo.currentText().startswith("Monthly")
            else HourlyAggregation()
        )

        # ----- ERA5 -----
        if self.source_combo.currentText() == "ERA5 NetCDF":
            path1 = self.era5_files.get(ind1)
            path2 = self.era5_files.get(ind2)

            if path1 is None or path2 is None:
                QMessageBox.warning(
                    self,
                    "Missing ERA5 File",
                    "Please load ERA5 files for both indicators."
                )
                return

            raw1, times = self.loader.load_variable(path1, self.variable_map[ind1])
            raw2, _ = self.loader.load_variable(path2, self.variable_map[ind2])

        # ----- CSV -----
        else:
            if self.csv_file is None:
                QMessageBox.warning(self, "Missing CSV File",
                                    "Please load a CSV file first.")
                return

            col1 = self.csv_temp.currentText()
            col2 = self.csv_precip.currentText()

            if not col1 or not col2:
                QMessageBox.warning(self, "Missing CSV Columns",
                                    "Please select both CSV columns.")
                return

            raw1, times = self.loader.load_csv(self.csv_file, col1)
            raw2, _ = self.loader.load_csv(self.csv_file, col2)

        x, y1 = strategy.aggregate(raw1, times)
        _, y2 = strategy.aggregate(raw2, times)

        if self.roll_slider.value() > 1:
            y1 = pd.Series(y1).rolling(self.roll_slider.value(), 1).mean().values
            y2 = pd.Series(y2).rolling(self.roll_slider.value(), 1).mean().values

        if self.normalize_cb.isChecked():
            y1 = (y1 - np.nanmean(y1)) / np.nanstd(y1)
            y2 = (y2 - np.nanmean(y2)) / np.nanstd(y2)

        self.visualization.plot(
            x, y1, y2,
            self.indicators[ind1],
            self.indicators[ind2],
            title=self.aggregation_combo.currentText(),
            predict=self.predict_cb.isChecked(),
            graph_type=graph,
            layout=self.layout_combo.currentText(),
            animate=self.animate_cb.isChecked(),
            normalized=self.normalize_cb.isChecked()
        )

    # ==================================================
    # EXPORTS
    # ==================================================
    def export_pdf(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "*.pdf")
        if p:
            self.visualization.save_pdf(p)

    def export_csv(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "*.csv")
        if p:
            self.visualization.save_csv(p)

    def export_gif(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export GIF", "", "*.gif")
        if p:
            self.visualization.save_gif(p)

    # ==================================================
    # SESSION
    # ==================================================
    def save_session(self):
        json.dump({
            "graph": self.graph_combo.currentText(),
            "layout": self.layout_combo.currentText(),
            "predict": self.predict_cb.isChecked(),
            "animate": self.animate_cb.isChecked(),
            "normalize": self.normalize_cb.isChecked(),
            "roll": self.roll_slider.value(),
            "dark": self.dark_cb.isChecked()
        }, open("session.json", "w"), indent=2)

    def load_session(self):
        if not os.path.exists("session.json"):
            return

        s = json.load(open("session.json"))
        self.graph_combo.setCurrentText(s.get("graph", "Line"))
        self.layout_combo.setCurrentText(s.get("layout", "Single"))
        self.predict_cb.setChecked(s.get("predict", True))
        self.animate_cb.setChecked(s.get("animate", False))
        self.normalize_cb.setChecked(s.get("normalize", False))
        self.roll_slider.setValue(s.get("roll", 1))
        self.dark_cb.setChecked(s.get("dark", False))
