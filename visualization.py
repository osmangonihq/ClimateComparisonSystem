from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


class VisualizationManager:
    """
    Visualization Manager

    Supports:
    - Line, Scatter, Bar, Area, Step
    - Histogram, Boxplot
    - Correlation Heatmap
    - Spatial (Geographic) Heatmap
    - Single / Dual axis layouts
    - Auto precipitation scaling
    - Axis coloring
    - Dual grids
    - Axis locking (normalized)
    - Animations + GIF export
    """

    def __init__(self):
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.ax1 = None
        self.ax2 = None
        self._anim = None

        self.colors = {
            "Temperature": "tab:blue",
            "Precipitation": "tab:orange"
        }

        self._last_state = None

    # --------------------------------------------------
    def _auto_fonts(self):
        h = self.figure.get_size_inches()[1]
        return {
            "title": max(12, h * 2.2),
            "label": max(10, h * 1.6),
            "legend": max(9, h * 1.3),
        }

    # --------------------------------------------------
    def _scale_precip(self, y):
        y = np.asarray(y, dtype=float)
        mean = np.nanmean(np.abs(y))

        if mean < 1e-3:
            return y * 1000, "mm"
        elif mean < 1:
            return y, "mm/day"
        else:
            return y, "mm"

    # ==================================================
    # TIME SERIES / COMPARISON PLOT
    # ==================================================
    def plot(
        self,
        x, y1, y2,
        ind1, ind2,
        title=None,
        predict=True,
        graph_type="Line",
        layout="Single",
        animate=False,
        normalized=False
    ):
        self.figure.clear()
        self._anim = None

        x = np.asarray(x)
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)

        self._last_state = (
            x, y1, y2, ind1, ind2,
            title, predict, graph_type, layout, normalized
        )

        fonts = self._auto_fonts()

        # ---------- Correlation Heatmap ----------
        if graph_type == "Heatmap":
            self._plot_correlation_heatmap(y1, y2, ind1, ind2, fonts)
            self.canvas.draw_idle()
            return

        # ---------- Axes setup ----------
        if graph_type in ("Histogram", "Boxplot"):
            self.ax1 = self.figure.add_subplot(111)
            self.ax2 = None
        elif layout == "Dual":
            self.ax1 = self.figure.add_subplot(211)
            self.ax2 = self.figure.add_subplot(212)
        elif normalized:
            self.ax1 = self.figure.add_subplot(111)
            self.ax2 = None
        else:
            self.ax1 = self.figure.add_subplot(111)
            self.ax2 = self.ax1.twinx()

        c1 = self.colors[ind1.name]
        c2 = self.colors[ind2.name]

        y2_scaled, precip_unit = self._scale_precip(y2)

        # ---------- Draw ----------
        def draw(alpha=1.0):
            self.ax1.clear()
            if self.ax2:
                self.ax2.clear()

            if graph_type == "Histogram":
                self.ax1.hist(y1[np.isfinite(y1)], bins=25, alpha=0.6, color=c1, label=ind1.name)
                self.ax1.hist(y2[np.isfinite(y2)], bins=25, alpha=0.6, color=c2, label=ind2.name)
                self.ax1.set_ylabel("Frequency")
                return

            if graph_type == "Boxplot":
                self.ax1.boxplot(
                    [y1[np.isfinite(y1)], y2[np.isfinite(y2)]],
                    labels=[ind1.name, ind2.name],
                    patch_artist=True
                )
                return

            self._plot_time_series(
                x, y1, y2_scaled,
                ind1, ind2,
                graph_type, alpha,
                fonts, predict,
                normalized, c1, c2,
                precip_unit
            )

            self.ax1.set_title(title or "Climate Indicator Comparison", fontsize=fonts["title"])

            # Grids
            if not normalized:
                self.ax1.grid(True, linestyle="-", alpha=0.3)
                if self.ax2:
                    self.ax2.grid(True, linestyle="--", alpha=0.3)

            # Axis locking
            if normalized:
                self.ax1.set_ylim(-3, 3)

            # Legends
            h1, l1 = self.ax1.get_legend_handles_labels()
            if h1:
                self.ax1.legend(h1, l1, loc="upper left", fontsize=fonts["legend"])

            if self.ax2:
                h2, l2 = self.ax2.get_legend_handles_labels()
                if h2:
                    self.ax2.legend(h2, l2, loc="upper right", fontsize=fonts["legend"])

        if animate:
            self._anim = FuncAnimation(
                self.figure,
                lambda i: draw(alpha=i / 20),
                frames=20,
                interval=30,
                repeat=False
            )
        else:
            draw()

        self.figure.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.88)
        self.canvas.draw_idle()

    # --------------------------------------------------
    def _plot_time_series(
        self, x, y1, y2,
        ind1, ind2,
        graph_type, alpha,
        fonts, predict,
        normalized, c1, c2,
        precip_unit
    ):
        ax1, ax2 = self.ax1, self.ax2

        def plot(ax, x, y, color, label):
            if graph_type == "Line":
                ax.plot(x, y, alpha=alpha, color=color, label=label)
            elif graph_type == "Scatter":
                ax.scatter(x, y, alpha=alpha, color=color, label=label)
            elif graph_type == "Bar":
                ax.bar(x, y, alpha=alpha, color=color, label=label)
            elif graph_type == "Area":
                ax.fill_between(x, y, alpha=0.4 * alpha, color=color, label=label)
            elif graph_type == "Step":
                ax.step(x, y, where="mid", alpha=alpha, color=color, label=label)

        plot(ax1, x, y1, c1, ind1.name)
        ax1.set_ylabel(f"{ind1.name} ({ind1.unit})", fontsize=fonts["label"], color=c1)
        ax1.tick_params(axis="y", colors=c1)

        if ax2:
            plot(ax2, x, y2, c2, ind2.name)
            ax2.set_ylabel(f"{ind2.name} ({precip_unit})", fontsize=fonts["label"], color=c2)
            ax2.tick_params(axis="y", colors=c2)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

        # Prediction
        if predict and graph_type in ("Line", "Scatter") and len(x) >= 6:
            idx = np.arange(len(x))
            future = np.arange(len(x), len(x) + max(5, len(x) // 5))

            if np.isfinite(y1).sum() >= 3:
                coef1 = np.polyfit(idx[np.isfinite(y1)], y1[np.isfinite(y1)], 1)
                ax1.plot(future, np.polyval(coef1, future), "--", color=c1, alpha=0.6)

            if ax2 and np.isfinite(y2).sum() >= 3:
                coef2 = np.polyfit(idx[np.isfinite(y2)], y2[np.isfinite(y2)], 1)
                ax2.plot(future, np.polyval(coef2, future), "--", color=c2, alpha=0.6)

    # ==================================================
    # CORRELATION HEATMAP
    # ==================================================
    def _plot_correlation_heatmap(self, y1, y2, ind1, ind2, fonts):
        ax = self.figure.add_subplot(111)
        corr = np.corrcoef(y1, y2)

        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([ind1.name, ind2.name])
        ax.set_yticklabels([ind1.name, ind2.name])
        ax.set_title("Correlation Heatmap", fontsize=fonts["title"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

        self.figure.colorbar(im, ax=ax)

    # ==================================================
    # SPATIAL (GEOGRAPHIC) HEATMAP
    # ==================================================
    def plot_spatial_heatmap(self, lons, lats, data, title, unit):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.contourf(lons, lats, data, levels=20, cmap="coolwarm")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)

        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label(unit)

        ax.grid(True, linestyle="--", alpha=0.4)
        self.canvas.draw_idle()

    # ==================================================
    # EXPORT / RESET
    # ==================================================
    def save_gif(self, file_path):
        if self._anim:
            self._anim.save(file_path, writer="pillow", fps=10)

    def save_pdf(self, file_path):
        self.figure.savefig(file_path, format="pdf", bbox_inches="tight")

    def save_csv(self, file_path):
        if not self._last_state:
            return
        x, y1, y2, ind1, ind2, *_ = self._last_state
        pd.DataFrame({
            "Index": x,
            ind1.name: y1,
            ind2.name: y2
        }).to_csv(file_path, index=False)

    def reset(self):
        self.figure.clear()
        self.ax1 = None
        self.ax2 = None
        self._anim = None
        self.canvas.draw_idle()
