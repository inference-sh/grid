import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional


class BarChartInput(BaseAppInput):
    """Simple bar chart — horizontal or vertical."""
    title: str = Field(description="Chart title")
    labels: List[str] = Field(description="Category labels")
    values: List[float] = Field(description="Values for each bar")
    colors: Optional[List[str]] = Field(None, description="Bar colors (hex or named)")
    xlabel: Optional[str] = Field(None, description="X-axis label")
    ylabel: Optional[str] = Field(None, description="Y-axis label")
    horizontal: bool = Field(default=False, description="Horizontal bars")
    value_labels: bool = Field(default=True, description="Show value on each bar")
    width: int = Field(default=800, description="Image width in pixels")
    height: int = Field(default=500, description="Image height in pixels")
    highlight_max: bool = Field(default=True, description="Bold the highest bar")


class BarChartOutput(BaseAppOutput):
    image: File = Field(description="Chart image (PNG)")


class GroupedBarInput(BaseAppInput):
    """Grouped bar chart — multiple series per category."""
    title: str = Field(description="Chart title")
    categories: List[str] = Field(description="Category labels (x-axis groups)")
    series: List[dict] = Field(description="List of {name, values, color?} dicts")
    xlabel: Optional[str] = Field(None, description="X-axis label")
    ylabel: Optional[str] = Field(None, description="Y-axis label")
    value_labels: bool = Field(default=True, description="Show value on each bar")
    width: int = Field(default=900, description="Image width in pixels")
    height: int = Field(default=500, description="Image height in pixels")


class GroupedBarOutput(BaseAppOutput):
    image: File = Field(description="Chart image (PNG)")


class ScatterInput(BaseAppInput):
    """Scatter plot with optional color grouping."""
    title: str = Field(description="Chart title")
    series: List[dict] = Field(description="List of {name, x, y, color?} dicts. x and y are lists of floats.")
    xlabel: Optional[str] = Field(None, description="X-axis label")
    ylabel: Optional[str] = Field(None, description="Y-axis label")
    width: int = Field(default=800, description="Image width in pixels")
    height: int = Field(default=600, description="Image height in pixels")


class ScatterOutput(BaseAppOutput):
    image: File = Field(description="Chart image (PNG)")


# Dark palette — vibrant on dark background
DEFAULT_COLORS = ["#60a5fa", "#f87171", "#9ca3af", "#4ade80", "#fbbf24", "#a78bfa"]

# Theme colors
BG = "#0f0f0f"
SURFACE = "#1a1a1a"
FG = "#e5e5e5"
FG_MUTED = "#a3a3a3"
GRID_COLOR = "#262626"


class App(BaseApp):

    async def setup(self, config=None):
        self.logger = logging.getLogger(__name__)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import urllib.request
        import os

        # Download Inter font
        font_path = "/tmp/Inter-Regular.ttf"
        font_bold_path = "/tmp/Inter-Bold.ttf"
        if not os.path.exists(font_path):
            urllib.request.urlretrieve("https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.ttf", font_path)
            urllib.request.urlretrieve("https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-700-normal.ttf", font_bold_path)
            fm.fontManager.addfont(font_path)
            fm.fontManager.addfont(font_bold_path)
            self.logger.info("Inter font downloaded and registered")

        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.rcParams.update({
            "font.family": "Inter",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 20,
            "axes.labelsize": 13,
            "axes.labelpad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": GRID_COLOR,
            "axes.linewidth": 0.8,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.grid": False,
            "xtick.color": FG_MUTED,
            "ytick.color": FG_MUTED,
            "axes.labelcolor": FG_MUTED,
            "text.color": FG,
            "figure.dpi": 150,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        })
        self.logger.info("matplotlib ready (dark theme)")

    def _save(self, fig, name):
        path = f"/tmp/{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor=BG, dpi=300, pad_inches=0.4)
        self.plt.close(fig)
        self.logger.info(f"saved {path}")
        return path

    async def bar_chart(self, input_data: BarChartInput) -> BarChartOutput:
        self.logger.info(f"bar_chart: {input_data.title}")
        plt = self.plt

        fig, ax = plt.subplots(figsize=(input_data.width / 100, input_data.height / 100))
        colors = input_data.colors or DEFAULT_COLORS[:len(input_data.labels)]

        if input_data.highlight_max:
            max_idx = input_data.values.index(max(input_data.values))
            alphas = [1.0 if i == max_idx else 0.6 for i in range(len(input_data.values))]
        else:
            alphas = [1.0] * len(input_data.values)

        if input_data.horizontal:
            bars = ax.barh(input_data.labels, input_data.values, color=colors, height=0.5, edgecolor="none")
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)
            if input_data.value_labels:
                for bar, val in zip(bars, input_data.values):
                    ax.text(val + max(input_data.values) * 0.025, bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=12, fontweight="bold", color=FG)
            if input_data.xlabel:
                ax.set_xlabel(input_data.xlabel)
            ax.invert_yaxis()
        else:
            bars = ax.bar(input_data.labels, input_data.values, color=colors, width=0.5, edgecolor="none")
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)
            if input_data.value_labels:
                for bar, val in zip(bars, input_data.values):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + max(input_data.values) * 0.025,
                            f"{val:.3f}", ha="center", fontsize=12, fontweight="bold", color=FG)
            if input_data.ylabel:
                ax.set_ylabel(input_data.ylabel)

        ax.set_title(input_data.title)
        path = self._save(fig, "bar_chart")
        return BarChartOutput(image=File(path=path))

    async def grouped_bar(self, input_data: GroupedBarInput) -> GroupedBarOutput:
        self.logger.info(f"grouped_bar: {input_data.title}")
        plt = self.plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(input_data.width / 100, input_data.height / 100))

        n_categories = len(input_data.categories)
        n_series = len(input_data.series)
        bar_width = 0.65 / n_series
        x = np.arange(n_categories)

        all_max = max(max(s["values"]) for s in input_data.series)

        for i, s in enumerate(input_data.series):
            color = s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            offset = (i - n_series / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, s["values"], bar_width * 0.9, label=s["name"], color=color, edgecolor="none")

            if input_data.value_labels:
                for bar, val in zip(bars, s["values"]):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + all_max * 0.02,
                            f"{val:.3f}", ha="center", fontsize=8, fontweight="bold", color=FG)

        ax.set_xticks(x)
        ax.set_xticklabels(input_data.categories, fontsize=11)
        if input_data.xlabel:
            ax.set_xlabel(input_data.xlabel)
        if input_data.ylabel:
            ax.set_ylabel(input_data.ylabel)
        ax.set_title(input_data.title)
        ax.legend(frameon=False, fontsize=11, labelcolor=FG, loc="upper right",
                  borderaxespad=1, handlelength=1.2, handletextpad=0.6)
        path = self._save(fig, "grouped_bar")
        return GroupedBarOutput(image=File(path=path))

    async def scatter(self, input_data: ScatterInput) -> ScatterOutput:
        self.logger.info(f"scatter: {input_data.title}")
        plt = self.plt

        fig, ax = plt.subplots(figsize=(input_data.width / 100, input_data.height / 100))

        for i, s in enumerate(input_data.series):
            color = s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            ax.scatter(s["x"], s["y"], label=s["name"], color=color, alpha=0.75, s=50, edgecolors="none")

        if input_data.xlabel:
            ax.set_xlabel(input_data.xlabel)
        if input_data.ylabel:
            ax.set_ylabel(input_data.ylabel)
        ax.set_title(input_data.title)
        ax.legend(frameon=False, fontsize=11, labelcolor=FG)
        path = self._save(fig, "scatter")
        return ScatterOutput(image=File(path=path))
