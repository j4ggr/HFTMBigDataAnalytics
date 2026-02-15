import logging
from functools import lru_cache
from math import pi

import daspi as dsp
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import rcParams

dsp.STR.LANGUAGE = "de"


def _configure_matplotlib_font() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]

    selected_font = next((font for font in preferred_fonts if font in available_fonts), "sans-serif")
    rcParams["font.family"] = selected_font
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


_configure_matplotlib_font()

PROBABILITY_CHART_DPI = 120
CORRELATION_CHART_DPI = 110


@lru_cache(maxsize=64)
def _theoretical_probabilities(n_dices: int) -> np.ndarray:
    single_die = np.ones(6, dtype=np.float64) / 6.0
    probabilities = single_die
    for _ in range(n_dices - 1):
        probabilities = np.convolve(probabilities, single_die)
    return probabilities


@lru_cache(maxsize=64)
def _empirical_counts(n_dices: int, n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42 + 31 * n_dices + n_samples)
    dice_rolls = rng.integers(1, 7, size=(n_samples, n_dices), dtype=np.int16).sum(axis=1)
    min_sum = n_dices
    max_sum = 6 * n_dices
    return np.bincount(dice_rolls, minlength=max_sum + 1)[min_sum : max_sum + 1]


def theoretical_dice_probability(n_dices: int) -> pd.DataFrame:
    min_sum = n_dices
    max_sum = 6 * n_dices
    sums = np.arange(min_sum, max_sum + 1)
    theoretical_probs = _theoretical_probabilities(n_dices)

    theoretical_data = pd.DataFrame({"sum": sums, "theoretical_probability": theoretical_probs})
    theoretical_data["theoretical_cum_probability"] = theoretical_data["theoretical_probability"].cumsum()
    return theoretical_data


def chart_dice_probability_distribution(n_dices: int, exp_samples: int, kind: str):
    hue = None
    dodge = False
    sub_title_addition = ""
    datasets: list[pd.DataFrame] = []

    n_samples = 10**exp_samples
    min_sum = n_dices
    max_sum = 6 * n_dices
    sums = np.arange(min_sum, max_sum + 1)

    if kind == "Empirisch":
        sub_title_addition = " (Empirisch)"
    elif kind == "Theoretisch":
        sub_title_addition = " (Theoretisch)"
    elif kind == "Beide":
        sub_title_addition = " (Empirisch vs. Theoretisch)"
        hue = "Datenquelle"
        dodge = (True, False, True, False)

    if kind in ["Empirisch", "Beide"]:
        counts = _empirical_counts(n_dices=n_dices, n_samples=n_samples)
        empirical_data = pd.DataFrame({"sum": sums, "count": counts})
        empirical_data["cum_count"] = empirical_data["count"].cumsum()
        empirical_data["probability"] = empirical_data["count"] / n_samples
        empirical_data["cum_probability"] = empirical_data["probability"].cumsum()
        empirical_data["Datenquelle"] = "Empirisch"
        datasets.append(empirical_data)

    if kind in ["Theoretisch", "Beide"]:
        theoretical_data = theoretical_dice_probability(n_dices)
        theoretical_data["count"] = theoretical_data["theoretical_probability"] * n_samples
        theoretical_data["cum_count"] = theoretical_data["count"].cumsum()
        theoretical_data["probability"] = theoretical_data["theoretical_probability"]
        theoretical_data["cum_probability"] = theoretical_data["theoretical_cum_probability"]
        theoretical_data["Datenquelle"] = "Theoretisch"
        datasets.append(theoretical_data)

    data = pd.concat(datasets, ignore_index=True)

    chart = dsp.JointChart(
            source=data,
            target=("count", "cum_count", "probability", "cum_probability"),
            feature="sum",
            hue=hue,
            dodge=dodge,
            sharex=True,
            categorical_feature=True,
            nrows=2,
            ncols=2,
            stretch_figsize=1.6,
            dpi=PROBABILITY_CHART_DPI,
        ).plot(dsp.Bar
        ).plot(dsp.Scatter
        ).plot(dsp.Line, on_last_axes=True
        ).plot(dsp.Bar
        ).plot(dsp.Scatter
        ).plot(dsp.Line, on_last_axes=True
        ).stripes(mean=True
        ).label(
            fig_title="Wahrscheinlichkeitsverteilung der Augensumme von Würfeln",
            sub_title=f"{n_samples:,} Würfe mit {n_dices} sechsseitigen Würfeln{sub_title_addition}".replace(",", "'"),
            target_label=(
                "Anzahl Beobachtungen",
                "Kumulative Beobachtungen",
                "Wahrscheinlichkeit",
                "Kumulative Wahrscheinlichkeit",),
            feature_label="Augensumme",)
    return chart.figure


def calculate_sine_wave(phase: float, duration: float, n_samples: int):
    timeline = np.linspace(0, duration, n_samples, endpoint=False)
    return timeline, np.sin(timeline + phase)


def calculate_cosine_wave(phase: float, duration: float, n_samples: int):
    timeline = np.linspace(0, duration, n_samples, endpoint=False)
    return timeline, np.cos(timeline + phase)


def phase_from_correlation_coefficient(correlation_coefficient: float) -> float:
    if not -1 <= correlation_coefficient <= 1:
        raise ValueError("Correlation coefficient must be in the range [-1, 1]")
    return np.arccos(correlation_coefficient)


def chart_correlation_coefficient(correlation_coefficient: float):
    n_samples = 500
    duration = 2 * pi

    phase = phase_from_correlation_coefficient(correlation_coefficient)
    timeline, sine_wave = calculate_sine_wave(phase=phase, duration=duration, n_samples=n_samples)
    _, cosine_wave = calculate_cosine_wave(phase=phase, duration=duration, n_samples=n_samples)
    _, sine_wave_base = calculate_sine_wave(phase=0, duration=duration, n_samples=n_samples)

    amplitude = np.sin(phase)
    noise_x = amplitude * np.random.normal(0, 0.4, n_samples)
    noise_y = amplitude * np.random.normal(0, 0.4, n_samples)

    df_wave = pd.DataFrame(
        {
            "Zeit": timeline,
            "Sinus": sine_wave,
            "Cosinus": cosine_wave,
            "Scatter_x": sine_wave_base + noise_x,
            "Scatter_y": sine_wave + noise_y,
        }
    )

    df_r = pd.DataFrame({"x": [0, phase], "y": [1, 1]})

    if phase > pi / 2:
        df_sides = pd.DataFrame(
            {
                "x_sin": [pi, phase],
                "y_sin": [abs(np.cos(phase)), 1],
                "x_cos": [0, pi],
                "y_cos": [0, abs(np.cos(phase))],
            }
        )
    else:
        df_sides = pd.DataFrame(
            {
                "x_sin": [0, phase],
                "y_sin": [np.cos(phase), 1],
                "x_cos": [0, 0],
                "y_cos": [0, np.cos(phase)],
            }
        )

    df_angle = pd.DataFrame({"x": np.linspace(0, phase, n_samples), "y": [0.2] * n_samples})

    chart = dsp.JointChart(
            source=df_wave,
            target=("", "Sinus", "Zeit", "Scatter_y"),
            feature=("", "Zeit", "Cosinus", "Scatter_x"),
            nrows=2,
            ncols=2,
            figsize=(5, 5),
            dpi=CORRELATION_CHART_DPI,
        ).plot(
            dsp.SkipSubplot
        ).plot(
            dsp.Line,
            hide_axis="feature",
            visible_spines="target",
            color=dsp.COLOR.BAD
        ).plot(
            dsp.Line,
            hide_axis="target",
            visible_spines="feature",
            color=dsp.COLOR.GOOD
        ).plot(
            dsp.LinearRegressionLine,
            show_scatter=True,
            show_fit_ci=True,
            show_pred_ci=True,
        ).label(
            axes_titles=(
                "",
                "Sinus",
                "Cosinus",
                "Korrelationskoeffizient R = {:.2f}".format(
                    np.corrcoef(df_wave["Scatter_y"], df_wave["Scatter_x"])[0, 1]),),)

    chart.figure.delaxes(chart.axes[0])
    chart.axes[1].set(xlim=(0, duration), ylim=(-1.1, 1.1))
    chart.axes[2].set(xlim=(-1.1, 1.1), ylim=(0, duration))
    chart.axes[3].set(xlim=(-2, 2), ylim=(-2, 2))

    ax_polar = chart.figure.add_subplot(2, 2, 1, polar=True)
    dsp.Stem(df_r, target="y", feature="x", bottom=0, ax=ax_polar)()
    ax_polar.plot(df_angle["x"], df_angle["y"], "k--", alpha=0.5)
    ax_polar.plot(df_sides["x_sin"], df_sides["y_sin"], c=dsp.COLOR.BAD)
    ax_polar.plot(df_sides["x_cos"], df_sides["y_cos"], c=dsp.COLOR.GOOD)
    ax_polar.set(yticks=[], ylim=(0, 1))

    return chart.figure


def chart_box_plot(point: float):
    data = pd.DataFrame({"samples": list(np.linspace(-10, 10, 9)) + [point]})
    data["is_dynamic"] = False
    data.loc[data.index[-1], "is_dynamic"] = True

    static_data = data.loc[~data["is_dynamic"]]
    dynamic_data = data.loc[data["is_dynamic"]]

    chart = (
        dsp.SingleChart(
            source=static_data,
            target="samples",
            target_on_y=False,
            categorical_feature=True,
            figsize=(12, 3),
        ).plot(
            dsp.Scatter,
            visible_spines="target",
            hide_axis="feature",
            kw_call={"s": 80},
        ).label(
            fig_title="Box Plot mit einem dynamischen Punkt",
            target_label="Beispielhafte Daten",))
    
    chart.axes[0].boxplot(data["samples"], vert=False, positions=[0], widths=6)
    chart.axes[0].scatter(
        dynamic_data["samples"],
        np.zeros(len(dynamic_data)),
        color=dsp.COLOR.GOOD,
        s=140,
        zorder=5,
    )
    chart.axes[0].set_ylim(-5, 5)
    chart.axes[0].set_xlim(-50, 50)

    return chart.figure