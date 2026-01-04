from weakref import finalize

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from neurokit2 import NeuroKitWarning, ppg_segment
from neurokit2.ppg.ppg_peaks import _ppg_peaks_plot
from neurokit2.signal.signal_rate import _signal_rate_plot
from scipy.fft import fftfreq, fft
import neurokit2 as nk

from warnings import warn

from glob import glob
import pandas as pd



def plot(ppg_signals, xf, mag, N, info=None, static=True, filename=None, coherence_score=None, lf_peak_freq=None):
    # Sanity-check input.
    if not isinstance(ppg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `ppg_signals` argument must"
            " be the DataFrame returned by `ppg_process()`."
        )

    trial_info = filename.split("_")
    subject_name = trial_info[1]
    trial_name = trial_info[2]
    attempt = trial_info[3].split('.')[0]

    # Extract Peaks.
    if info is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category=NeuroKitWarning,
        )
        info = {"sampling_rate": 1000}

    # Extract Peaks (take those from df as it might have been cropped)
    if "PPG_Peaks" in ppg_signals.columns:
        info["PPG_Peaks"] = np.where(ppg_signals["PPG_Peaks"] == 1)[0]

    if static:
        # Prepare figure
        gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[1 / 3, 1 / 3, 1 / 3])
        fig = plt.figure(constrained_layout=False)

        ax0 = fig.add_subplot(gs[0, 0:2])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[:, -1])
        ax3 = fig.add_subplot(gs[1, 1])

        fig.suptitle(f"Photoplethysmogram (PPG), Subject name: {subject_name}, Trial: {trial_name} (v{attempt})", fontweight="bold")

        # Plot cleaned and raw PPG
        ax0 = _ppg_peaks_plot(
            ppg_signals["PPG_Clean"].values,
            info=info,
            sampling_rate=info["sampling_rate"],
            raw=ppg_signals["PPG_Raw"].values,
            quality=ppg_signals["PPG_Quality"].values,
            ax=ax0,
        )

        # Plot Heart Rate
        ax1 = _signal_rate_plot(
            ppg_signals["PPG_Rate"].values,
            info["PPG_Peaks"],
            sampling_rate=info["sampling_rate"],
            title=f"Heart Rate",
            ytitle="Beats per minute (bpm)",
            color="#FB661C",
            color_mean="#FBB41C",
            color_points="#FF9800",
            ax=ax1,
        )

        # Plot individual heart beats
        ax2 = ppg_segment(
            ppg_signals["PPG_Clean"].values,
            info["PPG_Peaks"],
            info["sampling_rate"],
            show="return",
            ax=ax2,
        )

        ax3.plot(
            xf[1:N//2],
            2.0/N * mag[1:N//2],
            color="red",
            linestyle="dashdot",
            label="Power Spectral Density (PSD)",
        )
        ax3.set_title(f"Coherency score: {coherence_score:.2f}, LF Peak frequency {lf_peak_freq:.2f} Hz")
        ax3.set_ylabel("Power (ms2/Hz)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.axvline(x=0.04, color="blue", linestyle="--", label="HRV Coherency interval (0.04-0.26Hz)")
        ax3.axvline(x=0.26, color="blue", linestyle="--")
        ax3.legend(loc="upper right")

    else:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: ppg_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e

        # X-axis
        x_axis = np.linspace(
            0, len(ppg_signals) / info["sampling_rate"], len(ppg_signals)
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Raw and Cleaned Signal", "Rate"),
        )

        from_idx = -3000

        # Plot cleaned and raw PPG
        fig.add_trace(
            go.Scatter(x=x_axis[from_idx:], y=ppg_signals["PPG_Raw"][from_idx:], name="Raw"), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis[from_idx:],
                y=ppg_signals["PPG_Clean"][from_idx:],
                name="Cleaned",
                marker_color="#FB1CF0",
            ),
            row=1,
            col=1,
        )

        # Plot peaks
        fig.add_trace(
            go.Scatter(
                x=x_axis[info["PPG_Peaks"]][from_idx:],
                y=ppg_signals["PPG_Clean"][info["PPG_Peaks"]][from_idx:],
                name="Peaks",
                mode="markers",
                marker_color="#D60574",
            ),
            row=1,
            col=1,
        )

        # Rate
        ppg_rate_mean = ppg_signals["PPG_Rate"].mean()
        fig.add_trace(
            go.Scatter(
                x=x_axis[from_idx:],
                y=ppg_signals["PPG_Rate"][from_idx:],
                name="Rate",
                mode="lines",
                marker_color="#FB661C",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=ppg_rate_mean[from_idx:],
            line_dash="dash",
            line_color="#FBB41C",
            name="Mean",
            row=2,
            col=1,
        )
        fig.update_layout(title_text="Photoplethysmogram (PPG)", height=500, width=750)
        if info["sampling_rate"] is not None:
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        elif info["sampling_rate"] is None:
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_xaxes(title_text="Samples", row=2, col=1)
        return fig



def analyse(filename):
    x = np.load(filename, allow_pickle=True).item()

    raw = x["data"]
    # ibi = x["ibi"]

    df, info = nk.ppg_process(raw, sampling_rate=100)

    peaks = info["PPG_Peaks"]
    ibi = (peaks[1:] - peaks[:-1]) / info["sampling_rate"]

    record_length = np.sum(ibi)  # s
    interpolation_frequency = 2  # Hz (inter-beats interval)

    xinterp = np.linspace(0, record_length, np.int64(record_length * interpolation_frequency))
    xvals = np.linspace(0, np.int64(record_length), np.size(ibi))
    ibi = np.interp(xinterp, xvals, ibi) * 1000  # mHz

    fft_dat = x["fft"]

    ibi_cleaned = np.delete(ibi, np.where(ibi > 850))
    ibi_cleaned = np.delete(ibi, np.where(ibi < 700))

    #plt.plot(ibi)
    #plt.show()

    yf = fft(ibi_cleaned)
    N = len(ibi_cleaned)
    T = 1 / 2
    xf = fftfreq(N, T)

    record_length = np.sum(ibi_cleaned)  # s

    # df = 1.0 / record_length
    # nyquist = 1.0 / (2 * T)

    lf_peak_freq = 0

    # low-frequency range 0.04–0.26 Hz
    lf_mask = (xf >= 0.04) & (xf <= 0.26)
    window = 0.03  # Hz

    mag = np.abs(yf)

    peak_freq_idx = np.argmax(mag[lf_mask])
    if np.any(lf_mask):
        lf_peak_freq = xf[lf_mask][peak_freq_idx]
    else:
        lf_peak_freq = float("nan")

    win_mask = (xf >= lf_peak_freq - window / 2) & (xf >= lf_peak_freq + window / 2)
    below_mask = (xf >= 0.04) & (xf < lf_peak_freq - window / 2)
    above_mask = (xf <= 0.26) & (xf > lf_peak_freq + window / 2)


    peak_freq_power = np.sum(mag[win_mask])
    freq_power_below = np.sum(mag[below_mask])
    freq_power_above = np.sum(mag[above_mask])
    # print(f"{peak_freq_power=}, {mag=}, {peak_freq_idx=}, {freq_power_below=}, {freq_power_above=}")
    coherence_score = -1

    if freq_power_below:
        component1 = peak_freq_power / freq_power_below
        component2 = peak_freq_power / freq_power_above

        coherence_ratio = component1 * component2

        coherence_score = np.log(coherence_ratio + 1)

        print(f"{peak_freq_power=}, {freq_power_below=}, {freq_power_above=}, {component1=}, {component2=}, {coherence_score=}, {coherence_ratio=}, {np.log(0.8 + 1)=}")


        print(f"{coherence_score=}, {lf_peak_freq=}")

    """
    plt.plot(xf[lf_mask], mag[lf_mask])
    plt.show()
    
    plt.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]))
    plt.show()
    """

    plot(df, xf=xf, mag=mag, N=N, info=info, filename=filename, coherence_score=coherence_score, lf_peak_freq=lf_peak_freq)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19, 12)
    plt.savefig(f"graphs/{filename.split('.')[0]}.png", dpi=100)


for file in glob("recording*"):
    analyse(file)
