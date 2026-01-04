from collections import defaultdict
from weakref import finalize

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from charset_normalizer.cd import coherence_ratio
from neurokit2 import NeuroKitWarning, ppg_segment
from neurokit2.ppg.ppg_peaks import _ppg_peaks_plot
from neurokit2.signal.signal_rate import _signal_rate_plot
from scipy.fft import fftfreq, fft
import neurokit2 as nk

from warnings import warn

from glob import glob
import pandas as pd


coherence_scores = {}
lf_peak_freqs = {}


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
    coherence_score = np.nan

    if freq_power_below:
        component1 = peak_freq_power / freq_power_below
        component2 = peak_freq_power / freq_power_above

        coherence_ratio = component1 * component2

        coherence_score = np.log(coherence_ratio + 1)

        # print(f"{peak_freq_power=}, {freq_power_below=}, {freq_power_above=}, {component1=}, {component2=}, {coherence_score=}, {coherence_ratio=}, {np.log(0.8 + 1)=}")

        # print(f"{coherence_score=}, {lf_peak_freq=}")

    coherence_scores[filename] = coherence_score
    lf_peak_freqs[filename] = lf_peak_freq




for file in glob("recording*"):
    analyse(file)

from pprint import pprint

pprint(coherence_scores)
pprint(lf_peak_freqs)

valid_trials = [60, 90, 120, 150]
coherence_sorted = defaultdict(dict)
lf_peak_freq_sorted = defaultdict(dict)

for k, v in coherence_scores.items():
    invalid = 1
    found = 0

    for trial in valid_trials:
        if f"{trial}bpm" in k:
            invalid = 0
            found = trial
            break

    name = k.split("_")[1]

    if invalid:
        continue

    coherence_sorted[name][found] = v

pprint(coherence_sorted)

for k, v in lf_peak_freqs.items():
    invalid = 1
    found = 0

    for trial in valid_trials:
        if f"{trial}bpm" in k:
            invalid = 0
            found = trial
            break

    name = k.split("_")[1]

    if invalid:
        continue

    lf_peak_freq_sorted[name][found] = v

pprint(lf_peak_freq_sorted)

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(1, 2)
fig.suptitle(f"Heart Rate Variation (HRV) Analysis", fontweight="bold")


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

colors = {
    "mathias": "blue",
    "khalil": "#E91E63"
}

coherence_merged_x = []
coherence_merged_y = []
for k, v in coherence_sorted.items():
    data = []

    for trial in valid_trials:
        val = np.nan
        if trial in v:
            val = v[trial]
        data.append(val)

        if not np.isnan(val):
            coherence_merged_x.append(trial)
            coherence_merged_y.append(val)

    kw = {}
    if k in colors:
        kw["color"] = colors[k]
    ax1.plot(valid_trials, data, label=k, linewidth=2, **kw)

coherence_corr_coef = scipy.stats.spearmanr(coherence_merged_x, coherence_merged_y)

ax1.set_ylabel("Coherency score")
ax1.set_xlabel("Beats per minute")
ax1.set_xticks(valid_trials)
ax1.grid(axis='x')
ax1.grid(axis='y')
ax1.set_title(f"Coherency scores per bpm (correlation: {coherence_corr_coef[0]:.2f}, p={coherence_corr_coef[1]:.2f})")

ax1.legend(loc="upper right")

freq_merged_x = []
freq_merged_y = []
for k, v in lf_peak_freq_sorted.items():
    data = []

    for trial in valid_trials:
        val = np.nan
        if trial in v:
            val = v[trial]
        data.append(val)

        if not np.isnan(val):
            freq_merged_x.append(trial)
            freq_merged_y.append(val)

    kw = {}
    if k in colors:
        kw["color"] = colors[k]
    ax2.plot(valid_trials, data, linewidth=2, label=k, **kw)

freq_corr_coef = scipy.stats.spearmanr(freq_merged_x, freq_merged_y)

ax2.set_ylabel("LF Band Peak frequency (Hz) [0.04-0.26] Hz")
ax2.set_xlabel("Beats per minute")
ax2.set_xticks(valid_trials)
ax2.grid(axis='x')
ax2.grid(axis='y')
ax2.set_title(f"Peak frequencies per bpm (correlation: {freq_corr_coef[0]:.2f}, p={freq_corr_coef[1]:.2f})")

ax2.legend(loc="upper right")

figure = plt.gcf()  # get current figure
figure.set_size_inches(14, 8)
plt.savefig(f"graphs/correlations.png", dpi=100)

plt.show()