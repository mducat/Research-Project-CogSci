import os
import time

from PyQt5.QtCore import pyqtSlot, QThreadPool, QTimer, Qt
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QApplication, QHBoxLayout,
)

from scipy.fft import fft, fftfreq

import pyqtgraph as pg
import numpy as np


import platform
import sys


osDic = {
    "Darwin": f"MacOS/Intel{''.join(platform.python_version().split('.')[:2])}",
    "Linux": "Linux64",
    "Windows": f"Win{platform.architecture()[0][:2]}_{''.join(platform.python_version().split('.')[:2])}",
}
if platform.mac_ver()[0] != "":
    import subprocess
    from os import linesep

    p = subprocess.Popen("sw_vers", stdout=subprocess.PIPE)
    result = p.communicate()[0].decode("utf-8").split(str("\t"))[2].split(linesep)[0]
    if result.startswith("12."):
        print("macOS version is Monterrey!")
        osDic["Darwin"] = "MacOS/Intel310"
        if (
            int(platform.python_version().split(".")[0]) <= 3
            and int(platform.python_version().split(".")[1]) < 10
        ):
            print(f"Python version required is ≥ 3.10. Installed is {platform.python_version()}")
            exit()


sys.path.append(f"PLUX-API-Python3/{osDic[platform.system()]}")

# CONSTANTS


recording_frequency = 100  # Hz
recording_duration = 128 + 10  # 60s margin for errors

bypass_outstanding_inter_beat_duration_sort = False
local_maximum_window = True

# ECG params
# local_maximum_window_n_size = 2000
# spike_threshold_percentile = 98

# PPG params
local_maximum_window_n_size = 2000
spike_threshold_percentile = 85

interpolation_frequency = 2  # Hz (inter-beats interval)

subject_name = 'khalil'
current_trial = '150bpm'


import plux

global raw_recorded_data
global recorded_data
global inter_spike_duration
raw_recorded_data = np.array([])
recorded_data = np.array([])
inter_spike_duration = np.array([])

class NewDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.SignalsDev.__init__(address)
        self.duration = 0
        self.frequency = 0
        self.last_spike_time = -1
        self.during_spike = False

    def onRawFrame(self, nSeq, data):
        global raw_recorded_data
        global recorded_data
        global inter_spike_duration

        print(data)
        level = int(data[0])
        raw_recorded_data = np.append(raw_recorded_data, np.array([level]))

        # no raw process
        recorded_data = raw_recorded_data

        # override
        spike_threshold_bin = recorded_data
        if local_maximum_window:
            idx = max(len(spike_threshold_bin) - local_maximum_window_n_size, 0)
            spike_threshold_bin = spike_threshold_bin[idx:]

        spike_threshold = np.percentile(spike_threshold_bin, spike_threshold_percentile)

        if not self.during_spike and level >= spike_threshold:
            self.during_spike = True

            if self.last_spike_time > 0:
                time_unit = 1 / self.frequency
                duration = time_unit * (nSeq - self.last_spike_time)

                if len(inter_spike_duration) > 4:
                    low = np.percentile(inter_spike_duration, 1) * 1.5
                    high = np.percentile(inter_spike_duration, 99) * 1.5
                else:
                    low = 0
                    high = int(1e7)

                if low < duration < high or bypass_outstanding_inter_beat_duration_sort:  # discard otherwise

                    inter_spike_duration = np.append(inter_spike_duration, np.array([duration]))
                    if np.sum(inter_spike_duration) > recording_duration * 1000:
                        inter_spike_duration = inter_spike_duration[1:]

            self.last_spike_time = nSeq

        if self.during_spike and level < spike_threshold:
            self.during_spike = False

        if nSeq % 200 == 0:
            print(nSeq, *data)

        if len(raw_recorded_data) > 10000:
            raw_recorded_data = raw_recorded_data[1:]

        return nSeq > self.duration * self.frequency


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFixedSize(800, 600)
        self.setWindowTitle("Biofeedback")
        self.device = None

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.recording = False

        self.layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.thread_manager = QThreadPool()
        self.button_layout = QHBoxLayout()

        self.start_ack = QPushButton("Start acquisition")
        self.stop_ack  = QPushButton("Stop acquisition")

        self.hr_label = QLabel("HR: -- bpm, coherency score: --", alignment=Qt.AlignCenter)

        self.button_layout.addWidget(self.start_ack)
        self.button_layout.addWidget(self.stop_ack)
        self.button_layout.addWidget(self.hr_label)

        self.plot_graph = pg.PlotWidget(title="Raw PPG/ECG")
        self.interval_graph = pg.PlotWidget(title="Inter-beat intervals (IBIs)")
        self.fft_graph = pg.PlotWidget(title="HRV spectrum (LF band)")

        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.plot_graph)
        self.layout.addWidget(self.interval_graph)
        self.layout.addWidget(self.fft_graph)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.timer.timeout.connect(self.update_draw)
        self.start_ack.pressed.connect(self.start_ack_threaded)
        self.stop_ack.pressed.connect(self.stop_recording)

        self.plot_line = None
        self.interval_line = None
        self.fft_line = None

        self.data = None
        self.ibi = None
        self.fft = None

    @pyqtSlot()
    def update_draw(self):
        if self.plot_line:
            self.plot_line.clear()

        if self.interval_line:
            self.interval_line.clear()

        if self.fft_line:
            self.fft_line.clear()

        self.data = recorded_data
        self.plot_line = self.plot_graph.plot(recorded_data[-1000:])

        """try:
            data = biosppy.signals.ppg.ppg(recorded_data, show=False)
        except Exception as e:
            return

        filtered = data["filtered"]
        heart_rate = data["heart_rate"]
        peaks = data["peaks"]

        if len(heart_rate):
            hr = np.mean(heart_rate[:-5])  # last 5 values
            self.hr_label.setText(f"HR: {hr:.1f} bpm")

        print(heart_rate)

        self.plot_line = self.plot_graph.plot(filtered)

        print(peaks)

        ibi = (peaks[1:] - peaks[:1]) / recording_frequency  # s

        # interpolate

        record_length = np.sum(ibi)  # s

        xinterp = np.linspace(0, record_length, np.int64(record_length * interpolation_frequency))
        xvals = np.linspace(0, np.int64(record_length), np.size(ibi))
        inter_spike_duration_norm = np.interp(xinterp, xvals, ibi) * 1000

        self.interval_line = self.interval_graph.plot(inter_spike_duration_norm)

        # fourier

        yf = fft(inter_spike_duration_norm)
        N = len(inter_spike_duration_norm)
        T = 1 / interpolation_frequency
        xf = fftfreq(N, T)[:N//2]

        self.fft_line = self.fft_graph.plot(xf[1:], 2.0/N * np.abs(yf[1:N//2]))"""

        if len(inter_spike_duration) < 10:
            return

        # interpolate

        record_length = np.sum(inter_spike_duration)  # s

        xinterp = np.linspace(0, record_length, np.int64(record_length * interpolation_frequency))
        xvals = np.linspace(0, np.int64(record_length), np.size(inter_spike_duration))
        inter_spike_duration_norm = np.interp(xinterp, xvals, inter_spike_duration) * 1000

        self.interval_line = self.interval_graph.plot(inter_spike_duration_norm)
        self.ibi = inter_spike_duration_norm

        # fourier

        yf = fft(inter_spike_duration_norm)
        N = len(inter_spike_duration_norm)
        T = 1 / interpolation_frequency
        xf = fftfreq(N, T)

        self.fft = [xf, yf]

        self.fft_line = self.fft_graph.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]))

        df = 1.0 / record_length
        nyquist = 1.0 / (2 * T)

        lf_peak_freq = 0

        # low-frequency range 0.05–0.25 Hz
        lf_mask = (xf >= 0.05) & (xf <= 0.25)

        mag = np.abs(yf)

        # Compute HR from last 5 beats
        if np.sum(mag[lf_mask]) > 0:
            last5 = inter_spike_duration[-5:]
            avg_ibi = np.mean(last5)
            hr = 60.0 / avg_ibi if avg_ibi > 0 else 0

            peak_freq_idx = np.argmax(mag[lf_mask])
            if np.any(lf_mask):
                lf_peak_freq = xf[lf_mask][peak_freq_idx]
            else:
                lf_peak_freq = float("nan")

            peak_freq_power = mag[lf_mask][peak_freq_idx]
            freq_power_below = np.sum(mag[lf_mask][:peak_freq_idx])
            freq_power_above = np.sum(mag[lf_mask][peak_freq_idx:])
            # print(f"{peak_freq_power=}, {mag=}, {peak_freq_idx=}, {freq_power_below=}, {freq_power_above=}")
            coherence_score = -1

            if freq_power_below:

                component1 = peak_freq_power / freq_power_below
                component2 = peak_freq_power / freq_power_above

                coherence_score = np.log(component1 * component2 + 1)

            # @todo coherence score
            # CR = (Peak Power/Total power below the peak frequency)*(Peak Power/Total power above peak frequency)

            self.hr_label.setText(f"HR: {hr:.1f} bpm, coherency score: {coherence_score:.2f}")

        else:
            self.hr_label.setText("HR: -- bpm, coherency score: --")

        # print(f"{N=}, {interpolation_frequency=} Hz, {record_length=:.1f} s")
        # print(f"frequency bin (df) = {df:.5f} Hz, {nyquist=:.2f} Hz, {lf_peak_freq=:.2f} Hz")

    @pyqtSlot()
    def start_ack_threaded(self):
        if self.recording:
            return
        self.thread_manager.start(self.start_acquisition)
        self.timer.start()

    @pyqtSlot()
    def stop_recording(self):
        if not self.recording:
            return
        self.device.stop()
        self.device.close()
        self.recording = False

        self.save_data()

    @pyqtSlot()
    def start_acquisition(self):
        self.recording = True

        address = "BTH98:D3:41:FE:16:37"
        active_ports = [1]

        try:
            self.device = NewDevice(address)
            self.device.duration = int(recording_duration)
            self.device.frequency = int(recording_frequency)

            self.device.start(self.device.frequency, active_ports, 16)
            self.device.loop()
        except Exception as ex:
            print(ex, file=sys.stderr)

        self.stop_recording()

    def save_data(self):
        version = 0
        file_name = f'recording_{subject_name}_{current_trial}_{version}.npy'

        while os.path.exists(file_name):
            version += 1
            file_name = f'recording_{subject_name}_{current_trial}_{version}.npy'

        obj = {
            "data": self.data,
            "ibi": self.ibi,
            "fft": self.fft
        }
        np.save(file_name, obj)


if __name__ == "__main__":
    app = QApplication([])

    main_window = MainWindow()
    main_window.show()

    app.exec()
