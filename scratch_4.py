from PyQt5.QtCore import pyqtSlot, QThreadPool, QTimer, Qt
from PyQt5.QtWidgets import (
    QLabel, QWidget, QMainWindow, QPushButton,
    QVBoxLayout, QApplication
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
import plux

# ============================================================
#                    CONSTANTS
# ============================================================

recording_frequency = 100         # Hz sampling rate
ACQUISITION_DURATION = 120        # total seconds
interpolation_frequency = 2       # Hz for interpolated IBI signal

# *** R-peak detection threshold (percentile of amplitude) ***
# You can tweak this (e.g. 70, 75, 80, 85) to change sensitivity.
spike_threshold_percentile = 75   # was 85 before

# ============================================================
#               GLOBAL DATA ARRAYS
# ============================================================

raw_recorded_data = np.array([])
raw_timestamps = np.array([])
recorded_data = np.array([])

inter_spike_duration = np.array([])   # IBI durations (seconds)
rpeak_timestamps = np.array([])       # timestamps of detected R-peaks

# ============================================================
#                 DEVICE CLASS
# ============================================================

class NewDevice(plux.SignalsDev):
    def __init__(self, address):
        # NOTE: constructor kept exactly as in the original working code
        plux.SignalsDev.__init__(address)
        self.duration = 0
        self.frequency = 0
        self.last_spike_time = -1
        self.during_spike = False
        self.running = True

    def onRawFrame(self, nSeq, data):
        global raw_recorded_data, raw_timestamps, recorded_data
        global inter_spike_duration, rpeak_timestamps

        if not self.running:
            return True

        level = int(data[0])

        # Timestamp for this sample
        t_sec = nSeq / self.frequency if self.frequency > 0 else 0

        # Store raw PPG/ECG
        raw_recorded_data = np.append(raw_recorded_data, level)
        raw_timestamps = np.append(raw_timestamps, t_sec)
        recorded_data = raw_recorded_data

        # Sliding window (reduces UI lag)
        if len(raw_recorded_data) > 2000:   # keep ~20s at 100 Hz
            raw_recorded_data = raw_recorded_data[1:]
            raw_timestamps = raw_timestamps[1:]
            recorded_data = raw_recorded_data

        # ---------------- R-PEAK / PPG PEAK DETECTION ----------------
        # Threshold based on percentile of the *current* window
        if len(recorded_data) > 20:  # avoid percentile on tiny arrays
            spike_threshold = np.percentile(recorded_data, spike_threshold_percentile)
        else:
            spike_threshold = np.max(recorded_data)

        if not self.during_spike and level >= spike_threshold:
            # Entering spike
            self.during_spike = True

            if self.last_spike_time > 0:
                ibi = (nSeq - self.last_spike_time) / self.frequency  # seconds

                # Very simple artifact rejection:
                # For HR between 30–240 bpm -> IBI between 0.25–2.0 s
                if 0.25 < ibi < 2.0:
                    inter_spike_duration = np.append(inter_spike_duration, ibi)
                    rpeak_timestamps = np.append(rpeak_timestamps, t_sec)

            self.last_spike_time = nSeq

        if self.during_spike and level < spike_threshold:
            self.during_spike = False

        # Occasional debug print
        if nSeq % 500 == 0:
            print(f"DEBUG nSeq={nSeq}, level={level}")

        # Stop automatically after duration
        return nSeq > self.duration * self.frequency


# ============================================================
#                   GUI + ACQUISITION APP
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setFixedSize(900, 750)
        self.setWindowTitle("Biofeedback – BITalino HRV")

        # Timers
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(100)

        self.countdown_timer = QTimer()
        self.countdown_timer.setInterval(1000)

        self.recording = False
        self.device = None
        self.remaining_secs = 0

        # Layout + widgets
        self.layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.thread_manager = QThreadPool()

        self.start_ack = QPushButton("Start acquisition")
        self.stop_ack  = QPushButton("Stop acquisition")

        self.countdown_label = QLabel("Ready", alignment=Qt.AlignCenter)
        self.hr_label = QLabel("HR: -- bpm", alignment=Qt.AlignCenter)

        # Plots
        self.plot_graph = pg.PlotWidget(title="Raw PPG/ECG")
        self.interval_graph = pg.PlotWidget(title="Inter-beat intervals (IBIs)")
        self.fft_graph = pg.PlotWidget(title="HRV spectrum (LF band)")

        # UI assembly
        self.layout.addWidget(self.start_ack)
        self.layout.addWidget(self.stop_ack)
        self.layout.addWidget(self.countdown_label)
        self.layout.addWidget(self.hr_label)
        self.layout.addWidget(self.plot_graph)
        self.layout.addWidget(self.interval_graph)
        self.layout.addWidget(self.fft_graph)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        # Connections
        self.start_ack.clicked.connect(self.start_ack_threaded)
        self.stop_ack.clicked.connect(self.stop_acquisition)

        self.plot_timer.timeout.connect(self.update_plots)
        self.countdown_timer.timeout.connect(self.update_countdown)

        self.plot_line = None
        self.interval_line = None
        self.fft_line = None

    # ============================================================
    #                    GUI UPDATES
    # ============================================================

    @pyqtSlot()
    def update_plots(self):
        global recorded_data, inter_spike_duration

        # Clear previous
        if self.plot_line: self.plot_line.clear()
        if self.interval_line: self.interval_line.clear()
        if self.fft_line: self.fft_line.clear()

        # Raw signal plot
        self.plot_line = self.plot_graph.plot(recorded_data)

        # Compute HR from last 5 beats
        if len(inter_spike_duration) >= 5:
            last5 = inter_spike_duration[-5:]
            avg_ibi = np.mean(last5)
            hr = 60.0 / avg_ibi if avg_ibi > 0 else 0
            self.hr_label.setText(f"HR: {hr:.1f} bpm")
        else:
            self.hr_label.setText("HR: -- bpm")

        # IBIs + FFT
        if len(inter_spike_duration) > 3:
            ibi = inter_spike_duration * 1000  # ms
            self.interval_line = self.interval_graph.plot(ibi)

            # FFT processing
            total = np.sum(inter_spike_duration)
            if total > 1.0:
                xinterp = np.linspace(0, total, int(total * interpolation_frequency))
                xvals = np.linspace(0, len(inter_spike_duration), len(inter_spike_duration))
                ibi_interp = np.interp(xinterp, xvals, inter_spike_duration) * 1000

                yf = fft(ibi_interp)
                N = len(ibi_interp)
                T = 1 / interpolation_frequency
                xf = fftfreq(N, T)[:N // 2]

                self.fft_line = self.fft_graph.plot(
                    xf[1:], (2.0 / N) * np.abs(yf[1:N // 2])
                )

    # --------------------------------------------------------

    @pyqtSlot()
    def update_countdown(self):
        if not self.recording:
            return

        self.remaining_secs -= 1
        self.countdown_label.setText(f"{self.remaining_secs} s")

        if self.remaining_secs <= 0:
            self.stop_acquisition()

    # ============================================================
    #          ACQUISITION CONTROL
    # ============================================================

    def reset_data(self):
        global raw_recorded_data, raw_timestamps
        global recorded_data, inter_spike_duration, rpeak_timestamps

        raw_recorded_data = np.array([])
        raw_timestamps = np.array([])
        recorded_data = np.array([])
        inter_spike_duration = np.array([])
        rpeak_timestamps = np.array([])

    @pyqtSlot()
    def start_ack_threaded(self):
        if self.recording:
            print("Already recording")
            return

        print("Starting acquisition…")

        self.reset_data()

        self.remaining_secs = ACQUISITION_DURATION
        self.countdown_label.setText(f"{self.remaining_secs} s")

        self.recording = True
        self.plot_timer.start()
        self.countdown_timer.start()

        self.thread_manager.start(self.start_acquisition)

    # --------------------------------------------------------

    @pyqtSlot()
    def start_acquisition(self):
        address = "BTH98:D3:41:FE:16:37"
        active_ports = [1]   # A1

        try:
            self.device = NewDevice(address)
            self.device.duration = ACQUISITION_DURATION
            self.device.frequency = recording_frequency

            print("Connecting to BITalino…")
            self.device.start(self.device.frequency, active_ports, 16)
            self.device.loop()

        except Exception as ex:
            print("Acquisition error:", ex)

        finally:
            if self.device:
                try:
                    self.device.stop()
                    self.device.close()
                except:
                    pass

            self.recording = False
            self.plot_timer.stop()
            self.countdown_timer.stop()

            print("\n=== Acquisition Complete ===\n")
            self.print_summary()

    # --------------------------------------------------------

    @pyqtSlot()
    def stop_acquisition(self):
        print("Stop requested.")
        self.recording = False

        if self.device:
            self.device.running = False
            try:
                self.device.stop()
                self.device.close()
            except:
                pass

        self.plot_timer.stop()
        self.countdown_timer.stop()
        self.countdown_label.setText("Stopped")

        print("\n=== Manual Stop: computing HRV summary ===\n")
        self.print_summary()

    # ============================================================
    #                 HRV SUMMARY OUTPUT
    # ============================================================

    def print_summary(self):
        global inter_spike_duration, rpeak_timestamps

        print("\n========== HRV SUMMARY ==========")

        if len(inter_spike_duration) < 2:
            print("Not enough IBIs detected.\n")
            return

        ibis = inter_spike_duration
        mean_ibi = np.mean(ibis)
        mean_hr = 60.0 / mean_ibi
        sdnn = np.std(ibis * 1000)

        # FFT to find dominant frequency
        total = np.sum(ibis)
        xinterp = np.linspace(0, total, int(total * interpolation_frequency))
        xvals = np.linspace(0, len(ibis), len(ibis))
        ibi_interp = np.interp(xinterp, xvals, ibis)
        yf = fft(ibi_interp)
        N = len(ibi_interp)
        xf = fftfreq(N, 1 / interpolation_frequency)
        mag = np.abs(yf)

        # low-frequency range 0.04–0.15 Hz
        lf_mask = (xf >= 0.04) & (xf <= 0.15)
        if np.any(lf_mask):
            lf_peak_freq = xf[lf_mask][np.argmax(mag[lf_mask])]
        else:
            lf_peak_freq = float("nan")

        print(f"1. Mean Heart Rate: {mean_hr:.2f} bpm")
        print(f"2. SDNN: {sdnn:.2f} ms")
        print(f"3. Signature Frequency (0.04–0.15 Hz): {lf_peak_freq:.3f} Hz")
        print(f"4. Total Beats Detected: {len(ibis)}")
        print("=================================\n")


# ============================================================
#                       RUN APP
# ============================================================

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()