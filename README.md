# Heart Rate Variation & Coherence Analyzer

A Python-based toolkit for computing **Heart Rate Variation (HRV)** and **coherence scores** in real time using a **BITalino** device reading **PPG signals**, and performing **offline analysis** with customizable graphs.

The project supports:
- Live BITalino streaming from the PPG channel
- Real-time heart rate and coherence computation
- Signal preprocessing + peak detection
- HRV metrics extraction (time-domain + frequency-domain)
- Recording sessions to disk
- Offline analysis with visualizations

---

## 🚀 Features

### **Live Mode**
- Connects to BITalino device via Bluetooth
- Reads PPG signal from analog channel
- Filters raw signal (band-pass / detrend / smoothing)
- Detects heartbeats & extracts IBI (Inter-Beat Interval)
- Computes:
  - Heart Rate (HR)
  - HRV metrics (SDNN, RMSSD, etc.)
  - Coherence score (PSD-based, LF/HF may be included in future works)
- Displays live plots while recording (optional)
- Stores session data under "recoding_{subject}\_{session}\_{version}.npy" file

### **Offline Analysis**
- Loads recorded `.npy` session data
- Recomputes HRV metrics
- Generates visualizations such as:
  - PPG waveform
  - IBI / tachogram
  - RR interval histogram
  - HR over time
  - Power spectral density (PSD) + Coherency score
- Saves graphs as `.png`

---

## 🛠️ Requirements

### **Hardware**
- BITalino (e.g., BITalino (r)evolution Board)
- PPG sensor module
- Bluetooth or USB connectivity

### **Software**
- Python ≥ 3.9

### **Python Packages**
Core dependencies may include:

```bash
numpy
scipy
matplotlib
bitalino
neurokit
pyqtgraph (optional for live display)
````

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mducat/Research-Project-CogSci.git
cd Research-Project-CogSci
pip install -r requirements.txt
```

---

## ▶️ Usage

### **1. Live Mode**

Run the live acquisition and processing:

```bash
python biofeedback_nk.py
```

Parameters are hard-coded in the biofeedback_nk.py file (examples):

* `device_address` : BITalino MAC address
* `recording_frequency` : BITalino sampling rate (100, 1000 Hz)
* `recording_duration` : Duration of the recording session (in seconds)


---

### **2. Offline Analysis**

After recording a session, analyze it via:

```bash
python analyze.py
```

This script takes no argument. It will scan for .npy files in the "data" folder, and generates graphs accordingly.

---

## 📊 Example Outputs

Typical generated graphs may include:

* **Raw PPG waveform**
* **Filtered PPG waveform**
* **Detected peaks**
* **IBI series (tachogram)**
* **HR over time**
* **PSD (LF/HF power bands)**
* **Coherence score timeline**

---

## 📁 Project Structure

Example layout:

```
Research-Project-CogSci/
│
├─ biofeedback_nk.py        # Live acquisition & coherence computation
├─ analyze.py               # Offline data analysis
├─ correlation.py           # Correlations analysis for coherence & peak frequencies
├─ PLUX-API-Python3/        # Shared libraries to communicate with BITalino
├─ deprecated/              # Former versions of the live acquisition script
├─ data/                    # Recorded sessions
├─ graphs/                  # Output graphs
├─ requirements.txt
└─ README.md
```

---

## 📚 Coherence Metric Notes

Coherence is computed using power spectral density (PSD) in **Low Frequency (LF)** and **High Frequency (HF)** bands:

* LF band: 0.04–0.26 Hz
* HF band: 0.26–0.50 Hz

Coherence score method:

**Peak narrow-band LF power / total power**

---


## 📝 Disclaimer

This project is intended for **educational and research** purposes and **not** for medical diagnostics.
