# 🎵 Chord Detector Pro Studio

An offline-first, AI-powered **Automatic Chord Recognition, DAW MIDI Exporter & Lead Sheet Studio** built with Python, FastAPI, Librosa, and Demucs.

Designed for musicians, worship teams, guitarists, and music producers to transcribe chords from audio files or YouTube links, align chord transitions to musical time signatures (`4/4`, `3/4`, `6/8`), export multi-note MIDI tracks for DAWs (FL Studio, Ableton, Logic), and create printable lyric lead sheets with full Devanagari/Hindi & Latin typography support.

---

## ✨ Key Features

- **High-Resolution Harmonic & Bass Chroma Engine:** Combines 24-bin Constant-Q Transform (CQT) harmonic and sub-bass analysis (`C1–B8`) with automatic musical key estimation.
- **Meter-Aware Viterbi Decoding (`4/4`, `3/4`, `6/8`):** Automatically detects tempo (BPM) and time signature, or lets you manually switch meters (`4/4` ↔ `3/4` ↔ `6/8`) with confirmation to re-align chord changes onto true bar downbeats.
- **Optional Demucs AI Stem Separation:** Toggle Meta's `htdemucs` neural source separation to isolate harmonic instruments and bass from vocals and drums before detecting chords.
- **YouTube & Local Audio Support:** Analyze local `MP3`, `WAV`, `M4A`, and `FLAC` files or paste a YouTube/audio URL directly with real-time progress tracking.
- **Interactive Circle of Fifths & Instrument Voicings:** Live SVG Guitar fretboard diagrams, Piano chord voicings, Circle of Fifths key-family explorer, and instant Transpose / Capo calculator.
- **FL Studio / DAW MIDI Export (`.mid`):** One-click export of Type-0 multi-note MIDI chord tracks with embedded BPM and Time Signature metadata.
- **Dedicated Lead Sheet & Lyric Studio:** Distraction-free full-page workspace to paste lyrics (Hindi/Devanagari & English), stamp or drag chords onto exact syllables, and print/export clean PDF charts styled with **Poppins**.
- **Multiple Export Formats:** Export chord timelines as `.mid` (MIDI), `.txt`, `.csv`, or print-ready PDF chord charts.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Audio DSP & MIR:** Librosa, NumPy, SoundFile, Demucs (`htdemucs`), `yt-dlp`, FFmpeg
- **Frontend:** Vanilla JavaScript (60fps `requestAnimationFrame` audio sync), HTML5, CSS3, Dynamic SVG

---

## 📂 Project Structure

```text
ChordDetector/
├── app.py              # Complete FastAPI backend & interactive frontend UI
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes venv, cache, and local audio files
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

---

## 🚀 Quick Start (Local Installation)

### 1. Prerequisites
- **Python 3.10 or 3.11**
- **FFmpeg** installed and added to your system PATH:
  - **Windows:**
    ```bash
    winget install Gyan.FFmpeg
    ```
  - **macOS:**
    ```bash
    brew install ffmpeg
    ```
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt install ffmpeg
    ```

### 2. Clone & Install Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/ChordDetector.git](https://github.com/YOUR_USERNAME/ChordDetector.git)
cd ChordDetector

# Create and activate virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Install core packages
pip install -r requirements.txt
```

*(Optional)* To enable neural stem separation (`Demucs stems` toggle):
```bash
pip install demucs
```

### 3. Run the Application
```bash
python app.py
```
The server will start at `http://127.0.0.1:8000` and automatically open in your default browser.

---

## 📱 Run on Mobile / Tablet (Same Wi-Fi)

To access the studio from an iPad, Android tablet, or phone on the same Wi-Fi network:
1. In `app.py`, change `host="127.0.0.1"` to `host="0.0.0.0"` in the `uvicorn.run(...)` line at the bottom.
2. Find your PC's local IP address (`ipconfig` on Windows or `ifconfig` on macOS/Linux).
3. Open `http://<YOUR_LOCAL_IP>:8000` in your mobile/tablet browser.

---

## 🎹 Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `Space` | Play / Pause |
| `←` / `→` | Seek `-5s` / `+5s` |
| `↑` / `↓` | Transpose `+1` / `-1` semitone |
| `Double-Click Block` | Edit single chord or replace all occurrences in song |
| `Esc` | Close active modal |

---

## 🤝 Contributing

Contributions, feature requests, and pull requests are welcome! Feel free to open an issue if you have ideas for extended chord voicings (`7th`, `maj7`, `sus4`, slash chords), A-B section looping, or live microphone chord detection.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.