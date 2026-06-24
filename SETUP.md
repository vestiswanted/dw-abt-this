# Rofus AMD Edition – Setup Guide

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU | AMD RX 6600 XT (or any DirectX 12 GPU) |
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10.x |
| RAM | 8 GB |

---

## Step 1 – Install Python 3.10

Download from https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe

During install:
- **Tick "Add Python to PATH"**
- Click "Disable path length limit" if prompted

---

## Step 2 – Install Git (needed for BetterCam)

Download from https://git-scm.com/download/win and install with default settings.

---

## Step 3 – Run the installer

Double-click **`install.bat`**.  It will:

1. Upgrade pip
2. Install `onnxruntime-directml`, OpenCV, PyQt5, pynput, pywin32, etc.
3. Install BetterCam from GitHub
4. Create `lib/config/` directory

---

## Step 4 – Export the model (one-time)

Copy your existing `best.pt` into the `lib/` folder, then:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python export_model.py --pt lib/best.pt --out lib/best.onnx --size 640
```

After export succeeds you can free disk space by uninstalling torch:
```bat
pip uninstall torch torchvision -y
```

The ONNX model at `lib/best.onnx` is what the aimbot uses at runtime — no
internet download, no torch dependency.

---

## Step 5 – Launch

Double-click **`start.bat`** (or `python main.py`).

On first run you will be asked to enter your in-game sensitivities.  These are
saved to `lib/config/config.json`.  To re-run setup: `python main.py --setup`

---

## Keybinds

| Key | Action |
|-----|--------|
| Hold Right Mouse (default) | Activate aiming |
| F1 | Toggle aimbot on/off |
| F2 | Quit |

To change the aim keybind, open `config.py` and edit:
```python
keybind: int = 0x02   # 0x02 = right mouse button
```
Key codes: https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

---

## Tuning

All settings are in **`config.py`**:

```python
# Detection area size in pixels (square centred on screen)
detection_box_size: int = 400

# Mouse movement smoothness (lower = snappier, higher = smoother)
pixel_increment: float = 0.5
mouse_delay:     float = 0.0003

# Model confidence threshold (0–1)
confidence_thresh: float = 0.50

# Frame rate cap for screen capture
target_fps: int = 144
```

---

## Provider Selection

At startup the console prints which execution provider was selected:

```
[INFO] Provider : DmlExecutionProvider      ← AMD GPU (best)
[INFO] Provider : CUDAExecutionProvider     ← NVIDIA GPU
[INFO] Provider : CPUExecutionProvider      ← CPU fallback
```

The priority order is set in `config.py` under `InferenceConfig.provider_priority`.

---

## Troubleshooting

**`FileNotFoundError: lib/best.onnx`**  
Run `python export_model.py` first (see Step 4).

**`ImportError: onnxruntime`**  
Run `pip install onnxruntime-directml`.

**`ImportError: bettercam`**  
Run `pip install git+https://github.com/RootKit-Org/BetterCam.git` (needs git).

**Low FPS / CPU only**  
Confirm DirectX 12 is supported by your GPU (Device Manager → Display Adapters).
Ensure `onnxruntime-directml` is installed, *not* plain `onnxruntime`.

**Overlay not visible**  
PyQt5 must be installed.  Run `pip install PyQt5`.
