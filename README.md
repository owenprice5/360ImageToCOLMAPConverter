# Metashape → COLMAP Converter

**v0.29** · Windows · Python 3.10+

A GUI tool for converting Agisoft Metashape panoramic camera projects into COLMAP format for 3D Gaussian Splatting training. Supports GPU-accelerated equirectangular → perspective projection, job queuing, and a fully automated COLMAP rig pipeline for users without Metashape.

---

## Features

- **Metashape → COLMAP** — converts Metashape camera XML exports (spherical + frame cameras) into a complete COLMAP sparse reconstruction folder (`sparse/0/`, `images/`, `points3D.txt`)
- **GPU acceleration** — uses CuPy/CUDA for accelerated conversion on a modern GPU; falls back to CPU multiprocessing automatically
- **Equirectangular → perspective** — projects 360° panoramas into cubemap faces (front, back, left, right, up, down) plus optional diagonal and 45° views
- **COLMAP Rig pipeline** — for users without Metashape: converts raw equirectangular images and runs the full COLMAP feature extraction → rig configuration → matching → mapping pipeline automatically, with known camera intrinsics locked in
- **Job queue** — add multiple projects, run overnight, save/load queue to JSON
- **Post-conversion launchers** — automatically opens Lichtfeld Studio, Brush, or runs COLMAP `model_converter` (txt → bin) after each conversion
- **Preview & approve** — view rendered perspective crops before converting, toggle individual view directions, set yaw exclusion zones
- **Export toggles** — optionally skip cameras, images, or point cloud on re-runs
- **Auto-fill** — pick a project root folder and all paths fill automatically
- **Persistent settings** — launcher paths and toggles saved across sessions

---

## Requirements

### Required
- Python 3.10+ (other versions may work but was intended for Python 3.10. Other versions may conflict with numpy/Pillow)
- [numpy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)

### Optional
- [CuPy](https://cupy.dev/) (`cupy-cuda12x`) — GPU acceleration (NVIDIA CUDA 12.x)
- [COLMAP](https://colmap.github.io/) — for txt → bin conversion and the COLMAP Rig pipeline
- [Lichtfeld Studio](https://lichtfeld.studio/) — Gaussian Splatting viewer/trainer
- [Brush](https://github.com/ArthurBrussee/brush) — Gaussian Splatting viewer/trainer

All required Python packages can be installed from the **Dependencies** tab inside the app.

---

## Installation

ALL DEPENDENCIES CAN BE INSTALLED DIRECLTY FROM THE SCRIPT - No need for any coding.

If you want to install manually then here's how you do it.

```bash
# Clone or download the repository
git clone https://github.com/yourusername/metashape-colmap-converter.git
cd metashape-colmap-converter

# Install required dependencies
pip install numpy Pillow

# Optional: GPU acceleration (CUDA 12.x)
pip install cupy-cuda12x
```

Then run:
```bash
python import_47.py
```

No build step required. The app opens directly. Additionally, double clicking the python file will open it.

---

## Usage

### Metashape workflow

1. Export cameras from Metashape: `File → Export → Export Cameras` (save as XML)
2. Export sparse point cloud as .ply
3. Open the app and go to the **Metashape** tab
4. Either pick a **Project Root Folder** (auto-fills all paths) or set paths manually:
   - **Cameras XML** — Metashape camera export
   - **Spherical XML** — optional separate spherical camera export
   - **Images folder** — folder containing equirectangular panoramas
   - **Point Cloud** — optional `.ply` for `points3D.txt`
   - **Output folder** — where COLMAP output will be written
5. Configure view directions in **Views & Export** tab
6. Click **Convert →** or **+ Add to Queue**

Output structure:
```
output_001/
├── images/          ← rendered perspective crops
├── sparse/
│   └── 0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
```

### COLMAP Rig workflow (no Metashape)

For users with a 360° camera but no Metashape licence:

1. Go to the **COLMAP Rig** tab
2. Set your **Panoramas** folder (equirectangular `.jpg`/`.png` images)
3. Set an **Output** folder
4. Tick the pipeline steps to run (all enabled by default)
5. Choose a matcher (Sequential for walkthroughs, Exhaustive for small sets)
6. Click **▶ Run COLMAP Pipeline**

The script will:
- Convert panoramas to perspective crops using your Views & Export settings
- Write a `rig_config.json` with known camera rotations
- Run feature extraction with locked intrinsics (no focal length guessing)
- Run rig configurator (groups all crops from each panorama as one capture point)
- Run feature matching
- Run mapper with fixed intrinsics

Output is compatible with Lichtfeld Studio and Brush directly.

---

## Tabs

| Tab | Description |
|-----|-------------|
| **Metashape** | Main conversion — Metashape XML → COLMAP |
| **COLMAP Rig** | Automated pipeline for 360° cameras without Metashape |
| **Views & Export** | Camera rig configuration, FOV, output resolution, flip axes |
| **Performance** | Worker count, GPU batch size, tile height |
| **Queue** | Multi-project job queue with save/load |
| **Launchers** | Configure COLMAP, Lichtfeld Studio, and Brush for auto-launch |
| **Dependencies** | Install/update required Python packages with auto-restart |

---

## GPU Acceleration

With CuPy installed and a CUDA-capable GPU, the converter uses a pipelined GPU renderer:

```
Loader thread → GPU render thread → Writer threads
```

> **Note:** For best performance, write output to an SSD. HDD write speed can become the bottleneck at high GPU throughput.

---

## Troubleshooting

**Script opens but GPU not detected**
- Install CuPy: `pip install cupy-cuda12x`
- Check CUDA version matches: `nvcc --version`

**Conversion is slow / stops frequently**
- Write output to an SSD rather than HDD
- If using HDD, the GPU may outpace disk write speed bottlenecking performance
- Check Windows Defender isn't scanning output files in real-time (add output folder to exclusions)
- If consistently writing files, defragmenting the HDD will improve performance. Writing may start and stop if this is not done. (Only occurs when potentially writing  500,000 plus images in a short time span)

**Windows Security blocks CuPy**
- Go to Windows Security → Virus & threat protection → Exclusions
- Add your Python site-packages folder

**COLMAP returns code 255**
- Ensure COLMAP `.bat` or `.exe` path is set correctly in the Launchers tab
- Paths with spaces are handled automatically

**Multiple sparse model folders created**
- This happens when COLMAP can't connect all images into one reconstruction
- Ensure camera intrinsics are locked (`--Mapper.ba_refine_focal_length 0`) — this is done automatically by the COLMAP Rig pipeline
