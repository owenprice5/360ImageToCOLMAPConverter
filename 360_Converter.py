from __future__ import annotations
import os
import shutil
import re
import math
import struct
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import time
import queue
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
# Third-party imports — loaded safely so the UI can open even if missing
try:
    import numpy as np
    _numpy_ok = True
except ImportError:
    np = None
    _numpy_ok = False

try:
    from PIL import Image, ImageTk, ImageEnhance
    _pillow_ok = True
except ImportError:
    Image = ImageTk = ImageEnhance = None
    _pillow_ok = False

try:
    import cupy as cp
    _cupy_ok = True
except Exception:
    cp = None
    _cupy_ok = False
# -----------------------------
# Robust float parsing
# -----------------------------
FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
def parse_transform_4x4(text: str) -> np.ndarray:
    nums = re.findall(FLOAT_RE, text or "")
    if len(nums) < 16:
        raise ValueError(f"Transform has fewer than 16 numbers ({len(nums)})")
    nums = nums[:16]
    return np.array(list(map(float, nums)), dtype=np.float64).reshape((4, 4))
def rotmat_to_quat_wxyz(R: np.ndarray):
    """Convert rotation matrix to normalized quaternion (w,x,y,z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q.tolist()
def safe_basename(s: str) -> str:
    return os.path.basename((s or "").replace("\\", "/"))
def build_image_index(root_dir: str, exts=None):
    """Recursively index images under root_dir by lowercase filename and stem.
    Returns (by_name, by_stem) mapping to list of full paths.
    """
    if exts is None:
        exts = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp','.exr'}
    by_name = {}
    by_stem = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext and ext not in exts:
                continue
            full = os.path.join(dirpath, fn)
            name_l = fn.lower()
            stem_l = os.path.splitext(fn)[0].lower()
            by_name.setdefault(name_l, []).append(full)
            by_stem.setdefault(stem_l, []).append(full)
    return by_name, by_stem

def pick_best_match(paths):
    """Deterministic choice when multiple matches exist."""
    if not paths:
        return None
    # prefer shortest path (often closest / least nested)
    return sorted(paths, key=lambda p: (len(p), p.lower()))[0]

def find_image_path(by_name, by_stem, xml_label: str):
    """Match Metashape camera label/path to an image anywhere under input root.
    Tries exact filename match first; then stem match.
    Returns full path or None.
    """
    base = safe_basename(xml_label)
    base_l = base.lower()
    stem_l = os.path.splitext(base)[0].lower()
    # If XML label already includes an extension, try exact filename
    if os.path.splitext(base_l)[1]:
        p = pick_best_match(by_name.get(base_l, []))
        if p:
            return p
    # Fallback: stem match
    return pick_best_match(by_stem.get(stem_l, []))

# -----------------------------
# Parse Metashape XML cameras

# -----------------------------
def parse_metashape_project(xml_path: str):
    """Parse Metashape cameras.xml (mixed sensors).
    Returns:
      sensors: dict sensor_id -> dict(type, width, height, calib dict)
      cameras: list of dict(label, sensor_id, M_cw)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # sensors
    sensors = {}
    for s in root.findall('.//sensor'):
        sid = s.attrib.get('id')
        stype = (s.attrib.get('type') or '').lower()
        res = s.find('./resolution')
        w = int(res.attrib.get('width')) if res is not None and 'width' in res.attrib else None
        h = int(res.attrib.get('height')) if res is not None and 'height' in res.attrib else None

        calib_node = None
        for c in s.findall('./calibration'):
            # prefer adjusted if available
            if (c.attrib.get('class') or '').lower() == 'adjusted':
                calib_node = c
                break
        if calib_node is None:
            calib_node = s.find('./calibration')

        calib = {}
        if calib_node is not None:
            for child in list(calib_node):
                tag = child.tag.lower()
                if tag in ('resolution',):
                    continue
                if child.text is None:
                    continue
                t = child.text.strip()
                if not t:
                    continue
                try:
                    calib[tag] = float(t)
                except Exception:
                    pass

            # calibration resolution can override sensor resolution
            cres = calib_node.find('./resolution')
            if cres is not None:
                w = int(cres.attrib.get('width', w)) if w is not None else int(cres.attrib.get('width'))
                h = int(cres.attrib.get('height', h)) if h is not None else int(cres.attrib.get('height'))

        sensors[sid] = {'type': stype, 'width': w, 'height': h, 'calib': calib, 'label': s.attrib.get('label','')}

    # cameras
    cameras = []
    for cam in root.findall('.//camera'):
        tnode = cam.find('./transform')
        if tnode is None or not (tnode.text or '').strip():
            continue
        label = cam.attrib.get('label') or cam.attrib.get('path') or cam.attrib.get('name') or cam.attrib.get('id') or ''
        sensor_id = cam.attrib.get('sensor_id')
        M_cw = parse_transform_4x4(tnode.text)  # camera -> world
        cameras.append({'label': label, 'sensor_id': sensor_id, 'M_cw': M_cw})
    if not cameras:
        raise ValueError('No camera transforms found in XML.')
    return sensors, cameras

def parse_chunk_similarity(xml_path: str):
    """Parse Metashape <chunk><transform> similarity transform (chunk->world).
    Returns (R,T,S) where X_world = S * (R @ X_chunk) + T, or None if not found.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tnode = root.find('.//chunk/transform')
        if tnode is None:
            return None
        rnode = tnode.find('rotation')
        ttnode = tnode.find('translation')
        snode = tnode.find('scale')
        if rnode is None or ttnode is None or snode is None:
            return None
        rnums = re.findall(FLOAT_RE, rnode.text or '')
        tnums = re.findall(FLOAT_RE, ttnode.text or '')
        snums = re.findall(FLOAT_RE, snode.text or '')
        if len(rnums) < 9 or len(tnums) < 3 or len(snums) < 1:
            return None
        R = np.array(list(map(float, rnums[:9])), dtype=np.float64).reshape((3,3))
        T = np.array(list(map(float, tnums[:3])), dtype=np.float64)
        S = float(snums[0])
        if S == 0:
            return None
        return R, T, S
    except Exception:
        return None


def parse_spherical_labels(sphere_xml_path: str):
    """Parse optional camerassphere.xml and return a set of lowercase stems (camera labels) that are spherical."""
    if not sphere_xml_path or not os.path.isfile(sphere_xml_path):
        return None
    _, cams = parse_metashape_project(sphere_xml_path)
    stems = set()
    for c in cams:
        stems.add(os.path.splitext(safe_basename(c['label']))[0].lower())
    return stems

# -----------------------------
# PLY reader (ASCII or binary little endian)
# -----------------------------
def read_ply_points(ply_path: str):
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly.")
            s = line.decode("utf-8", errors="ignore").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt = None
        n_verts = 0
        props = []
        in_vertex = False
        for h in header:
            if h.startswith("format "):
                fmt = h.split()[1]
            elif h.startswith("element vertex "):
                n_verts = int(h.split()[2])
                in_vertex = True
                props = []
            elif h.startswith("element ") and not h.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and h.startswith("property "):
                parts = h.split()
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))
        if fmt is None or n_verts <= 0:
            raise ValueError("PLY missing format or vertex count.")
        type_map = {
            "char": "b", "int8": "b",
            "uchar": "B", "uint8": "B",
            "short": "h", "int16": "h",
            "ushort": "H", "uint16": "H",
            "int": "i", "int32": "i",
            "uint": "I", "uint32": "I",
            "float": "f", "float32": "f",
            "double": "d", "float64": "d",
        }
        names = [p[1].lower() for p in props]
        def idx(name):
            try:
                return names.index(name)
            except ValueError:
                return None
        ix, iy, iz = idx("x"), idx("y"), idx("z")
        ir, ig, ib = idx("red"), idx("green"), idx("blue")
        if ix is None or iy is None or iz is None:
            raise ValueError("PLY must include x,y,z vertex properties.")
        has_rgb = (ir is not None and ig is not None and ib is not None)
        if fmt == "ascii":
            xyz = np.zeros((n_verts, 3), dtype=np.float64)
            rgb = np.zeros((n_verts, 3), dtype=np.uint8) if has_rgb else None
            for i in range(n_verts):
                line = f.readline()
                if not line:
                    raise ValueError("PLY ASCII data ended early.")
                parts = line.decode("utf-8", errors="ignore").strip().split()
                xyz[i, 0] = float(parts[ix]); xyz[i, 1] = float(parts[iy]); xyz[i, 2] = float(parts[iz])
                if has_rgb:
                    rgb[i, 0] = int(float(parts[ir])); rgb[i, 1] = int(float(parts[ig])); rgb[i, 2] = int(float(parts[ib]))
            return xyz, rgb
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt} (need ascii or binary_little_endian)")
        fmt_str = "<" + "".join(type_map[t] for t, _ in props)
        stride = struct.calcsize(fmt_str)
        data = f.read(n_verts * stride)
        if len(data) < n_verts * stride:
            raise ValueError("PLY binary data ended early.")
        xyz = np.zeros((n_verts, 3), dtype=np.float64)
        rgb = np.zeros((n_verts, 3), dtype=np.uint8) if has_rgb else None
        off = 0
        for i in range(n_verts):
            values = struct.unpack_from(fmt_str, data, off)
            off += stride
            xyz[i, 0] = float(values[ix]); xyz[i, 1] = float(values[iy]); xyz[i, 2] = float(values[iz])
            if has_rgb:
                r = int(values[ir]); g = int(values[ig]); b = int(values[ib])
                rgb[i, 0] = max(0, min(255, r))
                rgb[i, 1] = max(0, min(255, g))
                rgb[i, 2] = max(0, min(255, b))
        return xyz, rgb
# -----------------------------
# Equirect bilinear sampling
# -----------------------------
def _bilinear_sample_rgb(pano_rgb, u, v, xp=None):
    """pano_rgb: HxWx3 uint8, u/v float arrays in pixel coords.
    xp: numpy or cupy module. If None, uses numpy.
    """
    if xp is None:
        xp = np
    H, W, _ = pano_rgb.shape
    u = xp.mod(u, W)
    v = xp.clip(v, 0.0, H - 1.0)
    u0 = xp.floor(u).astype(xp.int32)
    v0 = xp.floor(v).astype(xp.int32)
    u1 = (u0 + 1) % W
    v1 = xp.clip(v0 + 1, 0, H - 1)
    du = (u - u0)[..., None].astype(xp.float32)
    dv = (v - v0)[..., None].astype(xp.float32)
    p00 = pano_rgb[v0, u0].astype(xp.float32)
    p10 = pano_rgb[v0, u1].astype(xp.float32)
    p01 = pano_rgb[v1, u0].astype(xp.float32)
    p11 = pano_rgb[v1, u1].astype(xp.float32)
    out = (1-du)*(1-dv)*p00 + du*(1-dv)*p10 + (1-du)*dv*p01 + du*dv*p11
    return xp.clip(out, 0, 255).astype(xp.uint8)
# -----------------------------
# View rotations (COLMAP camera coords: x right, y down, z forward)
# -----------------------------
def _build_from_forward_up_yup(fwd, up):
    """Build a rotation (in y-up convention) from view coords to pano coords."""
    fwd = np.array(fwd, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    fwd /= np.linalg.norm(fwd) + 1e-12
    up /= np.linalg.norm(up) + 1e-12
    x = np.cross(up, fwd); x /= np.linalg.norm(x) + 1e-12
    y = np.cross(fwd, x);  y /= np.linalg.norm(y) + 1e-12
    R = np.stack([x, y, fwd], axis=1)  # columns = basis vectors
    return R
def view_rot_colmap_from_yaw_pitch(yaw_deg: float, pitch_deg: float = 0.0):
    """
    Return R_view (3x3) mapping view_cam -> pano_cam, in COLMAP y-down convention.
    yaw/pitch are defined in the y-up convention:
      - yaw: rotate around +Y (up), 0° = forward (+Z), 90° = right (+X)
      - pitch: rotate around +X (right), positive looks up
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    # In y-up coordinates
    # start with forward (+Z), then yaw around Y, pitch around X
    # forward = Ry(yaw) * Rx(pitch) * [0,0,1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Apply pitch then yaw (right-handed, y-up)
    # forward:
    fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float64)  # (x, y, z)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    R_yup = _build_from_forward_up_yup(fwd, up)
    # convert to y-down convention (same trick as cubemap builder)
    R_ydown = np.diag([1.,-1.,1.]) @ R_yup @ np.diag([1.,-1.,1.])
    return R_ydown.astype(np.float64)
def cubemap_face_rotations_colmap():
    """Return cubemap 6-face rotations view_cam -> pano_cam in COLMAP y-down convention."""
    def build_from_forward_up(fwd, up):
        fwd = np.array(fwd, dtype=np.float64)
        up = np.array(up, dtype=np.float64)
        fwd /= np.linalg.norm(fwd) + 1e-12
        up /= np.linalg.norm(up) + 1e-12
        x = np.cross(up, fwd); x /= np.linalg.norm(x) + 1e-12
        y = np.cross(fwd, x);  y /= np.linalg.norm(y) + 1e-12
        R = np.stack([x, y, fwd], axis=1)
        return R
    # y-up definitions
    faces_up = {
        "pz": ((0, 0, 1), (0, 1, 0)),   # forward
        "nz": ((0, 0,-1), (0, 1, 0)),   # back
        "px": ((1, 0, 0), (0, 1, 0)),   # right
        "nx": ((-1,0, 0), (0, 1, 0)),   # left
        "py": ((0, 1, 0), (0, 0,-1)),   # up
        "ny": ((0,-1, 0), (0, 0, 1)),   # down
    }
    out = {}
    for name, (fwd, up) in faces_up.items():
        R_up = build_from_forward_up(fwd, up)
        R_down = np.diag([1.,-1.,1.]) @ R_up @ np.diag([1.,-1.,1.])
        out[name] = R_down.astype(np.float64)
    return out
CUBE_FACE_ORDER = ["pz", "px", "nz", "nx", "py", "ny"]
# -----------------------------
# Parse yaw exclude ranges
# -----------------------------
def parse_exclude_ranges(text: str):
    """
    Parse strings like:
      "315-45" (wrap)
      "30-60, 150-210"
    Returns list of (start,end) degrees normalized to [0,360).
    """
    s = (text or "").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError("Exclude yaw ranges must be like 'start-end' (e.g. 315-45).")
        a, b = part.split("-", 1)
        start = float(a.strip()) % 360.0
        end = float(b.strip()) % 360.0
        out.append((start, end))
    return out
def yaw_in_ranges(yaw_deg: float, ranges):
    y = float(yaw_deg) % 360.0
    for start, end in ranges:
        if start <= end:
            if start <= y <= end:
                return True
        else:
            # wrap around 360
            if y >= start or y <= end:
                return True
    return False
# -----------------------------
# Equirect -> perspective (tiled, writable-safe)
# -----------------------------
def equirect_to_perspective_gpu(pano_rgb_gpu, size: int, R_view: np.ndarray,
                               fov_deg: float = 90.0):
    """GPU path using CuPy. Processes the entire output image in one pass — no tiling.
    pano_rgb_gpu: CuPy array HxWx3 uint8 already on GPU.
    Returns a CPU numpy HxWx3 uint8 array.
    """
    H, W, _ = pano_rgb_gpu.shape
    fov = math.radians(float(fov_deg))
    fx = fy = float(size) / (2.0 * math.tan(fov / 2.0))
    cx = cy = float(size) / 2.0

    # Build full pixel grid on GPU
    xs = (cp.arange(size, dtype=cp.float32) - cx) / fx          # (size,)
    ys = (cp.arange(size, dtype=cp.float32) - cy) / fy          # (size,)
    dx = cp.tile(xs[None, :], (size, 1))                         # (size, size)
    dy = cp.tile(ys[:, None], (1, size))
    dz = cp.ones((size, size), dtype=cp.float32)

    invn = 1.0 / cp.sqrt(dx*dx + dy*dy + dz*dz)
    dx *= invn; dy *= invn; dz *= invn

    R = R_view.astype(np.float32)
    vx = float(R[0,0])*dx + float(R[0,1])*dy + float(R[0,2])*dz
    vy = float(R[1,0])*dx + float(R[1,1])*dy + float(R[1,2])*dz
    vz = float(R[2,0])*dx + float(R[2,1])*dy + float(R[2,2])*dz

    vy_up = -vy
    lon = cp.arctan2(vx, vz)
    lat = cp.arcsin(cp.clip(vy_up, -1.0, 1.0))

    u = (lon + math.pi) / (2 * math.pi) * (W - 1)
    v = (math.pi/2 - lat) / math.pi * (H - 1)

    result_gpu = _bilinear_sample_rgb(pano_rgb_gpu, u, v, xp=cp)
    return cp.asnumpy(result_gpu), fx, fy, cx, cy


def equirect_to_perspective_tiled_from_rgb(pano_rgb: np.ndarray, size: int, R_view: np.ndarray,
                                          fov_deg: float = 90.0, tile_h: int = 256,
                                          use_gpu: bool = False):
    """
    Render a perspective view from an equirect panorama (CPU tiled path).
    If use_gpu=True and CuPy is available, delegates to the GPU path instead.
    """
    if use_gpu and _cupy_ok:
        pano_gpu = cp.asarray(pano_rgb)
        out_rgb, fx, fy, cx, cy = equirect_to_perspective_gpu(pano_gpu, size, R_view, fov_deg)
        return Image.fromarray(out_rgb, mode="RGB"), fx, fy, cx, cy

    pano = pano_rgb
    if pano.dtype != np.uint8 or pano.ndim != 3 or pano.shape[2] != 3:
        pano = np.array(pano, dtype=np.uint8, copy=False)

    H, W, _ = pano.shape
    fov = math.radians(float(fov_deg))
    fx = fy = float(size) / (2.0 * math.tan(fov / 2.0))
    cx = cy = float(size) / 2.0

    out = np.zeros((size, size, 3), dtype=np.uint8)
    xs = (np.arange(size, dtype=np.float32) - cx)[None, :] / fx

    for y0 in range(0, size, tile_h):
        y1 = min(size, y0 + tile_h)
        ys = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None] / fy

        dx = np.broadcast_to(xs, (y1 - y0, size)).copy()
        dy = np.broadcast_to(ys, (y1 - y0, size)).copy()
        dz = np.ones_like(dx, dtype=np.float32)

        invn = 1.0 / np.sqrt(dx*dx + dy*dy + dz*dz)
        dx *= invn; dy *= invn; dz *= invn

        vx = R_view[0,0]*dx + R_view[0,1]*dy + R_view[0,2]*dz
        vy = R_view[1,0]*dx + R_view[1,1]*dy + R_view[1,2]*dz
        vz = R_view[2,0]*dx + R_view[2,1]*dy + R_view[2,2]*dz

        vy_up = -vy
        lon = np.arctan2(vx, vz)
        lat = np.arcsin(np.clip(vy_up, -1.0, 1.0))

        u = (lon + math.pi) / (2 * math.pi) * (W - 1)
        v = (math.pi/2 - lat) / math.pi * (H - 1)

        tile_rgb = _bilinear_sample_rgb(pano, u, v, xp=np)

        for i in range(tile_rgb.shape[0]):
            out[y0 + i, :, :] = tile_rgb[i]

    return Image.fromarray(out, mode="RGB"), fx, fy, cx, cy


def equirect_to_perspective_tiled(pano_img: Image.Image, size: int, R_view: np.ndarray,
                                  fov_deg: float = 90.0, tile_h: int = 256):
    """
    Render a perspective view from an equirect panorama.

    Accepts either a PIL Image OR an already-decoded HxWx3 uint8 numpy array.
    For maximum performance when rendering many views per panorama, convert the
    panorama to numpy once and call equirect_to_perspective_tiled_from_rgb().
    """
    if isinstance(pano_img, np.ndarray):
        pano_rgb = pano_img
    else:
        pano_rgb = np.array(pano_img.convert("RGB"), dtype=np.uint8)
    return equirect_to_perspective_tiled_from_rgb(pano_rgb, size, R_view, fov_deg=fov_deg, tile_h=tile_h)

# -----------------------------
# View sets
# -----------------------------
def view_keys_only(base_mode: str, add_diag45: bool, add_upper45: bool, add_lower45: bool):
    """
    Returns just the list of view keys for the given configuration.
    Pure Python — no numpy. Safe to call before numpy is installed (used by UI).
    """
    keys = list(CUBE_FACE_ORDER)  # pz px nz nx py ny
    horizon_yaws = [0, 90, 180, 270]
    if add_diag45 or base_mode == "cubemap6_plus45":
        for yaw in (45, 135, 225, 315):
            keys.append(f"yaw{yaw:03d}")
            horizon_yaws.append(yaw)
    if add_upper45:
        for yaw in horizon_yaws:
            keys.append(f"up45_{yaw:03d}")
    if add_lower45:
        for yaw in horizon_yaws:
            keys.append(f"dn45_{yaw:03d}")
    return keys

def build_view_set(base_mode: str, add_diag45: bool, add_upper45: bool, add_lower45: bool):
    """
    Returns a list of view dicts:
      {
        "key": stable key,
        "suffix": filename suffix,
        "R": 3x3 view_cam->pano_cam,
        "yaw": yaw_deg or None,
        "pitch": pitch_deg or None,
        "yaw_sensitive": bool   # whether yaw-exclude applies
      }
    base_mode currently supports: "cubemap6"
    Extras:
      - add_diag45: adds horizon diagonals at yaw 45/135/225/315 (pitch 0)
      - add_upper45: adds an upper ring at pitch +45 for each horizon yaw sample
      - add_lower45: adds a lower ring at pitch -45 for each horizon yaw sample
    """
    views = []
    CUBE_FACE_ROT = cubemap_face_rotations_colmap()  # safe here — only called during conversion
    # Base cubemap 6 faces
    if base_mode not in ("cubemap6", "cubemap6_plus45"):
        base_mode = "cubemap6"
    for face in CUBE_FACE_ORDER:
        yaw = None
        pitch = None
        yaw_sensitive = False
        if face == "pz": yaw, pitch, yaw_sensitive = 0.0, 0.0, True
        elif face == "px": yaw, pitch, yaw_sensitive = 90.0, 0.0, True
        elif face == "nz": yaw, pitch, yaw_sensitive = 180.0, 0.0, True
        elif face == "nx": yaw, pitch, yaw_sensitive = 270.0, 0.0, True
        elif face == "py": yaw, pitch, yaw_sensitive = None, 90.0, False
        elif face == "ny": yaw, pitch, yaw_sensitive = None, -90.0, False
        views.append({
            "key": face,
            "suffix": face,
            "R": CUBE_FACE_ROT[face],
            "yaw": yaw,
            "pitch": pitch,
            "yaw_sensitive": yaw_sensitive,
        })
    # Collect horizon yaw samples (for diagonals and optional rings)
    horizon_yaws = [0.0, 90.0, 180.0, 270.0]
    if add_diag45 or base_mode == "cubemap6_plus45":
        for yaw in (45.0, 135.0, 225.0, 315.0):
            key = f"yaw{int(yaw):03d}"
            views.append({
                "key": key,
                "suffix": key,
                "R": view_rot_colmap_from_yaw_pitch(yaw, 0.0),
                "yaw": yaw,
                "pitch": 0.0,
                "yaw_sensitive": True,
            })
            horizon_yaws.append(yaw)
    # Upper / Lower rings at ±45° pitch
    if add_upper45:
        for yaw in horizon_yaws:
            key = f"up45_{int(yaw):03d}"
            views.append({
                "key": key,
                "suffix": key,
                "R": view_rot_colmap_from_yaw_pitch(yaw, 45.0),
                "yaw": yaw,
                "pitch": 45.0,
                "yaw_sensitive": True,
            })
    if add_lower45:
        for yaw in horizon_yaws:
            key = f"dn45_{int(yaw):03d}"
            views.append({
                "key": key,
                "suffix": key,
                "R": view_rot_colmap_from_yaw_pitch(yaw, -45.0),
                "yaw": yaw,
                "pitch": -45.0,
                "yaw_sensitive": True,
            })
    return views
# -----------------------------
# Worker: pano -> N views
# -----------------------------
def _save_jpeg_worker(args):
    """Top-level function for multiprocessing JPEG encode+save. GIL-free."""
    out_path, rgb_bytes, h, w = args
    try:
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
        Image.fromarray(arr, mode="RGB").save(out_path, quality=95, optimize=False)
        return True
    except Exception:
        return False


def _worker_convert_one(pano_path: str, out_images_dir: str, size: int, fov_deg: float,
                        view_payloads, tile_h: int, use_gpu: bool = False):
    """CPU worker for one panorama (used in multiprocess pool)."""
    pano_name = os.path.basename(pano_path)
    stem = os.path.splitext(pano_name)[0]
    pano_img = Image.open(pano_path).convert("RGB")
    pano_rgb = np.array(pano_img, dtype=np.uint8)
    try:
        pano_img.close()
    except Exception:
        pass
    out_files = []
    for suffix, Rflat in view_payloads:
        R = np.array(Rflat, dtype=np.float64).reshape((3, 3))
        out_name = f"{stem}_{suffix}.jpg"
        img, _, _, _, _ = equirect_to_perspective_tiled_from_rgb(
            pano_rgb, size, R, fov_deg=fov_deg, tile_h=tile_h)
        img.save(os.path.join(out_images_dir, out_name), quality=95)
        out_files.append((suffix, out_name))
    return pano_name, out_files


def _gpu_convert_batch(items_batch, out_images_dir: str, size: int, fov_deg: float,
                       view_payloads, gpu_batch_size: int = 4,
                       image_cb=None, stop_event=None):
    """
    GPU pipeline: loader thread -> GPU render thread -> writer threads.

    cp.asnumpy() releases the GIL during PCIe transfer, so writer threads
    run concurrently with GPU transfers automatically. Keep it simple and correct.
    """
    import queue as _queue

    SENTINEL   = object()
    N_WRITERS  = min(4, multiprocessing.cpu_count())
    LOAD_AHEAD = max(8, gpu_batch_size)

    load_q  = _queue.Queue(maxsize=LOAD_AHEAD)
    write_q = _queue.Queue(maxsize=LOAD_AHEAD * len(view_payloads) * 2)
    results = {}
    errors  = []

    # ── Loader thread: disk -> RAM ────────────────────────────────────────────
    def _loader():
        try:
            for pano_name, pano_path in items_batch:
                if stop_event and stop_event.is_set():
                    break
                try:
                    img = Image.open(pano_path).convert("RGB")
                    arr = np.array(img, dtype=np.uint8)
                    try: img.close()
                    except Exception: pass
                    load_q.put((pano_name, os.path.splitext(pano_name)[0], arr))
                except Exception as e:
                    errors.append(e)
                    load_q.put((pano_name, os.path.splitext(pano_name)[0], None))
        finally:
            load_q.put(SENTINEL)  # always unblock GPU thread

    # ── GPU thread: RAM -> VRAM -> render -> RAM -> write queue ───────────────
    def _gpu_render():
        try:
            # Build rotation matrices on GPU once
            R_list = [cp.asarray(np.array(Rflat, dtype=cp.float32).reshape(3, 3))
                      for _, Rflat in view_payloads]

            # Pre-compute normalised ray grid once — same for every panorama and view
            fov  = math.radians(float(fov_deg))
            fx   = fy = float(size) / (2.0 * math.tan(fov / 2.0))
            cx   = cy = float(size) / 2.0
            _xs  = (cp.arange(size, dtype=cp.float32) - cx) / fx
            _ys  = (cp.arange(size, dtype=cp.float32) - cy) / fy
            _X   = cp.tile(_xs[None, :], (size, 1))
            _Y   = cp.tile(_ys[:, None], (1, size))
            _Z   = cp.ones((size, size), dtype=cp.float32)
            _inv = 1.0 / cp.sqrt(_X*_X + _Y*_Y + _Z*_Z)
            dx   = (_X * _inv).astype(cp.float32)
            dy   = (_Y * _inv).astype(cp.float32)
            dz   = (_Z * _inv).astype(cp.float32)
            del _X, _Y, _Z, _inv, _xs, _ys

            while True:
                item = load_q.get()
                if item is SENTINEL:
                    break
                if stop_event and stop_event.is_set():
                    break

                pano_name, stem, pano_rgb = item
                pano_files = []

                if pano_rgb is None:
                    results[pano_name] = pano_files
                    continue

                try:
                    H_p, W_p, _ = pano_rgb.shape
                    pano_gpu = cp.asarray(pano_rgb)

                    for (suffix, _), R_gpu in zip(view_payloads, R_list):
                        vx  = R_gpu[0,0]*dx + R_gpu[0,1]*dy + R_gpu[0,2]*dz
                        vy  = R_gpu[1,0]*dx + R_gpu[1,1]*dy + R_gpu[1,2]*dz
                        vz  = R_gpu[2,0]*dx + R_gpu[2,1]*dy + R_gpu[2,2]*dz
                        lon = cp.arctan2(vx, vz)
                        lat = cp.arcsin(cp.clip(-vy, -1.0, 1.0))
                        u   = (lon + math.pi) / (2*math.pi) * (W_p - 1)
                        v   = (math.pi/2 - lat) / math.pi  * (H_p - 1)
                        out_gpu = _bilinear_sample_rgb(pano_gpu, u, v, xp=cp)
                        out_cpu = cp.asnumpy(out_gpu)
                        out_name = f"{stem}_{suffix}.jpg"
                        write_q.put((os.path.join(out_images_dir, out_name), out_cpu))
                        pano_files.append((suffix, out_name))

                    del pano_gpu
                except Exception as e:
                    import traceback
                    errors.append(Exception(f"Render error for {pano_name}: {e}\n{traceback.format_exc()}"))

                results[pano_name] = pano_files

        except Exception as e:
            import traceback
            errors.append(Exception(f"GPU render thread fatal: {e}\n{traceback.format_exc()}"))
        finally:
            # Always send sentinels — even if we crashed — so writers don't deadlock
            for _ in range(N_WRITERS):
                write_q.put(SENTINEL)
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    # ── Writer threads: RAM -> JPEG -> disk ───────────────────────────────────
    def _writer():
        while True:
            item = write_q.get()
            if item is SENTINEL:
                break
            out_path, arr = item
            try:
                Image.fromarray(arr, mode="RGB").save(out_path, quality=95)
                if image_cb:
                    image_cb(1)
            except Exception as e:
                errors.append(e)

    loader_t  = threading.Thread(target=_loader,     daemon=True)
    gpu_t     = threading.Thread(target=_gpu_render, daemon=True)
    writer_ts = [threading.Thread(target=_writer, daemon=True) for _ in range(N_WRITERS)]

    loader_t.start()
    gpu_t.start()
    for t in writer_ts: t.start()

    loader_t.join()
    gpu_t.join()
    for t in writer_ts: t.join()

    if errors:
        raise RuntimeError(str(errors[0]))
    return results

# -----------------------------
# Conversion main
# -----------------------------

def convert(xml_path, input_root, out_dir,
            sphere_xml_path,
            view_mode: str,
            add_diag45: bool,
            add_upper45: bool,
            add_lower45: bool,
            out_size: int,
            fov_deg: float,
            exclude_yaw_ranges_text: str,
            ply_path,
            workers,
            flip_world_180x,
            flip_world_180y,
            tile_h,
            enabled_view_keys=None,
            use_gpu=False,
            gpu_batch_size=8,
            skip_existing=False,
            dry_run=False,
            stop_event=None,
            pause_event=None,
            progress_cb=None):
    sensors, cameras_all = parse_metashape_project(xml_path)
    chunk_sim = parse_chunk_similarity(xml_path)
    spherical_label_stems = parse_spherical_labels(sphere_xml_path)
    # Build recursive image index once
    by_name, by_stem = build_image_index(input_root)

    CUBE_FACE_ROT = cubemap_face_rotations_colmap()  # safe here — numpy is required
    out_images = os.path.join(out_dir, "images")
    out_sparse = os.path.join(out_dir, "sparse", "0")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_sparse, exist_ok=True)
    # Global world flip — 180° about X and/or Y (optional, applied together)
    Rg = np.eye(3, dtype=np.float64)
    if flip_world_180x:
        Rx = np.array([[ 1, 0, 0],
                       [ 0,-1, 0],
                       [ 0, 0,-1]], dtype=np.float64)
        Rg = Rx @ Rg
    if flip_world_180y:
        Ry = np.array([[-1, 0, 0],
                       [ 0, 1, 0],
                       [ 0, 0,-1]], dtype=np.float64)
        Rg = Ry @ Rg
    # Build views + optionally exclude yaw ranges (horizon views only)
    views = build_view_set(view_mode, add_diag45, add_upper45, add_lower45)
    exclude_ranges = parse_exclude_ranges(exclude_yaw_ranges_text)
    # Optionally keep only approved view directions
    if enabled_view_keys is not None:
        enabled_set = set(enabled_view_keys)
        views = [v for v in views if v['key'] in enabled_set]
    if exclude_ranges:
        views = [v for v in views if not (v.get("yaw_sensitive", False) and v["yaw"] is not None and yaw_in_ranges(v["yaw"], exclude_ranges))]
        if not views:
            raise ValueError("All views were excluded by yaw ranges; nothing to export.")
    # Intrinsics (one shared camera model, same for every rendered image)
    fov = math.radians(float(fov_deg))
    fx = fy = float(out_size) / (2.0 * math.tan(fov / 2.0))
    cx = cy = float(out_size) / 2.0
    with open(os.path.join(out_sparse, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# Camera list\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
        # Camera ID 1: shared PINHOLE model for all rendered spherical views
        f.write(f"1 PINHOLE {out_size} {out_size} {fx} {fy} {cx} {cy}\n")

        # Frame sensors: write one camera model per Metashape sensor_id (PINHOLE; assumes images already undistorted)
        sensor_id_to_camid = {}
        next_camid = 2
        for sid, s in sensors.items():
            if (s.get("type") or "").lower() != "frame":
                continue
            w = s.get("width"); h = s.get("height")
            if not w or not h:
                continue
            calib = s.get("calib") or {}
            fxy = float(calib.get("f", 0.0))
            if fxy <= 0:
                continue
            cx_off = float(calib.get("cx", 0.0))
            cy_off = float(calib.get("cy", 0.0))
            cx_abs = (w / 2.0) + cx_off
            cy_abs = (h / 2.0) + cy_off
            k1 = float(calib.get("k1", 0.0)); k2 = float(calib.get("k2", 0.0))
            p1 = float(calib.get("p1", 0.0)); p2 = float(calib.get("p2", 0.0))
            k3 = float(calib.get("k3", 0.0))
            k4 = float(calib.get("k4", 0.0)); k5 = float(calib.get("k5", 0.0)); k6 = float(calib.get("k6", 0.0))
            f.write(f"{next_camid} PINHOLE {w} {h} {fxy} {fxy} {cx_abs} {cy_abs}\n")
            sensor_id_to_camid[sid] = next_camid
            next_camid += 1
    # Split cameras: spherical to convert, frame to pass-through
    spherical_cams = []
    frame_cams = []
    for cam in cameras_all:
        sid = cam.get('sensor_id')
        sensor = sensors.get(sid, {})
        stype = (sensor.get('type') or '').lower()
        stem = os.path.splitext(safe_basename(cam['label']))[0].lower()
        is_spherical = (stype == 'spherical')
        if spherical_label_stems is not None:
            is_spherical = stem in spherical_label_stems
        if is_spherical:
            spherical_cams.append(cam)
        else:
            frame_cams.append(cam)

    # Prepare spherical conversion items (source pano path + pose)
    items = []
    for cam in spherical_cams:
        src_path = find_image_path(by_name, by_stem, cam['label'])
        if not src_path:
            raise FileNotFoundError(f"Could not match spherical image for camera label: {cam['label']}")
        pano_name = os.path.basename(src_path)
        items.append((pano_name, src_path, cam['M_cw']))
    if skip_existing and not dry_run:
        filtered = []
        for pano_name, pano_path, M_cw in items:
            stem = os.path.splitext(pano_name)[0]
            if not all(os.path.isfile(os.path.join(out_images, f"{stem}_{v['suffix']}.jpg")) for v in views):
                filtered.append((pano_name, pano_path, M_cw))
        items = filtered
        frame_cams = [c for c in frame_cams
                      if not os.path.isfile(os.path.join(out_images, safe_basename(c['label'])))]
    if dry_run:
        items      = items[:3]
        frame_cams = frame_cams[:3]

    total_imgs = len(items) * len(views) + len(frame_cams)
    done_imgs = 0
    # Prepare payloads (keep args pickle-friendly for multiprocessing)
    view_payloads = [(v["suffix"], v["R"].reshape(-1).tolist()) for v in views]

    if use_gpu and _cupy_ok:
        # ── Auto-calculate optimal batch size from available VRAM ────────────
        if gpu_batch_size <= 0:
            gpu_batch_size = 8  # fallback default
        try:
            _mempool = cp.get_default_memory_pool()
            _dev = cp.cuda.Device(0)
            _dev.use()
            _free_bytes = _dev.mem_info[0]          # free VRAM in bytes
            # Estimate bytes per panorama: assume ~8K equirect (7680x3840x3) worst case
            # plus output views: out_size*out_size*3 per view
            _pano_bytes = 7680 * 3840 * 3
            _views_bytes = out_size * out_size * 3 * len(views)
            _bytes_per_pano = _pano_bytes + _views_bytes
            _headroom = 0.75   # use 75% of free VRAM to leave room for CuPy overhead
            gpu_batch_size = max(1, int((_free_bytes * _headroom) / _bytes_per_pano))
            gpu_batch_size = min(gpu_batch_size, 64)  # sanity cap
        except Exception:
            pass  # keep whatever gpu_batch_size was passed in

        # ── GPU pipeline: pipelined loader → GPU render → saver ─────────────
        GPU_BATCH = max(1, gpu_batch_size)
        results = {}
        batch_items = [(pano_name, pano_path) for pano_name, pano_path, _ in items]
        for i in range(0, len(batch_items), GPU_BATCH):
            if stop_event and stop_event.is_set():
                break
            if pause_event:
                while pause_event.is_set():
                    if stop_event and stop_event.is_set():
                        break
                    time.sleep(0.1)
            if stop_event and stop_event.is_set():
                break
            chunk = batch_items[i:i + GPU_BATCH]
            def _img_cb(n):
                nonlocal done_imgs
                done_imgs += n
                if progress_cb:
                    progress_cb(done_imgs, total_imgs)
            batch_results = _gpu_convert_batch(
                chunk, out_images, out_size, fov_deg, view_payloads,
                gpu_batch_size=GPU_BATCH,
                image_cb=_img_cb,
                stop_event=stop_event)
            results.update(batch_results)
    else:
        # ── CPU multiprocess pool ────────────────────────────────────────────
        futures = {}
        results = {}
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            for pano_name, pano_path, _ in items:
                if stop_event and stop_event.is_set():
                    break
                futures[ex.submit(_worker_convert_one, pano_path, out_images, out_size, fov_deg, view_payloads, tile_h)] = pano_name
            for fut in as_completed(futures):
                if stop_event and stop_event.is_set():
                    # Cancel pending futures
                    for f in futures:
                        f.cancel()
                    break
                if pause_event:
                    while pause_event.is_set():
                        if stop_event and stop_event.is_set():
                            break
                        time.sleep(0.1)
                pano_name, out_files = fut.result()
                results[pano_name] = out_files
                done_imgs += len(out_files)
                if progress_cb:
                    progress_cb(done_imgs, total_imgs)
    # Write images.txt with per-view camera extrinsics
    with open(os.path.join(out_sparse, "images.txt"), "w", encoding="utf-8") as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[]\n")
        img_id = 1
        for pano_name, _, M_cw in items:
            R_cw = M_cw[:3, :3]
            C_world = M_cw[:3, 3].astype(np.float64)
            if flip_world_180x or flip_world_180y:
                R_cw = Rg @ R_cw
                C_world = Rg @ C_world
            # results[pano_name] is in suffix order of view_payloads (worker loop)
            # Convert to dict for suffix->filename lookup
            suffix_to_file = {suf: fn for suf, fn in results[pano_name]}
            for v in views:
                suffix = v["suffix"]
                filename = suffix_to_file[suffix]
                R_view = v["R"]              # view_cam -> pano_cam
                R_view_cw = R_cw @ R_view    # view_cam -> world
                R_wc = R_view_cw.T
                t = -R_wc @ C_world
                qw, qx, qy, qz = rotmat_to_quat_wxyz(R_wc)
                tx, ty, tz = t.tolist()
                f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {filename}\n\n")
                img_id += 1
        # Pass-through frame cameras (copy images + write poses)
        for cam in frame_cams:
            src_path = find_image_path(by_name, by_stem, cam['label'])
            if not src_path:
                # Skip silently? Better to fail loudly so you don't get mismatched camera sets
                raise FileNotFoundError(f"Could not match frame image for camera label: {cam['label']}")
            base_name = os.path.basename(src_path)
            # Ensure unique output name
            out_name = base_name
            out_path = os.path.join(out_images, out_name)
            if os.path.exists(out_path):
                stem, ext = os.path.splitext(base_name)
                n = 2
                while True:
                    out_name = f"{stem}_dup{n}{ext}"
                    out_path = os.path.join(out_images, out_name)
                    if not os.path.exists(out_path):
                        break
                    n += 1
            shutil.copy2(src_path, out_path)

            M_cw = cam['M_cw']
            R_cw = M_cw[:3, :3]
            C_world = M_cw[:3, 3].astype(np.float64)
            if flip_world_180x or flip_world_180y:
                R_cw = Rg @ R_cw
                C_world = Rg @ C_world
            R_wc = R_cw.T
            t = -R_wc @ C_world
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R_wc)
            tx, ty, tz = t.tolist()
            sid = cam.get('sensor_id')
            camid = sensor_id_to_camid.get(sid)
            if camid is None:
                raise ValueError(f"Frame camera uses sensor_id={sid} but no camera model was written for it (missing/invalid calibration).")
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camid} {out_name}\n\n")
            img_id += 1
            done_imgs += 1
            if progress_cb:
                progress_cb(done_imgs, total_imgs)

    # points3D.txt (optional PLY passthrough)
    if ply_path and os.path.isfile(ply_path):
        xyz, rgb = read_ply_points(ply_path)
        # If the PLY was exported in Metashape 'world' coordinates, undo the chunk similarity transform
        # so points end up in the same 'chunk' coordinate frame as the cameras in cameras.xml.
        if chunk_sim is not None:
            R_chunk, T_chunk, S_chunk = chunk_sim  # chunk -> world
            xyz = (R_chunk.T @ (xyz - T_chunk).T).T / S_chunk

        if rgb is None:
            rgb = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)
        if flip_world_180x or flip_world_180y:
            xyz = (Rg @ xyz.T).T
        with open(os.path.join(out_sparse, "points3D.txt"), "w", encoding="utf-8") as f:
            f.write("# 3D point list\n")
            f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for pid in range(xyz.shape[0]):
                x, y, z = xyz[pid]
                r, g, b = rgb[pid].tolist()
                f.write(f"{pid+1} {x} {y} {z} {r} {g} {b} 1.0\n")
    else:
        with open(os.path.join(out_sparse, "points3D.txt"), "w", encoding="utf-8") as f:
            f.write("# Empty\n")
# -----------------------------
# GUI
# -----------------------------
# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metashape → COLMAP Converter")

        # ── taskbar / window icon ────────────────────────────────────────────
        try:
            import sys as _sys, ctypes
            if getattr(_sys, "frozen", False):
                _icon_path = os.path.join(os.path.dirname(_sys.executable), "app_icon.ico")
            else:
                _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico")
            if os.path.isfile(_icon_path):
                self.iconbitmap(_icon_path)
                # Tell Windows this is its own app (not python.exe) so taskbar shows our icon
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    u"MetashapeColmapConverter.App.1")
        except Exception:
            pass
        self.geometry("960x700")
        self.minsize(860, 580)
        self.resizable(True, True)

        # ── dark title bar (Windows 10 build 19041+ / Windows 11) ───────────
        try:
            import ctypes
            HWND = ctypes.windll.user32.GetParent(self.winfo_id())
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11 / 10 21H1+)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                HWND, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
        except Exception:
            pass  # silently ignore on non-Windows or older builds

        # ── colour palette ──────────────────────────────────────────────────
        BG       = "#12131a"
        CARD     = "#1a1b24"
        CARD2    = "#1f2030"
        ACCENT   = "#c8a96e"          # warm gold
        ACCENT2  = "#6e9ecf"          # steel blue
        ACCENT3  = "#4ecb8d"          # mint green (success/run)
        FG       = "#e2ddd6"          # warm white
        FG_DIM   = "#6b6a72"          # muted
        FG_MID   = "#a09b93"          # mid-tone
        BORDER   = "#282936"
        BORDER2  = "#2e2f3e"
        ENTRY_BG = "#0e0f16"
        ENTRY_BD = "#303148"
        RUN_BG   = "#4ecb8d"
        RUN_ACT  = "#3db578"
        BTN_BG   = "#252636"
        BTN_ACT  = "#2e3048"

        self.configure(bg=BG)

        # store palette for use in helper methods
        self._pal = dict(BG=BG, CARD=CARD, CARD2=CARD2, ACCENT=ACCENT,
                         ACCENT2=ACCENT2, ACCENT3=ACCENT3, FG=FG,
                         FG_DIM=FG_DIM, FG_MID=FG_MID, BORDER=BORDER,
                         BORDER2=BORDER2, ENTRY_BG=ENTRY_BG)

        style = ttk.Style(self)
        style.theme_use("clam")

        # ── global ttk overrides ────────────────────────────────────────────
        FONT_BODY = ("Calibri", 10)
        FONT_MONO = ("Consolas", 10)
        FONT_LABEL = ("Calibri", 9)
        FONT_BOLD = ("Calibri", 10, "bold")
        FONT_TITLE = ("Calibri", 14, "bold")
        FONT_SUB   = ("Calibri", 10)
        FONT_CAP   = ("Calibri", 8)

        style.configure(".",
            background=BG, foreground=FG,
            fieldbackground=ENTRY_BG, bordercolor=BORDER,
            troughcolor=CARD, selectbackground=ACCENT,
            font=FONT_BODY)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=FG, font=FONT_BODY)
        style.configure("Dim.TLabel", background=BG, foreground=FG_DIM, font=FONT_LABEL)
        style.configure("Card.TLabel", background=CARD, foreground=FG, font=FONT_BODY)
        style.configure("Card.Dim.TLabel", background=CARD, foreground=FG_DIM, font=FONT_LABEL)
        style.configure("TEntry",
            fieldbackground=ENTRY_BG, foreground=FG,
            insertcolor=ACCENT, bordercolor=ENTRY_BD, relief="flat", padding=(8, 5))
        style.map("TEntry",
            fieldbackground=[("focus", "#131420")],
            bordercolor=[("focus", ACCENT)])
        style.configure("TCheckbutton",
            background=CARD, foreground=FG, font=FONT_BODY,
            indicatorcolor=ENTRY_BG)
        style.map("TCheckbutton",
            background=[("active", CARD)],
            indicatorcolor=[("selected", ACCENT)])
        style.configure("TLabelframe",
            background=CARD, bordercolor=BORDER2, relief="flat", padding=0)
        style.configure("TLabelframe.Label",
            background=CARD, foreground=ACCENT, font=("Calibri", 9))

        # Browse buttons — ghost style
        style.configure("Browse.TButton",
            background=BTN_BG, foreground=FG_MID,
            bordercolor=BORDER2, relief="flat",
            font=FONT_LABEL, padding=(10, 5))
        style.map("Browse.TButton",
            background=[("active", BTN_ACT), ("pressed", BORDER)],
            foreground=[("active", FG)])

        # Action button
        style.configure("Action.TButton",
            background=ACCENT2, foreground="#ffffff",
            bordercolor=ACCENT2, relief="flat",
            font=FONT_BODY, padding=(14, 6))
        style.map("Action.TButton",
            background=[("active", "#5787b5"), ("pressed", "#4a76a3")])

        # Run button
        style.configure("Run.TButton",
            background=RUN_BG, foreground="#0a1a10",
            bordercolor=RUN_BG, relief="flat",
            font=("Calibri", 11, "bold"), padding=(24, 8))
        style.map("Run.TButton",
            background=[("active", RUN_ACT), ("pressed", RUN_ACT)])

        # Progress bar — thin gold line
        style.configure("TProgressbar",
            troughcolor=CARD, background=ACCENT,
            bordercolor=CARD, thickness=3)

        # Notebook
        style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=[0, 4, 0, 0])
        style.configure("TNotebook.Tab",
            background=BG, foreground=FG_DIM,
            font=("Calibri", 9), padding=(18, 8), bordercolor=BORDER)
        style.map("TNotebook.Tab",
            background=[("selected", CARD), ("active", "#161720")],
            foreground=[("selected", ACCENT), ("active", FG_MID)])

        # Combobox
        style.configure("TCombobox",
            fieldbackground=ENTRY_BG, background=ENTRY_BG,
            foreground=FG, arrowcolor=FG_DIM,
            bordercolor=ENTRY_BD, relief="flat")
        style.map("TCombobox",
            fieldbackground=[("readonly", ENTRY_BG)],
            selectbackground=[("readonly", ACCENT)])

        # Scrollbar
        style.configure("TScrollbar",
            background=CARD, troughcolor=ENTRY_BG,
            bordercolor=BORDER, arrowcolor=FG_DIM, relief="flat")

        # ── state vars ─────────────────────────────────────────────────────
        self.xml        = tk.StringVar()
        self.sphere_xml = tk.StringVar()
        self.panos      = tk.StringVar()
        self.ply        = tk.StringVar()
        self.out        = tk.StringVar()
        self.view_mode  = tk.StringVar(value="cubemap6")
        self.add_diag45   = tk.BooleanVar(value=True)
        self.add_upper45  = tk.BooleanVar(value=False)
        self.add_lower45  = tk.BooleanVar(value=False)
        self.out_size     = tk.IntVar(value=2048)
        self.fov_deg      = tk.DoubleVar(value=90.0)
        self.exclude_yaw  = tk.StringVar(value="")
        self.flip_world_180x = tk.BooleanVar(value=True)
        self.flip_world_180y = tk.BooleanVar(value=True)
        default_workers = max(1, (multiprocessing.cpu_count() or 4) - 1)
        self.workers = tk.IntVar(value=default_workers)
        self.tile_h  = tk.IntVar(value=256)
        self.project_dir      = tk.StringVar(value="")
        self._queue_jobs      = []   # list of job dicts
        self._queue_running   = False
        self._queue_stop      = threading.Event()
        self._queue_id_counter = 0
        self.output_folder_name = tk.StringVar(value="output")
        self.output_version      = tk.IntVar(value=1)
        self.output_folder_name.trace_add('write', self._on_output_name_changed)
        self.output_version.trace_add('write', lambda *_: self._on_output_name_changed())
        self.use_gpu        = tk.BooleanVar(value=True)
        self.gpu_batch_size = tk.IntVar(value=8)
        self.skip_existing  = tk.BooleanVar(value=False)
        self.dry_run        = tk.BooleanVar(value=False)
        self._dry_launch    = False
        self.colmap_exe       = tk.StringVar(value="")
        self.lichtfeld_exe    = tk.StringVar(value="")
        self.brush_exe        = tk.StringVar(value="")
        self.launch_colmap    = tk.BooleanVar(value=False)
        self.launch_lichtfeld = tk.BooleanVar(value=False)
        self.launch_brush     = tk.BooleanVar(value=False)
        self.rig_panos=tk.StringVar(value=''); self.rig_out=tk.StringVar(value='')
        self.rig_step_extract=tk.BooleanVar(value=True)
        self.rig_step_rig_cfg=tk.BooleanVar(value=True)
        self.rig_step_match=tk.BooleanVar(value=True)
        self.rig_step_map=tk.BooleanVar(value=True)
        self.rig_matcher=tk.StringVar(value='sequential')
        self.rig_running=False; self.rig_stop_event=threading.Event()
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        # Timer / rate tracking
        self._rate_hist     = []      # list of (timestamp, done_count)
        self._eta_seconds   = None    # current ETA in seconds
        self._eta_last_calc = 0.0     # when ETA was last recalculated
        self._eta_calc_time = 0.0     # time of last countdown tick

        # progress
        self._q          = queue.Queue()
        self._start_time = None
        self._total = 1
        self._done  = 0
        self._polling = False
        self._spin_i  = 0
        self._spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        # preview state
        self.view_enabled       = {}
        self._preview_win       = None
        self._preview_imgs      = {}
        self._preview_base_pil  = {}
        self._preview_last_sig  = None
        self._key_to_label      = {}

        # ── root layout ─────────────────────────────────────────────────────
        root_frame = tk.Frame(self, bg=BG)
        root_frame.pack(fill="both", expand=True)

        # ── header ──────────────────────────────────────────────────────────
        header = tk.Frame(root_frame, bg=BG)
        header.pack(fill="x", padx=0, pady=0)

        # top gold rule
        tk.Frame(header, bg=ACCENT, height=1).pack(fill="x")

        header_inner = tk.Frame(header, bg=BG)
        header_inner.pack(fill="x", padx=20, pady=(12, 10))

        # left: wordmark
        left_head = tk.Frame(header_inner, bg=BG)
        left_head.pack(side="left")

        tk.Label(left_head,
                 text="METASHAPE  ›  COLMAP",
                 bg=BG, fg=FG,
                 font=("Calibri", 14, "bold")).pack(anchor="w")
        tk.Label(left_head,
                 text="panoramic scene converter",
                 bg=BG, fg=FG_DIM,
                 font=("Calibri", 9)).pack(anchor="w", pady=(1, 0))

        # right: version badge
        badge = tk.Frame(header_inner, bg=CARD2, padx=10, pady=4)
        badge.pack(side="right", anchor="center")
        tk.Label(badge, text="v0.29", bg=CARD2, fg=ACCENT,
                 font=("Calibri", 8)).pack()

        # bottom border
        tk.Frame(header, bg=BORDER, height=1).pack(fill="x")

        # ── notebook ────────────────────────────────────────────────────────
        nb = ttk.Notebook(root_frame)
        self._nb = nb
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Global mousewheel scroll ──────────────────────────────────────────
        # Walks up the widget hierarchy from wherever the mouse is to find
        # the nearest Canvas with a yview command, then scrolls it.
        def _on_global_mousewheel(event):
            widget = event.widget
            # Walk up to find a canvas
            w = widget
            while w:
                if isinstance(w, tk.Canvas):
                    try:
                        w.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    except Exception:
                        pass
                    return
                try:
                    w = w.master
                except Exception:
                    break
        self.bind_all("<MouseWheel>", _on_global_mousewheel)

        def _section_header(parent, row, title):
            f=tk.Frame(parent,bg=BG)
            f.grid(row=row,column=0,sticky='ew',padx=20,pady=(0,4))
            tk.Label(f,text=title.upper(),bg=BG,fg=ACCENT,font=('Calibri',8,'bold')).pack(side='left')
            tk.Frame(f,bg=BORDER2,height=1).pack(side='left',fill='x',expand=True,padx=(10,0),pady=1)

        def _card(parent, row):
            c=tk.Frame(parent,bg=CARD,padx=16,pady=12)
            c.grid(row=row,column=0,sticky='ew',padx=20,pady=(0,8))
            c.columnconfigure(1,weight=1)
            return c

        def _make_scrollable(parent_tab):
            parent_tab.columnconfigure(0,weight=1)
            parent_tab.rowconfigure(0,weight=1)
            cv=tk.Canvas(parent_tab,bg=BG,highlightthickness=0)
            cv.grid(row=0,column=0,sticky='nsew')
            sb=ttk.Scrollbar(parent_tab,orient='vertical',command=cv.yview)
            sb.grid(row=0,column=1,sticky='ns')
            cv.configure(yscrollcommand=sb.set)
            inner=tk.Frame(cv,bg=BG)
            win=cv.create_window((0,0),window=inner,anchor='nw')
            inner.columnconfigure(0,weight=1)
            inner.bind('<Configure>',lambda e:cv.configure(scrollregion=cv.bbox('all')))
            cv.bind('<Configure>',lambda e:cv.itemconfigure(win,width=e.width))
            return inner

        # ════════════════════════════════════════════
        # TAB 1 – Files
        # ════════════════════════════════════════════
        tab_files = ttk.Frame(nb, style="TFrame")
        nb.add(tab_files, text="  Metashape  ")
        tab_files.columnconfigure(0, weight=1)

        tab_files.columnconfigure(0,weight=1); tab_files.rowconfigure(0,weight=1)
        scroll_canvas=tk.Canvas(tab_files,bg=BG,highlightthickness=0)
        scroll_canvas.grid(row=0,column=0,sticky='nsew')
        _files_sb=ttk.Scrollbar(tab_files,orient='vertical',command=scroll_canvas.yview)
        _files_sb.grid(row=0,column=1,sticky='ns')
        scroll_canvas.configure(yscrollcommand=_files_sb.set)
        files_inner=tk.Frame(scroll_canvas,bg=BG)
        _files_win=scroll_canvas.create_window((0,0),window=files_inner,anchor='nw')
        files_inner.columnconfigure(0,weight=1)
        files_inner.bind('<Configure>',lambda e:scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all')))
        scroll_canvas.bind('<Configure>',lambda e:scroll_canvas.itemconfigure(_files_win,width=e.width))
        self._files_inner=files_inner

        def _make_file_card(parent, row, badge_text, label, hint, var, browse_cmd):
            """A clean card-style file picker row."""
            card = tk.Frame(parent, bg=CARD, padx=0, pady=0)
            card.grid(row=row, column=0, sticky="ew", pady=(0, 2), padx=28)
            card.columnconfigure(1, weight=1)

            # left badge strip
            badge_strip = tk.Frame(card, bg=ACCENT if badge_text.startswith("xml") else
                                   ACCENT2 if badge_text.startswith("img") else
                                   ACCENT if badge_text == "ply" else ACCENT3,
                                   width=3)
            badge_strip.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 0))
            badge_strip.grid_propagate(False)

            # content
            content = tk.Frame(card, bg=CARD, padx=16, pady=10)
            content.grid(row=0, column=1, sticky="ew")
            content.columnconfigure(0, weight=1)

            top_row = tk.Frame(content, bg=CARD)
            top_row.pack(fill="x")

            tk.Label(top_row, text=label, bg=CARD, fg=FG,
                     font=("Calibri", 10, "bold")).pack(side="left")
            if hint:
                tk.Label(top_row, text="  " + hint, bg=CARD, fg=FG_DIM,
                         font=("Calibri", 8)).pack(side="left", pady=(2, 0))

            bottom_row = tk.Frame(content, bg=CARD)
            bottom_row.pack(fill="x", pady=(6, 0))
            bottom_row.columnconfigure(0, weight=1)

            entry_frame = tk.Frame(bottom_row, bg=ENTRY_BG)
            entry_frame.grid(row=0, column=0, sticky="ew")
            entry_frame.columnconfigure(0, weight=1)
            ttk.Entry(entry_frame, textvariable=var).grid(row=0, column=0, sticky="ew")

            ttk.Button(bottom_row, text="Browse", style="Browse.TButton",
                       command=browse_cmd).grid(row=0, column=1, padx=(8, 0))

        # ── Auto-fill project card ──────────────────────────────────────────
        proj_card = tk.Frame(files_inner, bg="#12131b", padx=0, pady=0)
        proj_card.grid(row=0, column=0, sticky="ew", pady=(10, 8), padx=28)
        proj_card.columnconfigure(1, weight=1)

        # gold left accent strip
        tk.Frame(proj_card, bg=ACCENT, width=3).grid(
            row=0, column=0, sticky="ns")

        proj_content = tk.Frame(proj_card, bg="#12131b", padx=16, pady=12)
        proj_content.grid(row=0, column=1, sticky="ew")
        proj_content.columnconfigure(0, weight=1)

        # Title row
        title_row = tk.Frame(proj_content, bg="#12131b")
        title_row.pack(fill="x")
        tk.Label(title_row, text="Project Root Folder",
                 bg="#12131b", fg=FG, font=("Calibri", 10, "bold")).pack(side="left")
        tk.Label(title_row,
                 text="  auto-fills all paths below",
                 bg="#12131b", fg=ACCENT, font=("Calibri", 8)).pack(side="left", pady=(2,0))

        # Explanation text
        tk.Label(proj_content,
                 text="Select your top-level project folder. The script will scan it and "
                      "fill in the XML, images folder, point cloud, and output path automatically. "
                      "You can still edit any field manually afterwards.",
                 bg="#12131b", fg=FG_DIM, font=("Calibri", 8),
                 wraplength=520, justify="left").pack(fill="x", pady=(4, 8))

        # Entry + Browse
        entry_row = tk.Frame(proj_content, bg="#12131b")
        entry_row.pack(fill="x")
        entry_row.columnconfigure(0, weight=1)

        proj_entry_bg = tk.Frame(entry_row, bg=ENTRY_BG)
        proj_entry_bg.grid(row=0, column=0, sticky="ew")
        proj_entry_bg.columnconfigure(0, weight=1)
        ttk.Entry(proj_entry_bg, textvariable=self.project_dir).grid(
            row=0, column=0, sticky="ew")

        ttk.Button(entry_row, text="Browse", style="Browse.TButton",
                   command=self.pick_project).grid(row=0, column=1, padx=(8, 0))

        # Output folder name row
        name_row = tk.Frame(proj_content, bg="#12131b")
        name_row.pack(fill="x", pady=(10, 0))
        tk.Label(name_row, text="Output folder name",
                 bg="#12131b", fg=FG_DIM,
                 font=("Calibri", 9)).pack(side="left", padx=(0, 10))

        # Base name entry
        name_entry_bg = tk.Frame(name_row, bg=ENTRY_BG)
        name_entry_bg.pack(side="left")
        ttk.Entry(name_entry_bg, textvariable=self.output_folder_name, width=18).pack()

        # Version spinbox — 001 to 999
        tk.Label(name_row, text="_", bg="#12131b", fg=FG_DIM,
                 font=("Calibri", 10)).pack(side="left", padx=(6, 0))
        tk.Spinbox(
            name_row,
            from_=1, to=999,
            textvariable=self.output_version,
            width=4,
            format="%03.0f",
            bg=ENTRY_BG, fg=FG,
            buttonbackground=CARD,
            relief="flat",
            highlightthickness=0,
            insertbackground=FG,
            font=("Calibri", 10),
            command=self._on_version_spin
        ).pack(side="left")

        tk.Label(name_row,
                 text="  spin to version up  —  each number is a separate folder",
                 bg="#12131b", fg=FG_DIM, font=("Calibri", 8)).pack(side="left", padx=(8, 0))

        # Hints row — one per auto-filled field
        hints_row = tk.Frame(proj_content, bg="#12131b")
        hints_row.pack(fill="x", pady=(8, 0))
        hints = [
            ("Cameras XML",   "looks for *.xml at root level"),
            ("Sphere XML",    "looks for *sphere*.xml at root level"),
            ("Images folder", "looks for folder named 'images/' — searches all subfolders for panoramas"),
            ("Point cloud",   "looks for *.ply anywhere inside the project"),
            ("Output folder", "creates <output folder name>/ inside the project — change name or version up above"),
        ]
        for i, (field, desc) in enumerate(hints):
            row_f = tk.Frame(hints_row, bg="#12131b")
            row_f.pack(fill="x", pady=1)
            tk.Label(row_f, text=f"  {field}",
                     bg="#12131b", fg=ACCENT, font=("Calibri", 8, "bold"),
                     width=16, anchor="w").pack(side="left")
            tk.Label(row_f, text=desc,
                     bg="#12131b", fg=FG_DIM, font=("Calibri", 8)).pack(side="left")

        # Thin divider
        tk.Frame(files_inner, bg=BORDER, height=1).grid(
            row=1, column=0, sticky="ew", padx=28, pady=(0, 6))

        _make_file_card(files_inner, 3, "xml1", "Metashape Cameras XML",
                  "mixed frame + spherical", self.xml, self.pick_xml)
        _make_file_card(files_inner, 4, "xml2", "Spherical-only Cameras XML",
                  "optional — camerassphere.xml", self.sphere_xml, self.pick_sphere_xml)
        _make_file_card(files_inner, 5, "img1", "Input Image Root Folder",
                  "searches all subfolders recursively for panoramas", self.panos, self.pick_panos)
        _make_file_card(files_inner, 6, "ply",  "Point Cloud  ·  PLY",
                  "optional", self.ply, self.pick_ply)
        _make_file_card(files_inner, 7, "out",  "Output Folder",
                  "COLMAP sparse/ + images/", self.out, self.pick_out)

        # ── Options strip ────────────────────────────────────────────────────
        opts_row = tk.Frame(files_inner, bg=BG)
        opts_row.grid(row=8, column=0, sticky="ew", padx=28, pady=(6, 0))

        skip_cb = tk.Checkbutton(opts_row,
            text="Skip existing frames",
            variable=self.skip_existing,
            bg=BG, fg=FG_DIM, selectcolor=CARD,
            activebackground=BG, activeforeground=FG,
            font=("Calibri", 9))
        skip_cb.pack(side="left", padx=(0, 8))
        tk.Label(opts_row,
            text="— skips panoramas whose output views already exist (resume interrupted jobs)",
            bg=BG, fg=FG_DIM, font=("Calibri", 8)).pack(side="left", padx=(0, 32))

        style.configure("DryRun.TButton",
            background="#1a2535", foreground=ACCENT2,
            bordercolor=ACCENT2, relief="flat",
            font=("Calibri", 9, "bold"), padding=(12, 5))
        style.map("DryRun.TButton",
            background=[("active", "#223040"), ("pressed", "#0f1820")],
            foreground=[("active", "#8abde8")])

        ttk.Button(opts_row, text="▶  Dry Run  (first 3)",
            style="DryRun.TButton",
            command=lambda: self._launch(dry=True)).pack(side="right")

        tk.Frame(files_inner, bg=BG, height=10).grid(row=9, column=0, sticky="ew")


        # COLMAP Rig tab
        tab_colmap=ttk.Frame(nb,style='TFrame')
        nb.add(tab_colmap,text='  COLMAP Rig  ')
        cr=_make_scrollable(tab_colmap)

        tk.Frame(cr,bg=BG,height=10).grid(row=0,column=0,sticky='ew')
        _section_header(cr,1,'360 Camera → COLMAP  (no Metashape required)')
        ic=_card(cr,2)
        tk.Label(ic,bg=CARD,fg=FG_DIM,font=('Calibri',9),justify='left',
            text='Converts equirectangular panoramas into perspective crops then runs the full COLMAP\n'
                 'rig pipeline. COLMAP is told all crops from each panorama share the same position\n'
                 '— only orientations differ — making matching faster than brute-force.\n'
                 'View directions from Views & Export tab. COLMAP exe from Launchers tab.'
        ).pack(anchor='w')
        _section_header(cr,3,'Folders')
        pc=_card(cr,4); pc.columnconfigure(1,weight=1)
        def _crr(parent,row,label,hint,var):
            tk.Label(parent,text=label,bg=CARD,fg=FG,font=('Calibri',9,'bold'),width=12,anchor='e'
            ).grid(row=row*2,column=0,sticky='e',padx=(0,10),pady=(8,0))
            ef=tk.Frame(parent,bg=CARD); ef.grid(row=row*2,column=1,sticky='ew',pady=(8,0))
            tk.Entry(ef,textvariable=var,bg=ENTRY_BG,fg=FG,insertbackground=FG,
                relief='flat',font=('Calibri',9)).pack(side='left',fill='x',expand=True,ipady=5,padx=(0,6))
            tk.Button(ef,text='Browse…',
                command=lambda v=var:v.set(filedialog.askdirectory() or v.get()),
                bg=CARD2,fg=FG_DIM,relief='flat',font=('Calibri',8),cursor='hand2').pack(side='left')
            tk.Label(parent,text=hint,bg=CARD,fg=FG_DIM,font=('Calibri',8)
            ).grid(row=row*2+1,column=1,sticky='w',pady=(0,2))
        _crr(pc,0,'Panoramas','folder with equirectangular images',self.rig_panos)
        _crr(pc,1,'Output','where images/, sparse/, rig_config.json will be written  (database.db goes one level up)',self.rig_out)
        _section_header(cr,5,'Pipeline Steps')
        sc2=_card(cr,6)
        def _sr(p,row,var,label,hint):
            tk.Checkbutton(p,variable=var,text=label,bg=CARD,fg=FG,selectcolor=ENTRY_BG,
                activebackground=CARD,activeforeground=FG,font=('Calibri',9,'bold')
            ).grid(row=row*2,column=0,sticky='w',pady=(6,0))
            tk.Label(p,text=f'   {hint}',bg=CARD,fg=FG_DIM,font=('Calibri',8)
            ).grid(row=row*2+1,column=0,sticky='w',pady=(0,3))
        _sr(sc2,0,self.rig_step_extract,'1  Feature extraction','colmap feature_extractor')
        _sr(sc2,1,self.rig_step_rig_cfg,'2  Rig configuration','colmap rig_configurator  —  locks known rotations')
        _sr(sc2,2,self.rig_step_match,'3  Feature matching','matches features between images')
        _sr(sc2,3,self.rig_step_map,'4  Mapping','colmap mapper  —  reconstructs camera positions')
        _section_header(cr,7,'Matching Options')
        mc2=_card(cr,8); mc2.columnconfigure(1,weight=1)
        tk.Label(mc2,text='Matcher',bg=CARD,fg=FG,font=('Calibri',9,'bold')
        ).grid(row=0,column=0,sticky='nw',padx=(0,12),pady=6)
        mf=tk.Frame(mc2,bg=CARD); mf.grid(row=0,column=1,sticky='w')
        for mv,ml,mt in[('sequential','Sequential','fast, good for walkthroughs'),
                         ('exhaustive','Exhaustive','better for small unordered sets'),
                         ('vocab_tree','Vocab Tree','scalable for very large datasets')]:
            tk.Radiobutton(mf,text=f'{ml}  —  {mt}',variable=self.rig_matcher,value=mv,
                bg=CARD,fg=FG,selectcolor=ENTRY_BG,activebackground=CARD,font=('Calibri',9)
            ).pack(anchor='w',pady=2)
        _section_header(cr,9,'Run')
        rc2=_card(cr,10); rc2.columnconfigure(0,weight=1)
        self.rig_status=tk.StringVar(value='Ready')
        style.configure('RigRun.TButton',background='#1a2e1a',foreground=ACCENT3,
            bordercolor=ACCENT3,relief='flat',font=('Calibri',11,'bold'),padding=(20,10))
        style.map('RigRun.TButton',background=[('active','#223222'),('disabled','#1a1a1a')],
            foreground=[('disabled','#445544')])
        style.configure('RigStop.TButton',background='#2e1a1a',foreground='#e06060',
            bordercolor='#e06060',relief='flat',font=('Calibri',10,'bold'),padding=(14,8))
        style.map('RigStop.TButton',background=[('active','#3d2020'),('disabled','#1a1a1a')],
            foreground=[('disabled','#664444')])
        br=tk.Frame(rc2,bg=CARD); br.pack(fill='x')
        self._btn_rig_run=ttk.Button(br,text='▶  Run COLMAP Pipeline',
            style='RigRun.TButton',command=self._rig_run)
        self._btn_rig_run.pack(side='left',padx=(0,10))
        self._btn_rig_stop=ttk.Button(br,text='■  Stop',
            style='RigStop.TButton',command=self._rig_stop)
        self._btn_rig_stop.pack(side='left')
        self._btn_rig_stop.state(['disabled'])
        tk.Label(rc2,textvariable=self.rig_status,bg=CARD,fg=FG,
            font=('Calibri',9)).pack(anchor='w',pady=(8,2))
        self._rig_log=tk.Text(rc2,height=10,bg=ENTRY_BG,fg=FG,
            font=('Calibri',8),relief='flat',state='disabled',wrap='word')
        self._rig_log.pack(fill='x',pady=(0,4))
        tk.Frame(cr,bg=BG,height=10).grid(row=11,column=0,sticky='ew')

        # ════════════════════════════════════════════
        # TAB 2 – Views & Export
        # ════════════════════════════════════════════
        tab_views=ttk.Frame(nb,style='TFrame')
        nb.add(tab_views,text='  Views & Export  ')
        views_inner=_make_scrollable(tab_views)

        tk.Frame(views_inner,bg=BG,height=10).grid(row=0,column=0,sticky='ew')

        _section_header(views_inner, 1, "Camera Rig")
        rig_card = _card(views_inner, 2)

        # Base view set row
        tk.Label(rig_card, text="Base set", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).grid(row=0, column=0, sticky="w", pady=(0, 8))
        cb = ttk.Combobox(rig_card, textvariable=self.view_mode,
                          values=["cubemap6"], state="readonly", width=20)
        cb.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 8))

        tk.Label(rig_card, text="Extras", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).grid(row=1, column=0, sticky="nw", pady=(4, 0))
        chk_frame = tk.Frame(rig_card, bg=CARD)
        chk_frame.grid(row=1, column=1, sticky="w", padx=(12, 0))
        for txt, var in [
            ("Horizon diagonals  ±45°", self.add_diag45),
            ("Upper ring  pitch +45°",  self.add_upper45),
            ("Lower ring  pitch −45°",  self.add_lower45),
        ]:
            cb2 = tk.Checkbutton(chk_frame, text=txt, variable=var,
                                  bg=CARD, fg=FG_MID, selectcolor=ENTRY_BG,
                                  activebackground=CARD, activeforeground=FG,
                                  font=("Calibri", 10))
            cb2.pack(anchor="w", pady=3)

        # diagram — on right side
        self.model_canvas = tk.Canvas(rig_card, width=140, height=100,
                                       bg=CARD, highlightthickness=1,
                                       highlightbackground=BORDER2)
        self.model_canvas.grid(row=0, column=2, rowspan=3, padx=(28, 0), pady=(0, 4), sticky="n")
        for _v in (self.add_diag45, self.add_upper45, self.add_lower45, self.view_mode):
            try: _v.trace_add("write", lambda *a: self._draw_model_diagram())
            except Exception: pass
        self.after(50, self._draw_model_diagram)

        _section_header(views_inner, 3, "Render Settings")
        rnd_card = _card(views_inner, 4)

        def _field_pair(parent, row, label1, var1, label2, var2, w=10):
            tk.Label(parent, text=label1, bg=CARD, fg=FG_DIM,
                     font=("Calibri", 9)).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(parent, textvariable=var1, width=w).grid(
                row=row, column=1, sticky="w", padx=(12, 30), pady=5)
            tk.Label(parent, text=label2, bg=CARD, fg=FG_DIM,
                     font=("Calibri", 9)).grid(row=row, column=2, sticky="w", pady=5)
            ttk.Entry(parent, textvariable=var2, width=w).grid(
                row=row, column=3, sticky="w", padx=(12, 0), pady=5)

        _field_pair(rnd_card, 0,
                    "Output resolution  px", self.out_size,
                    "FOV  degrees", self.fov_deg)

        tk.Label(rnd_card, text="Exclude yaw", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).grid(row=1, column=0, sticky="w", pady=5)
        yaw_entry = ttk.Entry(rnd_card, textvariable=self.exclude_yaw, width=26)
        yaw_entry.grid(row=1, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=5)
        tk.Label(rnd_card, text="e.g.  315-45  or  30-60,150-210",
                 bg=CARD, fg=FG_DIM, font=("Calibri", 8)).grid(
                 row=2, column=1, columnspan=3, sticky="w", padx=(12, 0))

        flip_row = tk.Frame(rnd_card, bg=CARD)
        flip_row.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 4))

        tk.Checkbutton(flip_row,
                       text="180° X-flip  (fix upside-down scene)",
                       variable=self.flip_world_180x,
                       bg=CARD, fg=FG_MID, selectcolor=ENTRY_BG,
                       activebackground=CARD, activeforeground=FG,
                       font=("Calibri", 10)).pack(side="left", padx=(0, 24))

        tk.Checkbutton(flip_row,
                       text="180° Y-flip  (fix left-right mirror)",
                       variable=self.flip_world_180y,
                       bg=CARD, fg=FG_MID, selectcolor=ENTRY_BG,
                       activebackground=CARD, activeforeground=FG,
                       font=("Calibri", 10)).pack(side="left")

        ttk.Button(rnd_card, text="Preview & Approve Directions",
                   style="Action.TButton",
                   command=self.open_preview_window)\
            .grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 2))

        # ════════════════════════════════════════════
        # TAB 3 – Performance
        # ════════════════════════════════════════════
        tab_perf=ttk.Frame(nb,style='TFrame')
        nb.add(tab_perf,text='  Performance  ')
        perf_inner=_make_scrollable(tab_perf)

        tk.Frame(perf_inner,bg=BG,height=10).grid(row=0,column=0,sticky='ew')
        _section_header(perf_inner, 1, "Parallel Processing")

        perf_card = _card(perf_inner, 2)

        cpu_count = multiprocessing.cpu_count() or 4
        cpu_bar = tk.Frame(perf_card, bg=CARD)
        cpu_bar.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 14))
        tk.Label(cpu_bar, text=f"{cpu_count}", bg=CARD, fg=ACCENT,
                 font=("Calibri", 13, "bold")).pack(side="left")
        tk.Label(cpu_bar, text="  CPU cores detected", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).pack(side="left", pady=(4, 0))

        def _lbl(p, t, r, c, **kw):
            tk.Label(p, text=t, bg=CARD, fg=FG_DIM,
                     font=("Calibri", 9), **kw).grid(
                     row=r, column=c, sticky="w", padx=(0, 8), pady=4)
        def _entry(p, v, r, c, w=9):
            ttk.Entry(p, textvariable=v, width=w).grid(
                row=r, column=c, sticky="w", padx=(0, 28), pady=4)

        _lbl(perf_card, "Worker processes", 1, 0)
        _entry(perf_card, self.workers, 1, 1, w=8)
        _lbl(perf_card, "Tile height  px", 1, 2)
        _entry(perf_card, self.tile_h, 1, 3, w=8)

        tk.Frame(perf_card, bg=BORDER, height=1).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(12, 10))

        tk.Label(perf_card,
                 text="Higher worker count → faster on multi-core machines.\n"
                      "Smaller tile height → reduced peak RAM per worker.",
                 bg=CARD, fg=FG_DIM, font=("Calibri", 9),
                 justify="left").grid(row=3, column=0, columnspan=4, sticky="w")

        # ── GPU section ──────────────────────────────────────────────────────
        tk.Frame(perf_inner, bg=BG, height=6).grid(row=3, column=0, sticky="ew")
        _section_header(perf_inner, 4, "GPU Acceleration")
        gpu_card = _card(perf_inner, 5)
        gpu_card.columnconfigure(1, weight=1)

        # Detect GPU status
        if _cupy_ok:
            try:
                import cupy as _cp
                _dev = _cp.cuda.Device(0)
                _dev.use()
                _gpu_name = _cp.cuda.runtime.getDeviceProperties(0).get("name", b"GPU")
                if isinstance(_gpu_name, bytes):
                    _gpu_name = _gpu_name.decode("utf-8", errors="replace").rstrip("\x00")
            except Exception:
                _gpu_name = "GPU detected"
            gpu_status_text  = f"✓  {_gpu_name}"
            gpu_status_color = ACCENT3
            gpu_note = ("GPU mode renders the full image in one pass on VRAM — much faster for large batches.\n"
                        "Worker count is automatically set to 1 when GPU mode is active.")
        else:
            gpu_status_text  = "✗  No CUDA GPU found  (CuPy not installed or no compatible GPU)"
            gpu_status_color = FG_DIM
            gpu_note = ("Install CuPy to enable GPU acceleration. See the Dependencies tab for instructions.\n"
                        "CPU mode with multiple workers is used as fallback.")

        tk.Label(gpu_card, text="GPU status", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).grid(row=0, column=0, sticky="w", padx=(0, 16), pady=(0, 8))
        tk.Label(gpu_card, text=gpu_status_text, bg=CARD, fg=gpu_status_color,
                 font=("Calibri", 10, "bold")).grid(row=0, column=1, sticky="w", pady=(0, 8))

        gpu_cb = tk.Checkbutton(gpu_card,
                                 text="Use GPU acceleration",
                                 variable=self.use_gpu,
                                 state="normal" if _cupy_ok else "disabled",
                                 bg=CARD, fg=FG_MID, selectcolor=ENTRY_BG,
                                 disabledforeground=FG_DIM,
                                 activebackground=CARD, activeforeground=FG,
                                 font=("Calibri", 10))
        gpu_cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Batch size — auto-detect from VRAM
        if _cupy_ok:
            try:
                _free_vram = cp.cuda.Device(0).mem_info[0]
                _vram_gb   = _free_vram / (1024**3)
                # Estimate for 2048px output, 8K pano source, 6 views
                _est_bytes = (7680*3840*3) + (2048*2048*3*6)
                _auto_batch = max(1, min(64, int((_free_vram * 0.75) / _est_bytes)))
                self.gpu_batch_size.set(_auto_batch)
                _vram_label = f"Auto-detected  —  {_vram_gb:.1f} GB free  →  batch {_auto_batch}"
            except Exception:
                _vram_label = "Could not read VRAM"
        else:
            _vram_label = "GPU not available"

        batch_frame = tk.Frame(gpu_card, bg=CARD)
        batch_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))
        tk.Label(batch_frame, text="Batch size", bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9)).pack(side="left")
        ttk.Entry(batch_frame, textvariable=self.gpu_batch_size, width=6).pack(side="left", padx=(10, 12))
        tk.Label(batch_frame, text=_vram_label, bg=CARD, fg=ACCENT if _cupy_ok else FG_DIM,
                 font=("Calibri", 8)).pack(side="left")

        tk.Label(gpu_card, text=gpu_note, bg=CARD, fg=FG_DIM,
                 font=("Calibri", 9), justify="left", wraplength=480).grid(
                 row=3, column=0, columnspan=2, sticky="w")

        # ════════════════════════════════════════════
        # TAB 4 – Dependencies
        # ════════════════════════════════════════════
        tab_deps=ttk.Frame(nb,style='TFrame')
        deps_inner=_make_scrollable(tab_deps)

        tk.Frame(deps_inner,bg=BG,height=10).grid(row=0,column=0,sticky='ew')
        _section_header(deps_inner, 1, "Required Packages")

        deps_card = _card(deps_inner, 2)
        deps_card.columnconfigure(1, weight=1)

        # Check status of each dependency
        deps = [
            {
                "package":  "numpy",
                "import":   "numpy",
                "install":  "pip install numpy",
                "purpose":  "Matrix maths, image array processing",
                "ok":       _numpy_ok,
            },
            {
                "package":  "Pillow",
                "import":   "PIL",
                "install":  "pip install Pillow",
                "purpose":  "Image loading, resizing, saving (PIL)",
                "ok":       _pillow_ok,
            },
            {
                "package":  "CuPy",
                "import":   "cupy",
                "install":  "pip install cupy-cuda12x",
                "purpose":  "GPU acceleration (optional — requires NVIDIA GPU + CUDA)",
                "ok":       _cupy_ok,
            },
        ]

        required_deps = [d for d in deps if "optional" not in d["purpose"].lower()]
        all_ok = all(d["ok"] for d in required_deps)

        # Status banner
        banner_bg = "#1a2e1a" if all_ok else "#2a1a1a"
        banner_fg = ACCENT3 if all_ok else "#e06060"
        banner_text = "All dependencies found — ready to convert." if all_ok                       else "One or more dependencies are missing. Install them below."
        banner = tk.Frame(deps_card, bg=banner_bg, padx=12, pady=8)
        banner.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        tk.Label(banner, text=banner_text, bg=banner_bg, fg=banner_fg,
                 font=("Calibri", 10, "bold")).pack(anchor="w")

        # Column headers
        for col, txt in enumerate(["Package", "Status", "Purpose"]):
            tk.Label(deps_card, text=txt.upper(), bg=CARD, fg=FG_DIM,
                     font=("Calibri", 8, "bold")).grid(
                     row=1, column=col, sticky="w",
                     padx=(0, 20) if col < 2 else 0, pady=(0, 6))
        tk.Frame(deps_card, bg=BORDER, height=1).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        for i, dep in enumerate(deps):
            row = i + 3
            ok = dep["ok"]
            status_color = ACCENT3 if ok else "#e06060"
            status_text  = "✓  Installed" if ok else "✗  Missing"

            tk.Label(deps_card, text=dep["package"], bg=CARD, fg=FG,
                     font=("Calibri", 10, "bold")).grid(
                     row=row, column=0, sticky="w", padx=(0, 20), pady=5)
            tk.Label(deps_card, text=status_text, bg=CARD, fg=status_color,
                     font=("Calibri", 10)).grid(
                     row=row, column=1, sticky="w", padx=(0, 20), pady=5)
            tk.Label(deps_card, text=dep["purpose"], bg=CARD, fg=FG_DIM,
                     font=("Calibri", 9)).grid(
                     row=row, column=2, sticky="w", pady=5)

        tk.Frame(deps_card, bg=BORDER, height=1).grid(
            row=len(deps)+3, column=0, columnspan=3, sticky="ew", pady=(8, 12))

        # ── Auto-install section ─────────────────────────────────────────────
        _section_header(deps_inner, 3, "Auto Install")
        inst_card = _card(deps_inner, 4)
        inst_card.columnconfigure(0, weight=1)

        tk.Label(inst_card,
                 text="Click a button to install the package automatically using pip.\n"
                      "A live log will show progress. Restart the script after installing.",
                 bg=CARD, fg=FG_DIM, font=("Calibri", 9),
                 justify="left").grid(row=0, column=0, sticky="w", pady=(0, 12))

        # One install button per package
        btn_row = tk.Frame(inst_card, bg=CARD)
        btn_row.grid(row=1, column=0, sticky="w", pady=(0, 12))

        style.configure("Install.TButton",
            background="#1e2a1e", foreground=ACCENT3,
            bordercolor=ACCENT3, relief="flat",
            font=("Calibri", 9, "bold"), padding=(14, 7))
        style.map("Install.TButton",
            background=[("active", "#253225"), ("pressed", "#111a11"),
                        ("disabled", "#1a1a1a")],
            foreground=[("disabled", "#445544")])

        style.configure("InstallOpt.TButton",
            background="#1a2535", foreground=ACCENT2,
            bordercolor=ACCENT2, relief="flat",
            font=("Calibri", 9, "bold"), padding=(14, 7))
        style.map("InstallOpt.TButton",
            background=[("active", "#223040"), ("pressed", "#0f1820"),
                        ("disabled", "#1a1a1a")],
            foreground=[("disabled", "#334455")])

        # Log output area
        log_frame = tk.Frame(inst_card, bg=ENTRY_BG)
        log_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)

        self._install_log = tk.Text(log_frame, height=8, bg=ENTRY_BG, fg=ACCENT,
                                    font=("Consolas", 9), relief="flat",
                                    state="disabled", wrap="word",
                                    insertbackground=FG)
        self._install_log.pack(fill="x", padx=10, pady=8)

        tk.Label(inst_card,
                 text="Python 3.10+ recommended  ·  For CuPy match your CUDA version: cupy-cuda11x, cupy-cuda12x etc.",
                 bg=CARD, fg=FG_DIM, font=("Calibri", 8)).grid(
                 row=3, column=0, sticky="w")

        def _run_install(pip_args, btn):
            """Run pip install in a background thread, stream output to log."""
            import subprocess, sys, os as _os

            btn.state(["disabled"])
            self._install_log.configure(state="normal")
            self._install_log.delete("1.0", "end")
            self._install_log.configure(state="disabled")

            def _find_python():
                """
                Find the best Python executable to use for pip.
                Tries in order:
                  1. sys.executable  — only if NOT a frozen .exe (PyInstaller)
                  2. 'py -3'         — Windows Python Launcher (works even if not on PATH)
                  3. 'python'        — may work if on PATH
                """
                import sys as _sys
                # Option 1: use the exact Python running us — but NOT if we're a frozen exe
                # (sys.executable would point to our .exe, not python.exe)
                if not getattr(_sys, "frozen", False):
                    if _sys.executable and _os.path.isfile(_sys.executable):
                        return [_sys.executable], False

                # Option 2: Windows py launcher
                try:
                    r = subprocess.run(["py", "-3", "--version"],
                                       capture_output=True, timeout=5)
                    if r.returncode == 0:
                        return ["py", "-3"], False
                except Exception:
                    pass

                # Option 3: bare python on PATH
                try:
                    r = subprocess.run(["python", "--version"],
                                       capture_output=True, timeout=5)
                    if r.returncode == 0:
                        return ["python"], False
                except Exception:
                    pass

                return None, False

            def _worker():
                py_cmd, _ = _find_python()

                if py_cmd is None:
                    self._log_install(
                        "✗  Could not locate Python.\n\n"
                        "Please install Python 3.10+ from https://python.org\n"
                        "Make sure to tick 'Add Python to PATH' during installation.\n"
                    )
                    self.after(0, lambda: btn.state(["!disabled"]))
                    return

                cmd = py_cmd + ["-m", "pip", "install"] + pip_args
                self._log_install(f"Using: {' '.join(cmd)}\n\n")

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    )
                    already_satisfied = []
                    for line in proc.stdout:
                        self._log_install(line)
                        if "already satisfied" in line.lower():
                            # Extract package name from pip's message
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                already_satisfied.append(parts[3])
                    proc.wait()

                    if proc.returncode == 0:
                        if already_satisfied:
                            self._log_install(
                                f"\n✓  Already installed: {', '.join(already_satisfied)}\n"
                                "   Nothing to do — package is up to date.\n"
                            )
                        else:
                            self._log_install(
                                "\n✓  Installation complete!\n"
                                "   Restarting script in 3 seconds...\n"
                            )
                            for i in (2, 1):
                                self.after((3 - i) * 1000,
                                    lambda n=i: self._log_install(f"   Restarting in {n}...\n"))
                            self.after(3000, self._restart_self)
                    else:
                        self._log_install(
                            f"\n✗  pip exited with code {proc.returncode}\n\n"
                            "Possible fixes:\n"
                            "  • Run the script as Administrator\n"
                            "  • Or install manually in a terminal:\n"
                            f"      python -m pip install {' '.join(pip_args)}\n"
                        )
                except FileNotFoundError:
                    self._log_install(
                        "✗  Could not launch pip.\n\n"
                        "Please open a Command Prompt and run:\n"
                        f"    python -m pip install {' '.join(pip_args)}\n"
                    )
                except Exception as e:
                    self._log_install(f"\n✗  Unexpected error: {e}\n")

                self.after(0, lambda: btn.state(["!disabled"]))

            threading.Thread(target=_worker, daemon=True).start()

        ttk.Button(btn_row, text="Install  numpy",
                   style="Install.TButton",
                   command=lambda: _run_install(["numpy"], _b_numpy)
                   ).pack(side="left", padx=(0, 8))
        _b_numpy = btn_row.winfo_children()[-1]

        ttk.Button(btn_row, text="Install  Pillow",
                   style="Install.TButton",
                   command=lambda: _run_install(["Pillow"], _b_pillow)
                   ).pack(side="left", padx=(0, 8))
        _b_pillow = btn_row.winfo_children()[-1]

        ttk.Button(btn_row, text="Install  numpy + Pillow  (both)",
                   style="Install.TButton",
                   command=lambda: _run_install(["numpy", "Pillow"], _b_both)
                   ).pack(side="left", padx=(0, 16))
        _b_both = btn_row.winfo_children()[-1]

        ttk.Button(btn_row, text="Install  CuPy  (CUDA 12)",
                   style="InstallOpt.TButton",
                   command=lambda: _run_install(["cupy-cuda12x"], _b_cupy)
                   ).pack(side="left", padx=(0, 0))
        _b_cupy = btn_row.winfo_children()[-1]

        # Fix button references now they exist (lambda captured by name above)
        _b_numpy  = btn_row.winfo_children()[0]
        _b_pillow = btn_row.winfo_children()[1]
        _b_both   = btn_row.winfo_children()[2]
        _b_cupy   = btn_row.winfo_children()[3]


        # ════════════════════════════════════════════
        # TAB 5 – Queue
        # ════════════════════════════════════════════
        tab_queue = ttk.Frame(nb, style="TFrame")
        nb.add(tab_queue, text="  Queue  ")
        tab_queue.columnconfigure(0, weight=1)
        tab_queue.rowconfigure(1, weight=1)

        tk.Frame(tab_queue, bg=BG, height=8).grid(row=0, column=0, sticky="ew")

        # ── Header row ────────────────────────────────────────────────────────
        q_header = tk.Frame(tab_queue, bg=BG)
        q_header.grid(row=0, column=0, sticky="ew", padx=28, pady=(10,4))

        tk.Label(q_header, text="Job Queue",
                 bg=BG, fg=FG, font=("Calibri", 11, "bold")).pack(side="left")
        tk.Label(q_header,
                 text="  set up paths in the Files tab, then click  '+ Add to Queue'  in the bottom bar",
                 bg=BG, fg=FG_DIM, font=("Calibri", 8)).pack(side="left", pady=(3,0))

        style.configure("QAdd.TButton",
            background="#1a2535", foreground=ACCENT2,
            bordercolor=ACCENT2, relief="flat",
            font=("Calibri", 9, "bold"), padding=(12,5))
        style.map("QAdd.TButton",
            background=[("active","#223040"),("pressed","#0f1820"),("disabled","#1a1a1a")],
            foreground=[("disabled","#334455")])
        style.configure("QRun.TButton",
            background="#1a2e1a", foreground=ACCENT3,
            bordercolor=ACCENT3, relief="flat",
            font=("Calibri", 9, "bold"), padding=(12,5))
        style.map("QRun.TButton",
            background=[("active","#223222"),("pressed","#111a11"),("disabled","#1a1a1a")],
            foreground=[("disabled","#445544")])
        style.configure("QStop.TButton",
            background="#2e1a1a", foreground="#e06060",
            bordercolor="#e06060", relief="flat",
            font=("Calibri", 9, "bold"), padding=(12,5))
        style.map("QStop.TButton",
            background=[("active","#3d2020"),("disabled","#1a1a1a")],
            foreground=[("disabled","#664444")])
        style.configure("QDel.TButton",
            background=CARD, foreground=FG_DIM,
            relief="flat", font=("Calibri", 8), padding=(6,3))
        style.map("QDel.TButton",
            background=[("active","#2a1a1a")],
            foreground=[("active","#e06060")])

        btn_bar = tk.Frame(tab_queue, bg=BG)
        btn_bar.grid(row=1, column=0, sticky="ew", padx=28, pady=(0,6))

        self._btn_q_save  = ttk.Button(btn_bar, text="Save Queue",
                                        style="QAdd.TButton",
                                        command=self._q_save)
        self._btn_q_save.pack(side="left", padx=(0,8))

        self._btn_q_load  = ttk.Button(btn_bar, text="Load Queue",
                                        style="QAdd.TButton",
                                        command=self._q_load)
        self._btn_q_load.pack(side="left", padx=(0,16))

        self._btn_q_run   = ttk.Button(btn_bar, text="▶  Run Queue",
                                        style="QRun.TButton",
                                        command=self._q_run)
        self._btn_q_run.pack(side="left", padx=(0,8))

        self._btn_q_stop  = ttk.Button(btn_bar, text="■  Stop",
                                        style="QStop.TButton",
                                        command=self._q_stop)
        self._btn_q_stop.pack(side="left")
        self._btn_q_stop.state(["disabled"])

        # ── Job list ──────────────────────────────────────────────────────────
        list_frame = tk.Frame(tab_queue, bg=BG)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0,8))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        tab_queue.rowconfigure(2, weight=1)

        self._q_canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
        self._q_canvas.grid(row=0, column=0, sticky="nsew")
        q_scroll = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self._q_canvas.yview)
        q_scroll.grid(row=0, column=1, sticky="ns")
        self._q_canvas.configure(yscrollcommand=q_scroll.set)

        self._q_inner = tk.Frame(self._q_canvas, bg=BG)
        self._q_win   = self._q_canvas.create_window((0,0), window=self._q_inner, anchor="nw")
        self._q_inner.bind("<Configure>",
            lambda e: self._q_canvas.configure(
                scrollregion=self._q_canvas.bbox("all")))
        self._q_canvas.bind("<Configure>",
            lambda e: self._q_canvas.itemconfigure(self._q_win, width=e.width))

        # summary label at bottom
        self._q_summary = tk.StringVar(value="No jobs in queue.")
        tk.Label(tab_queue, textvariable=self._q_summary,
                 bg=BG, fg=FG_DIM, font=("Calibri", 8)
                 ).grid(row=3, column=0, sticky="w", padx=28, pady=(0,4))

        # ════════════════════════════════════════════
        # TAB 6 – Launchers
        # ════════════════════════════════════════════
        tab_launch=ttk.Frame(nb,style='TFrame')
        nb.add(tab_launch,text='  Launchers  ')
        nb.add(tab_deps,text='  Dependencies  ')
        launch_inner=_make_scrollable(tab_launch)

        tk.Frame(launch_inner,bg=BG,height=10).grid(row=0,column=0,sticky='ew')
        _section_header(launch_inner, 1, "Post-Conversion App Launchers")

        launch_card = _card(launch_inner, 2)
        launch_card.columnconfigure(2, weight=1)

        tk.Label(launch_card,
            text="After each conversion finishes, ticked apps launch automatically.\n"
                 "COLMAP runs first (txt \u2192 bin), then viewers open. All are optional.",
            bg=CARD, fg=FG_DIM, font=("Calibri", 8), justify="left"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        def _make_smooth_toggle(parent, var):
            W, H, SCALE = 52, 26, 4
            SW, SH = W*SCALE, H*SCALE
            R = SH // 2
            COL_ON, COL_OFF = (78,203,141), (55,55,70)
            BG_RGB = tuple(int(CARD.lstrip('#')[i:i+2],16) for i in (0,2,4))
            def _render(on):
                from PIL import Image, ImageDraw
                img = Image.new('RGBA',(SW,SH),(0,0,0,0))
                d = ImageDraw.Draw(img)
                tc = COL_ON if on else COL_OFF
                d.ellipse([0,0,SH-1,SH-1],fill=(*tc,255))
                d.ellipse([SW-SH,0,SW-1,SH-1],fill=(*tc,255))
                d.rectangle([R,0,SW-R,SH-1],fill=(*tc,255))
                kx = SW-R-2*SCALE if on else R+2*SCALE
                ks = int(R*0.72)
                d.ellipse([kx-ks-SCALE,R-ks-SCALE,kx+ks+SCALE,R+ks+SCALE],fill=(0,0,0,60))
                d.ellipse([kx-ks,R-ks,kx+ks,R+ks],fill=(255,255,255,255))
                img = img.resize((W,H),Image.LANCZOS)
                bg = Image.new('RGB',(W,H),BG_RGB)
                bg.paste(img,mask=img.split()[3])
                return ImageTk.PhotoImage(bg)
            lbl = tk.Label(parent,bg=CARD,cursor='hand2')
            imgs = {}
            def _update(*_):
                key = 'on' if var.get() else 'off'
                if key not in imgs: imgs[key] = _render(var.get())
                lbl.configure(image=imgs[key])
            lbl.bind('<Button-1>', lambda e: (var.set(not var.get())))
            if _pillow_ok:
                _update()
                var.trace_add('write', _update)
            else:
                return tk.Checkbutton(parent,variable=var,bg=CARD,
                    selectcolor=ENTRY_BG,activebackground=CARD,fg=FG)
            return lbl

        def _app_row(parent, row, icon, label, hint, var_path, var_enable, pick_cmd):
            tog = _make_smooth_toggle(parent, var_enable)
            tog.grid(row=row, column=0, sticky='w', pady=8, padx=(0,4))
            name_f = tk.Frame(parent, bg=CARD)
            name_f.grid(row=row, column=1, sticky="w", padx=(4,16))
            tk.Label(name_f, text=icon, bg=CARD, fg=ACCENT,
                     font=("Calibri", 13)).pack(side="left")
            lbl_f = tk.Frame(name_f, bg=CARD)
            lbl_f.pack(side="left", padx=(6,0))
            tk.Label(lbl_f, text=label, bg=CARD, fg=FG,
                     font=("Calibri", 10, "bold"), anchor="w").pack(anchor="w")
            tk.Label(lbl_f, text=hint, bg=CARD, fg=FG_DIM,
                     font=("Calibri", 8), anchor="w").pack(anchor="w")
            entry_f = tk.Frame(parent, bg=CARD)
            entry_f.grid(row=row, column=2, sticky="ew", pady=6)
            tk.Entry(entry_f, textvariable=var_path,
                     bg=ENTRY_BG, fg=FG, insertbackground=FG,
                     relief="flat", font=("Calibri", 9)
            ).pack(side="left", fill="x", expand=True, ipady=5, padx=(0,6))
            tk.Button(entry_f, text="Browse\u2026", command=pick_cmd,
                      bg=CARD2, fg=FG_DIM, relief="flat",
                      font=("Calibri", 8), cursor="hand2"
            ).pack(side="left")
            def _oa(vp=var_path):
                p=vp.get().strip()
                if not p: messagebox.showwarning("Open","No path set"); return
                if not os.path.isfile(p): messagebox.showerror("Open",f"Not found:\n{p}"); return
                try:
                    import subprocess as _sp; _sp.Popen([p],creationflags=0x08000000)
                except Exception as e: messagebox.showerror("Open",str(e))
            tk.Button(entry_f,text="Open",command=_oa,bg=CARD2,fg=ACCENT2,
                relief="flat",font=("Calibri",8,"bold"),cursor="hand2"
            ).pack(side="left",padx=(6,0))

        def _pick_exe(var, title="Select executable"):
            p = filedialog.askopenfilename(
                title=title,
                filetypes=[("Executables", "*.exe *.bat *.cmd"), ("All files", "*.*")])
            if p:
                var.set(p)

        _app_row(launch_card, 1,
            icon="\u2b21", label="COLMAP",
            hint="model_converter: sparse txt \u2192 bin  (runs silently, no window)",
            var_path=self.colmap_exe, var_enable=self.launch_colmap,
            pick_cmd=lambda: _pick_exe(self.colmap_exe, "Select COLMAP executable"))

        tk.Frame(launch_card, bg=BORDER, height=1
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)

        _app_row(launch_card, 3,
            icon="\u2726", label="Lichtfeld Studio",
            hint="opens after COLMAP finishes",
            var_path=self.lichtfeld_exe, var_enable=self.launch_lichtfeld,
            pick_cmd=lambda: _pick_exe(self.lichtfeld_exe, "Select Lichtfeld Studio executable"))

        tk.Frame(launch_card, bg=BORDER, height=1
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)

        _app_row(launch_card, 5,
            icon="\u25c8", label="Brush",
            hint="Opens Brush without a dataset \u2014 your output folder is copied to clipboard automatically.\nIn Brush: click 'Directory', then Ctrl+V to paste the path and load your dataset.",
            var_path=self.brush_exe, var_enable=self.launch_brush,
            pick_cmd=lambda: _pick_exe(self.brush_exe, "Select Brush executable"))

        tk.Frame(launch_inner, bg=BG, height=8).grid(row=3, column=0, sticky="ew")
        _section_header(launch_inner, 4, "Launch Sequence")
        seq_card = _card(launch_inner, 5)
        seq_card.columnconfigure(0, weight=1)
        tk.Label(seq_card,
            text=(
                "1.  Metashape \u2192 COLMAP conversion  (this app)\n"
                "2.  COLMAP  model_converter  sparse/0  txt \u2192 bin  (silent, awaited)\n"
                "3.  Lichtfeld Studio opens with dataset pre-loaded\n"
                "4.  Brush opens  +  output folder copied to clipboard\n"
                "    \u2514\u2500  in Brush: click 'Directory', Ctrl+V, then set your training options\n\n"
                "Tip: enable COLMAP even if you only use a viewer \u2014 bin format loads faster."
            ),
            bg=CARD, fg=FG_DIM, font=("Calibri", 9), justify="left"
        ).pack(anchor="w")

        # ── bottom bar (always visible) ──────────────────────────────────────
        tk.Frame(root_frame, bg=BORDER, height=1).pack(fill="x")

        bottom = tk.Frame(root_frame, bg=BG)
        bottom.pack(fill="x", padx=20, pady=(8, 12))

        self.status = tk.StringVar(value="Ready")
        self.pb = ttk.Progressbar(bottom, style="TProgressbar", maximum=100)
        self.pb.pack(fill="x", pady=(0, 8))

        status_run = tk.Frame(bottom, bg=BG)
        status_run.pack(fill="x")

        tk.Label(status_run, textvariable=self.status, bg=BG, fg=FG_DIM,
                 font=("Calibri", 9)).pack(side="left")

        style.configure("Stop.TButton",
            background="#7a2020", foreground="#ffffff",
            bordercolor="#7a2020", relief="flat",
            font=("Calibri", 10, "bold"), padding=(14, 8))
        style.map("Stop.TButton",
            background=[("active", "#9a2828"), ("pressed", "#5a1818"),
                        ("disabled", "#3a1010")],
            foreground=[("disabled", "#666666")])

        style.configure("Pause.TButton",
            background="#3a3510", foreground="#c8a830",
            bordercolor="#3a3510", relief="flat",
            font=("Calibri", 10, "bold"), padding=(14, 8))
        style.map("Pause.TButton",
            background=[("active", "#4a4518"), ("pressed", "#2a2808"),
                        ("disabled", "#1e1e10")],
            foreground=[("disabled", "#555540")])

        self._btn_convert = ttk.Button(status_run, text="Convert →",
                   style="Run.TButton", command=self.run)
        self._btn_convert.pack(side="right", padx=(8, 0))

        style.configure("AddQueue.TButton",
            background="#1a2535", foreground=self._pal["ACCENT2"],
            bordercolor=self._pal["ACCENT2"], relief="flat",
            font=("Calibri", 10, "bold"), padding=(14, 8))
        style.map("AddQueue.TButton",
            background=[("active","#223040"),("pressed","#0f1820"),("disabled","#1a1a1a")],
            foreground=[("disabled","#334455")])

        self._btn_add_queue = ttk.Button(status_run, text="+ Add to Queue",
                   style="AddQueue.TButton", command=self._add_current_to_queue)
        self._btn_add_queue.pack(side="right", padx=(8, 0))

        self._btn_stop = ttk.Button(status_run, text="Stop",
                   style="Stop.TButton", command=self.stop_conversion)
        self._btn_stop.pack(side="right", padx=(8, 0))
        self._btn_stop.state(["disabled"])

        self._btn_pause = ttk.Button(status_run, text="Pause",
                   style="Pause.TButton", command=self.toggle_pause)
        self._btn_pause.pack(side="right", padx=(8, 0))
        self._btn_pause.state(["disabled"])

        self._ensure_view_vars()
        self._autofit_window()

        # If required packages are missing, jump straight to Dependencies tab
        # so the user sees the install buttons immediately
        if not _numpy_ok or not _pillow_ok:
            self.after(200, lambda: self._nb.select(3))

        self._config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'converter_config.json')
        self._load_config()
        self.protocol('WM_DELETE_WINDOW', self._on_close)


    # ════════════════════════════════════════════════════════════════════
    # Queue methods
    # ════════════════════════════════════════════════════════════════════

    def _load_config(self):
        import json
        try:
            with open(self._config_path, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            return
        self.colmap_exe.set(cfg.get('colmap_exe',''))
        self.lichtfeld_exe.set(cfg.get('lichtfeld_exe',''))
        self.brush_exe.set(cfg.get('brush_exe',''))
        self.launch_colmap.set(cfg.get('launch_colmap',False))
        self.launch_lichtfeld.set(cfg.get('launch_lichtfeld',False))
        self.launch_brush.set(cfg.get('launch_brush',False))

    def _save_config(self):
        import json
        cfg = {
            'colmap_exe':       self.colmap_exe.get(),
            'lichtfeld_exe':    self.lichtfeld_exe.get(),
            'brush_exe':        self.brush_exe.get(),
            'launch_colmap':    bool(self.launch_colmap.get()),
            'launch_lichtfeld': bool(self.launch_lichtfeld.get()),
            'launch_brush':     bool(self.launch_brush.get()),
        }
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _on_close(self):
        self._save_config()
        self.destroy()

    def _copy_to_clipboard(self, text: str):
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
        except Exception:
            pass

    def _next_job_id(self):
        self._queue_id_counter += 1
        return self._queue_id_counter

    def _add_current_to_queue(self):
        """Snapshot current Files tab paths into a new queue job."""
        xml    = self.xml.get().strip()
        panos  = self.panos.get().strip()
        out    = self.out.get().strip()

        if not xml or not os.path.isfile(xml):
            messagebox.showerror("Add to Queue", "Please set a valid Cameras XML first.")
            return
        if not panos or not os.path.isdir(panos):
            messagebox.showerror("Add to Queue", "Please set a valid Images folder first.")
            return
        if not out:
            messagebox.showerror("Add to Queue", "Please set an Output folder first.")
            return

        job = {
            "id":          self._next_job_id(),
            "project_dir": self.project_dir.get().strip() or os.path.dirname(xml),
            "xml":         xml,
            "sphere_xml":  self.sphere_xml.get().strip(),
            "panos":       panos,
            "ply":         self.ply.get().strip(),
            "out":         out,
            "status":      "waiting",
            "progress":    0,
            "message":     "",
        }
        self._queue_jobs.append(job)
        self._q_refresh_list()

        # Switch to Queue tab so user can see it was added
        self._nb.select(4)
        self.status.set(f"Added to queue  —  {len(self._queue_jobs)} job(s) waiting")

    def _q_add_job(self, project_dir=None):
        """Add a job by picking a project folder — auto-fills paths."""
        if project_dir is None:
            project_dir = filedialog.askdirectory(title="Select Project Folder for Queue")
        if not project_dir:
            return
        # Auto-detect paths using same logic as auto_fill
        import glob as _glob
        root = project_dir

        # XML
        xmls = [f for f in os.listdir(root) if f.lower().endswith('.xml')]
        cam_xmls    = [x for x in xmls if 'camera' in x.lower() and 'sphere' not in x.lower()]
        sphere_xmls = [x for x in xmls if 'sphere' in x.lower()]
        fallback    = [x for x in xmls if x not in cam_xmls and x not in sphere_xmls]
        xml_path    = os.path.join(root, (cam_xmls or fallback or [''])[0]) if (cam_xmls or fallback) else ''
        sphere_path = os.path.join(root, sphere_xmls[0]) if sphere_xmls else ''

        # Images folder
        img_dir = ''
        for entry in os.listdir(root):
            if os.path.isdir(os.path.join(root, entry)) and entry.lower() == 'images':
                img_dir = os.path.join(root, entry); break
        if not img_dir:
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full):
                    for sub in os.listdir(full):
                        if sub.lower() == 'images':
                            img_dir = os.path.join(full, sub); break
                if img_dir: break

        # PLY
        ply_path = ''
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith('.ply'):
                    preferred = any(k in fn.lower() for k in ('point','cloud','dense','fused','model'))
                    if not ply_path or preferred:
                        ply_path = os.path.join(dp, fn)
                    if preferred: break

        # Output
        out_name  = self._full_output_name() if hasattr(self, '_full_output_name') else 'output_001'
        out_path  = os.path.join(root, out_name)

        job = {
            "id":          self._next_job_id(),
            "project_dir": root,
            "xml":         xml_path,
            "sphere_xml":  sphere_path,
            "panos":       img_dir,
            "ply":         ply_path,
            "out":         out_path,
            "status":      "waiting",   # waiting / running / done / failed / skipped
            "progress":    0,
            "message":     "",
        }
        self._queue_jobs.append(job)
        self._q_refresh_list()

    def _q_refresh_list(self):
        """Rebuild the job list UI from self._queue_jobs."""
        for w in self._q_inner.winfo_children():
            w.destroy()

        FG_DIM = self._pal["FG_DIM"]
        STATUS_COLORS = {
            "waiting":  FG_DIM,
            "running":  self._pal["ACCENT2"],
            "done":     self._pal["ACCENT3"],
            "failed":   "#e06060",
            "skipped":  "#a08040",
        }
        STATUS_ICONS = {
            "waiting": "○",
            "running": "⠿",
            "done":    "✓",
            "failed":  "✗",
            "skipped": "⊘",
        }

        for i, job in enumerate(self._queue_jobs):
            CARD = self._pal["CARD"]
            FG   = self._pal["FG"]
            ACCENT = self._pal["ACCENT"]

            card = tk.Frame(self._q_inner, bg=CARD, pady=0)
            card.pack(fill="x", pady=(0,2))
            card.columnconfigure(1, weight=1)

            # Status strip
            sc = STATUS_COLORS.get(job["status"], FG_DIM)
            tk.Frame(card, bg=sc, width=3).grid(row=0, column=0, rowspan=2, sticky="ns")

            body = tk.Frame(card, bg=CARD, padx=12, pady=8)
            body.grid(row=0, column=1, sticky="ew")
            body.columnconfigure(0, weight=1)

            # Row 1: icon + project name + status + progress
            top_row = tk.Frame(body, bg=CARD)
            top_row.pack(fill="x")

            icon = STATUS_ICONS.get(job["status"], "○")
            tk.Label(top_row, text=f"{icon}  {os.path.basename(job['project_dir'])}",
                     bg=CARD, fg=sc, font=("Calibri", 10, "bold")).pack(side="left")

            if job["status"] == "running" and job["progress"] > 0:
                tk.Label(top_row, text=f"  {job['progress']}%",
                         bg=CARD, fg=sc, font=("Calibri", 9)).pack(side="left")

            if job["message"]:
                tk.Label(top_row, text=f"  — {job['message']}",
                         bg=CARD, fg=FG_DIM, font=("Calibri", 8)).pack(side="left")

            # Row 2: paths summary
            path_row = tk.Frame(body, bg=CARD)
            path_row.pack(fill="x", pady=(2,0))
            xml_name = os.path.basename(job["xml"]) if job["xml"] else "no XML found"
            out_name = os.path.basename(job["out"]) if job["out"] else "?"
            tk.Label(path_row,
                     text=f"  {job['project_dir']}   →   {out_name}   ·   {xml_name}",
                     bg=CARD, fg=FG_DIM, font=("Calibri", 8)).pack(side="left")

            # Delete button (only when not running)
            if job["status"] != "running":
                del_btn = tk.Label(body, text="✕", bg=self._pal["CARD"], fg=self._pal["FG_DIM"],
                                   font=("Calibri", 10), cursor="hand2")
                del_btn.place(relx=1.0, rely=0.0, anchor="ne")
                del_btn.bind("<Button-1>", lambda e, jid=job["id"]: self._q_remove_job(jid))

        total = len(self._queue_jobs)
        done  = sum(1 for j in self._queue_jobs if j["status"] == "done")
        fail  = sum(1 for j in self._queue_jobs if j["status"] == "failed")
        wait  = sum(1 for j in self._queue_jobs if j["status"] == "waiting")
        if total == 0:
            self._q_summary.set("No jobs in queue.")
        else:
            self._q_summary.set(
                f"{total} job{'s' if total!=1 else ''}  ·  "
                f"{wait} waiting  ·  {done} done  ·  {fail} failed")

    def _q_remove_job(self, job_id):
        self._queue_jobs = [j for j in self._queue_jobs if j["id"] != job_id]
        self._q_refresh_list()

    def _q_save(self):
        """Save queue to a JSON file."""
        import json
        path = filedialog.asksaveasfilename(
            title="Save Queue",
            defaultextension=".json",
            filetypes=[("Queue file", "*.json"), ("All files", "*.*")])
        if not path:
            return
        data = [{k: v for k, v in j.items() if k not in ("id",)} 
                for j in self._queue_jobs]
        # Reset statuses so saved queue can be re-run
        for d in data:
            d["status"] = "waiting"
            d["progress"] = 0
            d["message"] = ""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Queue saved", f"Saved {len(data)} jobs to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _q_load(self):
        """Load queue from a JSON file."""
        import json
        path = filedialog.askopenfilename(
            title="Load Queue",
            filetypes=[("Queue file", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                d["id"] = self._next_job_id()
                d.setdefault("status", "waiting")
                d.setdefault("progress", 0)
                d.setdefault("message", "")
            self._queue_jobs = data
            self._q_refresh_list()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _q_stop(self):
        self._queue_stop.set()
        self._q_summary.set("Stopping — finishing current job…")

    def _q_run(self):
        """Run all waiting jobs sequentially in a background thread."""
        if self._queue_running:
            return
        waiting = [j for j in self._queue_jobs if j["status"] == "waiting"]
        if not waiting:
            messagebox.showinfo("Queue", "No waiting jobs to run.\nAdd jobs or reset completed ones.")
            return

        self._queue_running = True
        self._queue_stop.clear()
        self._btn_q_run.state(["disabled"])
        self._btn_q_stop.state(["!disabled"])
        self._btn_add_queue.state(["disabled"])

        def _worker():
            for job in self._queue_jobs:
                if job["status"] != "waiting":
                    continue
                if self._queue_stop.is_set():
                    break

                # Mark running
                job["status"]  = "running"
                job["progress"] = 0
                job["message"] = ""
                self.after(0, self._q_refresh_list)

                # Snapshot current settings
                try:
                    xml_path        = job["xml"]
                    sphere_xml_path = job["sphere_xml"]
                    input_root      = job["panos"]
                    out_dir         = job["out"]
                    ply_path        = job["ply"]

                    if not xml_path or not os.path.isfile(xml_path):
                        raise FileNotFoundError(f"XML not found: {xml_path}")
                    if not input_root or not os.path.isdir(input_root):
                        raise FileNotFoundError(f"Images folder not found: {input_root}")

                    os.makedirs(out_dir, exist_ok=True)

                    view_mode   = self.view_mode.get()
                    add_diag45  = bool(self.add_diag45.get())
                    add_upper45 = bool(self.add_upper45.get())
                    add_lower45 = bool(self.add_lower45.get())
                    out_size    = int(self.out_size.get())
                    fov_deg     = float(self.fov_deg.get())
                    exclude_yaw = self.exclude_yaw.get().strip()
                    workers     = int(self.workers.get())
                    flip_x      = bool(self.flip_world_180x.get())
                    flip_y      = bool(self.flip_world_180y.get())
                    tile_h      = int(getattr(self, 'tile_h', tk.IntVar(value=512)).get())
                    use_gpu     = bool(self.use_gpu.get()) and _cupy_ok
                    gpu_batch   = int(self.gpu_batch_size.get())
                    skip_exist  = bool(self.skip_existing.get())
                    enabled_keys = self.get_enabled_view_keys()

                    stop_ev  = self._queue_stop
                    pause_ev = threading.Event()
                    pause_ev.set()

                    total_done = [0]
                    total_imgs = [1]

                    def _prog(done, total, rate=0, eta=0):
                        total_done[0] = done
                        total_imgs[0] = total
                        pct = int(done / max(total,1) * 100)
                        job["progress"] = pct
                        job["message"]  = f"{done}/{total}  {rate:.1f} img/s"
                        self.after(0, self._q_refresh_list)

                    convert(
                        xml_path=xml_path,
                        input_root=input_root,
                        out_dir=out_dir,
                        sphere_xml_path=sphere_xml_path or None,
                        view_mode=view_mode,
                        add_diag45=add_diag45,
                        add_upper45=add_upper45,
                        add_lower45=add_lower45,
                        out_size=out_size,
                        fov_deg=fov_deg,
                        exclude_yaw_ranges_text=exclude_yaw,
                        ply_path=ply_path or None,
                        workers=workers,
                        flip_world_180x=flip_x,
                        flip_world_180y=flip_y,
                        tile_h=tile_h,
                        enabled_view_keys=enabled_keys,
                        use_gpu=use_gpu,
                        gpu_batch_size=gpu_batch,
                        skip_existing=skip_exist,
                        dry_run=False,
                        stop_event=stop_ev,
                        pause_event=pause_ev,
                        progress_cb=_prog,
                    )

                    if self._queue_stop.is_set():
                        job["status"]  = "skipped"
                        job["message"] = "stopped by user"
                    else:
                        job["status"]  = "done"
                        job["message"] = f"{total_done[0]} images"

                except Exception as e:
                    job["status"]  = "failed"
                    job["message"] = str(e)[:80]

                self.after(0, self._q_refresh_list)

            # All done
            self._queue_running = False
            self.after(0, lambda: self._btn_q_run.state(["!disabled"]))
            self.after(0, lambda: self._btn_q_stop.state(["disabled"]))
            self.after(0, lambda: self._btn_add_queue.state(["!disabled"]))
            done  = sum(1 for j in self._queue_jobs if j["status"] == "done")
            fail  = sum(1 for j in self._queue_jobs if j["status"] == "failed")
            self.after(0, self._q_refresh_list)
            if not self._queue_stop.is_set():
                self.after(0, lambda: messagebox.showinfo(
                    "Queue complete",
                    f"All jobs finished.\n{done} succeeded  ·  {fail} failed"))

        threading.Thread(target=_worker, daemon=True).start()


    def _post_conversion_launch(self, out_dir: str):
        """Run COLMAP txt->bin then open viewers. Called after any successful conversion."""
        import subprocess

        colmap_path    = self.colmap_exe.get().strip()
        lichtfeld_path = self.lichtfeld_exe.get().strip()
        brush_path     = self.brush_exe.get().strip()
        do_colmap      = bool(self.launch_colmap.get())    and bool(colmap_path)
        do_lichtfeld   = bool(self.launch_lichtfeld.get()) and bool(lichtfeld_path)
        do_brush       = bool(self.launch_brush.get())     and bool(brush_path)

        if not (do_colmap or do_lichtfeld or do_brush):
            return

        sparse0 = os.path.join(out_dir, "sparse", "0")

        def _worker():
            # ── Step 1: COLMAP txt → bin ─────────────────────────────────────
            if do_colmap:
                try:
                    self.after(0, lambda: self.status.set("COLMAP: converting txt → bin…"))
                    CREATE_NO_WINDOW = 0x08000000
                    args = [colmap_path,'model_converter',
                            '--input_path',sparse0,
                            '--output_path',sparse0,
                            '--output_type','BIN']
                    is_bat = colmap_path.lower().endswith(('.bat','.cmd'))
                    if is_bat:
                        import subprocess as _sp
                        quoted = _sp.list2cmdline(args)
                        result = subprocess.run(f'cmd /c {quoted}',
                            capture_output=True,text=True,shell=True,creationflags=CREATE_NO_WINDOW)
                    else:
                        result = subprocess.run(args,capture_output=True,text=True,creationflags=CREATE_NO_WINDOW)
                    if result.returncode != 0:
                        err = (result.stderr or result.stdout or 'unknown error')[:300]
                        self.after(0, lambda e=err,c=result.returncode: messagebox.showwarning(
                            'COLMAP',f'model_converter returned code {c}:\n{e}'))
                    else:
                        self.after(0, lambda: self.status.set('COLMAP: txt → bin done'))
                except FileNotFoundError:
                    self.after(0, lambda: messagebox.showerror(
                        "COLMAP", f"Executable not found:\n{colmap_path}"))
                    return
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("COLMAP", str(e)))

            # ── Step 2: open viewers ─────────────────────────────────────────
            CREATE_NO_WINDOW = 0x08000000
            if do_lichtfeld:
                try:
                    splat_out = os.path.join(out_dir,'splat')
                    subprocess.Popen(
                        [lichtfeld_path,'--data-path',out_dir,'--output-path',splat_out],
                        creationflags=CREATE_NO_WINDOW)
                except Exception as e:
                    self.after(0, lambda err=e: messagebox.showerror('Lichtfeld Studio',str(err)))

            if do_brush:
                try:
                    brush_dir = os.path.dirname(brush_path)
                    subprocess.Popen([brush_path], cwd=brush_dir)
                    self.after(0, lambda d=out_dir: self._copy_to_clipboard(d))
                    self.after(0, lambda: self.status.set(
                        'Brush open — output path copied to clipboard. Click Directory in Brush and Ctrl+V.'))
                except Exception as e:
                    self.after(0, lambda err=e: messagebox.showerror('Brush',str(err)))

            launched = []
            if do_colmap:    launched.append("COLMAP bin")
            if do_lichtfeld: launched.append("Lichtfeld Studio")
            if do_brush:     launched.append("Brush")
            msg = "Launched: " + ", ".join(launched)
            self.after(0, lambda: self.status.set(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _rig_log_append(self,t):
        def _d():
            self._rig_log.configure(state='normal')
            self._rig_log.insert('end',t+'\n'); self._rig_log.see('end')
            self._rig_log.configure(state='disabled')
        self.after(0,_d)
    def _rig_stop(self):
        self.rig_stop_event.set()
        self.after(0,lambda:self.rig_status.set('Stopping…'))
    def _rig_done(self):
        self.rig_running=False
        self._btn_rig_run.state(['!disabled']); self._btn_rig_stop.state(['disabled'])
    def _rig_run(self):
        if self.rig_running: return
        pd=self.rig_panos.get().strip(); od=self.rig_out.get().strip()
        col=self.colmap_exe.get().strip()
        if not pd or not os.path.isdir(pd):
            messagebox.showerror('COLMAP Rig','Set a valid Panoramas folder.'); return
        if not od: messagebox.showerror('COLMAP Rig','Set an Output folder.'); return
        if not col or not os.path.isfile(col):
            messagebox.showerror('COLMAP Rig','COLMAP exe not set. Set it in Launchers tab.'); return
        if not any([self.rig_step_extract.get(),self.rig_step_rig_cfg.get(),
                    self.rig_step_match.get(),self.rig_step_map.get()]):
            messagebox.showwarning('COLMAP Rig','No steps selected.'); return
        self.rig_running=True; self.rig_stop_event.clear()
        self._btn_rig_run.state(['disabled']); self._btn_rig_stop.state(['!disabled'])
        self._rig_log.configure(state='normal'); self._rig_log.delete('1.0','end')
        self._rig_log.configure(state='disabled')
        de=bool(self.rig_step_extract.get()); dr=bool(self.rig_step_rig_cfg.get())
        dm=bool(self.rig_step_match.get()); dp=bool(self.rig_step_map.get())
        mat=self.rig_matcher.get()
        def _rc(args,label):
            import subprocess
            self.after(0,lambda:self.rig_status.set(f'{label}…'))
            self._rig_log_append(f'\n▶ {label}')
            self._rig_log_append('  '+' '.join(str(a) for a in args))
            CNW=0x08000000
            try:
                ib=str(args[0]).lower().endswith(('.bat','.cmd'))
                if ib:
                    import subprocess as _sp
                    proc=subprocess.Popen(f'cmd /c {_sp.list2cmdline(args)}',
                        shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=CNW)
                else:
                    proc=subprocess.Popen(args,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,creationflags=CNW)
                for line in proc.stdout:
                    line=line.rstrip()
                    if line: self._rig_log_append('  '+line)
                    if self.rig_stop_event.is_set(): proc.terminate(); return False
                proc.wait()
                if proc.returncode!=0:
                    self._rig_log_append(f'  ✗ Failed ({proc.returncode})'); return False
                self._rig_log_append('  ✓ Done'); return True
            except Exception as e:
                self._rig_log_append(f'  ✗ {e}'); return False
        def _worker():
            import json as _j,math as _m
            os.makedirs(od,exist_ok=True)
            db=os.path.join(os.path.dirname(od),'database.db'); rp=os.path.join(od,'rig_config.json')
            ir=os.path.join(od,'images'); sp=os.path.join(od,'sparse')
            self.after(0,lambda:self.rig_status.set('Converting panoramas…'))
            self._rig_log_append('\n▶ Converting panoramas → perspective crops')
            if not _numpy_ok or not _pillow_ok:
                self._rig_log_append('  ✗ numpy+Pillow required.')
                self.after(0,self._rig_done); return
            from PIL import Image as _I
            vws=build_view_set('cubemap6',bool(self.add_diag45.get()),
                bool(self.add_upper45.get()),bool(self.add_lower45.get()))
            sz=int(self.out_size.get()); fv=float(self.fov_deg.get())
            for v in vws: os.makedirs(os.path.join(ir,v['suffix']),exist_ok=True)
            exts={'.jpg','.jpeg','.png','.tif','.tiff','.webp'}
            pfs=sorted(os.path.join(pd,f) for f in os.listdir(pd)
                if os.path.splitext(f)[1].lower() in exts)
            if not pfs:
                for dp2,_,fns in os.walk(pd):
                    for fn in fns:
                        if os.path.splitext(fn)[1].lower() in exts:
                            p2=os.path.join(dp2,fn)
                            try:
                                w,h=_I.open(p2).size
                                if 1.8<=w/h<=2.2: pfs.append(p2)
                            except Exception: pass
                pfs.sort()
            self._rig_log_append(f'  Found {len(pfs)} panoramas')

            # Build view payloads (same format as main converter)
            view_payloads = [(v['suffix'], v['R'].reshape(-1).tolist()) for v in vws]
            use_gpu = bool(self.use_gpu.get()) and _cupy_ok

            # Output goes into flat images_rig/ — we reorganise into subfolders after
            # Actually write directly into subfolder structure using _worker_convert_one
            # For GPU: use _gpu_convert_batch then move files into subfolders
            items_for_convert = [(os.path.basename(p), p) for p in pfs]

            done_count = [0]
            total_count = len(pfs) * len(vws)

            def _progress(done, total):
                done_count[0] = done
                rate = done / max(time.time() - t_start, 0.001)
                self.after(0, lambda d=done,t=total,r=rate:
                    self.rig_status.set(f'Converting {d}/{t}  {r:.1f} img/s'))

            t_start = time.time()

            if use_gpu and _cupy_ok:
                self._rig_log_append(f'  Using GPU acceleration')
                # Use flat temp dir then reorganise into subfolders
                ir_flat = ir + '_tmp'
                os.makedirs(ir_flat, exist_ok=True)
                GPU_BATCH = max(1, int(self.gpu_batch_size.get()))
                for i in range(0, len(items_for_convert), GPU_BATCH):
                    if self.rig_stop_event.is_set(): break
                    chunk = items_for_convert[i:i+GPU_BATCH]
                    def _img_cb(n, _done=done_count):
                        _done[0] += n
                        _progress(_done[0], total_count)
                    _gpu_convert_batch(chunk, ir_flat, sz, fv, view_payloads,
                                       gpu_batch_size=GPU_BATCH,
                                       image_cb=_img_cb,
                                       stop_event=self.rig_stop_event)
                # Reorganise: ir_flat/stem_suffix.jpg -> ir/suffix/stem.jpg
                self._rig_log_append('  Reorganising into subfolders...')
                for fn in os.listdir(ir_flat):
                    if not fn.endswith('.jpg'): continue
                    name = fn[:-4]  # strip .jpg
                    for suf in [v['suffix'] for v in vws]:
                        if name.endswith('_' + suf):
                            stem = name[:-len('_' + suf)]
                            dst = os.path.join(ir, suf, stem + '.jpg')
                            if not os.path.isfile(dst):
                                try:
                                    os.link(os.path.join(ir_flat, fn), dst)
                                except OSError:
                                    import shutil as _sh
                                    _sh.copy2(os.path.join(ir_flat, fn), dst)
                            break
                import shutil as _sh2; _sh2.rmtree(ir_flat, ignore_errors=True)
            else:
                self._rig_log_append(f'  Using CPU ({max(1, int(self.workers.get()))} workers)')
                from concurrent.futures import ProcessPoolExecutor as _PPE, as_completed as _ac
                tile_h = int(getattr(self, 'tile_h', tk.IntVar(value=512)).get())
                with _PPE(max_workers=max(1, int(self.workers.get()))) as ex:
                    futs = {}
                    for pano_name, pano_path in items_for_convert:
                        if self.rig_stop_event.is_set(): break
                        # Write directly into subfolder structure
                        # _worker_convert_one writes to flat dir, so use per-suffix dirs
                        futs[ex.submit(_worker_convert_one, pano_path, ir + '_tmp2',
                                       sz, fv, view_payloads, tile_h)] = pano_name
                    ir_flat2 = ir + '_tmp2'
                    os.makedirs(ir_flat2, exist_ok=True)
                    for fut in _ac(futs):
                        if self.rig_stop_event.is_set():
                            for f in futs: f.cancel()
                            break
                        done_count[0] += len(vws)
                        _progress(done_count[0], total_count)
                # Reorganise
                for fn in os.listdir(ir_flat2):
                    if not fn.endswith('.jpg'): continue
                    name = fn[:-4]
                    for suf in [v['suffix'] for v in vws]:
                        if name.endswith('_' + suf):
                            stem = name[:-len('_' + suf)]
                            dst = os.path.join(ir, suf, stem + '.jpg')
                            if not os.path.isfile(dst):
                                try: os.link(os.path.join(ir_flat2, fn), dst)
                                except OSError:
                                    import shutil as _sh3; _sh3.copy2(os.path.join(ir_flat2, fn), dst)
                            break
                import shutil as _sh4; _sh4.rmtree(ir_flat2, ignore_errors=True)
            self._rig_log_append('  ✓ Conversion done')
            frad=_m.radians(fv); fx=fy=float(sz)/(2*_m.tan(frad/2)); cx=cy=float(sz)/2
            def _q(R):
                t=float(np.trace(R))
                if t>0:
                    s=(t+1)**0.5*2; return[0.25*s,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s]
                elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
                    s=(1+R[0,0]-R[1,1]-R[2,2])**0.5*2; return[(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s]
                elif R[1,1]>R[2,2]:
                    s=(1+R[1,1]-R[0,0]-R[2,2])**0.5*2; return[(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s]
                else:
                    s=(1+R[2,2]-R[0,0]-R[1,1])**0.5*2; return[(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s]
            rR=vws[0]['R']; cams=[]
            for i,v in enumerate(vws):
                c={'image_prefix':f"{v['suffix']}/",'camera_model_name':'PINHOLE',
                   'camera_params':[float(fx),float(fy),float(cx),float(cy)]}
                if i==0: c['ref_sensor']=True
                else:
                    q=_q(v['R']@rR.T)
                    c['cam_from_rig_rotation']=[round(x,10) for x in q]
                    c['cam_from_rig_translation']=[0.0,0.0,0.0]
                cams.append(c)
            with open(rp,'w',encoding='utf-8') as f: _j.dump([{'cameras':cams}],f,indent=2)
            self._rig_log_append(f'  ✓ rig_config.json ({len(vws)} views)')
            if self.rig_stop_event.is_set(): self.after(0,self._rig_done); return

            # Camera params string: fx,fy,cx,cy for PINHOLE
            cam_params_str = f'{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}'

            if de:
                if not _rc([col,'feature_extractor',
                    '--image_path',ir,'--database_path',db,
                    '--ImageReader.single_camera_per_folder','1',
                    '--ImageReader.camera_model','PINHOLE',
                    '--ImageReader.camera_params',cam_params_str,
                    ],'Feature extraction') \
                        or self.rig_stop_event.is_set():
                    self.after(0,self._rig_done); return
            if dr:
                if not _rc([col,'rig_configurator','--database_path',db,
                    '--rig_config_path',rp],'Rig configuration') or self.rig_stop_event.is_set():
                    self.after(0,self._rig_done); return
            if dm:
                mc={'sequential':'sequential_matcher','exhaustive':'exhaustive_matcher',
                    'vocab_tree':'vocab_tree_matcher'}.get(mat,'sequential_matcher')
                if not _rc([col,mc,'--database_path',db],f'Matching ({mat})') \
                        or self.rig_stop_event.is_set():
                    self.after(0,self._rig_done); return
            if dp:
                os.makedirs(sp,exist_ok=True)
                if not _rc([col,'mapper',
                    '--database_path',db,
                    '--image_path',ir,'--output_path',sp,
                    '--Mapper.ba_refine_focal_length','0',
                    '--Mapper.ba_refine_principal_point','0',
                    '--Mapper.ba_refine_extra_params','0',
                    ],'Mapping'):
                    self.after(0,self._rig_done); return
            self._rig_log_append('\n✓ Pipeline complete!')
            self.after(0,lambda:self.rig_status.set('✓ Pipeline complete!'))
            self.after(0,lambda:self._post_conversion_launch(od))
            self.after(0,self._rig_done)
        threading.Thread(target=_worker,daemon=True).start()

    def _autofit_window(self):
        """Resize window so Files tab content is fully visible without scrolling."""

        def _fit():
            self.update_idletasks()

            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()

            # files_inner is the actual content frame inside the scrollable canvas.
            # winfo_reqheight() gives its true needed height regardless of canvas size.
            try:
                content_h = self._files_inner.winfo_reqheight()
                content_w = self._files_inner.winfo_reqwidth()
            except Exception:
                content_h = 800
                content_w = 960

            # Add surrounding chrome:
            #   header (~60px) + notebook tab strip (~32px) + bottom bar (~70px) + padding
            total_h = content_h + 60 + 32 + 70 + 60
            total_w = max(960, content_w + 56)   # 56px for side padx=28 on both sides

            fit_w = max(900,  min(total_w, int(screen_w * 0.95)))
            fit_h = max(600,  min(total_h, int(screen_h * 0.92)))

            x = (screen_w - fit_w) // 2
            y = max(20, (screen_h - fit_h) // 2)
            self.geometry(f"{fit_w}x{fit_h}+{x}+{y}")

        # Delay slightly so all widgets have finished their layout pass
        self.after(100, _fit)

    def _ensure_view_vars(self):
        """Ensure BooleanVars and labels exist for current view set."""
        # Make friendly labels
        key_to_label = {
            "pz": "Front (0°)",
            "px": "Right (90°)",
            "nz": "Back (180°)",
            "nx": "Left (270°)",
            "py": "Up",
            "ny": "Down",
            "yaw045": "Diag 45°",
            "yaw135": "Diag 135°",
            "yaw225": "Diag 225°",
            "yaw315": "Diag 315°",
        }
        for yaw in (0,45,90,135,180,225,270,315):
            key_to_label[f"up45_{yaw:03d}"] = f"Upper +45°  {yaw}°"
            key_to_label[f"dn45_{yaw:03d}"] = f"Lower -45°  {yaw}°"
        self._key_to_label = key_to_label
        keys = view_keys_only(
            "cubemap6",
            bool(self.add_diag45.get()),
            bool(self.add_upper45.get()),
            bool(self.add_lower45.get()),
        )
        for k in keys:
            if k not in self.view_enabled:
                self.view_enabled[k] = tk.BooleanVar(value=True)
    def get_enabled_view_keys(self):
        """Return approved view keys for current configuration."""
        self._ensure_view_vars()
        keys = view_keys_only(
            "cubemap6",
            bool(self.add_diag45.get()),
            bool(self.add_upper45.get()),
            bool(self.add_lower45.get()),
        )
        return [k for k in keys if self.view_enabled.get(k, tk.BooleanVar(value=True)).get()]
    def _scan_panos(self):
        pano_dir = self.panos.get().strip()
        if not os.path.isdir(pano_dir):
            return []
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        candidates = []
        for dirpath, _, filenames in os.walk(pano_dir):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    candidates.append(os.path.join(dirpath, fn))
        candidates.sort()

        # Filter to 2:1 aspect ratio (equirectangular) using header-only reads — fast
        panos = []
        for p in candidates:
            try:
                with Image.open(p) as im:
                    im.verify()   # reads header only, does not decode pixels
            except Exception:
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                ratio = w / h if h else 0
                if 1.8 <= ratio <= 2.2:
                    panos.append(p)
            except Exception:
                pass

        return panos if panos else candidates
    def _draw_model_diagram(self):
        """Draw a simple schematic of the camera rig (rings on a sphere)."""
        c = getattr(self, "model_canvas", None)
        if c is None:
            return
        c.delete("all")
        w = int(c.winfo_width() or 150)
        h = int(c.winfo_height() or 110)
        cx, cy = w//2, h//2
        r = min(w, h)//2 - 10
        BG_C = "#1a1b24"; FG_C = "#4a4960"; ACC = "#c8a96e"; DOT = "#6e9ecf"
        c.configure(bg=BG_C)
        # Sphere outline
        c.create_oval(cx-r, cy-r, cx+r, cy+r, outline=FG_C, width=1)
        # Equator ring
        c.create_line(cx-r, cy, cx+r, cy, fill=FG_C, dash=(3,3))
        # Upper/lower ring hints
        if bool(self.add_upper45.get()):
            ry = int(r*0.55)
            c.create_oval(cx-r, cy-ry, cx+r, cy+ry, outline=ACC, dash=(2,3))
            c.create_text(cx, cy-ry-9, text="+45°", fill=ACC, font=("Calibri", 8))
        if bool(self.add_lower45.get()):
            ry = int(r*0.55)
            c.create_oval(cx-r, cy-ry, cx+r, cy+ry, outline=ACC, dash=(2,3))
            c.create_text(cx, cy+ry+9, text="-45°", fill=ACC, font=("Calibri", 8))
        # Top/bottom dots
        c.create_oval(cx-3, cy-r-3, cx+3, cy-r+3, fill=FG_C, outline="")
        c.create_oval(cx-3, cy+r-3, cx+3, cy+r+3, fill=FG_C, outline="")
        c.create_text(cx, cy-r-10, text="Up", fill=FG_C, font=("Calibri", 8))
        c.create_text(cx, cy+r+10, text="Dn", fill=FG_C, font=("Calibri", 8))
        # Points for horizon samples (base 4 + optional diagonals)
        yaws = [0,90,180,270]
        if bool(self.add_diag45.get()):
            yaws += [45,135,225,315]
        for yaw in yaws:
            ang = math.radians(yaw - 90)
            px = cx + int(r*math.cos(ang))
            py = cy
            c.create_oval(px-3, py-3, px+3, py+3, fill=DOT, outline="")
        c.create_text(cx, h-6, text="view directions", fill=FG_C, font=("Calibri", 8))
    # -------- preview window --------
    def open_preview_window(self):
        # Reuse window if already open
        if self._preview_win is not None and self._preview_win.winfo_exists():
            self._preview_win.lift()
            return
        self._ensure_view_vars()
        panos = self._scan_panos()
        if not panos:
            messagebox.showerror("Preview", "Pick a valid panorama folder with images first.")
            return
        w = tk.Toplevel(self)
        w.title("Preview / approve view directions")
        w.geometry("1200x780")
        w.minsize(900, 600)
        self._preview_win = w
        top = ttk.Frame(w, padding=10)
        top.pack(fill="both", expand=True)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Preview frame (choose any panorama; approvals apply to ALL frames):")\
            .grid(row=0, column=0, sticky="w")
        idx_var = tk.IntVar(value=0)
        name_var = tk.StringVar(value=os.path.basename(panos[0]))
        ttk.Label(top, textvariable=name_var).grid(row=0, column=1, sticky="w", padx=10)
        slider = tk.Scale(top, from_=0, to=max(0, len(panos)-1), orient="horizontal",
                          variable=idx_var, showvalue=True, length=520)
        slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 10))
        ttk.Label(top, text="Tip: change extras / FOV / output resolution in the main window; then refresh preview.")\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))
        # --- Equirect overview ---
        overview = ttk.LabelFrame(top, text="Equirectangular overview (drag lines to set yaw exclude)", padding=10)
        overview.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        overview.columnconfigure(0, weight=1)
        ov_canvas = tk.Canvas(overview, height=260, highlightthickness=1, highlightbackground="#999")
        ov_canvas.grid(row=0, column=0, sticky="ew")
        ov_readout = tk.StringVar(value="Start: –   End: –")
        ttk.Label(overview, textvariable=ov_readout).grid(row=1, column=0, sticky="w", pady=(6,0))
        btn_row = ttk.Frame(overview)
        btn_row.grid(row=2, column=0, sticky="w", pady=(6,0))
        start_deg = tk.DoubleVar(value=330.0)
        end_deg   = tk.DoubleVar(value=30.0)
        def _init_markers_from_text():
            s = (self.exclude_yaw.get() or "").strip()
            if not s:
                return
            first = s.split(",")[0].strip()
            if "-" not in first:
                return
            try:
                a,b = first.split("-", 1)
                start_deg.set(float(a) % 360.0)
                end_deg.set(float(b) % 360.0)
            except Exception:
                return
        _init_markers_from_text()
        def _append_or_set_range(set_mode: bool):
            a = float(start_deg.get()) % 360.0
            b = float(end_deg.get()) % 360.0
            rng = f"{a:.0f}-{b:.0f}"
            cur = (self.exclude_yaw.get() or "").strip()
            if set_mode or not cur:
                self.exclude_yaw.set(rng)
            else:
                self.exclude_yaw.set(cur + "," + rng)
            # refresh thumb visuals (no expensive re-render)
            try:
                _update_thumb_visuals_with_ranges(parse_exclude_ranges(self.exclude_yaw.get().strip()))
            except Exception:
                pass
        ttk.Button(btn_row, text="Set exclude to this range", command=lambda: _append_or_set_range(True)).pack(side="left")
        ttk.Button(btn_row, text="Add this range", command=lambda: _append_or_set_range(False)).pack(side="left", padx=6)
        ov_state = {"photo": None, "w": 1, "h": 1, "drag": None}
        def yaw_to_x(yaw_deg: float) -> float:
            W = max(1, int(ov_state["w"]))
            return (((float(yaw_deg) % 360.0) / 360.0) + 0.5) % 1.0 * W
        def x_to_yaw(x: float) -> float:
            W = max(1, int(ov_state["w"]))
            u = (float(x) / W) % 1.0
            yaw = ((u - 0.5) * 360.0) % 360.0
            return yaw
        def _redraw_lines():
            ov_canvas.delete("mark")
            W = max(1, int(ov_state["w"]))
            H = max(1, int(ov_state["h"]))
            xa = yaw_to_x(start_deg.get())
            xb = yaw_to_x(end_deg.get())
            # Wrap decided in X-space (seam behaviour matches what you see)
            if xb < xa:
                ov_canvas.create_rectangle(xa, 0, W, H, fill="#ff0000", stipple="gray25", width=0, tags=("mark",))
                ov_canvas.create_rectangle(0, 0, xb, H, fill="#ff0000", stipple="gray25", width=0, tags=("mark",))
            else:
                ov_canvas.create_rectangle(xa, 0, xb, H, fill="#ff0000", stipple="gray25", width=0, tags=("mark",))
            for yaw, lab in [(0,"0° Front"), (90,"90° Right"), (180,"180° Back"), (270,"270° Left")]:
                x = yaw_to_x(yaw)
                ov_canvas.create_line(x, 0, x, H, fill="#666", dash=(2,2), tags=("mark",))
                ov_canvas.create_text(x+4, 10, anchor="nw", text=lab, fill="#666", tags=("mark",), font=("Calibri", 9))
            ov_canvas.create_line(xa, 0, xa, H, fill="#ffcc00", width=2, tags=("mark","A"))
            ov_canvas.create_line(xb, 0, xb, H, fill="#00ccff", width=2, tags=("mark","B"))
            ov_canvas.create_text(xa+4, H-18, anchor="sw", text="Start", fill="#ffcc00", tags=("mark",), font=("Calibri", 9, "bold"))
            ov_canvas.create_text(xb+4, H-18, anchor="sw", text="End", fill="#00ccff", tags=("mark",), font=("Calibri", 9, "bold"))
            wrap = " (wrap)" if (yaw_to_x(end_deg.get()) < yaw_to_x(start_deg.get())) else ""
            a = float(start_deg.get()) % 360.0
            b = float(end_deg.get()) % 360.0
            ov_readout.set(f"Start: {a:0.0f}°   End: {b:0.0f}°{wrap}")
        def _draw_overview(pano_path: str):
            try:
                img = Image.open(pano_path).convert("RGB")
            except Exception:
                return
            tgt_h = 260
            aspect = img.width / max(1, img.height)
            tgt_w = int(tgt_h * aspect)
            tgt_w = max(520, min(1120, tgt_w))
            img_small = img.resize((tgt_w, tgt_h), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img_small)
            ov_canvas.configure(width=tgt_w, height=tgt_h)
            ov_canvas.delete("all")
            ov_state["photo"] = photo
            ov_state["w"] = tgt_w
            ov_state["h"] = tgt_h
            ov_canvas.create_image(0, 0, anchor="nw", image=photo)
            _redraw_lines()
            # Live thumb feedback while dragging (no re-render)
            try:
                _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
            except Exception:
                pass
            try: img.close()
            except Exception: pass
        def _hit(x: float, yaw_var: tk.DoubleVar) -> bool:
            return abs(x - yaw_to_x(yaw_var.get())) <= 10
        def _on_down(ev):
            x = ev.x
            if _hit(x, start_deg):
                ov_state["drag"] = "A"
            elif _hit(x, end_deg):
                ov_state["drag"] = "B"
            else:
                ov_state["drag"] = None
        def _on_drag(ev):
            if ov_state["drag"] is None:
                return
            x = max(0, min(ev.x, ov_state["w"]))
            yaw = x_to_yaw(x)
            if ov_state["drag"] == "A":
                start_deg.set(yaw)
            else:
                end_deg.set(yaw)
            _redraw_lines()
            # Live thumb feedback while dragging (no re-render)
            try:
                _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
            except Exception:
                pass
        def _on_up(_ev):
            ov_state["drag"] = None
        ov_canvas.bind("<ButtonPress-1>", _on_down)
        ov_canvas.bind("<B1-Motion>", _on_drag)
        ov_canvas.bind("<ButtonRelease-1>", _on_up)
        # --- Body: approvals + thumbs ---
        body = ttk.Frame(top)
        body.grid(row=4, column=0, columnspan=2, sticky="nsew")
        top.rowconfigure(4, weight=1)
        checks = ttk.LabelFrame(body, text="Approve directions", padding=10)
        checks.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        thumbs_box = ttk.LabelFrame(body, text="Preview (low-res) — click an image to approve/disapprove", padding=10)
        thumbs_box.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        thumbs_box.columnconfigure(0, weight=1)
        thumbs_box.rowconfigure(0, weight=1)
        thumbs_canvas = tk.Canvas(thumbs_box, highlightthickness=0)
        vscroll = ttk.Scrollbar(thumbs_box, orient="vertical", command=thumbs_canvas.yview)
        thumbs_canvas.configure(yscrollcommand=vscroll.set)
        thumbs_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        thumbs_inner = ttk.Frame(thumbs_canvas)
        window_id = thumbs_canvas.create_window((0,0), window=thumbs_inner, anchor="nw")
        def _sync_scrollregion(_ev=None):
            thumbs_canvas.configure(scrollregion=thumbs_canvas.bbox("all"))
        def _sync_inner_width(ev):
            thumbs_canvas.itemconfigure(window_id, width=ev.width)
        thumbs_inner.bind("<Configure>", _sync_scrollregion)
        thumbs_canvas.bind("<Configure>", _sync_inner_width)
        # Build view list for preview — pure Python, no numpy
        def _preview_views():
            face_yaws = {"pz":0,"px":90,"nz":180,"nx":270,"py":None,"ny":None}
            face_pitches = {"pz":0,"px":0,"nz":0,"nx":0,"py":90,"ny":-90}
            vs = [{"key":f,"yaw":face_yaws[f],"pitch":face_pitches[f],
                   "yaw_sensitive":face_yaws[f] is not None} for f in CUBE_FACE_ORDER]
            horizon = [0,90,180,270]
            if bool(self.add_diag45.get()):
                for y in (45,135,225,315):
                    vs.append({"key":f"yaw{y:03d}","yaw":float(y),"pitch":0,"yaw_sensitive":True})
                    horizon.append(y)
            if bool(self.add_upper45.get()):
                for y in horizon:
                    vs.append({"key":f"up45_{y:03d}","yaw":float(y),"pitch":45,"yaw_sensitive":True})
            if bool(self.add_lower45.get()):
                for y in horizon:
                    vs.append({"key":f"dn45_{y:03d}","yaw":float(y),"pitch":-45,"yaw_sensitive":True})
            return vs
        views = _preview_views()
        # Ensure vars exist
        for v in views:
            if v["key"] not in self.view_enabled:
                self.view_enabled[v["key"]] = tk.BooleanVar(value=True)
        # Checkboxes list
        for i, k in enumerate(v["key"] for v in views):
            lab = self._key_to_label.get(k, k)
            ttk.Checkbutton(
                checks, text=lab, variable=self.view_enabled[k],
                command=lambda: _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
            ).grid(row=i, column=0, sticky="w", pady=2)
        btns = ttk.Frame(checks)
        btns.grid(row=len(views)+1, column=0, sticky="ew", pady=(10,0))
        def set_all(val: bool):
            for v in views:
                self.view_enabled[v["key"]].set(bool(val))
            _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
        ttk.Button(btns, text="All on", command=lambda: set_all(True)).pack(side="left")
        ttk.Button(btns, text="All off", command=lambda: set_all(False)).pack(side="left", padx=6)
        # Thumbnail widgets
        thumb_labels = {}  # key -> ttk.Label
        # Fast visual update: tint/grey using cached base thumbnails (no re-render)
        def _update_thumb_visuals_with_ranges(ranges):
            if not thumb_labels:
                return
            # Use cached base PIL images; if not cached yet, do nothing.
            for v in views:
                k = v["key"]
                labw = thumb_labels.get(k)
                if labw is None:
                    continue
                base_img = self._preview_base_pil.get(k)
                if base_img is None:
                    continue
                excluded = False
                if ranges and v.get("yaw_sensitive", False) and v.get("yaw") is not None:
                    excluded = yaw_in_ranges(v["yaw"], ranges)
                img = base_img
                # If excluded, apply red tint + dim
                if excluded:
                    try:
                        img2 = ImageEnhance.Brightness(img).enhance(0.35)
                        red = Image.new("RGB", img2.size, (255, 0, 0))
                        img = Image.blend(img2, red, 0.18)
                    except Exception:
                        img = img2
                # If disabled (and not excluded), slightly dim for feedback
                enabled = self.view_enabled.get(k, tk.BooleanVar(value=True)).get()
                if (not excluded) and (not enabled):
                    try:
                        img = ImageEnhance.Brightness(img).enhance(0.55)
                    except Exception:
                        pass
                photo = ImageTk.PhotoImage(img)
                self._preview_imgs[k] = photo  # keep ref
                nice = self._key_to_label.get(k, k)
                if excluded:
                    status = "⛔"
                    suffix = " (excluded)"
                else:
                    status = "✓" if enabled else "✗"
                    suffix = ""
                labw.configure(image=photo, text=f"{status} {nice}{suffix}")
        # grid config
        cols = 4
        for c in range(cols):
            thumbs_inner.columnconfigure(c, weight=1)
        def _toggle_from_thumb(key: str):
            var = self.view_enabled.get(key)
            if var is None:
                return
            var.set(not var.get())
            _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
        # Create thumb placeholders in a grid
        for idx, v in enumerate(views):
            r = idx // cols
            c = idx % cols
            k = v["key"]
            nice = self._key_to_label.get(k, k)
            lab = ttk.Label(thumbs_inner, text=nice, compound="top", cursor="hand2")
            lab.grid(row=r, column=c, padx=8, pady=8, sticky="n")
            lab.bind("<Button-1>", lambda ev, kk=k: _toggle_from_thumb(kk))
            thumb_labels[k] = lab
        # Mousewheel scrolling (bind to this window only)
        def _on_mousewheel(ev):
            if hasattr(ev, "delta") and ev.delta:
                thumbs_canvas.yview_scroll(int(-1*(ev.delta/120)), "units")
        w.bind("<MouseWheel>", _on_mousewheel)
        _debounce_job = {"id": None}
        def on_change(_v=None):
            i = int(float(idx_var.get()))
            if not (0 <= i < len(panos)):
                return
            name_var.set(os.path.basename(panos[i]))
            _draw_overview(panos[i])
            # Debounce expensive thumb rendering while scrubbing
            if _debounce_job["id"] is not None:
                try:
                    w.after_cancel(_debounce_job["id"])
                except Exception:
                    pass
            def _do():
                self._render_preview(panos[i], thumb_labels)
                # After rendering base thumbs, apply current marker preview tint (single range) for live feel
                try:
                    _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
                except Exception:
                    pass
            _debounce_job["id"] = w.after(180, _do)
        slider.configure(command=on_change)
        _draw_overview(panos[0])
        self._render_preview(panos[0], thumb_labels)
        try:
            _update_thumb_visuals_with_ranges([(float(start_deg.get())%360.0, float(end_deg.get())%360.0)])
        except Exception:
            pass
    def _render_preview(self, pano_path: str, thumb_labels: dict):
        """Render base thumbnails once (expensive), then apply fast visual updates for enable/exclude.
        Heavy work = equirect->perspective renders. We cache base PIL thumbs for the current pano+view config
        so toggling checkboxes or dragging yaw markers stays responsive.
        """
        if self._preview_win is None or not self._preview_win.winfo_exists():
            return
        self._ensure_view_vars()
        fov_deg = float(self.fov_deg.get())
        preview_size = 320
        sig = (
            str(pano_path),
            float(fov_deg),
            bool(self.add_diag45.get()),
            bool(self.add_upper45.get()),
            bool(self.add_lower45.get()),
        )
        # If signature matches and we already have cached base thumbs, skip heavy render.
        if self._preview_last_sig == sig and self._preview_base_pil:
            try:
                ex_ranges = parse_exclude_ranges(self.exclude_yaw.get().strip())
            except Exception:
                ex_ranges = []
            # Apply visuals quickly using cached images
            # (This function exists in the preview window scope; if not, fall back to simple label update.)
            for k in view_keys_only("cubemap6", sig[2], sig[3], sig[4]):
                if k not in thumb_labels:
                    continue
                # We'll update images in-place by calling the preview-window helper if available.
            return
        try:
            pano_img = Image.open(pano_path)
        except Exception as e:
            messagebox.showerror("Preview", f"Failed to open image\n{e}")
            return
        views = build_view_set(
            "cubemap6",
            bool(self.add_diag45.get()),
            bool(self.add_upper45.get()),
            bool(self.add_lower45.get()),
        )
        self._preview_imgs = {}
        self._preview_base_pil = {}
        for v in views:
            key = v["key"]
            if key not in thumb_labels:
                continue
            try:
                img, *_ = equirect_to_perspective_tiled(pano_img, preview_size, v["R"], fov_deg=fov_deg, tile_h=128)
            except Exception:
                img = Image.new("RGB", (preview_size, preview_size), (40,40,40))
            self._preview_base_pil[key] = img
        self._preview_last_sig = sig
        try:
            pano_img.close()
        except Exception:
            pass
        # After base render, apply visuals using current exclude text.
        # We can't directly call the nested helper here, so we just update each label using cached base.
        try:
            ex_ranges = parse_exclude_ranges(self.exclude_yaw.get().strip())
        except Exception:
            ex_ranges = []
        for v in views:
            k = v["key"]
            labw = thumb_labels.get(k)
            if labw is None:
                continue
            base_img = self._preview_base_pil.get(k)
            if base_img is None:
                continue
            excluded = False
            if ex_ranges and v.get("yaw_sensitive", False) and v.get("yaw") is not None:
                excluded = yaw_in_ranges(v["yaw"], ex_ranges)
            img = base_img
            if excluded:
                try:
                    img2 = ImageEnhance.Brightness(img).enhance(0.35)
                    red = Image.new("RGB", img2.size, (255, 0, 0))
                    img = Image.blend(img2, red, 0.18)
                except Exception:
                    pass
            else:
                enabled = self.view_enabled.get(k, tk.BooleanVar(value=True)).get()
                if not enabled:
                    try:
                        img = ImageEnhance.Brightness(img).enhance(0.55)
                    except Exception:
                        pass
            photo = ImageTk.PhotoImage(img)
            self._preview_imgs[k] = photo
            nice = self._key_to_label.get(k, k)
            if excluded:
                status = "⛔"
                suffix = " (excluded)"
            else:
                enabled = self.view_enabled.get(k, tk.BooleanVar(value=True)).get()
                status = "✓" if enabled else "✗"
                suffix = ""
            labw.configure(image=photo, text=f"{status} {nice}{suffix}")
    # -------- file pickers --------
    def _full_output_name(self):
        base    = self.output_folder_name.get().strip() or "output"
        version = self.output_version.get()
        return f"{base}_{version:03d}"

    def _on_version_spin(self):
        self._on_output_name_changed()

    def _on_output_name_changed(self, *_):
        root = self.project_dir.get().strip()
        if not root:
            return
        name = self._full_output_name()
        new_path = os.path.join(root, name)
        self.out.set(new_path)  # folder created at conversion start, not here

    def _restart_self(self):
        """Relaunch this script as a fresh process then close the current one."""
        import subprocess, sys
        try:
            # sys.frozen is set by PyInstaller when running as a bundled exe
            if getattr(sys, "frozen", False):
                exe = sys.executable  # the .exe itself
                subprocess.Popen([exe])
            else:
                subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        except Exception as e:
            messagebox.showerror("Restart failed",
                f"Could not restart automatically:\n{e}\n\nPlease close and reopen manually.")
            return
        self.destroy()

    def _log_install(self, text):
        """Append text to the install log safely from any thread."""
        def _append():
            try:
                self._install_log.configure(state="normal")
                self._install_log.insert("end", text)
                self._install_log.see("end")
                self._install_log.configure(state="disabled")
            except Exception:
                pass
        self.after(0, _append)

    def pick_project(self):
        d = filedialog.askdirectory(title="Select Project Root Folder")
        if d:
            self.project_dir.set(d)
            self.auto_fill(d)

    def auto_fill(self, root):
        """Scan root folder and fill in whatever paths can be found automatically."""
        import glob as _glob

        # ── Cameras XML: any .xml at root level, prefer 'cameras' in name ──
        xmls = [f for f in os.listdir(root) if f.lower().endswith('.xml')]
        cam_xmls   = [x for x in xmls if 'camera' in x.lower() and 'sphere' not in x.lower()]
        sphere_xmls = [x for x in xmls if 'sphere' in x.lower() or 'spherical' in x.lower()]
        fallback_xmls = [x for x in xmls if x not in cam_xmls and x not in sphere_xmls]

        if cam_xmls:
            self.xml.set(os.path.join(root, cam_xmls[0]))
        elif fallback_xmls and not self.xml.get():
            self.xml.set(os.path.join(root, fallback_xmls[0]))

        if sphere_xmls:
            self.sphere_xml.set(os.path.join(root, sphere_xmls[0]))

        # ── Images folder: look for folder named 'images' (case-insensitive) ──
        # Also checks one level deep (e.g. root/data/images/)
        img_dir = None
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and entry.lower() == 'images':
                img_dir = full
                break
        if img_dir is None:
            # one level deeper
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full):
                    for sub in os.listdir(full):
                        if sub.lower() == 'images':
                            img_dir = os.path.join(full, sub)
                            break
                if img_dir:
                    break
        if img_dir:
            self.panos.set(img_dir)

        # ── Point cloud: any .ply anywhere in root (recursive) ──
        ply_hits = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.ply'):
                    ply_hits.append(os.path.join(dirpath, fn))
        if ply_hits:
            # Prefer files with 'point', 'cloud', 'dense', 'fused' in name
            preferred = [p for p in ply_hits if any(
                k in os.path.basename(p).lower()
                for k in ('point', 'cloud', 'dense', 'fused', 'model'))]
            self.ply.set((preferred or ply_hits)[0])

        # ── Output folder: use output_folder_name var, default 'output' ──────
        name          = self._full_output_name() if hasattr(self, '_full_output_name') else 'output_001'
        out_candidate = os.path.join(root, name)
        if not self.out.get():
            self.out.set(out_candidate)  # folder created at conversion start

    def pick_xml(self):
        self.xml.set(filedialog.askopenfilename(filetypes=[("XML", "*.xml"), ("All files", "*.*")]))
    def pick_sphere_xml(self):
        self.sphere_xml.set(filedialog.askopenfilename(filetypes=[("XML", "*.xml"), ("All files", "*.*")]))
    def pick_panos(self):
        self.panos.set(filedialog.askdirectory())
    def pick_ply(self):
        self.ply.set(filedialog.askopenfilename(filetypes=[("PLY", "*.ply"), ("All files", "*.*")]))
    def pick_out(self):
        self.out.set(filedialog.askdirectory())
    # -------- progress / run --------
    def _set_running(self, running):
        if running:
            self._btn_convert.state(["disabled"])
            self._btn_stop.state(["!disabled"])
            self._btn_pause.state(["!disabled"])
        else:
            self._btn_convert.state(["!disabled"])
            self._btn_stop.state(["disabled"])
            self._btn_pause.state(["disabled"])
            self._btn_pause.configure(text="Pause")
            self._pause_event.clear()

    def stop_conversion(self):
        self._stop_event.set()
        self._pause_event.clear()
        self._btn_stop.state(["disabled"])
        self._btn_pause.state(["disabled"])
        self.status.set("Stopping — finishing current image…")

    def toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._btn_pause.configure(text="Pause")
        else:
            self._pause_event.set()
            self._btn_pause.configure(text="Resume")

    def _reset_after_run(self):
        self.pb["value"] = 0
        self._stop_event.clear()
        self._pause_event.clear()
        self._polling = False
        self._rate_hist     = []
        self._eta_seconds   = None
        self._eta_last_calc = 0.0
        self._eta_calc_time = 0.0
        self._set_running(False)

    @staticmethod
    def _fmt_hms(seconds):
        """Format seconds as H:MM:SS or M:SS."""
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def _poll_queue(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "progress":
                    _, done, total = msg
                    self._done = int(done)
                    self._total = int(total)
                elif msg[0] == "done":
                    self._polling = False
                    self._reset_after_run()
                    is_dry = len(msg) > 1 and msg[1] == "dry"
                    if is_dry:
                        self._dry_launch = False
                        self.status.set("Dry run complete  —  check output, then hit Convert →")
                    else:
                        out_dir = self.out.get().strip()
                    self.status.set("Complete  —  ready for next conversion")
                    if out_dir:
                        self._post_conversion_launch(out_dir)
                    return
                elif msg[0] == "stopped":
                    self._polling = False
                    self._reset_after_run()
                    self.status.set("Stopped")
                    return
                elif msg[0] == "error":
                    self._polling = False
                    self._reset_after_run()
                    return
        except queue.Empty:
            pass

        if self._polling and self._start_time is not None and self._total:
            now     = time.time()
            elapsed = now - self._start_time
            done    = max(0, self._done)
            total   = max(1, self._total)

            self.pb["maximum"] = total
            self.pb["value"]   = done

            if self._pause_event.is_set():
                self.status.set(
                    f"Paused  —  {done}/{total}  |  elapsed {self._fmt_hms(elapsed)}")
                if self._polling:
                    self.after(100, self._poll_queue)
                return

            # ── imgs/sec: rolling window over last 10s ────────────────────
            self._rate_hist.append((now, done))
            # Prune entries older than 10s
            self._rate_hist = [(t, d) for t, d in self._rate_hist if now - t <= 10.0]
            if len(self._rate_hist) >= 2:
                dt = self._rate_hist[-1][0] - self._rate_hist[0][0]
                dd = self._rate_hist[-1][1] - self._rate_hist[0][1]
                imgs_per_sec = dd / dt if dt > 0 else 0.0
            else:
                imgs_per_sec = done / elapsed if elapsed > 0 else 0.0

            # ── ETA: recalculate every 5s, otherwise count down smoothly ──
            if now - self._eta_last_calc >= 5.0 and imgs_per_sec > 0:
                remaining = total - done
                self._eta_seconds   = remaining / imgs_per_sec
                self._eta_last_calc = now
                self._eta_calc_time = now
            elif self._eta_seconds is not None:
                # Count down from last calculation
                self._eta_seconds = max(
                    0, self._eta_seconds - (now - self._eta_calc_time))
                self._eta_calc_time = now

            # ── Build status string ───────────────────────────────────────
            spin = self._spinner[self._spin_i % len(self._spinner)]
            self._spin_i += 1
            elapsed_str  = self._fmt_hms(elapsed)
            rate_str     = f"{imgs_per_sec:.1f} img/s" if imgs_per_sec > 0 else "—"
            if self._eta_seconds is not None and imgs_per_sec > 0:
                eta_str = self._fmt_hms(self._eta_seconds)
                self.status.set(
                    f"{spin}  {done}/{total}  |  {rate_str}  |  "
                    f"elapsed {elapsed_str}  |  ETA {eta_str}")
            else:
                self.status.set(
                    f"{spin}  {done}/{total}  |  elapsed {elapsed_str}")

        if self._polling:
            self.after(100, self._poll_queue)
    def _launch(self, dry=False):
        self._dry_launch = dry
        self.run()

    def run(self):
        xml_path = self.xml.get().strip()
        sphere_xml_path = self.sphere_xml.get().strip()
        pano_dir = self.panos.get().strip()
        ply_path = self.ply.get().strip()
        out_dir = self.out.get().strip()
        view_mode = "cubemap6"
        out_size = int(self.out_size.get())
        fov_deg = float(self.fov_deg.get())
        exclude_yaw = self.exclude_yaw.get().strip()
        add_diag45 = bool(self.add_diag45.get())
        add_upper45 = bool(self.add_upper45.get())
        add_lower45 = bool(self.add_lower45.get())
        flip_world_180x = bool(self.flip_world_180x.get())
        flip_world_180y = bool(self.flip_world_180y.get())
        workers = int(self.workers.get())
        tile_h = int(self.tile_h.get())
        use_gpu        = bool(getattr(self, 'use_gpu', tk.BooleanVar(value=True)).get()) and _cupy_ok
        gpu_batch_size = int(getattr(self, 'gpu_batch_size', tk.IntVar(value=8)).get())
        skip_existing  = bool(self.skip_existing.get())
        dry_run        = bool(self._dry_launch)
        if not os.path.isfile(xml_path):
            messagebox.showerror("Error", "Select a valid Metashape XML.")
            return
        if sphere_xml_path and (not os.path.isfile(sphere_xml_path)):
            messagebox.showerror("Error", "Optional spherical-only XML was set but file not found.")
            return
        if not os.path.isdir(pano_dir):
            messagebox.showerror("Error", "Select a valid input root folder.")
            return
        if ply_path and not os.path.isfile(ply_path):
            messagebox.showerror("Error", "PLY selected but file not found.")
            return
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder:\n{e}")
                return
        if not out_dir:
            messagebox.showerror("Error", "Select a valid output folder.")
            return
        if out_size < 256:
            messagebox.showerror("Error", "Output resolution too small.")
            return
        if not (1.0 <= fov_deg <= 179.0):
            messagebox.showerror("Error", "FOV must be between 1 and 179 degrees.")
            return
        if workers < 1:
            workers = 1
        if tile_h < 64:
            tile_h = 64
        # NOTE: We no longer hard-cap workers here; see comments in the code.
        # Validate exclude format early
        try:
            _ = parse_exclude_ranges(exclude_yaw)
        except Exception as e:
            messagebox.showerror("Error", f"Exclude yaw ranges invalid:\n{e}")
            return
        self._ensure_view_vars()
        self._stop_event.clear()
        self._pause_event.clear()
        self._start_time    = time.time()
        self._done          = 0
        self._total         = 1
        self._polling       = True
        self._spin_i        = 0
        self._rate_hist     = []
        self._eta_seconds   = None
        self._eta_last_calc = 0.0
        self._eta_calc_time = 0.0
        self.pb["value"]    = 0
        self.status.set("Starting…")
        self._set_running(True)
        self.after(100, self._poll_queue)
        enabled_keys = self.get_enabled_view_keys()
        stop_ev  = self._stop_event
        pause_ev = self._pause_event
        def work():
            try:
                def prog(done, total):
                    self._q.put(("progress", done, total))
                convert(
                    xml_path=xml_path,
                    input_root=pano_dir,
                    sphere_xml_path=sphere_xml_path,
                    out_dir=out_dir,
                    view_mode=view_mode,
                    add_diag45=add_diag45,
                    add_upper45=add_upper45,
                    add_lower45=add_lower45,
                    out_size=out_size,
                    fov_deg=fov_deg,
                    exclude_yaw_ranges_text=exclude_yaw,
                    ply_path=ply_path,
                    workers=workers,
                    flip_world_180x=flip_world_180x,
                    flip_world_180y=flip_world_180y,
                    tile_h=tile_h,
                    enabled_view_keys=enabled_keys,
                    use_gpu=use_gpu,
                    gpu_batch_size=gpu_batch_size,
                    skip_existing=skip_existing,
                    dry_run=dry_run,
                    stop_event=stop_ev,
                    pause_event=pause_ev,
                    progress_cb=prog
                )
                if stop_ev.is_set():
                    self._q.put(("stopped",))
                else:
                    self._q.put(("done", "dry" if dry_run else "full"))
            except Exception as e:
                self._q.put(("error",))
                self.after(0, lambda: messagebox.showerror("Failed", str(e)))
        threading.Thread(target=work, daemon=True).start()
if __name__ == "__main__":
    multiprocessing.freeze_support()
    App().mainloop()