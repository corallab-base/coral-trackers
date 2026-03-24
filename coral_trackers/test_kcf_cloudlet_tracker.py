#!/usr/bin/env python3
"""
KCF "Cloudlet" RGB‑D point tracker (RealSense) with **OpenCV visualization + mouse picking**

What’s new vs the Open3D version:
  • Uses a cv2 window for live 2D visualization.
  • Click to add query points before starting (press 's' to start tracking).
  • Tracks each query using a small cloudlet of KCF subtrackers; median voting
    across the cloudlet yields a robust 2D/3D point per query.
  • Back‑projects each query’s median (u,v) using aligned depth → (X,Y,Z in m).
  • Optional NCC re‑acquisition around last median if a subtracker fails.

Usage
  pip install opencv-contrib-python numpy pyrealsense2
  python kcf_rgbd_cloudlet_tracker_cv2.py \
    --width 640 --height 480 --fps 30 \
    --grid 3 --bbox 24 --spacing 10 --search 32

Controls
  • Left‑click: add query point(s) before starting
  • s: start tracking
  • r: reset (clear points and trackers)
  • q / ESC: quit
  • d: toggle drawing sub‑bboxes (debug)

Notes
  • Requires Intel RealSense with depth aligned to color; camera assumed stationary.
  • For multiple targets: just click multiple times before 's'.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ----------------------------- RealSense wrapper ---------------------------- #
class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not available. Install it and connect a RealSense camera.")
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None, None
        bgr = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()) * self.depth_scale
        return bgr, depth_m

    def intrinsics_matrix(self) -> np.ndarray:
        return np.array([[self.intr.fx, 0, self.intr.ppx],
                         [0, self.intr.fy, self.intr.ppy],
                         [0, 0, 1]], dtype=np.float32)

    def stop(self):
        self.pipe.stop()

# ----------------------------- KCF utilities -------------------------------- #

def create_kcf():
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("OpenCV KCF tracker not available. Install opencv-contrib-python.")

@dataclass
class SubTracker:
    tracker: any
    bbox: Tuple[int,int,int,int]  # (x,y,w,h)
    ok: bool = True

@dataclass
class Cloudlet:
    subs: List[SubTracker]
    last_median_xy: Optional[Tuple[float,float]]
    trail: List[Tuple[int,int]]

# ----------------------------- Helpers ------------------------------------- #

def make_cloud_offsets(grid:int, spacing:int)->List[Tuple[int,int]]:
    r = range(-(grid//2), grid//2+1)
    return [(dx*spacing, dy*spacing) for dy in r for dx in r]


def clamp_bbox(x,y,w,h,W,H):
    x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
    w = int(max(2, min(w, W - x))); h = int(max(2, min(h, H - y)))
    return (x,y,w,h)


def bbox_center(bb):
    x,y,w,h = bb
    return (x + w/2.0, y + h/2.0)


def depth_at(depth: np.ndarray, u:int, v:int)->float:
    if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
        z = float(depth[v,u])
        if not np.isfinite(z) or z <= 0:
            y0,y1 = max(0,v-1), min(depth.shape[0], v+2)
            x0,x1 = max(0,u-1), min(depth.shape[1], u+2)
            patch = depth[y0:y1, x0:x1]
            good = patch[np.isfinite(patch) & (patch>0)]
            z = float(np.median(good)) if good.size>0 else 0.0
        return z
    return 0.0


def pix_to_xyz(u:float,v:float,z:float,K:np.ndarray):
    if z <= 0: return (np.nan,np.nan,np.nan)
    X = (u - K[0,2]) / K[0,0] * z
    Y = (v - K[1,2]) / K[1,1] * z
    return (float(X), float(Y), float(z))


def extract_template(gray: np.ndarray, bb: Tuple[int,int,int,int])->np.ndarray:
    x,y,w,h = bb
    return gray[y:y+h, x:x+w].copy()


def ncc_reacquire(gray: np.ndarray, templ: np.ndarray, center_xy, search:int):
    if templ.size == 0:
        return None
    h, w = gray.shape; th, tw = templ.shape
    cx, cy = map(int, map(round, center_xy))
    x0 = max(0, cx - search - tw//2)
    y0 = max(0, cy - search - th//2)
    x1 = min(w, cx + search + tw//2)
    y1 = min(h, cy + search + th//2)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    if maxVal < 0.5:
        return None
    top_left = (x0 + maxLoc[0], y0 + maxLoc[1])
    return (top_left[0], top_left[1], tw, th)

# ----------------------------- Main ---------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='KCF cloudlet RGB-D point tracker (cv2 vis)')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--grid', type=int, default=3, help='Cloudlet grid size (odd, e.g., 3,5)')
    ap.add_argument('--spacing', type=int, default=10, help='Pixel spacing between subtrackers')
    ap.add_argument('--bbox', type=int, default=24, help='Subtracker bbox size (square)')
    ap.add_argument('--search', type=int, default=32, help='Template re-acquire search radius')
    ap.add_argument('--max_trail', type=int, default=64)
    args = ap.parse_args()

    # RealSense
    cam = RealSenseRGBD(args.width, args.height, args.fps)
    K = cam.intrinsics_matrix()

    win = 'KCF Cloudlet — click to add, s=start, r=reset, q=quit, d=debug'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Mouse picking state
    clicks: List[Tuple[int,int]] = []
    started = False
    debug_draw = False

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, started
        if event == cv2.EVENT_LBUTTONDOWN and not started:
            clicks.append((x,y))
    cv2.setMouseCallback(win, on_mouse)

    # Wait for first frame to init trackers after we get user clicks
    bgr, depth = cam.read()
    if bgr is None:
        raise SystemExit('No frames from RealSense.')
    H, W = bgr.shape[:2]

    # Cloudlet containers
    cloudlets: List[Cloudlet] = []
    templates: List[List[np.ndarray]] = []  # mirror cloudlets layout
    cloud_offsets = make_cloud_offsets(args.grid, args.spacing)

    fps_ema = 0.0
    import time
    t_prev = time.time()

    try:
        while True:
            bgr, depth = cam.read()
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                started = False
                cloudlets.clear(); templates.clear(); clicks.clear()
            elif key == ord('s') and not started:
                # Initialize cloudlets from clicks
                cloudlets.clear(); templates.clear()
                for (qx,qy) in clicks:
                    subs: List[SubTracker] = []
                    templs: List[np.ndarray] = []
                    for (dx,dy) in cloud_offsets:
                        cx = int(round(qx + dx)); cy = int(round(qy + dy))
                        x = int(cx - args.bbox/2); y = int(cy - args.bbox/2)
                        bb = clamp_bbox(x, y, args.bbox, args.bbox, W, H)
                        tr = create_kcf(); tr.init(bgr, bb)
                        subs.append(SubTracker(tracker=tr, bbox=bb, ok=True))
                        templs.append(extract_template(gray, bb))
                    cloudlets.append(Cloudlet(subs=subs, last_median_xy=(qx,qy), trail=[]))
                    templates.append(templs)
                started = True
            elif key == ord('d'):
                debug_draw = not debug_draw

            # Draw current frame + UI
            vis = bgr.copy()
            if not started:
                cv2.putText(vis, 'Click to add points, press s to start', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                for (x,y) in clicks:
                    cv2.circle(vis, (x,y), 5, (0,0,255), -1)
                cv2.imshow(win, vis)
                continue

            # Update trackers
            med_uvz = []  # per cloudlet: (u,v,z)
            for ci, cloud in enumerate(cloudlets):
                centers_uv = []
                centers_z = []
                for si, sub in enumerate(cloud.subs):
                    ok, bb = sub.tracker.update(bgr)
                    if not ok:
                        # NCC re-acquisition
                        reacq = ncc_reacquire(gray, templates[ci][si], cloud.last_median_xy, args.search)
                        if reacq is not None:
                            sub.tracker = create_kcf(); sub.tracker.init(bgr, reacq)
                            sub.ok = True; sub.bbox = reacq
                        else:
                            sub.ok = False
                            continue
                    else:
                        sub.ok = True; sub.bbox = tuple(map(int, bb))

                    ux, uy = bbox_center(sub.bbox)
                    centers_uv.append((ux, uy))
                    z = depth_at(depth, int(round(ux)), int(round(uy)))
                    centers_z.append(z)

                    if debug_draw:
                        x,y,w,h = map(int, sub.bbox)
                        cv2.rectangle(vis, (x,y), (x+w,y+h), (80,255,80), 1)

                if len(centers_uv) == 0:
                    med_uvz.append((np.nan,np.nan,np.nan))
                    continue
                uv = np.array(centers_uv, dtype=np.float32)
                zarr = np.array(centers_z, dtype=np.float32)
                med_u, med_v = np.median(uv, axis=0)
                good = np.isfinite(zarr) & (zarr>0)
                med_z = float(np.median(zarr[good])) if np.any(good) else np.nan
                cloud.last_median_xy = (float(med_u), float(med_v))
                cloud.trail.append((int(round(med_u)), int(round(med_v))))
                if len(cloud.trail) > args.max_trail:
                    cloud.trail = cloud.trail[-args.max_trail:]
                med_uvz.append((med_u, med_v, med_z))

            # HUD & drawing
            for (med_u, med_v, med_z), cloud in zip(med_uvz, cloudlets):
                if np.isfinite(med_u) and np.isfinite(med_v):
                    cv2.circle(vis, (int(round(med_u)), int(round(med_v))), 6, (0,0,255), -1)
                    # trail
                    for i in range(1, len(cloud.trail)):
                        cv2.line(vis, cloud.trail[i-1], cloud.trail[i], (0,255,0), 2)
                    # 3D text
                    X,Y,Z = pix_to_xyz(med_u, med_v, med_z, K)
                    if np.isfinite(Z):
                        cv2.putText(vis, f"({X:.3f},{Y:.3f},{Z:.3f}) m", (int(med_u)+8, max(12,int(med_v)-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # FPS
            now = time.time(); dt = now - t_prev; t_prev = now
            if dt > 0: fps_ema = 0.9*fps_ema + 0.1*(1.0/dt)
            cv2.putText(vis, f"KCF cloudlets: {len(cloudlets)}   FPS: {fps_ema:.1f}", (15,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow(win, vis)

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# """
# KCF "Cloudlet" RGB‑D point tracker (RealSense) with Open3D visualization

# Goal: very fast, robust-ish 3D point tracking from a single stationary camera.
# Strategy:
#   • You provide one or more query pixels (x,y).
#   • Around each query we spawn a small **cloudlet** of KCF trackers (3×3 by default),
#     each on a tiny bbox (e.g., 24×24 px). This adds redundancy.
#   • Every frame we update all trackers, back‑project their centers via depth, and
#     compute a **median** 2D/3D for each query group (robust to outliers/occlusion).
#   • If a sub‑tracker fails, we try a light **template re‑acquisition** using NCC
#     in a small search radius around the last median.
#   • We visualize the point cloud and the median 3D point(s) as spheres in Open3D.

# No OpenCV GUI is used — only Open3D for rendering.

# Usage:
#   pip install opencv-contrib-python numpy open3d pyrealsense2
#   python kcf_rgbd_cloudlet_tracker_open3d.py \
#     --queries "320,240" --width 640 --height 480 --fps 30 \
#     --grid 3 --bbox 24 --spacing 10 --search 32

# Multiple points:
#   --queries "320,240;200,180"

# Tuning tips:
#   • Increase --bbox (e.g., 32) for larger texture patches.
#   • Increase --grid or --spacing for more redundancy (slower but sturdier).
#   • Increase --search for more forgiving re‑acquisition (slower).
# """

# import argparse
# from dataclasses import dataclass
# from typing import List, Tuple, Optional
# import numpy as np
# import cv2
# import open3d as o3d

# try:
#     import pyrealsense2 as rs
# except Exception:
#     rs = None

# # ----------------------------- RealSense wrapper ---------------------------- #
# class RealSenseRGBD:
#     def __init__(self, width=640, height=480, fps=30):
#         if rs is None:
#             raise RuntimeError("pyrealsense2 not available. Install it and connect a RealSense camera.")
#         self.pipe = rs.pipeline()
#         self.cfg = rs.config()
#         self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
#         self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
#         self.profile = self.pipe.start(self.cfg)
#         self.align = rs.align(rs.stream.color)
#         color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
#         self.intr = color_stream.get_intrinsics()
#         self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

#     def read(self):
#         frames = self.pipe.wait_for_frames()
#         aligned = self.align.process(frames)
#         color = aligned.get_color_frame()
#         depth = aligned.get_depth_frame()
#         if not color or not depth:
#             return None, None
#         bgr = np.asanyarray(color.get_data())
#         depth_m = np.asanyarray(depth.get_data()) * self.depth_scale
#         return bgr, depth_m

#     def intrinsics_matrix(self) -> np.ndarray:
#         return np.array([[self.intr.fx, 0, self.intr.ppx],
#                          [0, self.intr.fy, self.intr.ppy],
#                          [0, 0, 1]], dtype=np.float32)

#     def stop(self):
#         self.pipe.stop()

# # ----------------------------- KCF utilities -------------------------------- #

# def create_kcf():
#     # Handle OpenCV 4.x API differences
#     if hasattr(cv2, 'TrackerKCF_create'):
#         return cv2.TrackerKCF_create()
#     if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
#         return cv2.legacy.TrackerKCF_create()
#     raise RuntimeError("OpenCV KCF tracker not available. Install opencv-contrib-python.")

# @dataclass
# class SubTracker:
#     tracker: any
#     bbox: Tuple[int,int,int,int]  # (x,y,w,h)
#     ok: bool = True

# @dataclass
# class Cloudlet:
#     # a set of subtrackers around a single logical query point
#     subs: List[SubTracker]
#     last_median_xy: Optional[Tuple[float,float]]

# # ----------------------------- Helpers ------------------------------------- #

# def parse_queries(s: str) -> List[Tuple[float,float]]:
#     if not s:
#         return []
#     out = []
#     for part in s.split(';'):
#         x,y = part.split(',')
#         out.append((float(x), float(y)))
#     return out


# def make_cloud_offsets(grid:int, spacing:int)->List[Tuple[int,int]]:
#     # grid=3 -> offsets [-1,0,1]×[-1,0,1] * spacing
#     r = range(-(grid//2), grid//2+1)
#     return [(dx*spacing, dy*spacing) for dy in r for dx in r]


# def clamp_bbox(x,y,w,h,W,H):
#     x = int(np.clip(x, 0, W-1))
#     y = int(np.clip(y, 0, H-1))
#     w = int(max(2, min(w, W - x)))
#     h = int(max(2, min(h, H - y)))
#     return (x,y,w,h)


# def bbox_center(bb):
#     x,y,w,h = bb
#     return (x + w/2.0, y + h/2.0)


# def depth_at(depth: np.ndarray, u:int, v:int)->float:
#     if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
#         z = float(depth[v,u])
#         if not np.isfinite(z) or z <= 0:
#             y0,y1 = max(0,v-1), min(depth.shape[0], v+2)
#             x0,x1 = max(0,u-1), min(depth.shape[1], u+2)
#             patch = depth[y0:y1, x0:x1]
#             good = patch[np.isfinite(patch) & (patch>0)]
#             z = float(np.median(good)) if good.size>0 else 0.0
#         return z
#     return 0.0


# def pix_to_xyz(u:float,v:float,z:float,K:np.ndarray)->Tuple[float,float,float]:
#     if z <= 0: return (np.nan,np.nan,np.nan)
#     X = (u - K[0,2]) / K[0,0] * z
#     Y = (v - K[1,2]) / K[1,1] * z
#     return (float(X), float(Y), float(z))


# def extract_template(gray: np.ndarray, bb: Tuple[int,int,int,int])->np.ndarray:
#     x,y,w,h = bb
#     return gray[y:y+h, x:x+w].copy()


# def ncc_reacquire(gray: np.ndarray, templ: np.ndarray, center_xy: Tuple[float,float], search:int)->Optional[Tuple[int,int,int,int]]:
#     if templ.size == 0:
#         return None
#     h, w = gray.shape
#     th, tw = templ.shape
#     cx, cy = map(int, map(round, center_xy))
#     x0 = max(0, cx - search - tw//2)
#     y0 = max(0, cy - search - th//2)
#     x1 = min(w, cx + search + tw//2)
#     y1 = min(h, cy + search + th//2)
#     roi = gray[y0:y1, x0:x1]
#     if roi.shape[0] < th or roi.shape[1] < tw:
#         return None
#     res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
#     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
#     # require decent correlation
#     if maxVal < 0.5:
#         return None
#     top_left = (x0 + maxLoc[0], y0 + maxLoc[1])
#     return (top_left[0], top_left[1], tw, th)

# # ----------------------------- Open3D utilities ---------------------------- #

# def make_o3d_intrinsics_from_K(K: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
#     intr = o3d.camera.PinholeCameraIntrinsic()
#     intr.set_intrinsics(width, height, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
#     return intr


# def rgbd_to_pcd(rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> o3d.geometry.PointCloud:
#     rgb_o3 = o3d.geometry.Image(rgb[:, :, ::-1].copy())  # BGR->RGB
#     depth_o3 = o3d.geometry.Image(depth_m.astype(np.float32))
#     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         rgb_o3, depth_o3, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
#     )
#     intr = make_o3d_intrinsics_from_K(K, rgb.shape[1], rgb.shape[0])
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
#     return pcd


# def create_spheres(points_xyz: np.ndarray, radius=0.01, color=(1.0, 0.0, 0.0)):
#     spheres = []
#     for p in points_xyz:
#         m = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
#         m.paint_uniform_color(color)
#         m.translate(p.tolist())
#         m.compute_vertex_normals()
#         spheres.append(m)
#     return spheres


# def update_spheres(meshes, points_xyz: np.ndarray):
#     for m, p in zip(meshes, points_xyz):
#         current = np.asarray(m.vertices)
#         centroid = current.mean(axis=0)
#         m.translate((p - centroid).tolist())

# # ----------------------------- Main ---------------------------------------- #

# def main():
#     ap = argparse.ArgumentParser(description='KCF cloudlet RGB-D point tracker (Open3D)')
#     ap.add_argument('--queries', type=str, required=True, help='Semicolon-separated pixels: "x1,y1;x2,y2"')
#     ap.add_argument('--width', type=int, default=640)
#     ap.add_argument('--height', type=int, default=480)
#     ap.add_argument('--fps', type=int, default=30)
#     ap.add_argument('--grid', type=int, default=3, help='Cloudlet grid size (odd, e.g., 3,5)')
#     ap.add_argument('--spacing', type=int, default=10, help='Pixel spacing between subtrackers')
#     ap.add_argument('--bbox', type=int, default=24, help='Subtracker bbox size (square)')
#     ap.add_argument('--search', type=int, default=32, help='Template re-acquire search radius')
#     ap.add_argument('--sphere_radius', type=float, default=0.015)
#     args = ap.parse_args()

#     queries = parse_queries(args.queries)
#     if len(queries) == 0:
#         raise SystemExit('Provide at least one query via --queries "x,y"')

#     cam = RealSenseRGBD(args.width, args.height, args.fps)
#     K = cam.intrinsics_matrix()

#     # Initialize trackers after first frame
#     bgr, depth = cam.read()
#     if bgr is None:
#         raise SystemExit('No frames from RealSense.')
#     H, W = bgr.shape[:2]
#     gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

#     cloud_offsets = make_cloud_offsets(args.grid, args.spacing)

#     # One cloudlet per query
#     cloudlets: List[Cloudlet] = []
#     templates: List[List[np.ndarray]] = []  # per-sub template for re-acquire

#     for (qx, qy) in queries:
#         subs = []
#         templs = []
#         for (dx, dy) in cloud_offsets:
#             cx = int(round(qx + dx))
#             cy = int(round(qy + dy))
#             x = int(cx - args.bbox/2)
#             y = int(cy - args.bbox/2)
#             bb = clamp_bbox(x, y, args.bbox, args.bbox, W, H)
#             tracker = create_kcf()
#             tracker.init(bgr, bb)
#             subs.append(SubTracker(tracker=tracker, bbox=bb, ok=True))
#             templs.append(extract_template(gray0, bb))
#         cloudlets.append(Cloudlet(subs=subs, last_median_xy=(qx, qy)))
#         templates.append(templs)

#     # Open3D scene
#     vis = o3d.visualization.Visualizer()
#     vis.create_window('KCF Cloudlet — Open3D', 1280, 720)
#     pcd = o3d.geometry.PointCloud(); vis.add_geometry(pcd)
#     spheres = create_spheres(np.array([pix_to_xyz(qx,qy,depth_at(depth,int(qx),int(qy)),K) for (qx,qy) in queries]))
#     for s in spheres: vis.add_geometry(s)

#     try:
#         while True:
#             bgr, depth = cam.read()
#             if bgr is None:
#                 continue
#             gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

#             # Update each subtracker; try NCC re-acquire if it fails
#             med_xyz_list = []
#             for ci, cloud in enumerate(cloudlets):
#                 centers_uv = []
#                 centers_xyz = []
#                 for si, sub in enumerate(cloud.subs):
#                     ok, bb = sub.tracker.update(bgr)
#                     if not ok:
#                         # Try template re-acquire near last median
#                         if cloud.last_median_xy is not None:
#                             reacq = ncc_reacquire(gray, templates[ci][si], cloud.last_median_xy, args.search)
#                             if reacq is not None:
#                                 sub.tracker = create_kcf()
#                                 sub.tracker.init(bgr, reacq)
#                                 sub.ok = True
#                                 sub.bbox = reacq
#                             else:
#                                 sub.ok = False
#                                 continue
#                         else:
#                             sub.ok = False
#                             continue
#                     else:
#                         sub.ok = True
#                         sub.bbox = tuple(map(int, bb))

#                     ux, uy = bbox_center(sub.bbox)
#                     centers_uv.append((ux, uy))
#                     z = depth_at(depth, int(round(ux)), int(round(uy)))
#                     centers_xyz.append(pix_to_xyz(ux, uy, z, K))

#                 # Robust median over successful subs
#                 if len(centers_uv) == 0:
#                     med_xyz_list.append((np.nan, np.nan, np.nan))
#                     continue
#                 uv = np.array(centers_uv, dtype=np.float32)
#                 xyz = np.array(centers_xyz, dtype=np.float32)
#                 med_uv = np.median(uv, axis=0)
#                 # filter xyz rows with NaNs
#                 good = np.isfinite(xyz).all(axis=1)
#                 med_xyz = tuple(np.median(xyz[good], axis=0)) if np.any(good) else (np.nan, np.nan, np.nan)
#                 cloud.last_median_xy = (float(med_uv[0]), float(med_uv[1]))
#                 med_xyz_list.append(med_xyz)

#             # Update Open3D point cloud (from current RGB-D)
#             pcd_new = rgbd_to_pcd(bgr, depth, K)
#             if len(pcd_new.points) > 0:
#                 pcd_new = pcd_new.voxel_down_sample(voxel_size=0.01)
#             pcd.points = pcd_new.points; pcd.colors = pcd_new.colors; pcd.normals = pcd_new.normals
#             vis.update_geometry(pcd)

#             # Update spheres
#             for s, xyz in zip(spheres, med_xyz_list):
#                 if not np.isfinite(xyz[2]):  # if Z is nan
#                     continue
#                 current = np.asarray(s.vertices); centroid = current.mean(axis=0)
#                 s.translate((np.array(xyz) - centroid).tolist())
#                 vis.update_geometry(s)

#             vis.poll_events(); vis.update_renderer()

#     finally:
#         try:
#             cam.stop()
#         except Exception:
#             pass
#         vis.destroy_window()

# if __name__ == '__main__':
#     main()
