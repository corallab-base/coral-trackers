#!/usr/bin/env python3
"""
SpaTrackerV2 (ONLINE) — RealSense RGB‑D point tracking with **Open3D** 3D visualization (no OpenCV UI)

- Loads the online model: `Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")` (same family as the repo’s `inference.py`).
- Accepts **query pixels** via CLI (`--queries "320,240;100,200"`) or creates a **grid** (`--grid_size`).
- Runs **online** by repeatedly calling the model’s **`forward(...)`** on a sliding window, using the *same* argument pattern you shared (RGBD case):
  `model.forward(video_tensor, depth=..., intrs=..., extrs=..., queries=..., fps=..., iters_track=..., ...)`
- Visualizes live **3D spheres** at the tracked 3D point positions with **Open3D** (no cv2 UI).

Key API alignment with the provided `inference.py`:
- `from models.SpaTrackV2.models.predictor import Predictor`
- `from models.SpaTrackV2.models.utils import get_points_on_a_grid`
- Build `queries` as **(t, x, y)** with `t=0` → `query_xyt` (exactly like the script)
- Provide `video_tensor` (T,3,H,W), `depth_tensor` (T,H,W in meters), `intrs` (T,3,3), `extrs` (T,4,4)
- Optional: set `model.spatrack.track_num` to the number of grid/VO points

Usage examples:
  python spatrackerv2_realsense_online_open3d.py --repo ./SpaTrackerV2 --queries "320,240;220,180"
  python spatrackerv2_realsense_online_open3d.py --repo ./SpaTrackerV2 --grid_size 10 --vo_points 500

Controls: Open/close the Open3D window to run/stop. (No mouse picking in-window; use CLI queries for now.)
"""

import os
import sys
import time
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

import open3d as o3d


# ---------------------------- RealSense wrapper ---------------------------- #
class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not available. Install it and connect a RealSense camera.")
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(self.config)
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


# ---------------------------- Open3D helpers ------------------------------- #

def make_o3d_intrinsics_from_K(K: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
    return intr


def rgbd_to_pcd(rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> o3d.geometry.PointCloud:
    rgb_o3 = o3d.geometry.Image(rgb[:, :, ::-1].copy())  # BGR->RGB
    depth_o3 = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3, depth_o3, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    intr = make_o3d_intrinsics_from_K(K, rgb.shape[1], rgb.shape[0])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    return pcd


def create_spheres(points_xyz: np.ndarray, radius=0.01, color=(1.0, 0.0, 0.0)):
    spheres = []
    for p in points_xyz:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        m.paint_uniform_color(color)
        m.translate(p.tolist())
        m.compute_vertex_normals()
        spheres.append(m)
    return spheres


def update_spheres(meshes, points_xyz: np.ndarray):
    for m, p in zip(meshes, points_xyz):
        current = np.asarray(m.vertices)
        centroid = current.mean(axis=0)
        m.translate((p - centroid).tolist())


# ---------------------------- Query utils --------------------------------- #

def parse_queries(s: str) -> np.ndarray:
    if not s:
        return np.empty((0, 2), dtype=np.float32)
    pairs = []
    for part in s.split(';'):
        x, y = part.split(',')
        pairs.append([float(x), float(y)])
    return np.array(pairs, dtype=np.float32)


def make_grid_queries(W: int, H: int, n: int) -> np.ndarray:
    xs = np.linspace(W/(n+1), W - W/(n+1), n)
    ys = np.linspace(H/(n+1), H - H/(n+1), n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(np.float32)


def build_query_xyt(pix_xy: np.ndarray) -> np.ndarray:
    """Convert (Q,2) pixel coords into (Q,3) (t,x,y) with t=0, matching inference.py."""
    if pix_xy.size == 0:
        return np.empty((0,3), dtype=np.float32)
    t_col = np.zeros((pix_xy.shape[0], 1), dtype=np.float32)
    return np.concatenate([t_col, pix_xy], axis=1)


# ---------------------------- Main ---------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='SpaTrackerV2 ONLINE — Open3D live visualization (API-aligned)')
    ap.add_argument('--repo', default='./SpaTrackerV2', help='Path to SpaTrackerV2 repo root')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--queries', type=str, default='', help='Semicolon‑separated pixels: "x1,y1;x2,y2"')
    ap.add_argument('--grid_size', type=int, default=0, help='If >0, add an NxN grid of queries')
    ap.add_argument('--vo_points', type=int, default=500, help='model.spatrack.track_num for VO/grid points')
    ap.add_argument('--window', type=int, default=8, help='Frames per online chunk (clip)')
    ap.add_argument('--stride', type=int, default=4, help='Re-run forward every N frames')
    ap.add_argument('--sphere_radius', type=float, default=0.015)
    ap.add_argument('--max_draw', type=int, default=5000)
    args = ap.parse_args()

    device = 'cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'

    # Import from repo following the inference.py paths you provided
    sys.path.append(args.repo)
    try:
        from models.SpaTrackV2.models.predictor import Predictor
        from models.SpaTrackV2.models.utils import get_points_on_a_grid  # parity
    except Exception:
        from predictor import Predictor  # type: ignore
        def get_points_on_a_grid(gs, hw, device='cpu'):
            H, W = hw
            xs = np.linspace(W/(gs+1), W - W/(gs+1), gs)
            ys = np.linspace(H/(gs+1), H - H/(gs+1), gs)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
            return torch.from_numpy(pts)[None]

    # Load ONLINE model exactly like the script, then set track_num
    print('Loading SpaTrackerV2-Online …')
    model = Predictor.from_pretrained('Yuxihenry/SpatialTrackerV2-Online')
    model.eval()
    model.to(device)
    if hasattr(model, 'spatrack') and hasattr(model.spatrack, 'track_num'):
        model.spatrack.track_num = int(args.vo_points)

    cam = RealSenseRGBD(width=args.width, height=args.height, fps=args.fps)
    K_native = cam.intrinsics_matrix()

    # Build user queries (pixels)
    clicks_xy = parse_queries(args.queries)

    # Open3D scene
    vis = o3d.visualization.Visualizer()
    vis.create_window('SpaTrackerV2 ONLINE — Open3D', 1280, 720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # Adjust size as needed
        origin=[0.0, 0.0, 0.0]  # Adjust origin as needed
    )

    # Visualize the coordinate frame
    vis.add_geometry(coordinate_frame)

    marker_meshes = []
    last_count = 0

    # Online buffers
    rgb_buf: List[np.ndarray] = []
    d_buf: List[np.ndarray] = []
    intr_buf: List[np.ndarray] = []
    extr_buf: List[np.ndarray] = []

    frame_id = 0
    try:
        while True:
            bgr, depth_m = cam.read()
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            K = K_native

            rgb = bgr[:, :, ::-1]  # RGB uint8
            rgb_buf.append(rgb)
            d_buf.append(depth_m)
            intr_buf.append(K)
            extr_buf.append(np.eye(4, dtype=np.float32))  # stationary camera

            # Keep sliding window
            if len(rgb_buf) > args.window:
                rgb_buf = rgb_buf[-args.window:]
                d_buf = d_buf[-args.window:]
                intr_buf = intr_buf[-args.window:]
                extr_buf = extr_buf[-args.window:]

            do_run = (frame_id % max(1, args.stride) == 0) and (len(rgb_buf) == args.window)
            frame_id += 1

            if do_run:
                # Pack tensors per inference.py expectations (RGBD path)
                video_tensor = torch.from_numpy(np.stack(rgb_buf,0)).permute(0,3,1,2).float()  # T,3,H,W in [0..255]
                depth_tensor = torch.from_numpy(np.stack(d_buf,0)).float()                     # T,H,W (meters)
                intrs = torch.from_numpy(np.stack(intr_buf,0)).float()                         # T,3,3
                extrs = torch.from_numpy(np.stack(extr_buf,0)).float()                         # T,4,4

                # queries: clicks + optional grid → (Q,3) with t=0
                grid_xy = make_grid_queries(W, H, args.grid_size) if args.grid_size>0 else np.empty((0,2), np.float32)
                all_xy = np.concatenate([clicks_xy, grid_xy], axis=0) if (clicks_xy.size + grid_xy.size) > 0 else np.empty((0,2), np.float32)
                query_xyt = build_query_xyt(all_xy)

                with torch.no_grad():
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device=='cuda' else torch.float32):
                        (
                            c2w_traj, intrs_out, point_map, conf_depth,
                            track3d_pred, track2d_pred, vis_pred, conf_pred, video_out
                        ) = model.forward(
                            video_tensor, depth=depth_tensor.numpy(),
                            intrs=intrs.numpy(), extrs=extrs.numpy(),
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=None,
                            support_frame=len(rgb_buf)-1, replace_ratio=0.2
                        )

                # Take current 3D points (camera coords) at the end of the window
                if isinstance(track3d_pred, torch.Tensor):
                    xyz = track3d_pred[-1, :, :3].detach().cpu().numpy()
                else:
                    xyz = np.asarray(track3d_pred)[-1, :, :3]

                # Update point cloud from the most recent frame
                pcd_new = rgbd_to_pcd(bgr, depth_m, K)
                if len(pcd_new.points) > 0:
                    pcd_new = pcd_new.voxel_down_sample(voxel_size=0.01)
                pcd.points = pcd_new.points
                pcd.colors = pcd_new.colors
                pcd.normals = pcd_new.normals
                vis.update_geometry(pcd)

                # Update markers
                if xyz.shape[0] != last_count or not marker_meshes:
                    for m in marker_meshes:
                        vis.remove_geometry(m, reset_bounding_box=False)
                    marker_meshes = create_spheres(xyz[:args.max_draw], radius=args.sphere_radius, color=(1.0,0.2,0.2))
                    for m in marker_meshes:
                        vis.add_geometry(m)
                    last_count = xyz.shape[0]
                else:
                    update_spheres(marker_meshes, xyz[:len(marker_meshes)])
                    for m in marker_meshes:
                        vis.update_geometry(m)

            # Render
            vis.poll_events(); vis.update_renderer()

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        vis.destroy_window()


if __name__ == '__main__':
    main()
