#!/usr/bin/env python3
"""
SuperPoint (PyTorch) sparse 3D point tracking in RGB‑D (Intel RealSense)
using the **official PyTorch model and checkpoint** from rpautrat/SuperPoint.

Repo layout assumption (as you requested): this script sits next to the repo and weights:
.
├─ superpoint_rgbd_point_tracker.py   <-- this file
├─ SuperPoint/                        <-- cloned repo
│   ├─ superpoint_pytorch.py          <-- PyTorch model (provided link)
│   └─ weights/superpoint_v6_from_tf.pth

Run:
  pip install opencv-python numpy torch torchvision pyrealsense2
  python superpoint_rgbd_point_tracker.py \
      --sp_repo_dir ./SuperPoint \
      --weights ./SuperPoint/weights/superpoint_v6_from_tf.pth \
      --width 640 --height 480 --fps 30

Controls:
  - Click to select (snaps to nearest SuperPoint keypoint)
  - 'r' reselect, 'd' toggle keypoint overlay, 'q'/ESC quit

Notes:
  - Entirely **PyTorch**; no TensorFlow used.
  - Uses model outputs: "keypoints" (Nx2), "keypoint_scores" (N), "descriptors" (DxN).
  - Draws **all detected keypoints every frame** so you can debug density/coverage.
  - SuperPoint prefers image sizes divisible by 8 (e.g., 640×480).
"""

import os
import sys
import time
import argparse
from collections import deque

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ---------------------------- Camera (RealSense) ---------------------------- #
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
        self.color_intr = color_stream.get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None, None, None
        bgr = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()) * self.depth_scale
        ts = color.get_timestamp() / 1000.0
        return bgr, depth_m, ts

    def deproject(self, u, v, z):
        if z <= 0 or not np.isfinite(z):
            return None
        X = (u - self.color_intr.ppx) / self.color_intr.fx * z
        Y = (v - self.color_intr.ppy) / self.color_intr.fy * z
        return (float(X), float(Y), float(z))

    def stop(self):
        self.pipe.stop()


# ---------------------------- SuperPoint (Torch) ---------------------------- #
class TorchSuperPoint:
    def __init__(self, repo_dir: str, weights_path: str, use_cuda: bool = True):
        self.device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
        sys.path.append(repo_dir)
        from superpoint_pytorch import SuperPoint  # noqa: E402
        self.model = SuperPoint().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        state = state.get('state_dict', state)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def infer(self, gray_u8: np.ndarray):
        """Return (keypoints Nx2 float32, scores N float32, descriptors DxN float32)."""
        img = torch.from_numpy(gray_u8).float().unsqueeze(0).unsqueeze(0) / 255.0
        img = img.to(self.device)
        out = self.model({'image': img})
        kps = out['keypoints'][0].detach().cpu().numpy().astype(np.float32)        # Nx2 (x,y)
        scores = out['keypoint_scores'][0].detach().cpu().numpy().astype(np.float32)  # N
        desc = out['descriptors'][0].detach().cpu().numpy().astype(np.float32)     # DxN
        # Safety normalize descriptors column‑wise
        desc /= (np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8)
        return kps, scores, desc


# ---------------------------- Tracker -------------------------------------- #
class SPPointTracker:
    def __init__(self, sp_model: TorchSuperPoint, trail_len=128, snap_radius=12,
                 ratio_thresh=0.8, spatial_sigma=10.0, max_jump_px=80):
        self.sp = sp_model
        self.trail = deque(maxlen=trail_len)
        self.snap_radius = snap_radius
        self.ratio_thresh = ratio_thresh
        self.spatial_sigma = spatial_sigma
        self.max_jump_px = max_jump_px
        self.prev_desc = None   # Dx1
        self.prev_pt = None
        self.curr_pt = None
        self.has_target = False
        self.debug_kps = None

    def _clear(self):
        self.prev_desc = None
        self.prev_pt = None
        self.curr_pt = None
        self.trail.clear()
        self.has_target = False
        self.debug_kps = None

    def reset(self):
        self._clear()

    def set_target_from_click(self, gray: np.ndarray, x: int, y: int) -> bool:
        kps, scores, desc = self.sp.infer(gray)
        self.debug_kps = kps
        if kps.shape[0] == 0:
            self._clear(); return False
        d2 = np.sum((kps - np.array([[x, y]], dtype=np.float32))**2, axis=1)
        idx = int(np.argmin(d2))
        if np.sqrt(d2[idx]) > self.snap_radius:
            self._clear(); return False
        self.curr_pt = tuple(kps[idx].astype(float))
        self.prev_pt = self.curr_pt
        self.prev_desc = desc[idx:idx+1]  # Dx1
        self.trail.clear(); self.trail.append(self.curr_pt)
        self.has_target = True
        return True

    def step(self, gray: np.ndarray):
        kps, scores, desc = self.sp.infer(gray)
        self.debug_kps = kps
        if not self.has_target or self.prev_desc is None or kps.shape[0] == 0:
            return None, True, {}
        # Compute L2 distance to previous descriptor (correct broadcasting: DxN - Dx1)
        dists = np.linalg.norm(desc - self.prev_desc, axis=1)  # N
        if dists.size < 2:
            return None, True, {}
        order = np.argsort(dists)
        best, second = order[0], order[1]
        ratio_ok = (dists[best] / (dists[second] + 1e-6)) < self.ratio_thresh
        last = np.array(self.prev_pt, dtype=np.float32)
        pix_d = np.linalg.norm(kps - last, axis=1)
        # Gate extreme jumps to avoid wild hops
        pix_d_clipped = np.minimum(pix_d, self.max_jump_px)
        score = dists + (pix_d_clipped / self.spatial_sigma)
        best_idx = int(np.argmin(score)) if ratio_ok else int(best)
        # Optional: reject if jump too large
        if pix_d[best_idx] > self.max_jump_px:
            return None, True, {}
        self.curr_pt = tuple(kps[best_idx].astype(float))
        self.prev_pt = self.curr_pt
        self.prev_desc = desc[best_idx:best_idx+1]
        self.trail.append(self.curr_pt)
        return self.curr_pt, False, {}


# ---------------------------- Viz ------------------------------------------ #
def draw_overlay(frame, fps, point, point_3d, lost, trail, all_kps=None, show_kps=True):
    vis = frame
    h, w = vis.shape[:2]

    # Draw all detected keypoints (debug visualization)
    if show_kps and all_kps is not None and all_kps.size > 0:
        for (x, y) in all_kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 1, (120, 220, 220), -1)  # cyan points

    # Trail
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            cv2.line(vis, (int(trail[i-1][0]), int(trail[i-1][1])), (int(trail[i][0]), int(trail[i][1])), (0, 255, 0), 2)

    # Current point
    if point is not None:
        cv2.circle(vis, (int(point[0]), int(point[1])), 6, (0, 0, 255), -1)

    # HUD
    cv2.rectangle(vis, (5, 5), (w - 5, 90), (0, 0, 0), -1)
    cv2.putText(vis, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Keypoints: {0 if all_kps is None else all_kps.shape[0]}", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    status = "LOST (press 'r' to reselect)" if lost else "Tracking"
    cv2.putText(vis, status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if lost else (0, 255, 0), 2)
    if point_3d is not None:
        X, Y, Z = point_3d
        cv2.putText(vis, f"3D (m): X={X:.3f} Y={Y:.3f} Z={Z:.3f}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(vis, f"3D (m): unavailable", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    return vis


# ---------------------------- Main ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="SuperPoint (PyTorch) RGB‑D single‑point tracker (RealSense)")
    parser.add_argument('--sp_repo_dir', type=str, default='./SuperPoint', help='Path to rpautrat/SuperPoint repo')
    parser.add_argument('--weights', type=str, default='./SuperPoint/weights/superpoint_v6_from_tf.pth', help='Path to .pth checkpoint')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--snap_radius', type=int, default=12)
    parser.add_argument('--trail', type=int, default=128)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--spatial_sigma', type=float, default=10.0)
    parser.add_argument('--max_jump_px', type=float, default=80)
    parser.add_argument('--show_kps', action='store_true', default=True, help='Always draw all keypoints')
    args = parser.parse_args()

    sp = TorchSuperPoint(args.sp_repo_dir, args.weights, use_cuda=True)

    cam = RealSenseRGBD(width=args.width, height=args.height, fps=args.fps)
    tracker = SPPointTracker(sp, trail_len=args.trail, snap_radius=args.snap_radius,
                             ratio_thresh=args.ratio, spatial_sigma=args.spatial_sigma,
                             max_jump_px=int(args.max_jump_px))

    window = "SuperPoint (PyTorch) RGB (click to select)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_cache = {"gray": None}
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and frame_cache["gray"] is not None:
            ok = tracker.set_target_from_click(frame_cache["gray"], x, y)
            print("Selected SuperPoint" if ok else "No nearby SuperPoint (try another spot or increase --snap_radius)")
    cv2.setMouseCallback(window, on_mouse)

    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            bgr, depth_m, ts = cam.read()
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frame_cache["gray"] = gray

            pt, lost, _ = tracker.step(gray)

            # Depth -> 3D
            pt3 = None
            if pt is not None:
                u, v = int(round(pt[0])), int(round(pt[1]))
                if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]:
                    z = float(depth_m[v, u])
                    if z <= 0 or not np.isfinite(z):
                        y0, y1 = max(0, v-1), min(depth_m.shape[0], v+2)
                        x0, x1 = max(0, u-1), min(depth_m.shape[1], u+2)
                        patch = depth_m[y0:y1, x0:x1]
                        good = patch[np.isfinite(patch) & (patch > 0)]
                        z = float(np.median(good)) if good.size > 0 else 0.0
                    if z > 0:
                        pt3 = cam.deproject(u, v, z)

            now = time.time(); dt = now - t_prev; t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            vis = draw_overlay(bgr.copy(), fps, pt, pt3, lost, tracker.trail,
                               all_kps=tracker.debug_kps, show_kps=args.show_kps)
            cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                tracker.reset()
            elif key == ord('d'):
                args.show_kps = not args.show_kps

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
