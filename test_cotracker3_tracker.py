#!/usr/bin/env python3
"""
CoTracker3 ONLINE RGB‑D point tracker (RealSense)

This script uses Meta's CoTracker3 **online** mode via PyTorch Hub to track
user‑selected points from a live RGB stream, while lifting them to 3D using the
aligned depth stream from an Intel RealSense camera.

Design choices
- We run the official **online** API: `cotracker3_online` (sliding window).
- For maximum reliability with the public API, we start from a **regular grid**
  of points and, when you click, we bind your clicks to the **nearest grid
  points** at the start frame. From then on, we only visualize those selected
  tracks. (This avoids guessing undocumented query‑point arguments in online
  mode; it mirrors the repo's online demo behavior.)
- If you want explicit user‑defined points instead of grid points, we can switch
  to the model's `query_points` interface (supported in offline; often available
  in online too) — but the grid‑binding version is the path of least resistance
  and works out‑of‑the‑box.

Controls
- Left‑click: add a point *before* starting.
- s: start online tracking (initializes the model and begins streaming)
- r: reset (stop tracking, clear clicks, re‑arm)
- a: toggle drawing **all** grid tracks (default: only selected points)
- q or ESC: quit

Requirements
  pip install torch torchvision opencv-python numpy imageio[ffmpeg] pyrealsense2

GPU strongly recommended (CUDA). Reduce resolution to improve FPS.
"""

import os
import time
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import imageio.v3 as iio

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


# ---------------------------- Utilities ------------------------------------ #
def to_tensor_video(frames_bgr):
    """frames_bgr: list of HxWx3 uint8 BGR images. Returns 1xT x3xHxW float32 RGB tensor."""
    if len(frames_bgr) == 0:
        return None
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    arr = np.stack(rgb, axis=0)  # T x H x W x 3
    vid = torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float()  # 1 x T x 3 x H x W
    return vid


def draw_tracks(frame_bgr, pts_xy, depth_m=None, intr=None, show_all=False, all_pts_xy=None, trail=None):
    vis = frame_bgr
    h, w = vis.shape[:2]
    # all grid points (faint)
    if show_all and all_pts_xy is not None and len(all_pts_xy):
        for (x, y) in all_pts_xy:
            cv2.circle(vis, (int(x), int(y)), 1, (120, 220, 220), -1)
    # trail for selected
    if trail is not None and len(trail) >= 2:
        for i in range(1, len(trail)):
            cv2.line(vis, (int(trail[i-1][0]), int(trail[i-1][1])), (int(trail[i][0]), int(trail[i][1])), (0, 255, 0), 2)
    # current selected points
    for (x, y) in pts_xy:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        if depth_m is not None and intr is not None:
            u, v = int(round(x)), int(round(y))
            if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]:
                z = float(depth_m[v, u])
                if z <= 0 or not np.isfinite(z):
                    y0, y1 = max(0, v-1), min(depth_m.shape[0], v+2)
                    x0, x1 = max(0, u-1), min(depth_m.shape[1], u+2)
                    patch = depth_m[y0:y1, x0:x1]
                    good = patch[np.isfinite(patch) & (patch > 0)]
                    z = float(np.median(good)) if good.size > 0 else 0.0
                if z > 0:
                    X = (u - intr.ppx) / intr.fx * z
                    Y = (v - intr.ppy) / intr.fy * z
                    cv2.putText(vis, f"({X:.3f},{Y:.3f},{z:.3f}) m", (int(x)+8, int(y)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return vis


# ---------------------------- Main ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="CoTracker3 ONLINE RGB‑D point tracker (RealSense)")
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--grid_size', type=int, default=12, help='Initial grid density (e.g., 8, 10, 12, 16)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--show_all', action='store_true', help='Draw all grid tracks, not just selected')
    args = parser.parse_args()

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # Load CoTracker3 ONLINE from torch.hub (uses the repo's built-in checkpoints)
    print("Loading CoTracker3 online model via torch.hub …")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    cotracker.eval()
    step = getattr(cotracker, 'step', 8)
    window = step * 2
    print(f"Online step={step}, window={window}")

    # Camera
    cam = RealSenseRGBD(width=args.width, height=args.height, fps=args.fps)

    # Interaction state
    clicks_xy = []            # user clicks (x,y) before start
    selected_ids = None       # indices of grid tracks bound to clicks
    frame_buffer = deque(maxlen=window)
    last_draw_pts = []        # for a simple one‑point trail
    trail = deque(maxlen=128)
    tracking = False
    initialized = False

    win_name = "CoTracker3 ONLINE (s=start, r=reset, a=show all, click to add before start)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Clicks before start
    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicks_xy, tracking
        if event == cv2.EVENT_LBUTTONDOWN and not tracking:
            clicks_xy.append((float(x), float(y)))
            print(f"Added click at ({x},{y})")
    cv2.setMouseCallback(win_name, on_mouse)

    fps_ema = 0.0
    t_prev = time.time()

    try:
        while True:
            bgr, depth_m, ts = cam.read()
            if bgr is None:
                continue

            # Accumulate into sliding window
            frame_buffer.append(bgr.copy())

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('a'):
                args.show_all = not args.show_all
            elif key == ord('r'):
                tracking = False
                initialized = False
                clicks_xy.clear()
                selected_ids = None
                frame_buffer.clear()
                trail.clear()
                print("Reset. Click points, then press 's' to start.")
            elif key == ord('s') and not tracking:
                if len(clicks_xy) == 0:
                    print("No clicks; you can still start — select later via reset — tracking a grid only.")
                if len(frame_buffer) < window:
                    print(f"Waiting for {window} frames buffer before initialization…")
                tracking = True
                initialized = False

            # Not tracking yet: draw live with clicks
            if not tracking:
                vis = bgr.copy()
                for (x, y) in clicks_xy:
                    cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.putText(vis, "Click to add points, press 's' to start", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow(win_name, vis)
                continue

            # Start/continue online tracking once we have a full window
            if len(frame_buffer) < window:
                vis = bgr.copy()
                cv2.putText(vis, f"Buffering frames: {len(frame_buffer)}/{window}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.imshow(win_name, vis)
                continue

            # Prepare video chunk tensor [1, T, 3, H, W]
            video_chunk = to_tensor_video(list(frame_buffer)).to(device)

            if not initialized:
                # Initialize online processing with grid tracks. We'll bind user clicks
                # to nearest grid points after the first forward call.
                with torch.no_grad():
                    _ = cotracker(video_chunk=video_chunk, is_first_step=True, grid_size=args.grid_size)
                initialized = True
                selected_ids = None  # compute after first prediction below

            # Run the next online step
            with torch.no_grad():
                pred_tracks, pred_vis = cotracker(video_chunk=video_chunk)  # shapes: [1, T, N, 2], [1, T, N, 1]
            # current positions = the last frame of returned chunk
            pts_all = pred_tracks[0, -1].detach().cpu().numpy()  # N x 2 (x,y) in pixels

            if selected_ids is None:
                # Bind user clicks to nearest grid points at the *start* frame
                if len(clicks_xy) > 0:
                    # Use the first time index of the returned chunk (or last‑step midpoint); robustly pick closest
                    pts_start = pred_tracks[0, 0].detach().cpu().numpy()  # N x 2
                    selected_ids = []
                    for cx, cy in clicks_xy:
                        d2 = np.sum((pts_start - np.array([cx, cy]))**2, axis=1)
                        selected_ids.append(int(np.argmin(d2)))
                    print(f"Bound {len(selected_ids)} click(s) to grid IDs: {selected_ids}")
                else:
                    selected_ids = []

            # Compose drawing lists
            draw_pts = pts_all if (args.show_all and len(pts_all) < 8000) else []
            sel_pts = [tuple(pts_all[i]) for i in selected_ids] if selected_ids else []
            if len(sel_pts) == 1:
                trail.append(sel_pts[0])

            # FPS
            now = time.time(); dt = now - t_prev; t_prev = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            # Draw overlay and show
            vis = draw_tracks(bgr.copy(), sel_pts, depth_m=depth_m, intr=cam.color_intr,
                              show_all=args.show_all, all_pts_xy=draw_pts, trail=trail)
            cv2.putText(vis, f"FPS: {fps_ema:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(win_name, vis)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
