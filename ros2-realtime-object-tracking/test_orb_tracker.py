#!/usr/bin/env python3
"""
Real-time sparse 3D point tracking in RGB-D using ORB feature matching + Intel RealSense.

- Click on the RGB window to pick a point on your block. The script snaps to the nearest ORB keypoint
  and then tracks that keypoint across frames using descriptor matching (BFMatcher + ratio test).
- Depth is read from the aligned depth image to produce a 3D estimate in the color camera frame.
- Keys: 'r' to reselect, 'q'/ESC to quit, 'd' to toggle debug overlays.

Requirements:
  pip install opencv-python numpy pyrealsense2

Notes:
- ORB is fast on CPU and far more robust than KLT to larger motions/rotation.
- If you want SuperPoint later, we can swap the detector/descriptor while keeping this scaffolding.
"""

import time
import argparse
from collections import deque

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None


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


class ORBPointTracker:
    def __init__(self, nfeatures=1500, trail_len=96, snap_radius=8):
        # Create ORB and a BFMatcher for binary descriptors
        self.orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_kps = None
        self.prev_desc = None
        self.curr_pt = None          # (u,v) float
        self.curr_desc = None        # 1xD descriptor of the tracked feature
        self.has_target = False
        self.trail = deque(maxlen=trail_len)
        self.snap_radius = snap_radius
        self.debug = False

    def _detect(self, gray):
        kps, desc = self.orb.detectAndCompute(gray, None)
        return kps, desc

    def set_target_from_click(self, gray, x, y):
        """Snap the clicked location to the nearest ORB keypoint and set as target."""
        kps, desc = self._detect(gray)
        if not kps:
            self._clear()
            return False
        # Find nearest keypoint to click in pixel space
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        d2 = np.sum((pts - np.array([x, y], dtype=np.float32))**2, axis=1)
        idx = int(np.argmin(d2))
        if np.sqrt(d2[idx]) > self.snap_radius:
            # Too far from any keypoint; do not set
            self._clear()
            return False
        self.curr_pt = tuple(pts[idx])
        self.curr_desc = desc[idx:idx+1] if desc is not None else None
        self.prev_kps, self.prev_desc = kps, desc
        self.has_target = self.curr_desc is not None
        if self.has_target:
            self.trail.clear()
            self.trail.append(self.curr_pt)
        return self.has_target

    def _clear(self):
        self.prev_kps = None
        self.prev_desc = None
        self.curr_pt = None
        self.curr_desc = None
        self.has_target = False
        self.trail.clear()

    def reset(self):
        self._clear()

    def step(self, gray):
        """Advance one frame. Returns (pt, lost, dbg) where pt is (u,v) or None."""
        if not self.has_target or self.curr_desc is None:
            # Recompute keypoints for potential selection UI, but no tracking
            self.prev_kps, self.prev_desc = self._detect(gray)
            return None, True, {}

        # Detect in current frame
        kps, desc = self._detect(gray)
        dbg = {"kps": kps}
        if desc is None or len(kps) == 0:
            self.prev_kps, self.prev_desc = kps, desc
            return None, True, dbg

        # Match previous tracked descriptor to current descriptors (KNN, ratio test)
        matches = self.matcher.knnMatch(self.curr_desc, desc, k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if not good:
            # Try relaxed ratio or fallback to nearest
            nearest = self.matcher.match(self.curr_desc, desc)
            if nearest:
                m = min(nearest, key=lambda x: x.distance)
                good = [m]

        if not good:
            self.prev_kps, self.prev_desc = kps, desc
            return None, True, dbg

        # Choose best match; optionally bias toward spatial proximity
        # Add spatial proximity scoring
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        last_pt = np.array(self.curr_pt, dtype=np.float32)
        def score(m):
            # lower is better; combine Hamming distance and pixel distance
            pix = np.linalg.norm(pts[m.trainIdx] - last_pt)
            return m.distance + 0.5 * pix
        best = min(good, key=score)
        new_pt = tuple(pts[best.trainIdx])
        self.curr_pt = new_pt
        self.curr_desc = desc[best.trainIdx:best.trainIdx+1]
        self.prev_kps, self.prev_desc = kps, desc
        self.trail.append(self.curr_pt)
        return self.curr_pt, False, dbg


def draw_overlay(frame, fps, point, point_3d, lost, trail, show_debug=False, dbg_kps=None):
    vis = frame
    h, w = vis.shape[:2]

    # Keypoints overlay (debug)
    if show_debug and dbg_kps is not None:
        for kp in dbg_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(vis, (x, y), 2, (100, 100, 255), -1)

    # Trail
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            cv2.line(vis, (int(trail[i-1][0]), int(trail[i-1][1])), (int(trail[i][0]), int(trail[i][1])), (0, 255, 0), 2)

    # Current point
    if point is not None:
        cv2.circle(vis, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # HUD
    cv2.rectangle(vis, (5, 5), (w - 5, 80), (0, 0, 0), -1)
    cv2.putText(vis, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status = "LOST (press 'r' to reselect)" if lost else "Tracking"
    cv2.putText(vis, status, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if lost else (0, 255, 0), 2)
    if point_3d is not None:
        X, Y, Z = point_3d
        cv2.putText(vis, f"3D (m): X={X:.3f} Y={Y:.3f} Z={Z:.3f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(vis, f"3D (m): unavailable", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Real-time ORB-based 3D point tracking with RealSense RGB-D")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--nfeatures", type=int, default=1500)
    parser.add_argument("--snap_radius", type=int, default=10, help="Pixel radius for snapping click to nearest keypoint")
    parser.add_argument("--trail", type=int, default=96)
    parser.add_argument("--debug", action="store_true", help="Show detected keypoints")
    args = parser.parse_args()

    cam = RealSenseRGBD(width=args.width, height=args.height, fps=args.fps)
    tracker = ORBPointTracker(nfeatures=args.nfeatures, trail_len=args.trail, snap_radius=args.snap_radius)

    window = "RGB (click to select ORB keypoint)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Mouse selection
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            ok = tracker.set_target_from_click(cv2.cvtColor(frame_cache, cv2.COLOR_BGR2GRAY), x, y)
            print("Selected target" if ok else "Failed to select (no nearby keypoint)")
    cv2.setMouseCallback(window, on_mouse)

    fps = 0.0
    t_prev = time.time()
    global frame_cache
    frame_cache = None

    try:
        while True:
            bgr, depth_m, ts = cam.read()
            if bgr is None:
                continue
            frame_cache = bgr.copy()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            pt, lost, dbg = tracker.step(gray)

            # Depth to 3D
            pt3 = None
            if pt is not None:
                u, v = int(round(pt[0])), int(round(pt[1]))
                if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]:
                    z = float(depth_m[v, u])
                    if z <= 0 or not np.isfinite(z):
                        # median fallback
                        y0, y1 = max(0, v-1), min(depth_m.shape[0], v+2)
                        x0, x1 = max(0, u-1), min(depth_m.shape[1], u+2)
                        patch = depth_m[y0:y1, x0:x1]
                        good = patch[np.isfinite(patch) & (patch > 0)]
                        z = float(np.median(good)) if good.size > 0 else 0.0
                    if z > 0:
                        pt3 = cam.deproject(u, v, z)

            # FPS smoothing
            now = time.time(); dt = now - t_prev; t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            vis = draw_overlay(bgr.copy(), fps, pt, pt3, lost, tracker.trail, show_debug=args.debug, dbg_kps=dbg.get("kps"))
            cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                tracker.reset()
            elif key == ord('d'):
                args.debug = not args.debug

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
