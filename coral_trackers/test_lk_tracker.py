#!/usr/bin/env python3
"""
Real-time sparse 3D point tracking in RGB-D using Lucas–Kanade (OpenCV) + Intel RealSense.

- Click on the RGB window to choose the point on the block to track.
- The script tracks that pixel with KLT and reads its depth to report a 3D position.
- Press 'r' to reselect point, 'q' or ESC to quit.

Requirements:
  pip install opencv-python numpy pyrealsense2

Notes:
- Uses RealSense color stream aligned to depth. If alignment fails, check camera connection.
- 3D point is computed in the color camera frame.
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
            raise RuntimeError("pyrealsense2 not available. Please install it or use a machine with a RealSense connected.")
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Get intrinsics for color stream (after start)
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()  # fx, fy, ppx, ppy

        # Depth scale (meters per unit)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self):
        """Returns (bgr, depth_in_meters, timestamp_s)"""
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
        """Return 3D point (X,Y,Z) in meters in color camera frame from pixel (u,v) and depth z (m)."""
        if z <= 0 or np.isnan(z):
            return None
        X = (u - self.color_intrinsics.ppx) / self.color_intrinsics.fx * z
        Y = (v - self.color_intrinsics.ppy) / self.color_intrinsics.fy * z
        return (float(X), float(Y), float(z))

    def stop(self):
        self.pipe.stop()


class LK3DTracker:
    def __init__(self, trail_len=64):
        # LK params tuned for robustness vs. speed
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.point = None  # current point (u,v)
        self.p0 = None     # shape (1,1,2) float32 for LK
        self.prev_gray = None
        self.trail = deque(maxlen=trail_len)
        self.lost = True

    def set_point(self, u, v):
        self.point = (float(u), float(v))
        self.p0 = np.array([[[self.point[0], self.point[1]]]], dtype=np.float32)
        self.trail.clear()
        self.lost = False

    def reset(self):
        self.point = None
        self.p0 = None
        self.trail.clear()
        self.lost = True
        self.prev_gray = None

    def step(self, gray):
        if self.p0 is None:
            self.prev_gray = gray
            return None, True  # no point, nothing to track
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.point, False
        # LK optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
        ok = st is not None and st[0, 0] == 1
        if not ok:
            self.lost = True
            self.prev_gray = gray
            return None, True
        u, v = float(p1[0, 0, 0]), float(p1[0, 0, 1])
        self.point = (u, v)
        self.p0 = p1
        self.prev_gray = gray
        self.trail.append(self.point)
        return self.point, False


def draw_hud(frame, fps, point, point_3d, lost, trail):
    h, w = frame.shape[:2]
    hud = frame

    # Draw trail
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            cv2.line(hud, (int(trail[i-1][0]), int(trail[i-1][1])), (int(trail[i][0]), int(trail[i][1])), (0, 255, 0), 2)

    # Draw current point
    if point is not None:
        cv2.circle(hud, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Text overlay
    cv2.rectangle(hud, (5, 5), (w - 5, 70), (0, 0, 0), -1)
    cv2.putText(hud, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status = "LOST (press 'r' to reselect)" if lost else "Tracking"
    cv2.putText(hud, status, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if lost else (0, 255, 0), 2)
    if point_3d is not None:
        X, Y, Z = point_3d
        cv2.putText(hud, f"3D (m): X={X:.3f} Y={Y:.3f} Z={Z:.3f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(hud, f"3D (m): unavailable", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)

    return hud


def main():
    parser = argparse.ArgumentParser(description="Real-time 3D point tracking with LK + RealSense RGB-D")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--trail", type=int, default=64, help="Length of the on-screen trajectory trail")
    args = parser.parse_args()

    cam = RealSenseRGBD(width=args.width, height=args.height, fps=args.fps)
    tracker = LK3DTracker(trail_len=args.trail)

    window = "RGB (click to select point)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Mouse callback to select a point
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.set_point(x, y)

    cv2.setMouseCallback(window, on_mouse)

    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            bgr, depth_m, ts = cam.read()
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            point, lost = tracker.step(gray)

            # 3D deprojection if we have a point and valid depth
            pt3 = None
            if point is not None:
                u, v = int(round(point[0])), int(round(point[1]))
                if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]:
                    z = float(depth_m[v, u])
                    # Optionally perform small median filter around the pixel for robustness
                    if z == 0 or np.isnan(z) or not np.isfinite(z):
                        # try 3x3 median
                        y0, y1 = max(0, v-1), min(depth_m.shape[0], v+2)
                        x0, x1 = max(0, u-1), min(depth_m.shape[1], u+2)
                        patch = depth_m[y0:y1, x0:x1]
                        z = float(np.median(patch[np.isfinite(patch) & (patch > 0)])) if patch.size > 0 else 0.0
                    if z > 0:
                        pt3 = cam.deproject(u, v, z)

            # FPS
            t_now = time.time()
            dt = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t_prev = t_now

            # Draw HUD
            vis = draw_hud(bgr.copy(), fps, point, pt3, lost, tracker.trail)

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                tracker.reset()

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
