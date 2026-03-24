#!/usr/bin/env python3
import rclpy, cv2, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

# def backproject(mask, depth_m, K):
#     """mask: HxW uint8, depth_m: HxW float32 in meters, K: (fx, fy, cx, cy)
#        returns Nx3 xyz in camera frame"""
#     ys, xs = np.nonzero(mask)
#     if xs.size == 0: return np.empty((0,3), np.float32)
#     Z = depth_m[ys, xs]
#     good = np.isfinite(Z) & (Z > 0.05) & (Z < 5.0)
#     xs, ys, Z = xs[good], ys[good], Z[good]
#     if xs.size == 0: return np.empty((0,3), np.float32)
#     fx, fy, cx, cy = K
#     X = (xs - cx) * Z / fx
#     Y = (ys - cy) * Z / fy
#     return np.stack([X, Y, Z], axis=1).astype(np.float32)

# def robust_center(XYZ, max_sigma=2.5):
#     if XYZ.shape[0] < 10: return None, XYZ
#     med = np.median(XYZ, axis=0)
#     dev = np.linalg.norm(XYZ - med, axis=1)
#     mad = np.median(dev) + 1e-6
#     inliers = dev < max_sigma * 1.4826 * mad
#     Xf = XYZ[inliers]
#     if Xf.shape[0] < 10: return None, XYZ
#     return Xf.mean(axis=0), Xf

class ColorsTracker(Node):
    def __init__(self):
        super().__init__('colors_tracker')
        self.bridge = CvBridge()
        self.sub_img   = self.create_subscription(Image, 'image', self.on_img, 10)
        self.sub_info  = self.create_subscription(CameraInfo, 'camera_info', self.on_info, 10)
        self.sub_depth = self.create_subscription(Image, 'depth', self.on_depth, 10)

        self.pub_center = self.create_publisher(PointStamped, 'object_center', 10)
        self.pub_pose   = self.create_publisher(PoseStamped, 'object_pose_guess', 10)
        self.pub_mark   = self.create_publisher(Marker, 'object_points', 1)

        # HSV ranges per color (tune these!)
        # Format: (lower HSV), (upper HSV); H in [0,180] for OpenCV HSV.
        self.color_ranges = [
            # red (wrap-around handled via two ranges if needed)
            # ((0, 120, 80), (10, 255, 255)),
            # ((170, 120, 80), (180, 255, 255)),
            # # green
            # ((40, 80,  60), (80, 255, 255)),
            # # blue
            # ((100, 80, 60), (130, 255, 255)),
            # # add more faces as needed
        ]

        self.K = None    # (fx, fy, cx, cy)
        self.img = None  # last BGR
        self.depth = None # last depth meters

        # UI
        self.window="colors"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # ---- cursor sampling UI state ----
        self.cursor_xy = None                # (x, y) or None
        self.sample_hsv = []                 # list of sampled HSV (on left-click)
        self.proposed_ranges = []            # list of ((loH,loS,loV),(hiH,hiS,hiV)) or two for red wrap
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.timer = self.create_timer(0.05, self.render)

    def on_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy

    def on_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if d.dtype == np.uint16:
            depth_m = d.astype(np.float32) / 1000.0
        else:
            depth_m = d.astype(np.float32)
        self.depth = depth_m

    def on_img(self, msg: Image):
        if self.K is None or self.depth is None: return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = bgr
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        H, W = hsv.shape[:2]
        # table/ROI Z band (optional): compute from prior or set fixed window
        # Here we use a broad gate; tighten for stability.
        depth_m = self.depth
        zmin, zmax = 0.05, 3.0
        depth_gate = (depth_m > zmin) & (depth_m < zmax)

        # Build mask from all configured colors
        mask_all = np.zeros((H, W), np.uint8)
        for lo, hi in self.color_ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask_all |= m

        # Morphology to clean up
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # Depth gate
        mask &= depth_gate.astype(np.uint8) * 255

        # Backproject to 3D
        # XYZ = backproject(mask, depth_m, self.K)
        # center, inliers = robust_center(XYZ)
        # if center is None:
        #     return

        # # Publish center
        # pt = PointStamped()
        # pt.header = msg.header
        # pt.header.frame_id = 'camera_color_optical_frame'  # your camera frame
        # pt.point.x, pt.point.y, pt.point.z = map(float, center.tolist())
        # self.pub_center.publish(pt)

        # # Optional: rough pose by PCA
        # pose = PoseStamped()
        # pose.header = pt.header
        # # simple smoothing-free orientation guess:
        # cov = np.cov(inliers.T)
        # vals, vecs = np.linalg.eigh(cov)   # columns are eigenvectors (ascending)
        # # largest variance corresponds to longest visible axis; ensure right-handed:
        # R = vecs  # 3x3
        # if np.linalg.det(R) < 0: R[:,0] *= -1
        # # Convert R to quaternion:
        # qw = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
        # qx = (R[2,1] - R[1,2])/(4*qw); qy = (R[0,2] - R[2,0])/(4*qw); qz = (R[1,0] - R[0,1])/(4*qw)
        # pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pt.point.x, pt.point.y, pt.point.z
        # pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = float(qw), float(qx), float(qy), float(qz)
        # self.pub_pose.publish(pose)

        # # RViz scatter for debug
        # m = Marker()
        # m.header = pt.header
        # m.ns = "object_pts"; m.id = 0
        # m.type = Marker.POINTS; m.action = Marker.ADD
        # m.scale.x = 0.002; m.scale.y = 0.002
        # m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0
        # m.points = [type(pt.point)(x=float(x), y=float(y), z=float(z)) for x,y,z in inliers[:2000]]
        # self.pub_mark.publish(m)

    # ---- UI loop & key handling ----

    # --- Mouse callback: move to inspect; left-click to add sample; right-click to clear
    def on_mouse(self, event, x, y, flags, param):
        if self.img is None:  # nothing yet
            return
        self.cursor_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Grab BGR at cursor; convert to HSV (OpenCV HSV uses H in [0,180])
            bgr = self.img[y, x].reshape(1, 1, 3)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            self.sample_hsv.append(hsv.astype(np.uint16))
            self._update_proposed_ranges()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.sample_hsv.clear()
            self.proposed_ranges.clear()

    # --- Propose HSV ranges from samples (handles hue wrap-around for reds)
    def _update_proposed_ranges(self):
        if not self.sample_hsv:
            self.proposed_ranges = []
            return
        arr = np.array(self.sample_hsv, dtype=np.uint16)  # shape (N,3)
        H = arr[:, 0].astype(np.int32)  # [0,180]
        S = arr[:, 1]; V = arr[:, 2]

        # Pick 5th-95th percentiles for robustness
        def pct_range(x):
            lo = int(np.percentile(x, 5))
            hi = int(np.percentile(x, 95))
            return max(0, lo), min(int(255 if x is not H else 180), hi)

        # Handle hue circularity by trying raw vs shifted
        span_raw = (H.max() - H.min())
        H_shift = H.copy()
        # shift small hues up by +180 when that reduces span (typical for reds)
        H_shift[H_shift < 90] += 180
        span_shift = (H_shift.max() - H_shift.min())

        if span_raw <= span_shift:
            h_lo, h_hi = np.percentile(H, [5, 95]).astype(int)
            # single contiguous hue range
            s_lo, s_hi = pct_range(S)
            v_lo, v_hi = pct_range(V)
            self.proposed_ranges = [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
        else:
            # wrap-around case -> two ranges
            h_lo_s, h_hi_s = np.percentile(H_shift, [5, 95]).astype(int)
            # bring back to [0,180)
            h_lo = h_lo_s % 180
            h_hi = h_hi_s % 180
            s_lo, s_hi = pct_range(S)
            v_lo, v_hi = pct_range(V)
            # Example: h_lo=170, h_hi=20 -> two intervals: [0..20] and [170..180]
            if h_lo <= h_hi:
                # Shouldn't happen for wrap, but guard anyway
                self.proposed_ranges = [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
            else:
                self.proposed_ranges = [
                    ((0,   s_lo, v_lo), (h_hi, s_hi, v_hi)),
                    ((h_lo, s_lo, v_lo), (180, s_hi, v_hi))
                ]

        # Print to console so you can copy/paste into self.color_ranges
        txt = "Suggested HSV range(s): " + "  OR  ".join(
            [f"{lo} .. {hi}" for (lo, hi) in self.proposed_ranges]
        )
        self.get_logger().info(txt)

    def _draw_overlay(self, vis, x, y, bgr, hsv, depth_val):
        # Small panel in the top-left
        panel_h, panel_w = 115, 300
        cv2.rectangle(vis, (8, 8), (8+panel_w, 8+panel_h), (30, 30, 30), thickness=-1)
        cv2.rectangle(vis, (8, 8), (8+panel_w, 8+panel_h), (180, 180, 180), thickness=1)

        # Color swatch (BGR)
        cv2.rectangle(vis, (16, 16), (16+80, 16+80), tuple(int(c) for c in bgr.tolist()), thickness=-1)
        cv2.rectangle(vis, (16, 16), (16+80, 16+80), (255, 255, 255), 1)

        # Text lines
        def put(line, row):
            cv2.putText(vis, line, (110, 28 + 20*row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        put(f"BGR: ({int(bgr[0])}, {int(bgr[1])}, {int(bgr[2])})", 0)
        put(f"HSV: ({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})", 1)
        if depth_val is not None and np.isfinite(depth_val):
            put(f"Depth: {depth_val:0.3f} m", 2)
        put(f"Samples: {len(self.sample_hsv)}", 3)

        # Show proposed range(s)
        for i, (lo, hi) in enumerate(self.proposed_ranges[:2]):  # show up to two
            put(f"R{i+1}: {lo} .. {hi}", 4+i)

        # Crosshair at cursor
        cv2.drawMarker(vis, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

    # ---- UI loop & key handling ----
    def render(self):
        if self.img is None:
            self.get_logger().info("img is still none")
            return

        vis = self.img.copy()

        # If we have a cursor, show values at that pixel
        if self.cursor_xy is not None:
            x, y = self.cursor_xy
            h, w = vis.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                bgr = vis[y, x]  # B, G, R
                hsv = cv2.cvtColor(bgr.reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]
                depth_val = None
                if self.depth is not None and 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                    depth_val = float(self.depth[y, x])
                self._draw_overlay(vis, x, y, bgr, hsv, depth_val)

        cv2.imshow(self.window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # quick clear
            self.sample_hsv.clear()
            self.proposed_ranges.clear()
        elif key == ord('p'):  # print again
            if self.proposed_ranges:
                txt = "Suggested HSV range(s): " + "  OR  ".join(
                    [f"{lo} .. {hi}" for (lo, hi) in self.proposed_ranges]
                )
                self.get_logger().info(txt)

    # def render(self):
    #     if self.img is None:
    #         self.get_logger().info("img is still none")
    #         return

    #     vis = self.img.copy()
    #     cv2.imshow(self.window, vis)
    #     key = cv2.waitKey(1) & 0xFF

def main():
    rclpy.init()
    n = ColorsTracker()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
