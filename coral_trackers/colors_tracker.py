#!/usr/bin/env python3
import rclpy, cv2, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge


DEPTH_K = 5  # odd number (pixels)

# BGR colors for OpenCV
COLOR_BGR = {
    "red":    (0,   0, 255),
    "green":  (0, 255,   0),
    "blue":   (255, 0,   0),
    "yellow": (0, 255, 255),
    "cyan":   (255,255,  0),
    "magenta":(255, 0, 255),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
}

def _to_mask_u8(mask):
    """Return a single-channel uint8 mask in {0,255} with shape (H,W)."""
    m = mask
    if m.ndim == 3:  # if provided as (H,W,1)
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    # normalize to {0,255}
    if m.max() <= 1:
        m = (m > 0).astype(np.uint8) * 255
    return m

def _largest_component_u8(mask_u8: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Keep only the largest connected component (> min_area px) from a uint8 mask.
    Returns a uint8 mask with values in {0,255}.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return np.zeros((0, 0), np.uint8)
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros_like(m, dtype=np.uint8)
    # labels: 0 is background; choose argmax area among [1..num-1]
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if areas[idx - 1] < min_area:
        return np.zeros_like(m, dtype=np.uint8)
    kept = (labels == idx).astype(np.uint8) * 255
    return kept

def _mean_xyz_from_mask(mask_u8: np.ndarray, depth_m: np.ndarray, K) -> np.ndarray | None:
    """
    Compute mean 3D point (X,Y,Z) of all mask pixels using per-pixel depth and intrinsics K=(fx,fy,cx,cy).
    Returns (3,) float32 in camera frame, or None if insufficient valid depth.
    """
    fx, fy, cx, cy = K
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    zs = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0)
    if not np.any(valid):
        return None
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]
    Xs = (xs - cx) * zs / fx
    Ys = (ys - cy) * zs / fy
    # Optionally subsample for speed if huge:
    # if Xs.size > 20000: idx = np.random.choice(Xs.size, 20000, replace=False); Xs, Ys, zs = Xs[idx], Ys[idx], zs[idx]
    X = np.mean(Xs)
    Y = np.mean(Ys)
    Z = np.mean(zs)
    return np.array([X, Y, Z], dtype=np.float32)

def robust_depth_at_pixel(depth_m: np.ndarray, u: float, v: float, k: int = 5) -> float:
    """Median depth in a kxk window around (u, v). Returns 0.0 if none valid."""
    if depth_m is None:
        return 0.0
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    x0 = max(0, ui - k // 2); x1 = min(w, ui + k // 2 + 1)
    y0 = max(0, vi - k // 2); y1 = min(h, vi + k // 2 + 1)
    patch = depth_m[y0:y1, x0:x1]
    vals = patch[patch > 0.0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))

class ColorsTracker(Node):
    def __init__(self):
        super().__init__('colors_tracker')
        self.bridge = CvBridge()
        self.sub_img   = self.create_subscription(Image, 'image', self.on_img, 10)
        self.sub_info  = self.create_subscription(CameraInfo, 'camera_info', self.on_info, 10)
        self.sub_depth = self.create_subscription(Image, 'depth', self.on_depth, 10)

        # HSV ranges per color (tune these!)
        # Format: (lower HSV), (upper HSV); H in [0,180] for OpenCV HSV.
        self.color_ranges = {
            "yellow": ((21, 84, 168), (38, 255, 255)),
            "blue": ((95, 170, 80), (115, 255, 255)),
            "red": ((0, 121, 175), (18, 255, 255)),
            "green": ((39, 90, 87), (57, 255, 255)),
            "purple": ((109, 71, 98), (180, 255, 255)),
        }

        self.center_publishers = {
            k: self.create_publisher(PointStamped, f'{k}/center', 10)
            for k in self.color_ranges.keys()
        }

        self.centroid_publishers = {
            k: self.create_publisher(PointStamped, f'{k}/centroid', 10)
            for k in self.color_ranges.keys()
        }

        self.projected_centroid_publishers = {
            k: self.create_publisher(PointStamped, f'{k}/projected_centroid', 10)
            for k in self.color_ranges.keys()
        }

        self.mask_publishers = {
            k: self.create_publisher(Image, f'{k}/mask', 10)
            for k in self.color_ranges.keys()
        }

        def thresh_setter(cname, lo_or_hi_idx, hsv_idx):
            def f(x):
                self.color_ranges[cname][lo_or_hi_idx][hsv_idx] = x
            return f

        # Create one slider window for each color
        for cname, (lo, hi) in self.color_ranges.items():
            win = f"{cname}_sliders"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("H_low",  win, lo[0], 180, thresh_setter(cname, 0, 0))
            cv2.createTrackbar("S_low",  win, lo[1], 255, thresh_setter(cname, 0, 1))
            cv2.createTrackbar("V_low",  win, lo[2], 255, thresh_setter(cname, 0, 2))
            cv2.createTrackbar("H_high", win, hi[0], 180, thresh_setter(cname, 1, 0))
            cv2.createTrackbar("S_high", win, hi[1], 255, thresh_setter(cname, 1, 1))
            cv2.createTrackbar("V_high", win, hi[2], 255, thresh_setter(cname, 1, 2))

        self.K = None    # (fx, fy, cx, cy)
        self.img = None  # last BGR
        self.depth = None # last depth meters
        self.fx = self.fy = self.cx = self.cy = None

        self.masks = None
        self.mask_all = None

        # UI
        self.window="colors"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # Mouse selection state
        self.active_color = list(self.color_ranges.keys())[0] # Default active color
        self.hsv_img = None
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.timer = self.create_timer(0.05, self.render)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            self.is_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            self.drag_end = (x, y)
            self.update_hsv_from_patch()

    def update_hsv_from_patch(self):
        if self.hsv_img is None or self.drag_start is None or self.drag_end is None:
            return

        x1, y1 = self.drag_start
        x2, y2 = self.drag_end

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # If it's just a single click (or tiny box), expand it to a 5x5 patch
        if x_max - x_min < 2 or y_max - y_min < 2:
            x_min, x_max = max(0, x1 - 2), min(self.hsv_img.shape[1], x1 + 3)
            y_min, y_max = max(0, y1 - 2), min(self.hsv_img.shape[0], y1 + 3)

        patch = self.hsv_img[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return

        # Separate channels
        H_patch = patch[:, :, 0]
        S_patch = patch[:, :, 1]
        V_patch = patch[:, :, 2]

        # Calculate Saturation and Value normally
        min_s, max_s = np.min(S_patch), np.max(S_patch)
        min_v, max_v = np.min(V_patch), np.max(V_patch)

        # Handle Hue wrap-around (if values exist at both extremes)
        if np.any(H_patch < 30) and np.any(H_patch > 150):
            # True lower bound is the min of the HIGH values
            min_h = np.min(H_patch[H_patch > 90])
            # True upper bound is the max of the LOW values
            max_h = np.max(H_patch[H_patch < 90])
        else:
            min_h, max_h = np.min(H_patch), np.max(H_patch)

        # Apply buffers
        buffer_h, buffer_sv = 5, 30

        # S and V clamp between 0 and 255
        min_s = max(0, int(min_s) - buffer_sv)
        min_v = max(0, int(min_v) - buffer_sv)
        max_s = min(255, int(max_s) + buffer_sv)
        max_v = min(255, int(max_v) + buffer_sv)

        # Hue uses modulo 180 to wrap around safely (e.g., 2 - 5 = 177)
        min_h = (int(min_h) - buffer_h) % 180
        max_h = (int(max_h) + buffer_h) % 180

        cname = self.active_color
        win = f"{cname}_sliders"

        try:
            cv2.setTrackbarPos("H_low", win, min_h)
            cv2.setTrackbarPos("S_low", win, min_s)
            cv2.setTrackbarPos("V_low", win, min_v)
            cv2.setTrackbarPos("H_high", win, max_h)
            cv2.setTrackbarPos("S_high", win, max_s)
            cv2.setTrackbarPos("V_high", win, max_v)
            self.get_logger().info(f"Sampled patch! Updated '{cname}' HSV thresholds.")
        except cv2.error:
            self.get_logger().warn(f"Could not update trackbars. Is the {win} window closed?")

    def on_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def on_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if d.dtype == np.uint16:
            depth_m = d.astype(np.float32) / 1000.0
        else:
            depth_m = d.astype(np.float32)
        self.depth = depth_m

    def on_img(self, msg: Image):
        if self.K is None or self.depth is None:
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = bgr
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Save HSV for mouse sampling
        self.hsv_img = hsv.copy()

        H, W = hsv.shape[:2]
        depth_m = self.depth
        if depth_m.shape[:2] != (H, W):
            # Depth/rgb not aligned or missing; skip this frame cleanly.
            self.get_logger().warn("Depth/RGB size mismatch; skipping frame")
            return

        # Broad Z gate (tighten as needed for your table range)
        zmin, zmax = 0.05, 2.0
        depth_gate = (depth_m > zmin) & (depth_m < zmax)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        masks = {}
        centers_xyz = {}

        for name, (lo, hi) in self.color_ranges.items():
            # Initial color threshold: Handle Hue wrap-around for Red
            if lo[0] > hi[0]:
                # Split into two ranges: lo[0] to 180, and 0 to hi[0]
                lo1, hi1 = list(lo), list(hi)
                lo2, hi2 = list(lo), list(hi)

                hi1[0] = 180
                lo2[0] = 0

                m1 = cv2.inRange(hsv, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
                m2 = cv2.inRange(hsv, np.array(lo2, np.uint8), np.array(hi2, np.uint8))
                m = cv2.bitwise_or(m1, m2)
            else:
                # Normal continuous range
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

            # Clean with morphology
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

            # Depth gate
            m = (m > 0).astype(np.uint8) & depth_gate.astype(np.uint8)
            m *= 255  # back to {0,255}

            # Keep only largest island / blob
            m = _largest_component_u8(m, min_area=80)

            masks[name] = m

            # Compute 3D center of that blob & publish
            if cv2.countNonZero(m) > 0:
                # For Centroid
                M = cv2.moments(m)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    pt2d = PointStamped()
                    pt2d.header = msg.header
                    pt2d.point.x, pt2d.point.y = float(cx), float(cy)
                    self.centroid_publishers[name].publish(pt2d)

                    # For Projected Centroid
                    z = robust_depth_at_pixel(depth_m, cx, cy, k=DEPTH_K)
                    if z > 0 and np.isfinite(z):
                        X = (cx - self.cx) / self.fx * z
                        Y = (cy - self.cy) / self.fy * z
                        pt3d = PointStamped()
                        pt3d.header = msg.header
                        pt3d.point.x, pt3d.point.y, pt3d.point.z = float(X), float(Y), float(z)
                        self.projected_centroid_publishers[name].publish(pt3d)

                # For 3d center in camera frame
                center_xyz = _mean_xyz_from_mask(m, depth_m, self.K)
                if center_xyz is not None:
                    centers_xyz[name] = center_xyz
                    pt = PointStamped()
                    pt.header = msg.header
                    # Use the RGB optical frame (or whatever your image frame is)
                    if not pt.header.frame_id:
                        pt.header.frame_id = 'camera_color_optical_frame'
                    pt.point.x = float(center_xyz[0])
                    pt.point.y = float(center_xyz[1])
                    pt.point.z = float(center_xyz[2])
                    self.center_publishers[name].publish(pt)
                else:
                    self.get_logger().debug(f"No valid depth for {name} blob")

                # publish mask
                mask_msg = self.bridge.cv2_to_imgmsg(m, 'mono8')
                mask_msg.header = msg.header
                self.mask_publishers[name].publish(mask_msg)

            # else: no blob for this color

        # Stash filtered masks for visualization
        self.masks = masks
        self.mask_all = None

    # ---- UI loop & key handling ----
    def render(self):
        if not getattr(self, "masks", None) or len(self.masks) == 0:
            self.get_logger().info("masks is still none or empty")
            return

        # Determine canvas size from the first mask
        first_mask = next(iter(self.masks.values()))
        m0 = _to_mask_u8(first_mask)
        H, W = m0.shape[:2]

        base = getattr(self, "img", None)
        if base is None or base.shape[:2] != (H, W):
            base = np.zeros((H, W, 3), dtype=np.uint8)

        # Build a colored canvas of all masks
        color_canvas = np.zeros_like(base)
        for name, mask in self.masks.items():
            m = _to_mask_u8(mask)
            color = COLOR_BGR.get(name.lower(), (200, 200, 200))
            # paint mask pixels with this color
            color_canvas[m > 0] = color

        # Overlay colored masks on the base image
        alpha = 0.6
        vis = cv2.addWeighted(base, 1.0, color_canvas, alpha, 0)

        # Draw contours + labels for each mask
        for name, mask in self.masks.items():
            m = _to_mask_u8(mask)

            if cv2.countNonZero(m) == 0:
                continue
            color = COLOR_BGR.get(name.lower(), (200, 200, 200))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 2)

            # Label near the largest contour
            areas = [cv2.contourArea(c) for c in cnts]
            if areas:
                c = cnts[int(np.argmax(areas))]
                x, y, w, h = cv2.boundingRect(c)
                px = cv2.countNonZero(m)
                cv2.putText(
                    vis, f"{name} ({px} px)", (x, max(15, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                )

        # Optional legend
        y0 = 18
        for i, (name, color) in enumerate(COLOR_BGR.items()):
            cv2.rectangle(vis, (10, y0 + 22*i - 12), (30, y0 + 22*i + 8), color, -1)
            cv2.putText(vis, name, (36, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

        # Draw dragging rectangle
        if self.is_dragging and self.drag_start and self.drag_end:
            cv2.rectangle(vis, self.drag_start, self.drag_end, (255, 255, 255), 2)

        # Show active tuning color instructions
        msg_text = f"Active: {self.active_color} (Press 'c' to cycle colors)"
        cv2.putText(vis, msg_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.window, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            colors = list(self.color_ranges.keys())
            idx = colors.index(self.active_color)
            self.active_color = colors[(idx + 1) % len(colors)]
            self.get_logger().info(f"Switched active tuning color to: {self.active_color}")

def main():
    rclpy.init()
    n = ColorsTracker()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
