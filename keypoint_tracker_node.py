#!/usr/bin/env python3
import os
import sys
import time
import copy
from importlib.resources import files
from collections import deque
import dataclasses

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

import torch
import sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


# --- KLT/ORB params ---
KLT_WIN = (21, 21)
KLT_MAX_LEVEL = 3
KLT_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
ORB_N_FEATURES = 300
MASK_DILATE_PX = 5


# ---------- Image helpers (cv_bridge-free) ----------
def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        ch, dtype = 3, np.uint8
    elif enc in ("mono8",):
        ch, dtype = 1, np.uint8
    elif enc in ("16uc1", "mono16"):
        ch, dtype = 1, np.uint16
    elif enc in ("32fc1",):
        ch, dtype = 1, np.float32
    else:
        raise NotImplementedError(f"Unsupported encoding: {msg.encoding}")

    row_stride = msg.step // np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype=dtype)
    if ch == 1:
        img = arr.reshape((msg.height, row_stride))[:, :msg.width]
    else:
        img = arr.reshape((msg.height, row_stride // ch, ch))[:, :msg.width, :]
    return img

def numpy_to_imgmsg(img: np.ndarray, frame_id: str, encoding="bgr8") -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    step_channels = (img.shape[2] if img.ndim == 3 else 1)
    msg.step = msg.width * step_channels * img.dtype.itemsize
    msg.data = memoryview(img.tobytes())
    return msg

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

@dataclasses.dataclass
class TrackedPoint:
    oid: int
    pt: np.ndarray        # shape (2,) float32 [u, v]
    desc: np.ndarray      # ORB 1x32 (uint8), or None until computed
    valid: bool = True
    lost_count: int = 0

# =================== Config (env overrides ok) ===================
SAM2_CFG   = os.environ.get("SAM2_CFG", "sam2.1_hiera_l.yaml")
SAM2_CKPT  = str(files(sam2) / "checkpoints" / "sam2.1_hiera_large.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_CACHE_LIMIT  = 18     # rollover & reseed threshold
MASK_LOGIT_THR     = 0.0
SHOW_ANNOTATED     = True
WINDOW_NAME        = "SAM2 Click Tracker (click to add object | q=quit)"

# Depth sampling window (fallback if mask depth invalid)
DEPTH_K = 5  # odd number (pixels)
# ================================================================


def mask_centroid_bool(m: np.ndarray):
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


class SAM2ClickTracker:
    """
    Minimal click-to-track with SAM 2 using the 'video predictor' incremental API,
    plus per-object precise point tracking (KLT + ORB re-lock inside SAM mask).
    """

    def __init__(self, device=DEVICE, cfg=SAM2_CFG, ckpt=SAM2_CKPT):
        self.device = device
        sam_model = build_sam2(cfg, ckpt, device=device)
        self.img_predictor = SAM2ImagePredictor(sam_model)

        self.video_predictor = build_sam2_video_predictor(cfg, ckpt)
        self.state = self.video_predictor.init_state()
        self.state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.state["video_height"] = None
        self.state["video_width"] = None

        self.objects_count = 0
        self.latest_masks_by_id = {}   # oid -> bool HxW
        self.tracked_points = {}       # oid -> TrackedPoint
        self.prev_gray = None
        self.last_frame_idx = -1

        # ORB for re-acquisition
        self.orb = cv2.ORB_create(ORB_N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _ensure_size(self, image_rgb: np.ndarray):
        if self.state["video_height"] is None:
            H, W = image_rgb.shape[:2]
            self.state["video_height"] = H
            self.state["video_width"] = W

    def _compute_orb_desc_at(self, gray: np.ndarray, u: float, v: float):
        """Compute an ORB descriptor at (u, v) by placing a keypoint there."""
        kp = cv2.KeyPoint(x=float(u), y=float(v), size=16)
        kps, desc = self.orb.compute(gray, [kp])
        if desc is None:
            return None
        return desc[0:1, :]  # shape (1, 32)

    def _reacquire_inside_mask(self, gray: np.ndarray, mask_bool: np.ndarray, ref_desc: np.ndarray):
        """
        Detect ORB in a slightly dilated mask; return best keypoint (u,v) by Hamming distance to ref_desc.
        """
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        if MASK_DILATE_PX > 0:
            kernel = np.ones((MASK_DILATE_PX, MASK_DILATE_PX), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

        kps = self.orb.detect(gray, mask=mask_u8)
        if not kps:
            return None

        kps, descs = self.orb.compute(gray, kps)
        if descs is None or len(kps) == 0:
            return None

        matches = self.bf.match(ref_desc, descs)  # ref_desc is shape (1,32)
        if len(matches) == 0:
            return None
        best = min(matches, key=lambda m: m.distance)
        u, v = kps[best.trainIdx].pt
        return np.array([u, v], dtype=np.float32)

    def add_point(self, image_rgb: np.ndarray, x: int, y: int):
        """Add a positive point → seed a new object on this frame, and remember the exact clicked point."""
        self._ensure_size(image_rgb)
        self.img_predictor.set_image(image_rgb)

        pt = np.array([[x, y]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        masks, scores, logits = self.img_predictor.predict(
            point_coords=pt[None, ...],
            point_labels=lbl[None, ...],
            box=None,
            multimask_output=False,
        )
        m = (masks[0] > 0).astype(np.uint8)
        if m.sum() == 0:
            print("[sam2_tracker] Click produced empty mask; ignoring.")
            return

        self.objects_count += 1
        oid = self.objects_count
        m_bool = m.astype(bool)
        self.latest_masks_by_id[oid] = m_bool

        # Seed SAM2 state
        frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
        self.video_predictor.reset_state(self.state)
        _frame_idx, _, _ = self.video_predictor.add_new_mask(self.state, frame_idx, oid, m_bool)
        self.last_frame_idx = frame_idx

        # Initialize the precise tracked point & descriptor
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        desc = self._compute_orb_desc_at(gray, x, y)
        self.tracked_points[oid] = TrackedPoint(oid=oid, pt=np.array([float(x), float(y)], dtype=np.float32), desc=desc)

        print(f"[sam2_tracker] Added object id={oid} at ({x},{y})")

    def _rollover_reseed(self, image_rgb: np.ndarray) -> int:
        seeds = {int(oid): m for oid, m in self.latest_masks_by_id.items() if m is not None}

        self.state = self.video_predictor.init_state()
        self.state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        H, W = image_rgb.shape[:2]
        self.state["video_height"] = H
        self.state["video_width"] = W

        frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
        self.video_predictor.reset_state(self.state)
        for oid, m in seeds.items():
            _f, _, _ = self.video_predictor.add_new_mask(self.state, frame_idx, oid, m.astype(bool))
        self.last_frame_idx = frame_idx
        return frame_idx

    def add_frame(self, image_rgb: np.ndarray):
        """
        Append new frame, propagate masks, then update tracked points.
        Returns (frame_idx, masks_by_id, points_by_id)
        """
        if len(self.latest_masks_by_id) == 0:
            self.prev_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            return None, self.latest_masks_by_id, {}

        # bound memory
        if self.state["images"].shape[0] > FRAME_CACHE_LIMIT:
            frame_idx = self._rollover_reseed(image_rgb)
        else:
            frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)

        # SAM2 propagate masks first
        frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.state, frame_idx=frame_idx
        )
        for i, oid in enumerate(obj_ids):
            m_bool = (video_res_masks[i] > MASK_LOGIT_THR)[0].detach().cpu().numpy().astype(bool)
            self.latest_masks_by_id[int(oid)] = m_bool
        self.last_frame_idx = frame_idx

        # KLT flow for all tracked points
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        points_by_id = {}

        # Prepare arrays for vectorized KLT
        oids = []
        prev_pts = []
        for oid, tp in self.tracked_points.items():
            if tp.valid:
                oids.append(oid)
                prev_pts.append(tp.pt)
        if self.prev_gray is not None and prev_pts:
            prev_pts_np = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts_np, None,
                winSize=KLT_WIN, maxLevel=KLT_MAX_LEVEL, criteria=KLT_CRITERIA
            )
            for i, oid in enumerate(oids):
                tp = self.tracked_points[oid]
                if st[i][0] == 1 and np.isfinite(next_pts[i]).all():
                    new_pt = next_pts[i].reshape(2)
                    # If mask exists and point is out, try re-lock with ORB inside mask
                    m = self.latest_masks_by_id.get(oid, None)
                    if m is not None:
                        u, v = int(round(new_pt[0])), int(round(new_pt[1]))
                        in_bounds = (0 <= u < m.shape[1]) and (0 <= v < m.shape[0])
                        if not in_bounds or (in_bounds and not m[v, u]):
                            # Try re-acquisition using ORB descriptor; fall back to KLT result if none
                            ref_desc = tp.desc if tp.desc is not None else self._compute_orb_desc_at(self.prev_gray, *tp.pt)
                            snapped = self._reacquire_inside_mask(gray, m, ref_desc) if ref_desc is not None else None
                            if snapped is not None:
                                new_pt = snapped
                    tp.pt = new_pt.astype(np.float32)
                    tp.valid = True
                    tp.lost_count = 0
                else:
                    # KLT failed: try re-acquire with mask + ORB
                    m = self.latest_masks_by_id.get(oid, None)
                    ref_desc = tp.desc if tp.desc is not None else (self._compute_orb_desc_at(self.prev_gray, *tp.pt) if self.prev_gray is not None else None)
                    snapped = self._reacquire_inside_mask(gray, m, ref_desc) if (m is not None and ref_desc is not None) else None
                    if snapped is not None:
                        tp.pt = snapped.astype(np.float32)
                        tp.valid = True
                        tp.lost_count = 0
                    else:
                        tp.lost_count += 1
                        # Optional: after N fails, mark invalid
                        if tp.lost_count > 10:
                            tp.valid = False

                # Update descriptor when valid
                if tp.valid:
                    desc_now = self._compute_orb_desc_at(gray, *tp.pt)
                    if desc_now is not None:
                        tp.desc = desc_now

        # Output points_by_id for convenience
        for oid, tp in self.tracked_points.items():
            if tp.valid:
                points_by_id[oid] = tp.pt.copy()

        self.prev_gray = gray

        return frame_idx, self.latest_masks_by_id, points_by_id


class SAM2ClickTrackerNode(Node):
    """
    ROS 2 node:
      - Subscribes to color images (e.g., /camera/.../color/image_raw), aligned depth, and camera info
      - Click-to-add points (OpenCV window)
      - Publishes:
          • 2D pixel centroids on ~/centroids_px (sensor_msgs/PointCloud; z=0)
          • 3D centroids on ~/centroids_3d (sensor_msgs/PointCloud; meters)
          • (optional) annotated image on ~/annotated
    """

    def __init__(self):
        super().__init__("sam2_click_tracker_node")
        # Parameters
        self.declare_parameter("frozen_points", False)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("depth_unit_scale", 0.001)  # 16UC1 millimeters -> meters

        frozen_points = self.get_parameter("frozen_points").get_parameter_value().bool_value
        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic  = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").get_parameter_value().double_value)

        # Subscribers
        self.sub_color = self.create_subscription(Image, color_topic, self.color_cb, 10)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.sub_info  = self.create_subscription(CameraInfo, info_topic, self.info_cb, 10)

        # Publishers
        self.centroids_px_pub  = self.create_publisher(PointCloud, "~/centroids_px", 10)
        self.centroids_3d_pub  = self.create_publisher(PointCloud, "~/centroids_3d", 10)
        self.annotated_pub     = self.create_publisher(Image, "~/annotated", 10) if self.publish_annotated else None

        self.frozen_points = frozen_points
        self.cached_frozen_result = None
        self.tracker = SAM2ClickTracker()

        # OpenCV window + click handling
        self.pending_clicks = []
        if SHOW_ANNOTATED:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            def on_mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.pending_clicks.append((x, y))
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        self.ui_timer = self.create_timer(0.001, self._ui_spin)

        # Depth/intrinsics cache
        self.last_depth_m = None        # float32 meters
        self.last_depth_shape = None
        self.fx = self.fy = self.cx = self.cy = None

        self.get_logger().info(f"Subscribed to: color={color_topic}, depth={depth_topic}, info={info_topic}")
        self.get_logger().info("Publishing 2D centroids on: ~/centroids_px and 3D on: ~/centroids_3d")

    # --------- Callbacks ----------

    def info_cb(self, msg: CameraInfo):
        # Intrinsics from K:
        # [ fx, 0, cx,
        #   0, fy, cy,
        #   0,  0,  1 ]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def depth_cb(self, msg: Image):
        depth = imgmsg_to_numpy(msg)
        enc = msg.encoding.lower()
        if enc == "16uc1" or enc == "mono16":
            self.last_depth_m = depth.astype(np.float32) * self.depth_unit_scale
        elif enc == "32fc1":
            # already meters (typical for some pipelines)
            # Some drivers may publish NaNs for invalid pixels; keep as-is
            self.last_depth_m = depth.astype(np.float32)
        else:
            # Unexpected encoding; try best-effort cast
            self.last_depth_m = depth.astype(np.float32)
        self.last_depth_shape = self.last_depth_m.shape

    def _ui_spin(self):
        if SHOW_ANNOTATED:
            key = cv2.waitKey(1) & 0xFF   # only call once
            if key == ord('q'):
                self.get_logger().info("Quit signal (q) received; shutting down.")
                rclpy.shutdown()
            elif key == ord('r'):
                self.get_logger().info("Got r")
                self.tracker.objects_count = 0
                self.tracker.latest_masks_by_id = {}
                self.tracker.tracked_points = {}
                self.cached_frozen_result = None

    def color_cb(self, msg: Image):
        # Convert color → RGB/BGR depending on encoding
        arr = imgmsg_to_numpy(msg)
        enc = msg.encoding.lower()
        if enc == "bgr8":
            frame_bgr = arr
            frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif enc == "rgb8":
            frame_rgb = arr
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            self.get_logger().warn(f"Unexpected color encoding '{msg.encoding}', trying to treat as RGB.")
            frame_rgb = arr
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        added_points = False

        # Handle clicks on THIS frame
        if self.pending_clicks:
            added_points = True
            for (x, y) in self.pending_clicks:
                self.tracker.add_point(frame_rgb, x, y)
            self.pending_clicks.clear()

        # Run tracker

        if self.frozen_points and not added_points and self.cached_frozen_result is not None:
            frame_idx, masks_by_id, points_by_id = self.cached_frozen_result
        else:
            frame_idx, masks_by_id, points_by_id = self.tracker.add_frame(frame_rgb)
            self.cached_frozen_result = frame_idx, masks_by_id, points_by_id

        # 2D pixel points (was: centroids)
        cloud_px = PointCloud()
        cloud_px.header = msg.header
        ids_channel_px = ChannelFloat32()
        ids_channel_px.name = "id"

        # 3D points (was: centroids)
        cloud_3d = PointCloud()
        cloud_3d.header = msg.header
        ids_channel_3d = ChannelFloat32()
        ids_channel_3d.name = "id"

        have_depth = (self.last_depth_m is not None) and (self.fx is not None)
        H, W = frame_rgb.shape[:2]
        depth_ok = have_depth and (self.last_depth_shape == (H, W))

        # Publish the precise tracked points
        if points_by_id:
            for oid, pt in points_by_id.items():
                u, v = float(pt[0]), float(pt[1])

                # 2D
                cloud_px.points.append(Point32(x=u, y=v, z=0.0))
                ids_channel_px.values.append(float(oid))

                # 3D
                if depth_ok:
                    z = robust_depth_at_pixel(self.last_depth_m, u, v, k=DEPTH_K)
                    if z > 0 and np.isfinite(z):
                        X = (u - self.cx) / self.fx * z
                        Y = (v - self.cy) / self.fy * z
                        cloud_3d.points.append(Point32(x=float(X), y=float(Y), z=float(z)))
                        ids_channel_3d.values.append(float(oid))

        cloud_px.channels.append(ids_channel_px)
        self.centroids_px_pub.publish(cloud_px)

        if len(ids_channel_3d.values) > 0:
            cloud_3d.channels.append(ids_channel_3d)
            self.centroids_3d_pub.publish(cloud_3d)

        if SHOW_ANNOTATED or self.publish_annotated:
            vis = frame_bgr.copy()
            # Draw masks
            for oid, m in masks_by_id.items():
                m_u8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

            # Draw precise points
            for oid, pt in points_by_id.items():
                u, v = int(round(pt[0])), int(round(pt[1]))
                cv2.drawMarker(vis, (u, v), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
                label = f"id{oid}"
                if have_depth and self.last_depth_shape == vis.shape[:2]:
                    z = robust_depth_at_pixel(self.last_depth_m, u, v, k=DEPTH_K)
                    if z > 0 and np.isfinite(z) and (self.fx is not None):
                        label += f"  Z={z:.3f}m"
                cv2.putText(vis, label, (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            if SHOW_ANNOTATED:
                cv2.imshow(WINDOW_NAME, vis)
            if self.publish_annotated and self.annotated_pub is not None:
                self.annotated_pub.publish(numpy_to_imgmsg(vis, msg.header.frame_id, encoding="bgr8"))

    # --------- Helpers ----------

    def _robust_depth_at_mask(self, depth_m: np.ndarray, mask_bool: np.ndarray, u: float, v: float) -> float:
        """
        Robust depth estimate:
          1) median depth over the mask (valid pixels only)
          2) fallback: median over a kxk window around (u,v)
        Returns depth in meters (float), or 0.0 if not available.
        """
        # 1) median over mask
        valid = (depth_m > 0.0) & mask_bool
        vals = depth_m[valid]
        if vals.size > 0:
            return float(np.median(vals))

        # 2) fallback window
        h, w = depth_m.shape[:2]
        u_i = int(round(u)); v_i = int(round(v))
        k = DEPTH_K
        x0 = max(0, u_i - k // 2); x1 = min(w, u_i + k // 2 + 1)
        y0 = max(0, v_i - k // 2); y1 = min(h, v_i + k // 2 + 1)
        patch = depth_m[y0:y1, x0:x1]
        vals = patch[patch > 0.0]
        if vals.size == 0:
            return 0.0
        return float(np.median(vals))


def main():
    rclpy.init()
    node = SAM2ClickTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
