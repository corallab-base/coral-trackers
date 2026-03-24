#!/usr/bin/env python3
import os
import sys
import time
import threading
import copy
from importlib.resources import files

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

import torch

# --- RealSense ---
try:
    import pyrealsense2 as rs
except Exception:
    print("Please `pip install pyrealsense2` for your system Python.", file=sys.stderr)
    raise

# --- SAM 2 ---
import sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


# =================== Config (env overrides ok) ===================
SAM2_CFG   = os.environ.get("SAM2_CFG", "sam2.1_hiera_s.yaml")
SAM2_CKPT  = str(files(sam2) / "checkpoints" / "sam2.1_hiera_small.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# RealSense stream config
RS_COLOR_W = int(os.environ.get("RS_COLOR_W", "640"))
RS_COLOR_H = int(os.environ.get("RS_COLOR_H", "480"))
RS_FPS     = int(os.environ.get("RS_FPS", "30"))
ALIGN_TO_COLOR = True  # keep in case you later want depth

# SAM2/video settings
MASK_LOGIT_THR     = 0.0
FRAME_CACHE_LIMIT  = 18     # rollover & reseed after this many frames cached inside SAM2
SHOW_ANNOTATED     = True   # set False for headless
WINDOW_NAME        = "SAM2 Tracker (click to add | q=quit)"

# ================================================================


def mask_centroid_bool(m: np.ndarray):
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


class SAM2ClickTracker:
    """
    Minimal click-to-track with SAM 2 using the 'video predictor' incremental API.

    - add_point(image_rgb, x, y): seed a new object with a positive point → mask via SAM2ImagePredictor.
    - add_frame(image_rgb): append frame, propagate all masks; auto rollover+reseed when cache limit exceeded.
    """

    def __init__(self, device=DEVICE, cfg=SAM2_CFG, ckpt=SAM2_CKPT):
        self.device = device

        # Image predictor for clicks
        sam_model = build_sam2(cfg, ckpt, device=device)
        self.img_predictor = SAM2ImagePredictor(sam_model)

        # Video predictor (incremental API)
        self.video_predictor = build_sam2_video_predictor(cfg, ckpt)
        self.state = self.video_predictor.init_state()
        self.state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.state["video_height"] = None
        self.state["video_width"] = None

        self.objects_count = 0
        self.latest_masks_by_id = {}  # obj_id -> bool HxW
        self.last_frame_idx = -1

    def _ensure_size(self, image_rgb: np.ndarray):
        if self.state["video_height"] is None:
            H, W = image_rgb.shape[:2]
            self.state["video_height"] = H
            self.state["video_width"] = W

    def add_point(self, image_rgb: np.ndarray, x: int, y: int):
        """Add a positive point → create/seed a new object on this frame."""
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

        # Add this frame and seed mask
        frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
        self.video_predictor.reset_state(self.state)
        _frame_idx, _, _ = self.video_predictor.add_new_mask(self.state, frame_idx, oid, m_bool)
        self.last_frame_idx = frame_idx

        print(f"[sam2_tracker] Added object id={oid} at ({x},{y})")

    def _rollover_reseed(self, image_rgb: np.ndarray) -> int:
        """Bound memory: rebuild state, add current frame once, reseed all latest masks."""
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
        """Append new frame and propagate masks. Returns (frame_idx, latest_masks_by_id)."""
        if len(self.latest_masks_by_id) == 0:
            return None, self.latest_masks_by_id

        # Bound memory
        if self.state["images"].shape[0] > FRAME_CACHE_LIMIT:
            frame_idx = self._rollover_reseed(image_rgb)
        else:
            frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)

        # Propagate
        frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.state, frame_idx=frame_idx
        )

        # Update latest per-object masks
        for i, oid in enumerate(obj_ids):
            m_bool = (video_res_masks[i] > MASK_LOGIT_THR)[0].detach().cpu().numpy().astype(bool)
            self.latest_masks_by_id[int(oid)] = m_bool

        self.last_frame_idx = frame_idx
        return frame_idx, self.latest_masks_by_id


class SAM2RealSenseNode(Node):
    """
    ROS 2 node that *internally* captures RealSense frames via pyrealsense2,
    runs SAM 2 click-to-track, and publishes outputs:
      - ~/centroids_px : sensor_msgs/PointCloud  (pixel centroids; ids in 'id' channel)
      - ~/annotated    : sensor_msgs/Image       (optional)
    """

    def __init__(self):
        super().__init__("sam2_rs_click_tracker_node")
        self.declare_parameter("publish_annotated", True)
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value

        self.centroids_pub = self.create_publisher(PointCloud, "~/centroids_px", 10)
        self.annotated_pub = self.create_publisher(Image, "~/annotated", 10) if self.publish_annotated else None

        # Tracker
        self.tracker = SAM2ClickTracker()

        # OpenCV window & click handling
        self.pending_clicks = []
        self._click_lock = threading.Lock()
        if SHOW_ANNOTATED:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        # RealSense pipeline
        self.running = True
        self.rs_thread = threading.Thread(target=self._rs_loop, daemon=True)
        self.rs_thread.start()

        # Tiny timer to service OpenCV events / 'q' key
        self.ui_timer = self.create_timer(0.001, self._ui_spin)

        self.get_logger().info("Started: internal RealSense capture + SAM2 tracking")
        self.get_logger().info("Publishing centroids on: ~/centroids_px (sensor_msgs/PointCloud)")

    # ------------------------ UI ------------------------

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._click_lock:
                self.pending_clicks.append((x, y))

    def _ui_spin(self):
        if not SHOW_ANNOTATED:
            return
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quit signal (q) received; shutting down.")
            rclpy.shutdown()

    # --------------------- RealSense loop ---------------------

    def _rs_loop(self):
        # Setup RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, RS_COLOR_W, RS_COLOR_H, rs.format.bgr8, RS_FPS)
        if ALIGN_TO_COLOR:
            config.enable_stream(rs.stream.depth, RS_COLOR_W, RS_COLOR_H, rs.format.z16, RS_FPS)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color) if ALIGN_TO_COLOR else None

        try:
            while rclpy.ok() and self.running:
                frames = pipeline.wait_for_frames()
                if ALIGN_TO_COLOR:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_bgr = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # clicks → seed new objects on THIS frame
                with self._click_lock:
                    if self.pending_clicks:
                        for (x, y) in self.pending_clicks:
                            self.tracker.add_point(frame_rgb, x, y)
                        self.pending_clicks.clear()

                # tracker step
                frame_idx, masks_by_id = self.tracker.add_frame(frame_rgb)

                # Build & publish centroids (pixel coords)
                cloud = PointCloud()
                # No camera header here; you can set a frame_id if you want:
                # cloud.header.frame_id = "camera_color_optical_frame"
                ids_channel = ChannelFloat32()
                ids_channel.name = "id"

                if masks_by_id:
                    for oid, m in masks_by_id.items():
                        c = mask_centroid_bool(m)
                        if c is None:
                            continue
                        cloud.points.append(Point32(x=float(c[0]), y=float(c[1]), z=0.0))
                        ids_channel.values.append(float(oid))
                cloud.channels.append(ids_channel)
                self.centroids_pub.publish(cloud)

                # Annotated preview + optional ROS image
                if SHOW_ANNOTATED or self.publish_annotated:
                    vis = frame_bgr.copy()
                    for oid, m in masks_by_id.items():
                        m_u8 = (m.astype(np.uint8) * 255)
                        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
                        c = mask_centroid_bool(m)
                        if c is not None:
                            cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
                            cv2.putText(vis, f"id{oid}", (int(c[0]) + 6, int(c[1]) - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    if SHOW_ANNOTATED:
                        cv2.imshow(WINDOW_NAME, vis)
                    if self.publish_annotated and self.annotated_pub is not None:
                        self.annotated_pub.publish(numpy_to_imgmsg(vis, msg.header.frame_id, encoding="bgr8"))

        except Exception as e:
            self.get_logger().error(f"RealSense loop error: {e}")
        finally:
            pipeline.stop()

    # --------------------- Shutdown ---------------------

    def destroy_node(self):
        self.running = False
        try:
            if self.rs_thread.is_alive():
                self.rs_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if SHOW_ANNOTATED:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SAM2RealSenseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
