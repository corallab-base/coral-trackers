#!/usr/bin/env python3
import os
import sys
import threading
from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, RegionOfInterest

# SAMURAI (SAM2-based) tracker — imported after sys.path setup
_samurai_path = os.environ.get(
    'SAMURAI_PATH',
    os.path.expanduser('~/phd/software/samurai/sam2'),
)
if _samurai_path not in sys.path:
    sys.path.insert(0, _samurai_path)

import torch  # noqa: E402, I100

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402, I100

_FrameEntry = Tuple[np.ndarray, object]  # (rgb_hwc_uint8, ros_header)


class SamuraiTrackerNode(Node):
    """ROS 2 node that runs SAMURAI 2D object tracking and publishes centroids."""

    def __init__(self):
        """Declare parameters, load SAMURAI model, start background thread."""
        super().__init__('samurai_tracker')

        self.declare_parameter('rgb_topic', 'image')
        self.declare_parameter('object_name', 'obj')
        self.declare_parameter('samurai_checkpoint', '')
        self.declare_parameter('samurai_config', 'configs/samurai/sam2.1_hiera_b+.yaml')
        self.declare_parameter('publish_mask', True)

        self._rgb_topic = self.get_parameter('rgb_topic').value
        self._object_name = self.get_parameter('object_name').value
        checkpoint = self.get_parameter('samurai_checkpoint').value
        config = self.get_parameter('samurai_config').value
        self._publish_mask = bool(self.get_parameter('publish_mask').value)

        if not checkpoint:
            self.get_logger().warn(
                'samurai_checkpoint not set — SAMURAI will fail to load weights.')

        self.get_logger().info(f'Loading SAMURAI model from {checkpoint} ...')
        self.predictor = build_sam2_video_predictor(
            config, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info('SAMURAI model loaded.')

        self.bridge = CvBridge()

        # Tracking state (all guarded by _lock)
        self._state = None                          # SAM2 inference_state
        self._initial_bbox: Optional[np.ndarray] = None
        self._pending_init: bool = False
        self._latest_frame: Optional[_FrameEntry] = None

        self._lock = threading.Lock()
        self._frame_event = threading.Event()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._sub_rgb = self.create_subscription(
            Image, self._rgb_topic, self._image_callback, sensor_qos)
        self._sub_bbox = self.create_subscription(
            RegionOfInterest, f'{self._object_name}/bbox', self._bbox_callback, 10)

        self._pub_centroid = self.create_publisher(
            PointStamped, f'{self._object_name}_centroid', 10)
        self._pub_mask = (
            self.create_publisher(Image, f'{self._object_name}_mask', 10)
            if self._publish_mask else None
        )

        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True)
        self._process_thread.start()

        self.get_logger().info(
            f"SAMURAI tracker ready. Publish bbox to '/{self._object_name}/bbox' to start.")

    # ---------------------- callbacks ----------------------

    def _image_callback(self, msg: Image):
        """Store the latest RGB frame; overwrite any unprocessed frame."""
        frame = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        with self._lock:
            self._latest_frame = (frame, msg.header)
        self._frame_event.set()

    def _bbox_callback(self, msg: RegionOfInterest):
        """Accept a new bounding box and trigger (re-)initialization."""
        x1 = float(msg.x_offset)
        y1 = float(msg.y_offset)
        x2 = float(msg.x_offset + msg.width)
        y2 = float(msg.y_offset + msg.height)
        with self._lock:
            self._initial_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            self._pending_init = True
            self._state = None
        self._frame_event.set()
        self.get_logger().info(
            f'Received bbox ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}); '
            'waiting for next frame to initialize ...')

    # ---------------------- background processing ----------------------

    def _process_loop(self):
        """Background thread: track each arriving frame as soon as it is available."""
        while rclpy.ok():
            self._frame_event.wait(timeout=0.1)
            self._frame_event.clear()

            with self._lock:
                item = self._latest_frame
                self._latest_frame = None
                pending_init = self._pending_init
                bbox = self._initial_bbox.copy() if self._initial_bbox is not None else None
                state = self._state

            if item is None:
                continue
            frame_np, header = item

            if pending_init and bbox is not None:
                self.get_logger().info('Initializing SAMURAI on first frame ...')
                try:
                    state = self.predictor.init_state(
                        frames=[frame_np],
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=0, box=bbox)
                    for _, _, video_res_masks in self.predictor.propagate_in_video(state):
                        self._publish_mask_result(video_res_masks, header)
                    with self._lock:
                        self._state = state
                        self._pending_init = False
                    self.get_logger().info('SAMURAI initialized; streaming tracking started.')
                except Exception as e:
                    self.get_logger().error(f'SAMURAI init failed: {e}')

            elif state is not None:
                try:
                    video_res_masks = self.predictor.track_new_frame(state, frame_np)
                    self._publish_mask_result(video_res_masks, header)
                except Exception as e:
                    self.get_logger().error(f'SAMURAI track_new_frame failed: {e}')

    # ---------------------- publishing ----------------------

    def _publish_mask_result(self, video_res_masks, header):
        """Publish centroid and optionally mask from a (1, 1, H, W) mask tensor."""
        mask_np = video_res_masks[0, 0].cpu().numpy() > 0.0
        cx, cy = self._mask_to_centroid(mask_np)
        if cx < 0:
            return

        centroid_msg = PointStamped()
        centroid_msg.header = header
        centroid_msg.point.x = cx
        centroid_msg.point.y = cy
        centroid_msg.point.z = 0.0
        self._pub_centroid.publish(centroid_msg)

        if self._pub_mask is not None:
            mono = (mask_np.astype(np.uint8) * 255)
            mask_msg = self.bridge.cv2_to_imgmsg(mono, encoding='mono8')
            mask_msg.header = header
            self._pub_mask.publish(mask_msg)

    @staticmethod
    def _mask_to_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Return (cx, cy) pixel centroid of a boolean mask; (-1,-1) if empty."""
        pts = np.argwhere(mask)  # (N, 2) in row-col order
        if pts.size == 0:
            return -1.0, -1.0
        cy, cx = pts.mean(axis=0)
        return float(cx), float(cy)


def main():
    """Spin the SAMURAI tracker node."""
    rclpy.init()
    node = SamuraiTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
