#!/usr/bin/env python3
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, RegionOfInterest

_WINDOW = 'BBox Selector'
_HINT = "Press 'S' to select bbox, 'Q' to quit"


class BboxSelectorNode(Node):
    """ROS 2 node: display live image, publish a user-drawn bounding box and its binary mask."""

    def __init__(self):
        super().__init__('bbox_selector')

        self.declare_parameter('image_topic', 'image')
        self.declare_parameter('obj_name', 'obj')

        image_topic = self.get_parameter('image_topic').value
        obj_name = self.get_parameter('obj_name').value
        self._bbox_topic = f'{obj_name}/bbox'
        self._mask_topic = f'{obj_name}/mask'

        self._bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self.last_bbox = None  # (x, y, w, h) — drawn on live feed after selection

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, image_topic, self._image_callback, sensor_qos)
        self._pub_bbox = self.create_publisher(RegionOfInterest, self._bbox_topic, 10)
        self._pub_mask = self.create_publisher(Image, self._mask_topic, 10)

        self.get_logger().info(
            f"Subscribed to '{image_topic}'. "
            f"Publishing bbox on '{self._bbox_topic}', mask on '{self._mask_topic}'.")

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self._frame_lock:
            self._latest_frame = frame

    def latest_frame(self):
        """Return a copy of the most recently received frame, or None."""
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def publish_bbox(self, x: int, y: int, w: int, h: int):
        """Publish a RegionOfInterest, a binary mask image, and cache bbox for overlay."""
        msg = RegionOfInterest()
        msg.x_offset = x
        msg.y_offset = y
        msg.width = w
        msg.height = h
        self._pub_bbox.publish(msg)
        self.last_bbox = (x, y, w, h)
        self.get_logger().info(f'Bbox published: ({x}, {y}, {w}x{h})')

        frame = self.latest_frame()
        if frame is not None:
            h_img, w_img = frame.shape[:2]
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            mask[y:y + h, x:x + w] = 255
            mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
            self._pub_mask.publish(mask_msg)
            self.get_logger().info(f'Mask published to {self._mask_topic}')


def main():
    """Spin BboxSelectorNode and run the OpenCV GUI on the main thread."""
    rclpy.init()
    node = BboxSelectorNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            frame = node.latest_frame()
            if frame is None:
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                continue

            display = frame.copy()
            cv2.putText(display, _HINT, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if node.last_bbox is not None:
                x, y, w, h = node.last_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(_WINDOW, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                roi = cv2.selectROI(_WINDOW, frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = (int(v) for v in roi)
                if w > 0 and h > 0:
                    node.publish_bbox(x, y, w, h)
                else:
                    node.get_logger().warn('Empty ROI — bbox not published.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
