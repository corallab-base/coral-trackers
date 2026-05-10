#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


def _mean_xyz_from_mask(mask_u8, depth_m, K, mad_threshold=2.5):
    """Mean 3D point of all mask pixels using per-pixel depth. Returns (3,) float32 or None.

    Depth outliers (mask edges with mixed foreground/background readings) are
    rejected via median absolute deviation before averaging.
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

    # Reject depth outliers using MAD
    median_z = float(np.median(zs))
    mad = float(np.median(np.abs(zs - median_z)))
    if mad > 0:
        inliers = np.abs(zs - median_z) <= mad_threshold * mad
        xs, ys, zs = xs[inliers], ys[inliers], zs[inliers]
    if zs.size == 0:
        return None

    return np.array([(xs - cx) @ zs / (fx * zs.size),
                     (ys - cy) @ zs / (fy * zs.size),
                     zs.mean()], dtype=np.float32)


class MaskCenterTrackerNode(Node):
    """Compute and publish a filtered 3D center from a segmentation mask + depth."""

    def __init__(self):
        super().__init__('mask_center_tracker')

        self.declare_parameter('object_name', 'obj')
        self.declare_parameter('depth_topic', 'depth')
        self.declare_parameter('camera_info_topic', 'camera_info')
        self.declare_parameter('alpha', 0.3)
        self.declare_parameter('dist_threshold', 0.3)

        name = self.get_parameter('object_name').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.alpha = float(self.get_parameter('alpha').value)
        self.dist_threshold = float(self.get_parameter('dist_threshold').value)

        self.bridge = CvBridge()
        self.K = None       # (fx, fy, cx, cy)
        self.depth = None   # float32 metres (H, W)
        self.filtered_xyz = None
        self.gate_count = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(CameraInfo, info_topic, self._on_info, 10)
        self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos)
        self.create_subscription(Image, f'{name}_mask', self._on_mask, 10)

        self._pub = self.create_publisher(PointStamped, f'{name}/center', 10)
        self.get_logger().info(
            f"Ready. Subscribing to '{name}_mask' and '{depth_topic}'.")

    def _on_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)

    def _on_mask(self, msg: Image):
        if self.K is None or self.depth is None:
            return
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        if mask.shape[:2] != self.depth.shape[:2]:
            self.get_logger().warn('Mask/depth size mismatch; skipping')
            return

        raw_xyz = _mean_xyz_from_mask(mask, self.depth, self.K)
        if raw_xyz is None:
            return

        if self.filtered_xyz is None:
            self.filtered_xyz = raw_xyz
        else:
            dist = float(np.linalg.norm(raw_xyz - self.filtered_xyz))
            if dist < self.dist_threshold or self.gate_count > 90:
                self.filtered_xyz = (self.alpha * raw_xyz
                                     + (1.0 - self.alpha) * self.filtered_xyz)
                self.gate_count = 0
            else:
                self.get_logger().warn(f'Outlier rejected: {dist:.2f} m jump')
                self.gate_count += 1
                return  # suppress bad estimate

        pt = PointStamped()
        pt.header = msg.header
        pt.header.frame_id = msg.header.frame_id or 'camera_color_optical_frame'
        pt.point.x, pt.point.y, pt.point.z = (float(v) for v in self.filtered_xyz)
        self._pub.publish(pt)


def main():
    rclpy.init()
    node = MaskCenterTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
