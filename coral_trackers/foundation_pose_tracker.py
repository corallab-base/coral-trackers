#!/usr/bin/env python3
"""FoundationPose tracker: register() on first mask, track_one() every frame after."""
import os
import sys
from typing import Dict, Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
import torch
import trimesh

# FoundationPose is not installed as a package; inject path before importing
_fp_ros_path = os.environ.get(
    'FOUNDATIONPOSE_PATH',
    '/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2',
)
if _fp_ros_path not in sys.path:
    sys.path.append(_fp_ros_path)

_fp_path = os.path.join(_fp_ros_path, "FoundationPose")
if _fp_path not in sys.path:
    sys.path.append(_fp_path)

from FoundationPose.estimater import (  # noqa: E402, I100
    dr,
    draw_posed_3d_box,
    draw_xyz_axis,
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
)


class _ObjectState:
    def __init__(self, name, mesh_path, apply_scale, force_apply_color, apply_color,
                 viz_enable):
        self.name = name
        self.viz_enable = viz_enable

        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(apply_scale)
        if force_apply_color and apply_color is not None:
            colors = np.tile(np.array(apply_color, dtype=np.uint8),
                             (mesh.vertices.shape[0], 1))
            mesh.visual.vertex_colors = colors
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=2,
        )

        self.initialized = False
        self.pending_mask: Optional[np.ndarray] = None  # latest mask waiting for register()


class FoundationPoseTrackerNode(Node):
    """Register once on the first mask, then call track_one() every depth frame."""

    def __init__(self):
        super().__init__('foundation_pose_tracker')

        self.declare_parameter('rgb_topic', 'image')
        self.declare_parameter('depth_topic', 'depth')
        self.declare_parameter('camera_info_topic', 'camera_info')
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('objects', ['obj'])
        self.declare_parameter('mesh_file_path', '')
        self.declare_parameter('apply_scale', 1.0)
        self.declare_parameter('force_apply_color', False)
        self.declare_parameter('apply_color', [0, 159, 237])
        self.declare_parameter('register_iter', 5)
        self.declare_parameter('track_refine_iter', 4)
        self.declare_parameter('publish_viz', True)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        object_names = list(self.get_parameter('objects').value)
        mesh_file_path = self.get_parameter('mesh_file_path').value
        apply_scale = float(self.get_parameter('apply_scale').value)
        force_apply_color = bool(self.get_parameter('force_apply_color').value)
        apply_color = list(self.get_parameter('apply_color').value)
        self.register_iter = int(self.get_parameter('register_iter').value)
        self.track_refine_iter = int(self.get_parameter('track_refine_iter').value)
        self.publish_viz = bool(self.get_parameter('publish_viz').value)

        self.objects: Dict[str, _ObjectState] = {}
        for name in object_names:
            if not mesh_file_path:
                self.get_logger().warn(f"No mesh_file_path set; skipping '{name}'.")
                continue
            self.objects[name] = _ObjectState(
                name=name,
                mesh_path=mesh_file_path,
                apply_scale=apply_scale,
                force_apply_color=force_apply_color,
                apply_color=apply_color,
                viz_enable=self.publish_viz,
            )
            self.get_logger().info(f"Object '{name}' loaded from {mesh_file_path}")

        self.K: Optional[np.ndarray] = None
        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, rgb_topic, self._on_rgb, sensor_qos)
        self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos)
        self.create_subscription(CameraInfo, info_topic, self._on_info, 10)

        self.mask_subs = {}
        for name in self.objects:
            self.mask_subs[name] = self.create_subscription(
                Image, f'{name}_mask',
                lambda msg, n=name: self._on_mask(n, msg), 10,
            )

        self.pose_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.viz_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        for name in self.objects:
            self.pose_pubs[name] = self.create_publisher(PoseStamped, f'{name}/pose', 10)
            if self.publish_viz:
                self.viz_pubs[name] = self.create_publisher(Image, f'{name}/pose_viz', 10)

        self.get_logger().info('FoundationPose tracker ready.')

    # ---------------------- callbacks ----------------------

    def _on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'Intrinsics received: fx={self.K[0,0]:.1f}')

    def _on_rgb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def _on_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.depth = d * self.depth_unit_scale
        self._process(msg.header)

    def _on_mask(self, name: str, msg: Image):
        st = self.objects.get(name)
        if st is None:
            return
        mask = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        st.pending_mask = mask > 0  # bool (H, W)

    # ---------------------- processing ----------------------

    def _process(self, header):
        if self.rgb is None or self.depth is None or self.K is None:
            return

        for name, st in self.objects.items():
            if not st.initialized:
                if st.pending_mask is None:
                    continue  # wait for first mask
                self.get_logger().info(f"Registering '{name}' ...")
                try:
                    pose = st.est.register(
                        K=self.K,
                        rgb=self.rgb,
                        depth=self.depth,
                        ob_mask=st.pending_mask,
                        iteration=self.register_iter,
                    )
                except Exception as e:
                    self.get_logger().error(f"register() failed for '{name}': {e}")
                    continue
                st.pending_mask = None
                st.initialized = True
                self.get_logger().info(f"'{name}' registered.")
            else:
                try:
                    pose = st.est.track_one(
                        rgb=self.rgb,
                        depth=self.depth,
                        K=self.K,
                        iteration=self.track_refine_iter,
                    )
                except Exception as e:
                    self.get_logger().error(f"track_one() failed for '{name}': {e}")
                    continue

            self._publish_pose(name, pose, header)

            if st.viz_enable and name in self.viz_pubs:
                center_pose = pose @ np.linalg.inv(st.to_origin)
                vis = draw_posed_3d_box(self.K, img=self.rgb.copy(),
                                        ob_in_cam=center_pose, bbox=st.bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                viz_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                viz_msg.header = header
                self.viz_pubs[name].publish(viz_msg)

        torch.cuda.empty_cache()

    def _publish_pose(self, name: str, T: np.ndarray, header):
        msg = PoseStamped()
        msg.header = header
        msg.header.frame_id = header.frame_id or 'camera_color_optical_frame'
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.pose_pubs[name].publish(msg)


def main():
    rclpy.init()
    node = FoundationPoseTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
