#!/usr/bin/env python3
import os
import sys
from typing import Dict, Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
import torch
import trimesh

from .utils.kalman_filter_6d import KalmanFilter6D

# FoundationPose is not installed as a package; inject path before importing
_fp_ros_path = os.environ.get(
    'FOUNDATIONPOSE_PATH',
    '/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2',
)
if _fp_ros_path not in sys.path:
    sys.path.append(_fp_ros_path)

_fp_path = os.environ.get(
    'FOUNDATIONPOSE_PATH',
    '/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2/FoundationPose',
)
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


# ----------------------------- helpers -----------------------------
def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
    """Convert 6D pose array [tx,ty,tz,rx,ry,rz] to 4x4 SE(3) matrix."""
    xyz = pose_arr[:3]
    euler = pose_arr[3:]
    R = Rotation.from_euler('xyz', euler, degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def get_6d_pose_arr_from_mat(pose) -> np.ndarray:
    """Convert 4x4 SE(3) matrix (numpy or torch) to 6D pose array."""
    if torch.is_tensor(pose):
        T = pose[0].detach().cpu().numpy() if pose.ndim == 3 else pose.detach().cpu().numpy()
    else:
        T = pose
    xyz = T[:3, 3]
    euler = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler]




# ---------------------- per-object tracker state --------------------
class ObjectState:
    """Runtime state for a single tracked object."""

    def __init__(self, name: str, mesh_path: str, apply_scale: float,
                 force_apply_color: bool, apply_color: Optional[list],
                 kf_enable: bool, kf_noise_scale: float,
                 viz_enable: bool):
        """Initialize object state."""
        self.name = name
        self.mesh_path = mesh_path
        self.apply_scale = apply_scale
        self.force_apply_color = force_apply_color
        self.apply_color = (
            np.array(apply_color, dtype=np.uint8) if apply_color is not None else None
        )
        self.kf_enable = kf_enable
        self.kf_noise_scale = kf_noise_scale
        self.viz_enable = viz_enable

        self.est: Optional[FoundationPose] = None
        self.kf: Optional[KalmanFilter6D] = KalmanFilter6D(kf_noise_scale) if kf_enable else None
        self.kf_mean = None
        self.kf_cov = None

        self.to_origin = None
        self.bbox = None

        self.center_xyz: Optional[np.ndarray] = None  # filtered 3D center from mask_center_tracker
        self.initialized = False

    def load_mesh(self):
        """Load mesh from disk, apply scale and optional color override."""
        mesh = trimesh.load(self.mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(self.apply_scale)
        if self.force_apply_color and self.apply_color is not None:
            colors = np.tile(self.apply_color, (mesh.vertices.shape[0], 1))
            mesh.visual.vertex_colors = colors
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.to_origin = to_origin
        self.bbox = bbox
        return mesh

    def build_estimator(self, mesh):
        """Construct the FoundationPose estimator for this object."""
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


# ---------------------------- main node -----------------------------
class FoundationPosePlusPlusNode(Node):
    """ROS 2 node implementing FoundationPose++ 6DOF tracking."""

    def __init__(self):
        """Declare parameters, build object states, and create topics."""
        super().__init__('foundationpose_plus_plus_tracker')

        self.declare_parameter('rgb_topic', 'image')
        self.declare_parameter('depth_topic', 'depth')
        self.declare_parameter('camera_info_topic', 'camera_info')
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('objects', ['obj'])
        self.declare_parameter('mesh_file_path', '')
        self.declare_parameter('apply_scale', 1.0)
        self.declare_parameter('force_apply_color', False)
        self.declare_parameter('apply_color', [0, 159, 237])
        self.declare_parameter('use_kalman_filter', False)
        self.declare_parameter('kf_measurement_noise_scale', 0.05)
        self.declare_parameter('track_refine_iter', 4)
        self.declare_parameter('publish_viz', True)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self.object_names = list(self.get_parameter('objects').value)
        mesh_file_path = self.get_parameter('mesh_file_path').value

        self.apply_scale = float(self.get_parameter('apply_scale').value)
        self.force_apply_color = bool(self.get_parameter('force_apply_color').value)
        self.apply_color = list(self.get_parameter('apply_color').value)
        self.use_kf = bool(self.get_parameter('use_kalman_filter').value)
        self.kf_noise = float(self.get_parameter('kf_measurement_noise_scale').value)
        self.track_refine_iter = int(self.get_parameter('track_refine_iter').value)
        self.publish_viz = bool(self.get_parameter('publish_viz').value)

        self.objects: Dict[str, ObjectState] = {}
        for name in self.object_names:
            if not mesh_file_path:
                self.get_logger().warn(f"No mesh_file_path set; skipping object '{name}'.")
                continue
            st = ObjectState(
                name=name,
                mesh_path=mesh_file_path,
                apply_scale=self.apply_scale,
                force_apply_color=self.force_apply_color,
                apply_color=self.apply_color,
                kf_enable=self.use_kf,
                kf_noise_scale=self.kf_noise,
                viz_enable=self.publish_viz,
            )
            mesh = st.load_mesh()
            st.build_estimator(mesh)
            self.objects[name] = st
            self.get_logger().info(f"Object '{name}' loaded from {mesh_file_path}")

        self.K_np: Optional[np.ndarray] = None

        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_rgb = self.create_subscription(
            Image, self.rgb_topic, self.image_callback, sensor_qos)
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos)
        self.sub_caminfo = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.center_subs = {}
        for name in self.objects.keys():
            self.center_subs[name] = self.create_subscription(
                PointStamped, f'{name}/center',
                lambda msg, n=name: self.on_center(n, msg), 10,
            )

        self.pose_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.viz_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        for name in self.objects.keys():
            self.pose_publishers[name] = self.create_publisher(
                PoseStamped, f'{name}/pose', 10)
            if self.publish_viz:
                self.viz_publishers[name] = self.create_publisher(
                    Image, f'{name}/pose_viz', 10)

        self.rgb = None
        self.depth = None

        self.get_logger().info('FoundationPose++ tracker node ready.')

    # ---------------------- callbacks ----------------------
    def camera_info_callback(self, msg):
        """Store camera intrinsics on first message."""
        if self.K_np is None:
            self.K_np = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(
                f'Camera intrinsics received: fx={self.K_np[0, 0]:.1f}')

    def image_callback(self, msg):
        """Buffer the latest RGB frame."""
        self.rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def depth_callback(self, msg):
        """Buffer the latest depth frame and trigger processing."""
        self.depth = self.bridge.imgmsg_to_cv2(msg, '32FC1') * self.depth_unit_scale
        self.try_process_frame(msg.header)

    def on_center(self, name: str, msg: PointStamped):
        """Store the filtered 3D center from mask_center_tracker."""
        st = self.objects.get(name)
        if st is None:
            return
        st.center_xyz = np.array(
            [msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)

    # ---------------------- processing ----------------------
    def try_process_frame(self, header):
        """Run one tracking step for all initialized objects."""
        if self.rgb is None or self.depth is None or self.K_np is None:
            return

        color = self.rgb
        depth = self.depth
        K = self.K_np

        for name, st in self.objects.items():
            # 1) Init: set initial pose directly from the filtered 3D mask center.
            if not st.initialized:
                if st.center_xyz is None:
                    continue  # wait for first center from mask_center_tracker
                tx, ty, tz = st.center_xyz
                pose_init = np.eye(4, dtype=np.float32)
                pose_init[:3, 3] = [tx, ty, tz]
                st.est.pose_last = torch.from_numpy(pose_init).unsqueeze(0)
                if st.kf_enable:
                    st.kf_mean, st.kf_cov = st.kf.initiate(
                        get_6d_pose_arr_from_mat(pose_init))
                st.initialized = True
                self.get_logger().info(
                    f"'{name}' initialized: XYZ=({tx:.3f}, {ty:.3f}, {tz:.3f})")

            # 2) Anchor translation to filtered mask center before refining.
            if st.center_xyz is not None and st.est.pose_last is not None:
                if st.kf_enable and st.kf_mean is not None:
                    st.kf_mean, st.kf_cov = st.kf.update(
                        st.kf_mean, st.kf_cov,
                        get_6d_pose_arr_from_mat(st.est.pose_last))
                    st.kf_mean[:3] = st.center_xyz  # anchor translation to mask center
                    st.est.pose_last = (
                        torch.from_numpy(get_mat_from_6d_pose_arr(st.kf_mean[:6]))
                        .unsqueeze(0).to(st.est.pose_last.device)
                    )
                else:
                    p_np = st.est.pose_last[0].detach().cpu().numpy().copy()
                    p_np[:3, 3] = st.center_xyz
                    st.est.pose_last = (
                        torch.from_numpy(p_np).unsqueeze(0).to(st.est.pose_last.device)
                    )

            pose = st.est.track_one(
                rgb=color, depth=depth, K=K, iteration=self.track_refine_iter)

            if st.kf_enable and st.kf_mean is not None:
                st.kf_mean, st.kf_cov = st.kf.predict(st.kf_mean, st.kf_cov)

            self.publish_pose(name, pose, header)

            if st.viz_enable:
                center_pose = pose @ np.linalg.inv(st.to_origin)
                vis = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=center_pose,
                                        bbox=st.bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                viz_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                viz_msg.header = header
                self.viz_publishers[name].publish(viz_msg)

        torch.cuda.empty_cache()

    def publish_pose(self, name: str, T: np.ndarray, header):
        """Publish a 4x4 SE(3) matrix as PoseStamped."""
        msg = PoseStamped()
        msg.header = header
        msg.header.frame_id = header.frame_id or 'camera_color_optical_frame'
        R = T[:3, :3]
        q = Rotation.from_matrix(R).as_quat()  # x,y,z,w
        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.pose_publishers[name].publish(msg)


def main():
    """Spin the tracker node."""
    rclpy.init()
    node = FoundationPosePlusPlusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
