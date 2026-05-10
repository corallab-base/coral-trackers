# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

This is a ROS2 `ament_python` package. The workspace root is `~/phd/software/ros_workspaces/test_ws`.

```bash
# Build (from workspace root)
cd ~/phd/software/ros_workspaces/test_ws
colcon build --packages-select coral_trackers
source install/setup.bash

# Run a node
ros2 run coral_trackers colors_tracker
ros2 run coral_trackers samurai_tracker
ros2 run coral_trackers bbox_selector

# Lint / tests (from workspace root or package dir)
colcon test --packages-select coral_trackers
colcon test-result --verbose

# Run a single test file directly
python3 -m pytest test/test_flake8.py -v
```

## Architecture

### Node Overview

Four production nodes handle different tracking modalities:

| Node | File | What it does |
|------|------|--------------|
| `colors_tracker` | `colors_tracker.py` | HSV-based multi-color blob tracking |
| `samurai_tracker` | `samurai_tracker.py` | SAM2 2D segmentation tracking |
| `bbox_selector` | `bbox_selector.py` | Interactive bounding box selection tool |
| `mask_center_tracker` | `mask_center_tracker.py` | Filtered 3D center from mask + depth |
| `foundation_pose_tracker` | `foundation_pose_tracker.py` | FoundationPose: register() once, track_one() every frame |
| `foundationpose_plus_plus_tracker` | `foundation_pose_plus_plus_tracker.py` | 6DOF pose estimation with mask-center anchoring and optional Kalman filter |

A shared utility lives in `utils/kalman_filter_6d.py`.

### Typical Pipeline

```
bbox_selector ──(RegionOfInterest)──► samurai_tracker ──({name}_mask)──► mask_center_tracker ──({name}/center)──► foundation_pose_plus_plus_tracker
                                                                                                                           │
colors_tracker ─────────────────────────────────────────────────────────────────────────────(centroid)────────────────────┘
```

`bbox_selector` provides an interactive OpenCV window for the user to draw a bbox; `samurai_tracker` uses that bbox to initialize SAM2 tracking and emits per-frame masks; `mask_center_tracker` lifts each mask to a filtered 3D center using per-pixel depth; `foundation_pose_plus_plus_tracker` uses that center to anchor its pose estimate and refines with FoundationPose.

### Threading Model

`SamuraiTrackerNode` and `BboxSelectorNode` both run OpenCV or heavy GPU inference outside the ROS2 spin thread:

- **`samurai_tracker`**: A daemon background thread (`_process_loop`) waits on `_frame_event` and processes one frame at a time. A single `_latest_frame` slot (not a queue) holds the most recent unprocessed frame; newer arrivals overwrite it, which is correct for live cameras. All shared state (`_state`, `_latest_frame`, `_pending_init`) is protected by `_lock`.
- **`bbox_selector`**: `rclpy.spin()` runs in a daemon thread; the OpenCV display loop runs on the main thread (required for GUI on most platforms).

### Topics & Interfaces

**`colors_tracker`** (subscribes): `image` (BGR), `camera_info`, `depth`  
Publishes per-color: `{color}/center` (3D EMA-filtered), `{color}/centroid` (2D pixel), `{color}/projected_centroid` (3D at centroid depth), `{color}/mask`

**`samurai_tracker`** (subscribes): `{rgb_topic}` (default `image`), `{object_name}/bbox` (RegionOfInterest)  
Publishes: `{object_name}_centroid` (PointStamped, pixel coords), `{object_name}_mask` (mono8)

**`bbox_selector`** (subscribes): `{image_topic}` (default `image`)  
Publishes: `{bbox_topic}` (default `obj/bbox`, RegionOfInterest). Press **S** to open selectROI, **Q** to quit.

**`mask_center_tracker`** (subscribes): `{object_name}_mask` (mono8), `{depth_topic}` (default `depth`), `{camera_info_topic}` (default `camera_info`)  
Publishes: `{object_name}/center` (PointStamped, filtered 3D camera-frame XYZ)

**`foundation_pose_tracker`** (subscribes): `image`, `depth`, `camera_info`, `{name}_mask` (mono8)  
Publishes: `{name}/pose` (PoseStamped), `{name}/pose_viz` (annotated image, optional)  
Parameters: `register_iter` (default 5), `track_refine_iter` (default 4)

**`foundation_pose_plus_plus_tracker`** (subscribes): `image`, `depth`, `camera_info`, `{name}/center`  
Publishes: `{name}/pose` (PoseStamped), `{name}/pose_viz` (annotated image, optional)

### External Dependencies (not pip-installed)

Both heavyweight trackers inject paths via environment variables at import time:

- **`SAMURAI_PATH`** — path to the SAM2 checkout (default: `~/phd/software/samurai/sam2`). Must contain the `sam2` Python package.
- **`FOUNDATIONPOSE_PATH`** — path to the FoundationPose checkout (default: `/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2/FoundationPose`).

Set these before running if the defaults don't match your environment.

### Key Implementation Details

- **Hue wrap-around** (`colors_tracker`): When `lo[0] > hi[0]` (e.g., red straddles 0°/360°), the tracker splits into two `cv2.inRange` calls and ORs them. The drag-to-sample UI applies the same wrap-around logic when computing min/max hue from a patch.
- **Distance-gated EMA filter** (`colors_tracker`): Large jumps (> `dist_threshold` meters) are rejected as outliers. After 90 consecutive rejections the gate resets, allowing a genuine object relocation to be accepted.
- **SAM2 streaming inference** (`samurai_tracker`): On the first bbox, `init_state(frames=[frame])` is called with a single frame, `add_new_points_or_box` seeds it, and `propagate_in_video` runs on that one frame to establish the conditioning memory. Every subsequent frame calls `predictor.track_new_frame(state, frame_np)` — a method added to `SAM2VideoPredictor` (`~/phd/software/samurai/sam2/sam2/sam2_video_predictor.py`) that appends the frame to `inference_state["images"]` and calls `_run_single_frame_inference` directly, returning a mask without re-initializing state. Raw frame tensors are freed after feature extraction. A new bbox resets `_state = None` and triggers re-initialization on the next frame.
- **Mask-based 3D center** (`mask_center_tracker`): Back-projects every mask pixel to 3D using per-pixel depth and intrinsics, rejects depth outliers via median absolute deviation (MAD) to discard mixed foreground/background readings at mask edges, then applies a distance-gated EMA filter before publishing. Parameters: `alpha` (EMA weight, default 0.3), `dist_threshold` (outlier gate in metres, default 0.3).
- **Standard FoundationPose** (`foundation_pose_tracker`): Calls `register()` once on the first received mask (hypothesis generation + scoring), then `track_one()` every depth frame thereafter. No KalmanFilter, no `mask_center_tracker` dependency — the simplest pipeline when you just need reliable 6DOF pose. A new mask (via `{name}_mask`) re-runs `register()` to reinitialize.
- **FoundationPose++ init** (`foundation_pose_plus_plus_tracker`): Skips the expensive `register()` call. Waits for the first `{name}/center` message from `mask_center_tracker` and uses it directly as the initial XYZ translation, then refines with `track_one`. Each subsequent frame anchors the pose translation to the latest center before refinement. An optional `KalmanFilter6D` smooths the rotation; its translation state is overridden by the mask center each frame.
- **QoS**: Camera subscriptions use `BEST_EFFORT` reliability + `KEEP_LAST` to tolerate dropped frames from real-time sensors.
