#!/usr/bin/env python3
"""
TAPNext (JAX) RealSense online tracker — **no OpenCV UI**

This version uses **matplotlib** for:
  • interactive point picking on the first frame
  • live 2D visualization of tracked points (occlusion-aware)

It keeps the TAPNext JAX model (as in the demo you shared) and back‑projects
(u, v) with aligned RealSense depth to (X, Y, Z) in camera coordinates.

Run:
  pip install tapnet flax jax jaxlib numpy pillow matplotlib pyrealsense2 einops
  python tapnext_realsense_online_matplotlib.py \
      --ckpt tapnet/checkpoints/bootstapnext_ckpt.npz --width 640 --height 480 --fps 30

Controls:
  • Click: add query points on the picker window
  • s: start tracking (from the picker window)
  • q: quit (from either window)
  • v: toggle occlusion gating (during live)
  • r: reset (return to picker using current frame)
"""

import os
import io
import time
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# JAX / Flax / TapNext
import jax
import jax.numpy as jnp
import flax.linen as nn
import einops
import jax.nn as jnn

# ----------------------------- RealSense wrapper ---------------------------- #
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not available. Install it and connect a RealSense camera.")
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None, None
        bgr = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()) * self.depth_scale
        return bgr, depth_m

    def intrinsics_matrix(self) -> np.ndarray:
        return np.array([[self.intr.fx, 0, self.intr.ppx],
                         [0, self.intr.fy, self.intr.ppy],
                         [0, 0, 1]], dtype=np.float32)

    def stop(self):
        self.pipe.stop()

# ----------------------------- TAPNext model (from demo) -------------------- #
class MlpBlock(nn.Module):

  @nn.compact
  def __call__(self, x):
    d = x.shape[-1]
    x = nn.gelu(nn.Dense(4 * d)(x))
    return nn.Dense(d)(x)

class ViTBlock(nn.Module):
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    y = nn.LayerNorm()(x)
    y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
    x = x + y
    y = nn.LayerNorm()(x)
    y = MlpBlock()(y)
    x = x + y
    return x

class Einsum(nn.Module):
  width: int = 768

  def setup(self):
    self.w = self.param("w", nn.initializers.zeros_init(), (2, self.width, self.width * 4))
    self.b = self.param("b", nn.initializers.zeros_init(), (2, 1, 1, self.width * 4))[:, 0]

  def __call__(self, x):
    return jnp.einsum("...d,cdD->c...D", x, self.w) + self.b

class RMSNorm(nn.Module):
  width: int = 768

  def setup(self):
    self.scale = self.param("scale", nn.initializers.zeros_init(), (self.width,))

  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + 1e-6)
    scale = jnp.expand_dims(self.scale, axis=range(len(x.shape) - 1))
    return normed_x * (scale + 1)

class Conv1D(nn.Module):
  width: int = 768
  kernel_size: int = 4

  def setup(self):
    self.w = self.param("w", nn.initializers.zeros_init(), (self.kernel_size, self.width))
    self.b = self.param("b", nn.initializers.zeros_init(), (self.width,))

  def __call__(self, x, state):
    if state is None:
      state = jnp.zeros((x.shape[0], self.kernel_size - 1, x.shape[1]), dtype=x.dtype)
    x = jnp.concatenate([state, x[:, None]], axis=1)
    out = (x * self.w[None]).sum(axis=-2) + self.b[None]
    state = x[:, 1 - self.kernel_size :]
    return out, state

class BlockDiagonalLinear(nn.Module):
  width: int = 768
  num_heads: int = 12

  def setup(self):
    width = self.width // self.num_heads
    self.w = self.param("w", nn.initializers.zeros_init(), (self.num_heads, width, width))
    self.b = self.param("b", nn.initializers.zeros_init(), (self.num_heads, width))

  def __call__(self, x):
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_heads)
    y = jnp.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_heads)

class RGLRU(nn.Module):
  width: int = 768
  num_heads: int = 12

  def setup(self):
    self.a_real_param = self.param("a_param", nn.initializers.zeros_init(), (self.width,))
    self.input_gate = BlockDiagonalLinear(self.width, self.num_heads, name="input_gate")
    self.a_gate = BlockDiagonalLinear(self.width, self.num_heads, name="a_gate")

  def __call__(self, x, state):
    gate_x = jnn.sigmoid(self.input_gate(x))
    if state is None:
      return gate_x * x
    else:
      gate_a = jnn.sigmoid(self.a_gate(x))
      log_a = -8.0 * gate_a * jnn.softplus(self.a_real_param)
      a = jnp.exp(log_a)
      scale_factor = jnp.sqrt(1 - jnp.exp(2 * log_a))
      return a * state + gate_x * x * scale_factor

class MLPBlock(nn.Module):
  width: int = 768

  def setup(self):
    self.ffw_up = Einsum(self.width, name="ffw_up")
    self.ffw_down = nn.Dense(self.width, name="ffw_down")

  def __call__(self, x):
    out = self.ffw_up(x)
    return self.ffw_down(nn.gelu(out[0]) * out[1])

class RecurrentBlock(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4

  def setup(self) -> None:
    self.linear_y = nn.Dense(self.width, name="linear_y")
    self.linear_x = nn.Dense(self.width, name="linear_x")
    self.conv_1d = Conv1D(self.width, self.kernel_size, name="conv_1d")
    self.lru = RGLRU(self.width, self.num_heads, name="rg_lru")
    self.linear_out = nn.Dense(self.width, name="linear_out")

  def __call__(self, x, state):
    y = jax.nn.gelu(self.linear_y(x))
    x = self.linear_x(x)
    x, conv1d_state = self.conv_1d(x, None if state is None else state["conv1d_state"])
    rg_lru_state = self.lru(x, None if state is None else state["rg_lru_state"])
    x = self.linear_out(rg_lru_state * y)
    return x, {"rg_lru_state": rg_lru_state, "conv1d_state": conv1d_state}

class ResidualBlock(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4

  def setup(self):
    self.temporal_pre_norm = RMSNorm(self.width)
    self.recurrent_block = RecurrentBlock(self.width, self.num_heads, self.kernel_size, name="recurrent_block")
    self.channel_pre_norm = RMSNorm(self.width)
    self.mlp = MLPBlock(self.width, name="mlp_block")

  def __call__(self, x, state):
    y = self.temporal_pre_norm(x)
    y, state = self.recurrent_block(y, state)
    x = x + y
    y = self.mlp(self.channel_pre_norm(x))
    x = x + y
    return x, state

class ViTSSMBlock(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4

  def setup(self):
    self.ssm_block = ResidualBlock(self.width, self.num_heads, self.kernel_size)
    self.vit_block = ViTBlock(self.num_heads)

  def __call__(self, x, state):
    b = x.shape[0]
    x = einops.rearrange(x, "b n c -> (b n) c")
    x, state = self.ssm_block(x, state)
    x = einops.rearrange(x, "(b n) c -> b n c", b=b)
    x = self.vit_block(x)
    return x, state

class ViTSSMBackbone(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4
  num_blocks: int = 12

  def setup(self):
    self.blocks = [
        ViTSSMBlock(self.width, self.num_heads, self.kernel_size, name=f"encoderblock_{i}")
        for i in range(self.num_blocks)
    ]
    self.encoder_norm = nn.LayerNorm()

  def __call__(self, x, state):
    new_states = []
    for i in range(self.num_blocks):
      x, new_state = self.blocks[i](x, None if state is None else state[i])
      new_states.append(new_state)
    x = self.encoder_norm(x)
    return x, new_states

# Pose embeddings + TAPNext

def posemb_sincos_2d(h, w, width):
  y, x = jnp.mgrid[0:h, 0:w]
  freqs = jnp.linspace(0, 1, num=width // 4, endpoint=True)
  inv_freq = 1.0 / (10_000 ** freqs)
  y = jnp.einsum("h w, d -> h w d", y, inv_freq)
  x = jnp.einsum("h w, d -> h w d", x, inv_freq)
  pos_emb = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1)
  return pos_emb

class Backbone(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4
  num_blocks: int = 12

  def setup(self):
    self.lin_proj = nn.Conv(self.width, (1, 8, 8), strides=(1, 8, 8), padding="VALID", name="embedding")
    self.mask_token = self.param("mask_token", nn.initializers.zeros_init(), (1, 1, 1, self.width))[:, 0]
    self.unknown_token = self.param("unknown_token", nn.initializers.zeros_init(), (1, 1, self.width))
    self.point_query_token = self.param("point_query_token", nn.initializers.zeros_init(), (1, 1, 1, self.width))[:, 0]
    self.image_pos_emb = self.param("pos_embedding", nn.initializers.zeros_init(), (1, 256 // 8 * 256 // 8, self.width))
    self.encoder = ViTSSMBackbone(self.width, self.num_heads, self.kernel_size, self.num_blocks, name="Transformer")

  def __call__(self, frame, query_points, step, state):
    x = self.lin_proj(frame)
    b, h, w, c = x.shape
    query_points = jnp.concatenate([query_points[..., :1] - step, query_points[..., 1:]], axis=-1)
    posemb2d = posemb_sincos_2d(256, 256, self.width)

    def interp(x, y):
      return jax.scipy.ndimage.map_coordinates(x, y.T - 0.5, order=1, mode="nearest")

    interp_fn = jax.vmap(interp, in_axes=(-1, None), out_axes=-1)
    interp_fn = jax.vmap(interp_fn, in_axes=(None, 0), out_axes=0)
    point_tokens = self.point_query_token + interp_fn(posemb2d, query_points[..., 1:])
    query_timesteps = query_points[..., 0:1].astype(jnp.int32)
    query_tokens = jnp.where(query_timesteps > 0, self.unknown_token, self.mask_token)
    query_tokens = jnp.where(query_timesteps == 0, point_tokens, query_tokens)
    image_tokens = (jnp.reshape(x, [b, h * w, c]) + self.image_pos_emb)
    x = jnp.concatenate([image_tokens, query_tokens], axis=-2)
    x, state = self.encoder(x, state)
    _, q, _ = query_points.shape
    x = x[:, -q:, :]
    return x, state


class TAPNext(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4
  num_blocks: int = 12

  def setup(self):
    self.backbone = Backbone(self.width, self.num_heads, self.kernel_size, self.num_blocks)
    self.visible_head = nn.Sequential([
        nn.Dense(256), nn.LayerNorm(), nn.gelu, nn.Dense(256), nn.LayerNorm(), nn.gelu, nn.Dense(1),
    ])
    self.coordinate_head = nn.Sequential([
        nn.Dense(256), nn.LayerNorm(), nn.gelu, nn.Dense(256), nn.LayerNorm(), nn.gelu, nn.Dense(512),
    ])

  @nn.compact
  def __call__(self, frame, query_points, step, state):
    feat, state = self.backbone(frame, query_points, step, state)
    track_logits = self.coordinate_head(feat)
    visible_logits = self.visible_head(feat)
    position_x, position_y = jnp.split(track_logits, 2, axis=-1)
    position = jnp.stack([position_x, position_y], axis=-2)
    index = jnp.arange(position.shape[-1])[None, None, None]
    argmax = jnp.argmax(position, axis=-1, keepdims=True)
    mask = jnp.abs(argmax - index) <= 20
    probs = jnn.softmax(position * 0.5, axis=-1) * mask
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    tracks = jnp.sum(probs * index, axis=-1) + 0.5
    visible = (visible_logits > 0).astype(jnp.float32)
    return tracks, visible, state

model = TAPNext()

@jax.jit
def forward(params, frame, query_points, step, state):
  tracks, visible, state = model.apply({"params": params}, frame, query_points, step, state)
  return tracks, visible, state

# ----------------------------- Checkpoint helpers --------------------------- #

def npload(fname):
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    with open(fname, "rb") as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)

def recover_tree(flat_dict):
  tree = {}
  for k, v in flat_dict.items():
    parts = k.split("/")
    node = tree
    for part in parts[:-1]:
      if part not in node:
        node[part] = {}
      node = node[part]
    node[parts[-1]] = v
  return tree

# ----------------------------- Utility ------------------------------------- #

def scale_intrinsics(K: np.ndarray, in_w: int, in_h: int, out_w: int, out_h: int) -> np.ndarray:
    sx, sy = out_w / in_w, out_h / in_h
    K2 = K.copy().astype(np.float32)
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def resize_rgb_bilinear(rgb_bgr: np.ndarray, new_hw=(256,256)) -> np.ndarray:
    # bgr uint8 -> RGB uint8 resized via PIL bilinear
    h, w = rgb_bgr.shape[:2]
    img = Image.fromarray(rgb_bgr[:, :, ::-1], mode='RGB')
    img = img.resize((new_hw[1], new_hw[0]), resample=Image.BILINEAR)
    return np.asarray(img)

def resize_nn_depth(depth: np.ndarray, new_hw=(256,256)) -> np.ndarray:
    h, w = depth.shape[:2]
    new_h, new_w = new_hw
    rr = (np.arange(new_h) * (h / new_h)).astype(np.int32)
    cc = (np.arange(new_w) * (w / new_w)).astype(np.int32)
    return depth[rr][:, cc]

def depth_at(depth: np.ndarray, u: int, v: int) -> float:
    if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            y0, y1 = max(0, v-1), min(depth.shape[0], v+2)
            x0, x1 = max(0, u-1), min(depth.shape[1], u+2)
            patch = depth[y0:y1, x0:x1]
            good = patch[np.isfinite(patch) & (patch > 0)]
            z = float(np.median(good)) if good.size > 0 else 0.0
        return z
    return 0.0

def pix_to_xyz(u: float, v: float, z: float, K: np.ndarray):
    if z <= 0: return (np.nan, np.nan, np.nan)
    X = (u - K[0,2]) / K[0,0] * z
    Y = (v - K[1,2]) / K[1,1] * z
    return (float(X), float(Y), float(z))

# ----------------------------- Main ---------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='TAPNext RealSense online tracker (matplotlib UI)')
    ap.add_argument('--ckpt', type=str, default='tapnet/bootstapnext_ckpt.npz')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--gate_occlusion', action='store_true')
    args = ap.parse_args()

    # Load checkpoint
    print(f"Loading TAPNext params from {args.ckpt} …")
    params = recover_tree(npload(args.ckpt))

    # Warm up JIT
    _ = forward(params, jnp.zeros((1,256,256,3), jnp.float32), jnp.zeros((1,1,3), jnp.float32), 0, None)

    cam = RealSenseRGBD(args.width, args.height, args.fps)
    K_native = cam.intrinsics_matrix()

    # Grab the first frame for picking
    bgr0, depth0 = cam.read()
    if bgr0 is None:
        raise SystemExit('No frames from RealSense.')
    H_in, W_in = bgr0.shape[:2]

    # Prepare picker window (256×256)
    rgb256 = resize_rgb_bilinear(bgr0)
    depth256 = resize_nn_depth(depth0)
    K256 = scale_intrinsics(K_native, W_in, H_in, 256, 256)

    picks: List[Tuple[int,int]] = []
    started = False
    quit_flag = False

    fig_pick, ax_pick = plt.subplots(num='TAPNext — pick points')
    im_pick = ax_pick.imshow(rgb256)
    ax_pick.set_title('Click to add points, press "s" to start, "q" to quit')

    def onclick(event):
        nonlocal picks
        if event.inaxes is not ax_pick:
            return
        if event.button == 1 and not started:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            picks.append((x, y))
            ax_pick.add_patch(Circle((x, y), radius=3, color='red'))
            fig_pick.canvas.draw_idle()

    def onkey(event):
        nonlocal started, quit_flag
        if event.key == 's' and not started:
            started = True
            plt.close(fig_pick)
        elif event.key == 'q':
            quit_flag = True
            plt.close(fig_pick)

    cid_click = fig_pick.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig_pick.canvas.mpl_connect('key_press_event', onkey)
    plt.show(block=True)

    if quit_flag:
        cam.stop(); return
    if len(picks) == 0:
        print('No points selected. Exiting.')
        cam.stop(); return

    # Build query tensor (t=0,x,y) in 256 coords
    query_xyt = np.stack([
        np.zeros(len(picks), dtype=np.float32),
        np.array([p[0] for p in picks], dtype=np.float32),
        np.array([p[1] for p in picks], dtype=np.float32)
    ], axis=1)[None]

    # Live figure
    fig, ax = plt.subplots(num='TAPNext — live')
    ax.set_axis_off()
    gate_occ = args.gate_occlusion

    state = None
    step = 0
    last_xyz = [None] * len(picks)

    # Initialize artists
    im = ax.imshow(rgb256)
    scat_vis = ax.scatter([], [], s=30, c='red')
    scat_occ = ax.scatter([], [], s=30, facecolors='none', edgecolors='yellow')
    texts = [ax.text(0,0,'', color='white', fontsize=8) for _ in picks]

    t_prev = time.time(); fps_ema = 0.0

    while True:
        bgr, depth = cam.read()
        if bgr is None:
            continue
        rgb256 = resize_rgb_bilinear(bgr)
        depth256 = resize_nn_depth(depth)
        K256 = scale_intrinsics(K_native, W_in, H_in, 256, 256)

        # JAX input
        frame_jax = (rgb256.astype(np.float32) / 255.0) * 2.0 - 1.0
        frame_jax = jnp.asarray(frame_jax[None, ...], dtype=jnp.float32)
        queries_jax = jnp.asarray(query_xyt, dtype=jnp.float32)

        tracks, visible, state = forward(params, frame_jax, queries_jax, step, state)

        step += 1
        xy = np.array(tracks)[0][..., ::-1]          # (Q,2) (x,y) in [0,256)
        vis = np.array(visible)[0, :, 0]  # (Q,)

        # Back‑project + update texts
        vis_xy = []; occ_xy = []
        for i, (u, v) in enumerate(xy):
            if gate_occ and not bool(vis[i]):
                occ_xy.append([u, v])
                # retain last valid text position
                continue
            uu, vv = int(round(u)), int(round(v))
            z = depth_at(depth256, uu, vv)
            X, Y, Z = pix_to_xyz(u, v, z, K256)
            texts[i].set_position((u+5, max(8, v-5)))
            if np.isfinite(Z) and Z > 0:
                texts[i].set_text(f'({X:.3f},{Y:.3f},{Z:.3f})')
                last_xyz[i] = (u, v, (X, Y, Z))
            else:
                texts[i].set_text('')
            if bool(vis[i]):
                vis_xy.append([u, v])
            else:
                occ_xy.append([u, v])

        im.set_data(rgb256)
        if len(vis_xy):
            scat_vis.set_offsets(np.array(vis_xy))
        else:
            scat_vis.set_offsets(np.empty((0,2)))
        if len(occ_xy):
            scat_occ.set_offsets(np.array(occ_xy))
        else:
            scat_occ.set_offsets(np.empty((0,2)))

        # HUD title with FPS & gate status
        now = time.time(); dt = now - t_prev; t_prev = now
        if dt > 0: fps_ema = 0.9*fps_ema + 0.1*(1.0/dt)
        ax.set_title(f'FPS: {fps_ema:.1f}  Q: {len(picks)}  gate_occ: {gate_occ}   (q=quit, v=toggle, r=reset)')

        plt.pause(0.001)

        # Handle key presses from the live figure
        # (matplotlib doesn’t provide non-blocking key state; use event connection)
        pressed = {'q': False, 'v': False, 'r': False}
        def onkey_live(event):
            if event.key in pressed:
                pressed[event.key] = True
        cid_live = fig.canvas.mpl_connect('key_press_event', onkey_live)
        plt.gcf().canvas.flush_events()
        fig.canvas.mpl_disconnect(cid_live)

        if pressed['q']:
            break

        if pressed['v']:
            gate_occ = not gate_occ

        if pressed['r']:
            # return to picker, using current frame as backdrop
            plt.close(fig)
            # rebuild picker state
            picks = []
            fig_pick2, ax_pick2 = plt.subplots(num='TAPNext — pick points')
            im_pick2 = ax_pick2.imshow(rgb256)
            ax_pick2.set_title('Click to add points, press "s" to start, "q" to quit')
            def onclick2(event):
                if event.inaxes is not ax_pick2:
                    return
                if event.button == 1:
                    x, y = int(round(event.xdata)), int(round(event.ydata))
                    picks.append((x, y))
                    ax_pick2.add_patch(Circle((x, y), radius=3, color='red'))
                    fig_pick2.canvas.draw_idle()
            started2 = False; quit2 = False
            def onkey2(event):
                nonlocal started2, quit2
                if event.key == 's':
                    started2 = True; plt.close(fig_pick2)
                elif event.key == 'q':
                    quit2 = True; plt.close(fig_pick2)
            fig_pick2.canvas.mpl_connect('button_press_event', onclick2)
            fig_pick2.canvas.mpl_connect('key_press_event', onkey2)
            plt.show(block=True)
            if quit2 or len(picks)==0:
                break
            # rebuild query tensor & artists
            query_xyt = np.stack([
                np.zeros(len(picks), dtype=np.float32),
                np.array([p[0] for p in picks], dtype=np.float32),
                np.array([p[1] for p in picks], dtype=np.float32)
            ], axis=1)[None]
            state = None; step = 0
            fig, ax = plt.subplots(num='TAPNext — live')
            ax.set_axis_off()
            im = ax.imshow(rgb256)
            scat_vis = ax.scatter([], [], s=30, c='red')
            scat_occ = ax.scatter([], [], s=30, facecolors='none', edgecolors='yellow')
            texts = [ax.text(0,0,'', color='white', fontsize=8) for _ in picks]

    cam.stop(); plt.close('all')

if __name__ == '__main__':
    main()
