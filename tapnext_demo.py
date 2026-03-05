#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Install Code and Dependencies {form-width: "25%"}
get_ipython().system('pip install git+https://github.com/google-deepmind/tapnet.git')


# In[ ]:


# @title Download Model {form-width: "25%"}

get_ipython().run_line_magic('mkdir', '-p tapnet/checkpoints')
get_ipython().system('wget -P tapnet/checkpoints --no-check-certificate https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz')
get_ipython().run_line_magic('ls', 'tapnet/checkpoints')


# In[ ]:


# @title Base Imports {form-width: "10%"}

import io
import os

import cv2
import einops
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib
import mediapy as media
import numpy as np
from tapnet import evaluation_datasets


# In[ ]:


# @title Base Layers {form-width: "25%"}


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
    self.w = self.param(
        "w", nn.initializers.zeros_init(), (2, self.width, self.width * 4)
    )
    self.b = self.param(
        "b", nn.initializers.zeros_init(), (2, 1, 1, self.width * 4)
    )[:, 0]

  def __call__(self, x):
    return jnp.einsum("...d,cdD->c...D", x, self.w) + self.b


class RMSNorm(nn.Module):
  width: int = 768

  def setup(self):
    self.scale = self.param("scale", nn.initializers.zeros_init(), (self.width))

  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + 1e-6)
    scale = jnp.expand_dims(self.scale, axis=range(len(x.shape) - 1))
    return normed_x * (scale + 1)


class Conv1D(nn.Module):
  width: int = 768
  kernel_size: int = 4

  def setup(self):
    self.w = self.param(
        "w", nn.initializers.zeros_init(), (self.kernel_size, self.width)
    )
    self.b = self.param("b", nn.initializers.zeros_init(), (self.width))

  def __call__(self, x, state):
    if state is None:
      state = jnp.zeros(
          (x.shape[0], self.kernel_size - 1, x.shape[1]), dtype=x.dtype
      )
    x = jnp.concatenate([state, x[:, None]], axis=1)  # shape: (b, k, c)
    out = (x * self.w[None]).sum(axis=-2) + self.b[None]  # shape: (b, c)
    state = x[:, 1 - self.kernel_size :]  # shape: (b, k - 1, c)
    return out, state


class BlockDiagonalLinear(nn.Module):
  width: int = 768
  num_heads: int = 12

  def setup(self):
    width = self.width // self.num_heads
    self.w = self.param(
        "w", nn.initializers.zeros_init(), (self.num_heads, width, width)
    )
    self.b = self.param(
        "b", nn.initializers.zeros_init(), (self.num_heads, width)
    )

  def __call__(self, x):
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_heads)
    y = jnp.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_heads)


class RGLRU(nn.Module):
  width: int = 768
  num_heads: int = 12

  def setup(self):
    self.a_real_param = self.param(
        "a_param", nn.initializers.zeros_init(), (self.width)
    )
    self.input_gate = BlockDiagonalLinear(
        self.width, self.num_heads, name="input_gate"
    )
    self.a_gate = BlockDiagonalLinear(self.width, self.num_heads, name="a_gate")

  def __call__(self, x, state):
    gate_x = jnn.sigmoid(self.input_gate(x))
    if state is None:
      return gate_x * x  # No memory accumulation, return directly
    else:
      gate_a = jnn.sigmoid(self.a_gate(x))
      log_a = -8.0 * gate_a * jnn.softplus(self.a_real_param)
      a = jnp.exp(log_a)
      scale_factor = jnp.sqrt(1 - jnp.exp(2 * log_a))  # Compute decay factor
      return a * state + gate_x * x * scale_factor  # Memory update


# In[ ]:


# @title ViTSSM Blocks {form-width: "25%"}


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
    x, conv1d_state = self.conv_1d(
        x, None if state is None else state["conv1d_state"]
    )
    rg_lru_state = self.lru(x, None if state is None else state["rg_lru_state"])
    x = self.linear_out(rg_lru_state * y)
    return x, {"rg_lru_state": rg_lru_state, "conv1d_state": conv1d_state}


class ResidualBlock(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4

  def setup(self):
    self.temporal_pre_norm = RMSNorm(self.width)
    self.recurrent_block = RecurrentBlock(
        self.width, self.num_heads, self.kernel_size, name="recurrent_block"
    )
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
        ViTSSMBlock(
            self.width,
            self.num_heads,
            self.kernel_size,
            name=f"encoderblock_{i}",
        )
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


# In[ ]:


# @title TAPNext Model {form-width: "25%"}


def posemb_sincos_2d(h, w, width):
  """Compute 2D sine-cosine positional embeddings following MoCo v3 logic."""
  y, x = jnp.mgrid[0:h, 0:w]
  freqs = jnp.linspace(0, 1, num=width // 4, endpoint=True)
  inv_freq = 1.0 / (10_000**freqs)
  y = jnp.einsum("h w, d -> h w d", y, inv_freq)
  x = jnp.einsum("h w, d -> h w d", x, inv_freq)
  pos_emb = jnp.concatenate(
      [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1
  )
  return pos_emb


class Backbone(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4
  num_blocks: int = 12

  def setup(self):
    self.lin_proj = nn.Conv(
        self.width,
        (1, 8, 8),
        strides=(1, 8, 8),
        padding="VALID",
        name="embedding",
    )
    self.mask_token = self.param(
        "mask_token", nn.initializers.zeros_init(), (1, 1, 1, self.width)
    )[:, 0]
    self.unknown_token = self.param(
        "unknown_token", nn.initializers.zeros_init(), (1, 1, self.width)
    )
    self.point_query_token = self.param(
        "point_query_token", nn.initializers.zeros_init(), (1, 1, 1, self.width)
    )[:, 0]
    self.image_pos_emb = self.param(
        "pos_embedding",
        nn.initializers.zeros_init(),
        (1, 256 // 8 * 256 // 8, self.width),
    )
    self.encoder = ViTSSMBackbone(
        self.width,
        self.num_heads,
        self.kernel_size,
        self.num_blocks,
        name="Transformer",
    )

  def __call__(self, frame, query_points, step, state):
    x = self.lin_proj(frame)  # x: (b, h, w, c)
    b, h, w, c = x.shape
    query_points = jnp.concatenate(
        [query_points[..., :1] - step, query_points[..., 1:]], axis=-1
    )  # (b, q, 3)
    posemb2d = posemb_sincos_2d(256, 256, self.width)  # posemb2d: (256, 256, c)

    def interp(x, y):
      return jax.scipy.ndimage.map_coordinates(
          x, y.T - 0.5, order=1, mode="nearest"
      )

    interp_fn = jax.vmap(interp, in_axes=(-1, None), out_axes=-1)
    interp_fn = jax.vmap(interp_fn, in_axes=(None, 0), out_axes=0)
    point_tokens = self.point_query_token + interp_fn(
        posemb2d, query_points[..., 1:]
    )  # (b, q, c)
    # Query tokens
    query_timesteps = query_points[..., 0:1].astype(jnp.int32)  # (b, q, 1)
    query_tokens = jnp.where(
        query_timesteps > 0, self.unknown_token, self.mask_token
    )  # (b, q, c)
    query_tokens = jnp.where(
        query_timesteps == 0, point_tokens, query_tokens
    )  # (b, q, c)
    # Image tokens
    image_tokens = (
        jnp.reshape(x, [b, h * w, c]) + self.image_pos_emb
    )  # x: (b, h*w, c)

    x = jnp.concatenate(
        [image_tokens, query_tokens], axis=-2
    )  # x: (b, h*w + q, c)
    x, state = self.encoder(x, state)
    _, q, _ = query_points.shape
    x = x[:, -q:, :]  # x: (b, q, c)

    return x, state


class TAPNext(nn.Module):
  width: int = 768
  num_heads: int = 12
  kernel_size: int = 4
  num_blocks: int = 12

  def setup(self):
    self.backbone = Backbone(
        self.width, self.num_heads, self.kernel_size, self.num_blocks
    )
    self.visible_head = nn.Sequential([
        nn.Dense(256),
        nn.LayerNorm(),
        nn.gelu,
        nn.Dense(256),
        nn.LayerNorm(),
        nn.gelu,
        nn.Dense(1),
    ])
    self.coordinate_head = nn.Sequential([
        nn.Dense(256),
        nn.LayerNorm(),
        nn.gelu,
        nn.Dense(256),
        nn.LayerNorm(),
        nn.gelu,
        nn.Dense(512),
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
  tracks, visible, state = model.apply(
      {"params": params}, frame, query_points, step, state
  )
  return tracks, visible, state


# In[ ]:


# @title Load parameters {form-width: "25%"}


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
  tree = (
      {}
  )  # Initialize an empty dictionary to store the resulting tree structure
  for (
      k,
      v,
  ) in (
      flat_dict.items()
  ):  # Iterate over each key-value pair in the flat dictionary
    parts = k.split(
        "/"
    )  # Split the key into parts using "/" as a delimiter to build the tree structure
    node = tree  # Start at the root of the tree
    for part in parts[
        :-1
    ]:  # Loop through each part of the key, except the last one
      if (
          part not in node
      ):  # If the current part doesn't exist as a key in the node, create an empty dictionary for it
        node[part] = {}
      node = node[part]  # Move down the tree to the next level
    node[parts[-1]] = v  # Set the value at the final part of the key
  return tree  # Return the reconstructed tree


ckpt_path = "tapnet/checkpoints/bootstapnext_ckpt.npz"
params = recover_tree(npload(ckpt_path))


# In[ ]:


# @title Visualization function {form-width: "25%"}


def plot_2d_tracks(
    video,
    points,
    visibles,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False,
):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = (np.array(color_map(cmap_norm(i)))[:3] * 255).astype(
        np.uint8
    )

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[
        max(0, t - tracks_leave_trace) : t + 1
    ]
    for s in range(line_tracks.shape[0] - 1):
      img = frame.copy()

      for i in range(num_points):
        if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
        elif (
            show_occ
            and line_infront_cameras[s, i]
            and line_infront_cameras[s + 1, i]
        ):  # occluded
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], -1, cv2.LINE_AA)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], 1, cv2.LINE_AA)

    frames.append(frame)
  frames = np.stack(frames)
  return frames


# In[ ]:


# @title Download TAPVid-DAVIS Dataset {form-width: "25%"}
get_ipython().system('wget --no-check-certificate https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip')
get_ipython().system('unzip tapvid_davis.zip')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# @title Qualitative results on dense tracks\n\ndavis_dataset = evaluation_datasets.create_davis_dataset(\n    davis_points_path='./tapvid_davis/tapvid_davis.pkl',\n    query_mode='first',\n    full_resolution=False,\n    resolution=(256, 256),\n)\n\nfor i, sample in enumerate(davis_dataset):\n  frames = sample['davis']['video']\n  ys, xs = np.meshgrid(np.linspace(8, 256, 32), np.linspace(8, 256, 32))\n  query_points = np.stack(\n      [np.zeros(len(xs.flatten())), xs.flatten(), ys.flatten()], axis=1\n  )[None]\n\n  tracks, visibles, state = [], [], None\n  for t in range(0, frames.shape[1]):\n    pred_tracks, pred_visible, state = forward(\n        params, frames[:, t], query_points, t, state\n    )\n    tracks.append(pred_tracks)\n    visibles.append(pred_visible)\n  tracks = np.stack(tracks, axis=2)[..., ::-1]\n  visibles = np.stack(visibles, axis=2).squeeze(-1)\n\n  frames = ((frames[0] + 1) / 2 * 255).astype(np.uint8)\n  video_viz = plot_2d_tracks(\n      frames, tracks[0].transpose(1, 0, 2), visibles[0].transpose(1, 0)\n  )\n  media.show_video(video_viz, fps=15)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# @title Qualitative results on sparse tracks\n\ndavis_dataset = evaluation_datasets.create_davis_dataset(\n    davis_points_path='./tapvid_davis/tapvid_davis.pkl',\n    query_mode='first',\n    full_resolution=False,\n    resolution=(256, 256),\n)\n\ncnt = 0\nfor i, sample in enumerate(davis_dataset):\n  batch = sample['davis']\n  tracks, visibles, state = [], [], None\n  for t in range(0, batch['video'].shape[1]):\n    pred_tracks, pred_visible, state = forward(\n        params, batch['video'][:, t], batch['query_points'], t, state\n    )\n    tracks.append(pred_tracks)\n    visibles.append(pred_visible)\n  tracks = np.stack(tracks, axis=2)[..., ::-1]\n  visibles = np.stack(visibles, axis=2).squeeze(-1)\n\n  frames = ((batch['video'][0] + 1) / 2 * 255).astype(np.uint8)\n  eye = np.eye(batch['video'].shape[1], dtype=np.int32)\n  query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye\n  query_frame = (batch['query_points'][0, :, 0]).astype(np.int32)\n  evaluation_mask = query_frame_to_eval_frames[query_frame]\n  visibles *= evaluation_mask[None]\n\n  video_viz = np.concatenate(\n      [\n          plot_2d_tracks(\n              frames, tracks[0].transpose(1, 0, 2), visibles[0].transpose(1, 0)\n          ),\n          plot_2d_tracks(\n              frames,\n              batch['target_points'][0].transpose(1, 0, 2),\n              ~batch['occluded'][0].transpose(1, 0),\n          ),\n      ],\n      axis=2,\n  )\n  media.show_video(video_viz, fps=15)\n\n  cnt += 1\n  if cnt > 5:\n    break\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# @title Quantitative results\n\ndavis_dataset = evaluation_datasets.create_davis_dataset(\n    davis_points_path='./tapvid_davis/tapvid_davis.pkl',\n    query_mode='first',\n    full_resolution=False,\n    resolution=(256, 256),\n)\n\nscores = []\nfor i, sample in enumerate(davis_dataset):\n  batch = sample['davis']\n  print('video', i, jax.tree_util.tree_map(lambda x: x.shape, batch))\n  tracks, visibles, state = [], [], None\n  for t in range(0, batch['video'].shape[1]):\n    pred_tracks, pred_visible, state = forward(\n        params, batch['video'][:, t], batch['query_points'], t, state\n    )\n    tracks.append(pred_tracks)\n    visibles.append(pred_visible)\n  tracks = np.stack(tracks, axis=2)[..., ::-1]\n  visibles = np.stack(visibles, axis=2).squeeze(-1)\n\n  scalars = evaluation_datasets.compute_tapvid_metrics(\n      batch['query_points'],\n      batch['occluded'],\n      batch['target_points'],\n      np.logical_not(visibles),\n      tracks,\n      query_mode='first',\n  )\n  scalars = jax.tree.map(lambda x: np.array(np.sum(x, axis=0)), scalars)\n  scores.append(scalars)\n\nprint(np.mean([scores[k]['average_jaccard'] for k in range(len(scores))]))\nprint(np.mean([scores[k]['occlusion_accuracy'] for k in range(len(scores))]))\nprint(\n    np.mean(\n        [scores[k]['average_pts_within_thresh'] for k in range(len(scores))]\n    )\n)\n")

