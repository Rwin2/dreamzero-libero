#!/usr/bin/env python3
"""Send first frame of each task to the server and compare predictions.

This tests whether the model produces different actions for different scenes
(scene-conditioned) or always the same trajectory (unconditional).
"""
import numpy as np
import pandas as pd
import websocket
import msgpack
import cv2
from pathlib import Path


def _encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def _decode_numpy(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


TASKS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]

data_root = Path("/workspace/dreamzero_work/data/libero_spatial_lerobot")
video_dir = data_root / "videos" / "chunk-000"

ws = websocket.create_connection("ws://localhost:5000")
metadata_raw = ws.recv()
metadata = msgpack.unpackb(metadata_raw, object_hook=_decode_numpy, raw=False)

labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]

all_first_actions = []  # first action of each task's prediction

for task_idx in range(10):
    ep_idx = task_idx * 50  # first episode of each task
    pq = pd.read_parquet(data_root / f"data/chunk-000/episode_{ep_idx:06d}.parquet")
    expert_actions = np.array(pq["action"].tolist())
    expert_states = np.array(pq["observation.state"].tolist())

    # Read first frame
    def read_first_frame(key):
        path = str(video_dir / key / f"episode_{ep_idx:06d}.mp4")
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ext1 = read_first_frame("observation.images.exterior_image_1_left")
    ext2 = read_first_frame("observation.images.exterior_image_2_left")
    wrist = read_first_frame("observation.images.wrist_image_left")

    # Reset session for each task
    reset_msg = msgpack.packb(
        {"endpoint": "reset", "session_id": f"diag-task-{task_idx}"},
        default=_encode_numpy
    )
    ws.send(reset_msg, opcode=websocket.ABNF.OPCODE_BINARY)
    ws.recv()

    # Send observation
    obs = {
        "endpoint": "infer",
        "session_id": f"diag-task-{task_idx}",
        "observation/exterior_image_0_left": ext1,
        "observation/exterior_image_1_left": ext2,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": expert_states[0, :7].astype(np.float64),
        "observation/cartesian_position": np.zeros(6, dtype=np.float64),
        "observation/gripper_position": expert_states[0, 7:9].astype(np.float64),
        "prompt": TASKS[task_idx],
    }

    packed = msgpack.packb(obs, default=_encode_numpy)
    ws.send(packed, opcode=websocket.ABNF.OPCODE_BINARY)
    response_raw = ws.recv()

    try:
        response = msgpack.unpackb(response_raw, object_hook=_decode_numpy, raw=False)
    except:
        print(f"Task {task_idx}: Server error")
        print(response_raw[:200] if isinstance(response_raw, bytes) else response_raw)
        continue

    if isinstance(response, dict):
        pred_actions = response.get("actions", response.get("action"))
    elif isinstance(response, np.ndarray):
        pred_actions = response
    else:
        print(f"Task {task_idx}: Unexpected response: {type(response)}")
        continue

    pred_first = pred_actions[0]  # first predicted action
    expert_first = expert_actions[0]
    all_first_actions.append(pred_first)

    # Cosine similarity of first 6 dims
    cos = np.dot(pred_first[:6], expert_first[:6]) / (
        np.linalg.norm(pred_first[:6]) * np.linalg.norm(expert_first[:6]) + 1e-8
    )

    print(f"\nTask {task_idx}: {TASKS[task_idx][:60]}")
    print(f"  Pred  step 0: [{', '.join(f'{v:+.3f}' for v in pred_first)}]")
    print(f"  Expert step 0: [{', '.join(f'{v:+.3f}' for v in expert_first)}]")
    print(f"  Cosine sim: {cos:+.3f}  L2: {np.linalg.norm(pred_first[:6] - expert_first[:6]):.3f}")

ws.close()

# Check if predictions vary across tasks or are all the same
all_preds = np.array(all_first_actions)
print("\n\n=== Cross-Task Analysis ===")
print(f"Number of tasks with predictions: {len(all_preds)}")

if len(all_preds) > 1:
    print("\nPer-dimension variance across tasks (higher = more scene-dependent):")
    for d in range(min(7, all_preds.shape[1])):
        var = all_preds[:, d].var()
        mean = all_preds[:, d].mean()
        print(f"  {labels[d]:8s}: mean={mean:+.4f}  var={var:.6f}  "
              f"range=[{all_preds[:, d].min():+.4f}, {all_preds[:, d].max():+.4f}]")

    # Pairwise cosine similarity between predictions
    print("\nPairwise cosine similarity between task predictions (first 6 dims):")
    sims = []
    for i in range(len(all_preds)):
        for j in range(i+1, len(all_preds)):
            s = np.dot(all_preds[i, :6], all_preds[j, :6]) / (
                np.linalg.norm(all_preds[i, :6]) * np.linalg.norm(all_preds[j, :6]) + 1e-8
            )
            sims.append(s)
    print(f"  Mean: {np.mean(sims):+.4f}  Min: {np.min(sims):+.4f}  Max: {np.max(sims):+.4f}")
    if np.mean(sims) > 0.9:
        print("  → HIGH similarity: model produces nearly identical actions for all tasks (not scene-conditioned)")
    elif np.mean(sims) > 0.5:
        print("  → MODERATE similarity: model is somewhat scene-conditioned")
    else:
        print("  → LOW similarity: model produces varied actions per task (good scene conditioning)")
