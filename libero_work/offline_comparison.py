#!/usr/bin/env python3
"""Offline action comparison: fine-tuned model vs expert across all tasks.

Sends the first frame of each task's first 5 episodes to the server,
compares predicted first-step actions vs expert first-step actions.
Reports L2 distance, cosine similarity, and per-dim correlation.
"""
import numpy as np
import pandas as pd
import websocket
import msgpack
import cv2
from pathlib import Path
import json


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
_ = ws.recv()  # metadata

all_pred = []
all_expert = []
all_l2 = []
all_cos = []
per_task_results = []

n_episodes_per_task = 3  # compare first 3 episodes per task

for task_idx in range(10):
    task_l2s = []
    task_coss = []

    for ep_offset in range(n_episodes_per_task):
        ep_idx = task_idx * 50 + ep_offset
        pq_path = data_root / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            continue

        pq = pd.read_parquet(pq_path)
        expert_actions = np.array(pq["action"].tolist())
        expert_states = np.array(pq["observation.state"].tolist())

        def read_first_frame(key):
            path = str(video_dir / key / f"episode_{ep_idx:06d}.mp4")
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ext1 = read_first_frame("observation.images.exterior_image_1_left")
        ext2 = read_first_frame("observation.images.exterior_image_2_left")
        wrist = read_first_frame("observation.images.wrist_image_left")

        # Reset
        reset_msg = msgpack.packb(
            {"endpoint": "reset", "session_id": f"offline-{task_idx}-{ep_offset}"},
            default=_encode_numpy
        )
        ws.send(reset_msg, opcode=websocket.ABNF.OPCODE_BINARY)
        ws.recv()

        # Infer
        obs = {
            "endpoint": "infer",
            "session_id": f"offline-{task_idx}-{ep_offset}",
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
        resp_raw = ws.recv()

        try:
            resp = msgpack.unpackb(resp_raw, object_hook=_decode_numpy, raw=False)
        except Exception:
            continue

        if isinstance(resp, dict):
            pred = resp.get("actions", resp.get("action"))
        elif isinstance(resp, np.ndarray):
            pred = resp
        else:
            continue

        # Compare first 8 steps (or action chunk vs expert)
        n_compare = min(8, pred.shape[0], expert_actions.shape[0])
        for s in range(n_compare):
            p = pred[s, :6]
            e = expert_actions[s, :6]
            l2 = np.linalg.norm(p - e)
            cos = np.dot(p, e) / (np.linalg.norm(p) * np.linalg.norm(e) + 1e-8)
            all_pred.append(p)
            all_expert.append(e)
            all_l2.append(l2)
            all_cos.append(cos)
            task_l2s.append(l2)
            task_coss.append(cos)

    if task_l2s:
        per_task_results.append({
            "task_id": task_idx,
            "task": TASKS[task_idx][:60],
            "mean_l2": float(np.mean(task_l2s)),
            "mean_cos": float(np.mean(task_coss)),
            "n_samples": len(task_l2s),
        })
        print(f"Task {task_idx}: L2={np.mean(task_l2s):.4f}  cos={np.mean(task_coss):+.4f}  (n={len(task_l2s)})")

ws.close()

all_pred = np.array(all_pred)
all_expert = np.array(all_expert)

print(f"\n=== Aggregate Results ({len(all_l2)} samples) ===")
print(f"Mean L2 distance: {np.mean(all_l2):.4f}")
print(f"Mean cosine sim:  {np.mean(all_cos):+.4f}")
print(f"Median L2:        {np.median(all_l2):.4f}")
print(f"Median cosine:    {np.median(all_cos):+.4f}")

labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
print(f"\nPer-dimension:")
for d in range(6):
    corr = np.corrcoef(all_pred[:, d], all_expert[:, d])[0, 1]
    l2_d = np.mean(np.abs(all_pred[:, d] - all_expert[:, d]))
    print(f"  {labels[d]:8s}: pred_mean={all_pred[:, d].mean():+.4f}  expert_mean={all_expert[:, d].mean():+.4f}  corr={corr:+.4f}  MAE={l2_d:.4f}")

# Cross-task prediction variance
print(f"\nPrediction variance across all samples:")
for d in range(6):
    print(f"  {labels[d]:8s}: pred_std={all_pred[:, d].std():.4f}  expert_std={all_expert[:, d].std():.4f}")

# Save results
results = {
    "aggregate": {
        "mean_l2": float(np.mean(all_l2)),
        "mean_cosine_sim": float(np.mean(all_cos)),
        "median_l2": float(np.median(all_l2)),
        "n_samples": len(all_l2),
    },
    "per_task": per_task_results,
}
out_path = "/workspace/dreamzero_work/output/eval_finetuned/offline_comparison.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
