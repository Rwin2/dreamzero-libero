#!/usr/bin/env python3
"""Diagnose action output: compare model predictions vs expert actions.

Sends the first frame of episode 0 to the server and compares
the predicted action chunk against the expert actions.
"""
import argparse
import json
import numpy as np
import pandas as pd
import websocket
import msgpack

# --- msgpack helpers (same as eval_dreamzero.py) ---
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # 1. Load expert data for episode 0
    pq = pd.read_parquet(
        "/workspace/dreamzero_work/data/libero_spatial_lerobot/data/chunk-000/episode_000000.parquet"
    )
    expert_actions = np.array(pq["action"].tolist())  # (T, 7)
    expert_states = np.array(pq["observation.state"].tolist())  # (T, 9)

    # 2. Load first frame images from the video files
    import cv2
    video_dir = "/workspace/dreamzero_work/data/libero_spatial_lerobot/videos/chunk-000"

    def read_first_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
        raise RuntimeError(f"Cannot read {video_path}")

    ext1 = read_first_frame(f"{video_dir}/observation.images.exterior_image_1_left/episode_000000.mp4")
    ext2 = read_first_frame(f"{video_dir}/observation.images.exterior_image_2_left/episode_000000.mp4")
    wrist = read_first_frame(f"{video_dir}/observation.images.wrist_image_left/episode_000000.mp4")

    print(f"Image shapes: ext1={ext1.shape} ext2={ext2.shape} wrist={wrist.shape}")
    print(f"Expert states[0] (joint 7 + gripper 2): {expert_states[0]}")

    # 3. Connect to server
    ws = websocket.create_connection(f"ws://{args.host}:{args.port}")
    metadata_raw = ws.recv()
    metadata = msgpack.unpackb(metadata_raw, object_hook=_decode_numpy, raw=False)
    print(f"Server metadata: {metadata}")

    # 4. Send reset
    reset_msg = msgpack.packb(
        {"endpoint": "reset", "session_id": "diag-001"},
        default=_encode_numpy
    )
    ws.send(reset_msg, opcode=websocket.ABNF.OPCODE_BINARY)
    ws.recv()

    # 5. Send first observation
    joint_states = expert_states[0, :7]
    gripper_states = expert_states[0, 7:9]

    obs = {
        "endpoint": "infer",
        "session_id": "diag-001",
        "observation/exterior_image_0_left": ext1,
        "observation/exterior_image_1_left": ext2,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_states.astype(np.float64),
        "observation/cartesian_position": np.zeros(6, dtype=np.float64),
        "observation/gripper_position": gripper_states.astype(np.float64),
        "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    }

    packed = msgpack.packb(obs, default=_encode_numpy)
    ws.send(packed, opcode=websocket.ABNF.OPCODE_BINARY)
    response_raw = ws.recv()

    # Try to decode
    try:
        response = msgpack.unpackb(response_raw, object_hook=_decode_numpy, raw=False)
    except Exception:
        response = response_raw
        if isinstance(response, (str, bytes)):
            print(f"Server error: {response}")
            ws.close()
            return

    if isinstance(response, dict):
        pred_actions = response.get("actions", response.get("action", None))
    elif isinstance(response, np.ndarray):
        pred_actions = response
    else:
        print(f"Unexpected response type: {type(response)}")
        print(f"Response: {response}")
        ws.close()
        return

    ws.close()

    print(f"\nPredicted action chunk shape: {pred_actions.shape}")
    print(f"Expert actions shape: {expert_actions.shape}")

    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    # The server returns 7-dim: [joint_pos(6), gripper(1)]
    # or 8-dim: [joint_pos(7), gripper(1)] depending on model
    n_action_dims = pred_actions.shape[-1]
    print(f"\nModel output dims: {n_action_dims}")

    print("\n=== Step-by-step comparison (first 8 steps) ===")
    n_compare = min(8, pred_actions.shape[0], expert_actions.shape[0])

    for i in range(n_compare):
        pred = pred_actions[i]
        expert = expert_actions[i]
        print(f"\nStep {i}:")
        print(f"  Model:  [{', '.join(f'{v:+.4f}' for v in pred)}]")
        print(f"  Expert: [{', '.join(f'{v:+.4f}' for v in expert)}]")

        # Compare first 6 dims (delta EE)
        n = min(6, len(pred), len(expert))
        l2 = np.linalg.norm(pred[:n] - expert[:n])
        cosine = np.dot(pred[:n], expert[:n]) / (np.linalg.norm(pred[:n]) * np.linalg.norm(expert[:n]) + 1e-8)
        print(f"  L2 dist (first {n} dims): {l2:.4f}")
        print(f"  Cosine sim (first {n} dims): {cosine:.4f}")

    # Overall stats
    print("\n=== Per-dimension comparison (first 8 steps) ===")
    for d in range(min(n_action_dims, 7)):
        pred_vals = pred_actions[:n_compare, d]
        if d < 7:
            expert_vals = expert_actions[:n_compare, d]
            label = labels[d] if d < len(labels) else f"dim{d}"
            corr = np.corrcoef(pred_vals, expert_vals)[0, 1] if len(pred_vals) > 1 else 0
            print(f"  {label:8s}: pred=[{pred_vals.min():+.4f}, {pred_vals.max():+.4f}] "
                  f"expert=[{expert_vals.min():+.4f}, {expert_vals.max():+.4f}] corr={corr:+.4f}")
        else:
            print(f"  dim{d:d}    : pred=[{pred_vals.min():+.4f}, {pred_vals.max():+.4f}]")


if __name__ == "__main__":
    main()
