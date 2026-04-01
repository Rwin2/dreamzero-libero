#!/usr/bin/env python
"""
LIBERO evaluation client for DreamZero inference server.

Bridges the LIBERO simulation environment with DreamZero's WebSocket-based
inference server. Designed to run from the LIBERO conda env (Python 3.8)
with only lightweight extra dependencies: websockets, msgpack, numpy.

Usage:
    # 1. Start DreamZero inference server (in dreamzero env):
    #    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    #        --standalone --nproc_per_node=2 \
    #        inference_server.py --port 5000 --enable-dit-cache \
    #        --model-path ./checkpoints/dreamzero_libero_lora
    #
    # 2. Run this eval client (in libero env):
    #    python eval_dreamzero.py --host localhost --port 5000 --suite libero_spatial

    # Save videos and results with local plots:
    #    python eval_dreamzero.py --port 5000 --suite libero_spatial \
    #        --save-videos --output-dir ./eval_results
"""

import argparse
import functools
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import msgpack
import numpy as np
import torch
import websockets.sync.client

# ---------------------------------------------------------------------------
# Inline msgpack-numpy helpers (replaces openpi_client.msgpack_numpy)
# We inline these so we don't need the openpi_client package in the libero env.
# Protocol is identical to the DreamZero server's serialization format.
# ---------------------------------------------------------------------------

def _pack_ndarray(obj):
    """Custom msgpack packer for numpy arrays."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError("Unsupported dtype: {}".format(obj.dtype))
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_ndarray(obj):
    """Custom msgpack unpacker for numpy arrays."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_ndarray)
msgpack_packb = functools.partial(msgpack.packb, default=_pack_ndarray)
msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_ndarray)

# ---------------------------------------------------------------------------
# WebSocket client for DreamZero server
# ---------------------------------------------------------------------------

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

logger = logging.getLogger(__name__)


class DreamZeroClient:
    """Lightweight WebSocket client that talks to the DreamZero inference server.

    Implements the same wire protocol as eval_utils/policy_client.py:
      - On connect, receives server metadata (PolicyServerConfig) via msgpack.
      - infer(): sends obs dict with endpoint="infer", receives action array.
      - reset(): sends dict with endpoint="reset", receives ack string.
    """

    def __init__(self, host="localhost", port=6000, open_loop_horizon=8,
                 controller="OSC_POSE"):
        self._uri = "ws://{}:{}".format(host, port)
        self._packer = MsgpackPacker()
        self._open_loop_horizon = open_loop_horizon
        self._controller = controller
        self._ws = None
        self._server_metadata = None

        # Action chunk state
        self._pred_action_chunk = None
        self._chunk_step = 0
        self._session_id = str(uuid.uuid4())

        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        """Connect to the DreamZero server and receive metadata."""
        logger.info("Connecting to DreamZero server at %s ...", self._uri)
        try:
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                close_timeout=PING_TIMEOUT_SECS,
            )
        except Exception:
            # Retry with wss://
            logger.info("ws:// failed, trying wss:// ...")
            self._uri = "wss://" + self._uri.split("//")[1]
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                close_timeout=PING_TIMEOUT_SECS,
            )

        # Server sends its config as the first message
        self._server_metadata = msgpack_unpackb(self._ws.recv())
        logger.info("Connected. Server metadata: %s", self._server_metadata)

    # ------------------------------------------------------------------
    # Reset / Infer
    # ------------------------------------------------------------------

    def reset(self):
        """Reset server-side policy state for a new episode."""
        self._pred_action_chunk = None
        self._chunk_step = 0
        self._session_id = str(uuid.uuid4())

        reset_data = {
            "endpoint": "reset",
            "session_id": self._session_id,
        }
        self._ws.send(self._packer.pack(reset_data))
        resp = self._ws.recv()
        logger.debug("Reset response: %s", resp)

    def _send_infer(self, obs_dict):
        """Send an observation to the server and return the raw action chunk.

        Args:
            obs_dict: Dict with the observation keys expected by the server
                      (observation/exterior_image_0_left, etc.).

        Returns:
            np.ndarray of shape (N, 7) or (N, 8) -- the action chunk.
        """
        obs_dict["endpoint"] = "infer"
        obs_dict["session_id"] = self._session_id

        data = self._packer.pack(obs_dict)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError("Server error:\n{}".format(response))
        result = msgpack_unpackb(response)

        # The server wraps actions in a dict with key "actions" or returns raw array
        if isinstance(result, dict):
            actions = result.get("actions", result.get(b"actions", None))
            if actions is None:
                # Fallback: try to find any array-valued key
                for v in result.values():
                    if isinstance(v, np.ndarray) and v.ndim == 2:
                        actions = v
                        break
            if actions is None:
                raise RuntimeError("Could not find actions in server response: {}".format(
                    list(result.keys())))
        else:
            actions = result

        assert isinstance(actions, np.ndarray), "Expected ndarray, got {}".format(type(actions))
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return actions

    def get_action(self, agentview_rgb, eye_in_hand_rgb, joint_states,
                   gripper_states, language_instruction):
        """Get a single action from the DreamZero server.

        Manages open-loop action chunking: only queries the server every
        `open_loop_horizon` steps; otherwise replays from the cached chunk.

        Args:
            agentview_rgb: (H, W, 3) uint8 -- third-person camera image.
            eye_in_hand_rgb: (H, W, 3) uint8 -- wrist camera image.
            joint_states: (7,) float64 -- arm joint positions.
            gripper_states: (2,) float64 -- gripper finger positions.
            language_instruction: str -- task language description.

        Returns:
            action: (7,) float64 -- delta action [pos(3), rot(3), gripper(1)].
        """
        need_new_chunk = (
            self._pred_action_chunk is None
            or self._chunk_step >= self._open_loop_horizon
            or self._chunk_step >= len(self._pred_action_chunk)
        )

        if need_new_chunk:
            self._chunk_step = 0

            # Send images at native 128x128 — the server's transform pipeline
            # handles resizing to the model's internal resolution.
            # DreamZero expects 3 cameras. We duplicate agentview as the
            # second exterior camera (LIBERO only has 2 cameras).
            request_data = {
                "observation/exterior_image_0_left": agentview_rgb,
                "observation/exterior_image_1_left": agentview_rgb.copy(),
                "observation/wrist_image_left": eye_in_hand_rgb,
                "observation/joint_position": joint_states.astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": gripper_states[:2].astype(np.float64),
                "prompt": language_instruction,
            }

            self._pred_action_chunk = self._send_infer(request_data)
            logger.debug("Received action chunk shape: %s", self._pred_action_chunk.shape)

        action = self._pred_action_chunk[self._chunk_step]
        self._chunk_step += 1

        # DreamZero oxe_droid returns 8-dim: [joint_pos(7), gripper(1)]
        # Map to LIBERO action format based on controller type:
        gripper_val = 1.0 if action[-1] > 0.5 else -1.0

        if self._controller == "JOINT_POSITION":
            # JOINT_POSITION: 7 joint deltas + 1 gripper = 8 dim
            # DreamZero outputs radians (q99 range ~±0.7), but the controller
            # expects [-1,1] input scaled by output_max=0.05 rad.
            # Rescale: divide by output_max so model's radian values map correctly.
            joint_actions = action[:7] / 0.05
            action = np.concatenate([joint_actions, [gripper_val]])
        else:
            # OSC_POSE: 6 delta EE (pos+rot) + 1 gripper = 7 dim
            action = np.concatenate([action[:6], [gripper_val]])

        return action.astype(np.float64)

    def close(self):
        """Close the WebSocket connection."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize_image(img, target_h, target_w):
    """Resize image with padding to target size, preserving aspect ratio.

    Args:
        img: (H, W, 3) uint8 numpy array.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (target_h, target_w, 3) uint8 numpy array.
    """
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size (center the image)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# LIBERO evaluation
# ---------------------------------------------------------------------------

SUITE_MAP = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_10": "LIBERO_10",
    "libero_90": "LIBERO_90",
}


def evaluate_libero(args):
    """Main evaluation loop across all tasks in the selected LIBERO suite."""

    # -- Load LIBERO benchmark ----------------------------------------------
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_cls = get_benchmark(SUITE_MAP[args.suite])
    benchmark = benchmark_cls(task_order_index=0)
    n_tasks = benchmark.get_num_tasks()
    task_names = benchmark.get_task_names()

    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")

    logger.info("Suite: %s  |  Tasks: %d", args.suite, n_tasks)

    # -- Connect to DreamZero server ----------------------------------------
    client = DreamZeroClient(
        host=args.host,
        port=args.port,
        open_loop_horizon=args.open_loop_horizon,
        controller=args.controller,
    )

    # -- Output directory ---------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    # -- Evaluation loop ----------------------------------------------------
    all_results = {}
    all_success_rates = []

    for task_id in range(n_tasks):
        task = benchmark.get_task(task_id)
        task_name = task.name
        language = task.language

        logger.info("=" * 70)
        logger.info("Task %d/%d: %s", task_id + 1, n_tasks, language)
        logger.info("=" * 70)

        # Create environment
        bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
            "controller": args.controller,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()

        # Load fixed init states
        init_states = benchmark.get_task_init_states(task_id)
        num_trials = min(args.num_trials, len(init_states))

        num_successes = 0
        task_video_dir = video_dir / "task_{:02d}".format(task_id) if args.save_videos else None

        for trial_idx in range(num_trials):
            if task_video_dir is not None:
                task_video_dir.mkdir(parents=True, exist_ok=True)

            # Reset environment to the fixed init state
            obs = env.set_init_state(init_states[trial_idx])

            # Let physics settle (5 zero-action steps, same as LIBERO eval protocol)
            action_dim = 8 if args.controller == "JOINT_POSITION" else 7
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(action_dim))

            # Reset DreamZero server state
            client.reset()

            success = False
            frames = [] if args.save_videos else None

            for step in range(args.max_steps):
                # Extract observations from the LIBERO env
                # LIBERO env returns keys: agentview_image, robot0_eye_in_hand_image,
                # robot0_joint_pos, robot0_gripper_qpos, etc.
                agentview_rgb = obs["agentview_image"]
                eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]
                joint_states = obs["robot0_joint_pos"]
                gripper_states = obs["robot0_gripper_qpos"]

                if args.save_videos and step % 2 == 0:
                    frames.append(agentview_rgb.copy())

                # Query DreamZero for action
                action = client.get_action(
                    agentview_rgb=agentview_rgb,
                    eye_in_hand_rgb=eye_in_hand_rgb,
                    joint_states=joint_states,
                    gripper_states=gripper_states,
                    language_instruction=language,
                )

                # Step the environment with the 7-dim delta action
                obs, reward, done, info = env.step(action)

                if reward > 0:
                    success = True
                    break
                if done:
                    break

            if success:
                num_successes += 1

            # Save video for this trial
            if args.save_videos and frames and len(frames) > 0:
                _save_video(
                    frames,
                    str(task_video_dir / "trial_{:03d}{}.mp4".format(
                        trial_idx, "_success" if success else "")),
                    fps=10,
                )

            logger.info(
                "  Trial %d/%d: %s  (running: %d/%d = %.1f%%)",
                trial_idx + 1, num_trials,
                "SUCCESS" if success else "fail",
                num_successes, trial_idx + 1,
                100.0 * num_successes / (trial_idx + 1),
            )

        env.close()

        success_rate = num_successes / num_trials if num_trials > 0 else 0.0
        all_success_rates.append(success_rate)
        all_results[task_name] = {
            "task_id": task_id,
            "language": language,
            "success_rate": success_rate,
            "num_successes": num_successes,
            "num_trials": num_trials,
        }

        logger.info(
            "Task %d result: %d/%d = %.1f%%",
            task_id, num_successes, num_trials, 100.0 * success_rate,
        )

        # Log per-task metrics incrementally to JSON
        _save_incremental_results(output_dir, task_id, language, success_rate,
                                  num_successes, num_trials)

    # -- Aggregate results --------------------------------------------------
    mean_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0

    logger.info("=" * 70)
    logger.info("OVERALL RESULTS: %s", args.suite)
    logger.info("=" * 70)
    for task_name, res in all_results.items():
        logger.info(
            "  %-60s  %.1f%% (%d/%d)",
            res["language"],
            100.0 * res["success_rate"],
            res["num_successes"],
            res["num_trials"],
        )
    logger.info("-" * 70)
    logger.info("  Mean success rate: %.1f%%", 100.0 * mean_success_rate)
    logger.info("=" * 70)

    # -- Generate local visualization plots -----------------------------------
    _generate_eval_plots(all_results, mean_success_rate, args.suite, output_dir)

    # -- Save JSON results --------------------------------------------------
    results_path = output_dir / "results_{}.json".format(args.suite)
    results_json = {
        "suite": args.suite,
        "mean_success_rate": mean_success_rate,
        "tasks": all_results,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    with open(str(results_path), "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # -- Cleanup ------------------------------------------------------------
    client.close()

    return mean_success_rate


# ---------------------------------------------------------------------------
# Local visualization (replaces W&B)
# ---------------------------------------------------------------------------

def _save_incremental_results(output_dir, task_id, language, success_rate,
                              num_successes, num_trials):
    """Append per-task results to a JSONL file for live monitoring."""
    jsonl_path = output_dir / "eval_progress.jsonl"
    entry = {
        "task_id": task_id,
        "language": language,
        "success_rate": success_rate,
        "num_successes": num_successes,
        "num_trials": num_trials,
        "timestamp": datetime.now().isoformat(),
    }
    with open(str(jsonl_path), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _generate_eval_plots(all_results, mean_success_rate, suite, output_dir):
    """Generate matplotlib bar chart of per-task success rates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot generation.")
        return

    tasks = list(all_results.values())
    labels = [t["language"][:40] + "..." if len(t["language"]) > 40 else t["language"]
              for t in tasks]
    rates = [t["success_rate"] * 100 for t in tasks]

    fig, ax = plt.subplots(figsize=(14, max(6, len(tasks) * 0.5)))
    colors = ["#2ecc71" if r >= 50 else "#e74c3c" if r < 25 else "#f39c12" for r in rates]
    bars = ax.barh(range(len(labels)), rates, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("DreamZero on {} — Mean: {:.1f}%".format(suite, mean_success_rate * 100))
    ax.set_xlim(0, 105)
    ax.axvline(x=mean_success_rate * 100, color="blue", linestyle="--", alpha=0.7,
               label="Mean: {:.1f}%".format(mean_success_rate * 100))
    ax.legend()

    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                "{:.0f}%".format(rate), va="center", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "success_rates_{}.png".format(suite)
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    logger.info("Saved evaluation plot to %s", plot_path)


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def _save_video(frames, path, fps=10):
    """Save a list of RGB frames as an MP4 video using pyav (H.264)."""
    if not frames:
        return
    import av
    h, w = frames[0].shape[:2]
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}
    for frame_rgb in frames:
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    logger.debug("Saved video: %s (%d frames)", path, len(frames))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DreamZero on LIBERO benchmark via WebSocket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server connection
    parser.add_argument("--host", type=str, default="localhost",
                        help="DreamZero inference server host.")
    parser.add_argument("--port", type=int, default=5000,
                        help="DreamZero inference server port.")

    # LIBERO suite
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=list(SUITE_MAP.keys()),
                        help="LIBERO task suite to evaluate.")

    # Eval parameters
    parser.add_argument("--num-trials", type=int, default=50,
                        help="Number of init states (trials) per task.")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum env steps per trial.")
    parser.add_argument("--open-loop-horizon", type=int, default=8,
                        help="Number of actions to execute per server query.")

    # Output
    parser.add_argument("--save-videos", action="store_true", default=False,
                        help="Save evaluation rollout videos.")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory for results JSON and videos.")

    # Controller
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        choices=["OSC_POSE", "JOINT_POSITION"],
                        help="Robosuite controller type.")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("DreamZero-LIBERO Evaluation Client")
    logger.info("Server: %s:%d", args.host, args.port)
    logger.info("Suite: %s  |  Trials/task: %d  |  Max steps: %d",
                args.suite, args.num_trials, args.max_steps)

    t_start = time.time()
    mean_sr = evaluate_libero(args)
    elapsed = time.time() - t_start

    logger.info("Evaluation complete in %.1f minutes. Mean success rate: %.1f%%",
                elapsed / 60.0, 100.0 * mean_sr)


if __name__ == "__main__":
    main()
