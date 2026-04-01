#!/usr/bin/env python3
"""Test the eval transform pipeline with a dummy observation.

Loads ONLY the transform config + dataset metadata (fast, no 14B model),
then runs a fake observation through it to catch shape/resolution errors
before committing to a full server restart.

Usage:
    conda activate dreamzero
    python validate_transforms.py --model-path output/dreamzero_libero_lora_droid/checkpoint-200 --embodiment-tag libero_sim
"""
import argparse
import json
import numpy as np
import torch
import traceback

from hydra.utils import instantiate

from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


def build_eval_transform(model_path: str, embodiment_tag: str):
    """Reconstruct the eval transform from a checkpoint, same as sim_policy.py."""
    import safetensors
    from pathlib import Path
    from omegaconf import OmegaConf

    ckpt_path = Path(model_path)
    # Checkpoint stores config in experiment_cfg/
    cfg_path = ckpt_path / "experiment_cfg" / "conf.yaml"
    if not cfg_path.exists():
        cfg_path = ckpt_path / "train_config.yaml"
    train_cfg = OmegaConf.load(cfg_path)

    # Load dataset metadata
    metadata_path = ckpt_path / "experiment_cfg" / "metadata.json"
    if not metadata_path.exists():
        metadata_path = ckpt_path / "dataset_metadata.json"

    from groot.vla.data.schema.lerobot import DatasetMetadata
    raw = json.loads(metadata_path.read_text())
    # metadata.json may be keyed by embodiment tag
    if embodiment_tag in raw:
        raw = raw[embodiment_tag]
    metadata = DatasetMetadata.model_validate(raw)

    # Build transform
    tag = embodiment_tag
    assert tag in train_cfg.transforms, f"{tag} not in {list(train_cfg.transforms.keys())}"
    eval_transform = instantiate(train_cfg.transforms[tag])
    assert isinstance(eval_transform, ComposedModalityTransform)
    eval_transform.set_metadata(metadata)
    eval_transform.eval()

    return eval_transform, metadata, train_cfg


def make_dummy_obs(metadata):
    """Build a dummy observation matching what eval_dreamzero.py sends."""
    obs = {}

    # Images at native dataset resolution
    for vkey, vmeta in metadata.modalities.video.items():
        w, h = vmeta.resolution
        obs[f"video.{vkey}"] = np.zeros((1, 1, h, w, 3), dtype=np.uint8)

    # State
    for skey, smeta in metadata.modalities.state.items():
        dim = smeta.shape[0] if smeta.shape else 1
        obs[f"state.{skey}"] = np.zeros((1, 1, dim), dtype=np.float64)

    # Language
    obs["annotation.language.language_instruction"] = "pick up the black bowl"

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    args = parser.parse_args()

    print("Loading transform config...")
    eval_transform, metadata, train_cfg = build_eval_transform(
        args.model_path, args.embodiment_tag
    )

    print(f"\n=== Dataset Metadata ===")
    print(f"Video keys: {list(metadata.modalities.video.keys())}")
    for k, v in metadata.modalities.video.items():
        print(f"  {k}: resolution={v.resolution}, channels={v.channels}")
    print(f"State keys: {list(metadata.modalities.state.keys())}")
    for k, v in metadata.modalities.state.items():
        print(f"  {k}: shape={v.shape} absolute={v.absolute}")
    print(f"Action keys: {list(metadata.modalities.action.keys())}")
    for k, v in metadata.modalities.action.items():
        print(f"  {k}: shape={v.shape} absolute={v.absolute}")

    print(f"\n=== Transform Pipeline ({len(eval_transform.transforms)} transforms) ===")
    for i, t in enumerate(eval_transform.transforms):
        print(f"  [{i}] {type(t).__name__}: apply_to={getattr(t, 'apply_to', 'N/A')}")

    obs = make_dummy_obs(metadata)
    print(f"\n=== Dummy Observation ===")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {v!r}")

    print(f"\n=== Running transforms one by one ===")
    data = dict(obs)
    for i, t in enumerate(eval_transform.transforms):
        tname = type(t).__name__
        try:
            data = t(data)
            print(f"  [{i}] {tname}: OK")
            # Show shapes of any changed keys
            for k, v in data.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    shape = tuple(v.shape)
                    dtype = v.dtype
                    print(f"       {k}: {shape} {dtype}")
        except Exception as e:
            print(f"  [{i}] {tname}: FAILED")
            print(f"       Error: {e}")
            print(f"       Current data keys & shapes:")
            for k, v in data.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    print(f"         {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"         {k}: {type(v).__name__}")
            traceback.print_exc()
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
