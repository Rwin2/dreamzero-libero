# SINGER: DreamZero on LIBERO Benchmark

Zero-shot and fine-tuned evaluation of [DreamZero](https://github.com/dreamzero0/dreamzero) (14B World Action Model) on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark.

## Structure

```
dreamzero_work/   # DreamZero inference server, training scripts, data conversion, configs
libero_work/      # LIBERO eval client (connects to DreamZero server via WebSocket)
docs/             # Findings, architecture notes, quick reference
```

## Quick Start

See [docs/DREAMZERO_LIBERO_QUICKREF.md](docs/DREAMZERO_LIBERO_QUICKREF.md) for terminal commands.

## Key Files

| File | Purpose |
|---|---|
| `dreamzero_work/socket_test_optimized_AR.py` | Inference server (loads 14B model, serves via WebSocket) |
| `dreamzero_work/scripts/train/libero_training.sh` | LoRA fine-tuning launcher |
| `dreamzero_work/scripts/data/convert_libero_to_lerobot.py` | LIBERO HDF5 to LeRobot v2 conversion |
| `libero_work/eval_dreamzero.py` | Eval client (LIBERO env <-> WebSocket <-> DreamZero) |

## Requirements

- 2x H100 GPUs (inference + training)
- DreamZero-DROID checkpoint
- LIBERO datasets (downloaded separately)
- Conda envs: `dreamzero` (server/training), `libero` (eval)
