from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from .config import ViFactConfig
from .utils.logging import setup_logging
from .data.loader import load_examples
from .data.preprocess import preprocess_example
from .data.validate import validate_example


def cmd_process(args: argparse.Namespace) -> None:
    cfg = ViFactConfig.from_yaml(args.config)
    setup_logging(cfg.system.get("log_level", "INFO"))
    out_dir = Path(args.output or cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = args.shard_size or cfg.data.get("shards", {}).get("shard_size", 50000)
    lower = cfg.preprocess.get("lower", False)

    n = 0
    shard_idx = 0
    shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
    f_out = open(shard_path, "w", encoding="utf-8")

    with tqdm(total=None, desc="processing") as pbar:
        for ex_id, ex in load_examples(args.input):
            ex = preprocess_example(ex, lower=lower)
            ok, errs = validate_example(ex_id, ex)
            if not ok:
                if args.skip_invalid:
                    pbar.write(f"skip {ex_id}: {errs}")
                    continue
                else:
                    raise ValueError(f"Invalid example {ex_id}: {errs}")
            ex_out: Dict = {"id": ex_id, **ex}
            f_out.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
            n += 1
            pbar.update(1)
            if shard_size and n % shard_size == 0:
                f_out.close()
                shard_idx += 1
                shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
                f_out = open(shard_path, "w", encoding="utf-8")

    f_out.close()
    print(f"Wrote {n} examples into {shard_idx+1} shard(s) at {out_dir}")


def cmd_split(args: argparse.Namespace) -> None:
    setup_logging("INFO")
    import random

    src = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0, "test": 0}

    def decide() -> str:
        r = rng.random()
        if r < args.train:
            return "train"
        if r < args.train + args.val:
            return "val"
        return "test"

    writers = {split: open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") for split in counts}

    def iter_jsonl(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line

    for shard in sorted(src.glob("*.jsonl")):
        for line in iter_jsonl(shard):
            split = decide()
            writers[split].write(line)
            counts[split] += 1

    for f in writers.values():
        f.close()
    print("Splits counts:", counts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vifact", description="ViFact data utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_process = sub.add_parser("process", help="Process raw dataset to sharded JSONL")
    p_process.add_argument("--input", required=True, help="Path to raw JSON/JSONL file")
    p_process.add_argument("--output", help="Output directory for shards")
    p_process.add_argument("--config", default="config/vifact_config.yaml")
    p_process.add_argument("--shard-size", type=int, default=None)
    p_process.add_argument("--skip-invalid", action="store_true")
    p_process.set_defaults(func=cmd_process)

    p_split = sub.add_parser("split", help="Split processed shards into train/val/test")
    p_split.add_argument("--input", required=True, help="Directory with processed shards (*.jsonl)")
    p_split.add_argument("--output", required=True, help="Output directory for splits")
    p_split.add_argument("--train", type=float, default=0.8)
    p_split.add_argument("--val", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
