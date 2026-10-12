"""Publish an explicitly user-accepted Host checkpoint for frozen Routing use."""
from __future__ import annotations

import argparse
import sys

import config as conf
from schedulers.shared_host import accept_host_bundle, atomic_json, read_manifest


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default=conf.SHARED_HOST_CHECKPOINT)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    target = accept_host_bundle(args.checkpoint, args.output_dir, reason=args.reason)
    _, metadata = read_manifest(target)
    atomic_json({"checkpoint": str(target), "status": metadata["status"],
                 "episode": metadata["training"]["episode"]}, target.parent / "latest.json")
    print(f"Host 人工验收发布完成，第 {metadata['training']['episode']} 轮：{target}")
    print("网络权重和验证指标保持原样；人工验收不代表自动收敛检查已通过。")


if __name__ == "__main__":
    main()
