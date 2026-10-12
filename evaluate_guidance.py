"""Paired evaluation of one v5 Routing checkpoint with a fixed frozen Host.

Example:
python -B evaluate_guidance.py --algorithm BCGH-MASAC --routing-checkpoint model/.../final.pt --host-checkpoint model/shared_host/final --environment environment/env_keep/... --output analysis_outputs/guidance_v5_eval.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys

import torch

from schedulers.shared_host import load_frozen_hosts


def evaluate_variants(env_factory, routing, hosts, settings, evaluator, seeds):
    """Test-time ablations, with identical tasks and Host weights in each pair."""
    variants = {
        'full': {},
        'no_guidance': {'enable_heuristic_guidance': False},
        'no_congestion': {'resource_penalty_enabled': False},
        'no_path_risk': {'transit_penalty_enabled': False},
        'legacy_scoring': {'scoring_mode': 'legacy'},
    }
    for seed in seeds:
        for name, overrides in variants.items():
            env = env_factory(seed)
            try:
                metrics = evaluator(env, routing, hosts, replace(settings, **overrides),
                                    seed=int(seed), deterministic=True)
                lifecycle = metrics.pop('absorption', {})
                yield {'variant': name, 'seed': int(seed), **metrics,
                       **{'lifecycle_'+k: v for k, v in lifecycle.items()}}
            finally:
                env.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='固定 Host 和任务种子的引导模块消融评估（不训练）')
    parser.add_argument('--algorithm', choices=['BCGH-MASAC', 'MR-BCGH-MASAC'], required=True)
    parser.add_argument('--routing-checkpoint', type=Path, required=True)
    parser.add_argument('--host-checkpoint', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--seeds', default='101,102,103,104,105')
    parser.add_argument('--jobs', type=int, help='可选的小规模冒烟任务数；各组相同')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在，请指定新的路径。')
    metadata = json.loads(args.routing_checkpoint.with_suffix('.trainer.json').read_text(encoding='utf-8'))
    semantics = metadata.get('bgh_guidance', {})
    if semantics.get('version') != 5 or semantics.get('guidance_context_version') != 1:
        parser.error('此评估入口要求 v5/context-v1 训练检查点；请重新训练旧 Routing 权重。')
    prefix = 'schedulers.'+args.algorithm
    support = importlib.import_module(prefix+'.training_support')
    networks = importlib.import_module(prefix+'.h_masac_agent')
    settings = replace(support.TrainConfig(), **semantics['parameters'],
                       host_checkpoint=str(args.host_checkpoint))
    checkpoint = torch.load(args.routing_checkpoint, map_location='cpu', weights_only=False)
    route_config = networks.RoutingMASACConfig(**checkpoint['config'])
    structure = checkpoint['structure'] if args.algorithm == 'MR-BCGH-MASAC' else checkpoint
    dimensions = {key: structure[key] for key in ('local_obs_dim', 'global_state_dim', 'action_dim', 'num_agents')}
    if args.algorithm == 'MR-BCGH-MASAC':
        dimensions.update(edge_dc_ids=structure['edge_dc_ids'], action_target_dc_ids=structure['action_target_dc_ids'])
    routing = networks.RoutingMASAC(**dimensions, config=route_config)
    routing.load(args.routing_checkpoint, load_optimizers=False)
    routing.eval_mode()
    def factory(seed):
        env = support.build_environment(int(seed), str(args.environment))
        if args.jobs is not None:
            if args.jobs <= 0:
                parser.error('--jobs 必须为正整数')
            env.job_num = args.jobs
        return env
    env = factory(101)
    try:
        hosts = load_frozen_hosts(env, args.host_checkpoint, routing.device)
        from schedulers.shared_host import validate_scheduler_resume
        validate_scheduler_resume(str(args.routing_checkpoint), hosts, settings)
    finally:
        env.close()
    seeds = [int(value) for value in args.seeds.split(',')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(evaluate_variants(factory, routing, hosts, settings, support.evaluate_episode, seeds))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    print(f'已保存 {len(rows)} 次配对评估：{args.output.resolve()}')


if __name__ == '__main__':
    main()
