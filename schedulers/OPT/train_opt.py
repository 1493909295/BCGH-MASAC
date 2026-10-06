"""Train OPT with the shared Host-pretrain / Routing-train / joint schedule.

Run from the project root: python -m schedulers.OPT.train_opt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import inspect
import json
from pathlib import Path
import sys
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "schedulers.OPT"

import config as conf
from ._base import trainer, agents, RoutingMASACConfig, HostSACConfig
from .opt_agent import OPTRoutingMASAC
from .routing_observation import RoutingObservationBuilder


@dataclass(frozen=True)
class TrainConfig(trainer.TrainConfig):
    checkpoint_dir: str = conf.OPT_CHECKPOINT_DIR
    episode_log_csv_path: str = conf.OPT_EPISODE_LOG_CSV_PATH
    dc_log_csv_path: str = conf.OPT_DC_LOG_CSV_PATH
    resume_checkpoint: str | None = conf.OPT_RESUME_CHECKPOINT
    host_init_checkpoint: str | None = conf.OPT_HOST_INIT_CHECKPOINT


def project_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(conf.BASE_DIR) / path


def model_config(config_type, prefix, *, seed, device):
    """Read the same ROUTING_*/HOST_* settings as the H-MASAC entry point."""
    values = {item.name: getattr(conf, f"{prefix}_{item.name.upper()}")
              for item in fields(config_type) if item.name not in {"seed", "device", "allow_cpu"}}
    return config_type(**values, seed=seed, device=device, allow_cpu=str(device) == "cpu")


def require_cpu_capable_shared_modules() -> None:
    """Reject a partial deployment before constructing OPT's CPU agents."""
    missing = []
    for config_type in (RoutingMASACConfig, HostSACConfig):
        if "allow_cpu" not in {item.name for item in fields(config_type)}:
            missing.append(f"{config_type.__name__}.allow_cpu")
    if "allow_cpu" not in inspect.signature(agents.resolve_training_device).parameters:
        missing.append("resolve_training_device(..., allow_cpu=...)")
    if "device" not in inspect.signature(trainer.set_global_random_seeds).parameters:
        missing.append("set_global_random_seeds(..., device=...)")
    if missing:
        raise RuntimeError(
            "OPT 的 CPU 训练需要同步更新 H-MASAC 共用文件；当前缺少 "
            + ", ".join(missing)
            + f"。请将本项目的 {Path(agents.__file__).resolve()} 和 "
            + f"{Path(trainer.__file__).resolve()} 同步到训练环境后重试。"
        )


def validate_opt_device(device_name: str) -> str:
    """Exercise real CUDA kernels before creating an environment or models."""
    device = torch.device(device_name)
    if device.type == "cpu":
        require_cpu_capable_shared_modules()
        return "cpu"
    if device.type != "cuda":
        raise ValueError(f"OPT supports CPU or CUDA, got {device_name!r}")
    try:
        return str(agents.resolve_training_device(device_name))
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\nOPT 也可显式使用 --device cpu 运行。") from exc


def choose_opt_device(requested: str) -> str:
    """Auto prefers the configured GPU and falls back visibly to CPU."""
    if requested != "auto":
        return validate_opt_device(requested)
    preferred = str(conf.DEVICE or "cuda:0")
    try:
        return validate_opt_device(preferred)
    except RuntimeError as exc:
        validate_opt_device("cpu")
        print(
            f"⚠️ CUDA 设备预检未通过（{exc}）。OPT 将使用 CPU 继续训练；"
            "完整规模训练会明显更慢。",
            flush=True,
        )
        return "cpu"


def initialize_hosts(env, host_agents, checkpoint) -> None:
    """Import only Host weights, after validating DC and Host action ordering."""
    path = project_path(checkpoint)
    metadata = json.loads(trainer.checkpoint_state_path(path).read_text(encoding="utf-8"))
    structure = metadata.get("structure", {})
    expected_ids = list(map(str, env.edge_dc_ids))
    if structure.get("edge_dc_ids") != expected_ids:
        raise RuntimeError("Host initialization checkpoint DC order does not match")
    dc_map = {str(dc.dc_id): dc for dc in env.base_datacenters}
    host_dir = trainer.host_checkpoint_dir(path)
    for dc_id, agent in host_agents.items():
        host_ids = [str(host.host_id) for host in dc_map[dc_id].host_list]
        if structure.get("host_ids_per_dc", {}).get(dc_id) != host_ids:
            raise RuntimeError(f"Host initialization checkpoint Host order mismatch: {dc_id}")
        if structure.get("hosts", {}).get(dc_id) != {"obs_dim": agent.obs_dim, "action_dim": agent.action_dim}:
            raise RuntimeError(f"Host initialization checkpoint dimensions mismatch: {dc_id}")
        if not (host_dir / f"{dc_id}.pt").is_file():
            raise FileNotFoundError(host_dir / f"{dc_id}.pt")
    for dc_id, agent in host_agents.items():
        agent.load(host_dir / f"{dc_id}.pt", load_optimizers=False)
        # Imported weights are initialization, not a resumed training run.
        agent.update_step = 0


def train(train_config: TrainConfig | None = None,
          routing_masac_config: RoutingMASACConfig | None = None,
          host_sac_config: HostSACConfig | None = None):
    settings = train_config or TrainConfig()
    if settings.resume_checkpoint and settings.host_init_checkpoint:
        raise ValueError("Use either a full OPT resume or Host initialization, not both")
    route_device = routing_masac_config.device if routing_masac_config is not None else None
    host_device = host_sac_config.device if host_sac_config is not None else None
    if route_device is not None and host_device is not None and torch.device(route_device) != torch.device(host_device):
        raise ValueError("OPT Routing 和 Host 必须使用相同设备")
    device = choose_opt_device(str(route_device or host_device or "auto"))
    if routing_masac_config is None:
        routing_masac_config = model_config(RoutingMASACConfig, "ROUTING", seed=settings.seed, device=device)
    else:
        routing_masac_config = replace(
            routing_masac_config, device=device,
            allow_cpu=routing_masac_config.allow_cpu or device == "cpu",
        )
    if host_sac_config is None:
        host_sac_config = model_config(HostSACConfig, "HOST", seed=settings.seed, device=device)
    else:
        host_sac_config = replace(
            host_sac_config, device=device,
            allow_cpu=host_sac_config.allow_cpu or device == "cpu",
        )
    if settings.old_env_path is None:
        if conf.NUM_HOST < conf.NUM_DATACENTERS:
            raise ValueError(
                f"NUM_HOST={conf.NUM_HOST} 小于 NUM_DATACENTERS={conf.NUM_DATACENTERS}；"
                "请检查 config.py，或通过 --old-env 加载已有环境。"
            )
        if not Path(conf.HOST_DATASET_PATH).is_file():
            raise FileNotFoundError(
                f"Host 数据集不存在：{conf.HOST_DATASET_PATH}。"
                "请检查 config.HOST_DATASET_PATH，或传入 --host-dataset；"
                "已有环境快照可用 --old-env 加载。"
            )
    if not Path(conf.JOB_DATASET_PATH).is_file():
        raise FileNotFoundError(
            f"任务数据集不存在：{conf.JOB_DATASET_PATH}。"
            "请检查 config.JOB_DATASET_PATH，或传入 --job-dataset。"
        )
    initializer = None
    if settings.host_init_checkpoint:
        initializer = lambda env, hosts: initialize_hosts(env, hosts, settings.host_init_checkpoint)
    return trainer.train(
        settings, routing_masac_config, host_sac_config,
        routing_observation_builder_type=RoutingObservationBuilder,
        routing_agent_type=OPTRoutingMASAC,
        host_initializer=initializer,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train OPT with global DC-load routing observations")
    parser.add_argument("--host-pretrain-episodes", type=int)
    parser.add_argument("--routing-train-episodes", type=int)
    parser.add_argument("--joint-finetune-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--old-env", dest="old_env_path")
    parser.add_argument("--host-dataset", help="Host CSV path for a newly generated environment")
    parser.add_argument("--job-dataset", help="Job CSV path")
    parser.add_argument("--resume", dest="resume_checkpoint")
    parser.add_argument("--host-init-checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--episode-log", dest="episode_log_csv_path")
    parser.add_argument("--dc-log", dest="dc_log_csv_path")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N (default: auto)")
    args = vars(parser.parse_args(argv))
    device = choose_opt_device(args.pop("device"))
    host_dataset = args.pop("host_dataset")
    job_dataset = args.pop("job_dataset")
    if host_dataset is not None:
        conf.HOST_DATASET_PATH = str(project_path(host_dataset).resolve())
    if job_dataset is not None:
        conf.JOB_DATASET_PATH = str(project_path(job_dataset).resolve())
    settings = replace(TrainConfig(), **{key: value for key, value in args.items() if value is not None})
    settings = replace(settings, num_episodes=settings.host_pretrain_episodes
                       + settings.routing_train_episodes + settings.joint_finetune_episodes)
    routing_config = model_config(RoutingMASACConfig, "ROUTING", seed=settings.seed, device=device)
    host_config = model_config(HostSACConfig, "HOST", seed=settings.seed, device=device)
    train(settings, routing_config, host_config)


if __name__ == "__main__":
    main()
