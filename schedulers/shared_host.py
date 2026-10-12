"""One versioned Host bundle shared by all five routing schedulers.

The H-MASAC Host implementation is the canonical network/observation contract.
Routing trainers use FrozenHostPolicy, which has no optimizers or replay memory.
"""
from __future__ import annotations

import csv
import copy
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

import config as conf

_agents = importlib.import_module("schedulers.H-MASAC.h_masac_agent")
_observations = importlib.import_module("schedulers.H-MASAC.observations")
HostSACConfig = _agents.HostSACConfig
LocalHostSAC = _agents.LocalHostSAC
HostObservationBuilder = _observations.HostObservationBuilder
ROOT = Path(conf.BASE_DIR)
SCHEMA_VERSION = 1
NORMALIZATION_FIELDS = (
    "norm_eps", "max_job_cpu", "max_job_gpu", "max_job_duration",
    "max_host_cpu", "max_host_gpu", "queue_length_scale", "queue_workload_scale",
    "sla_deadline_ratio", "drop_deadline_ratio",
)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def shared_host_mode(settings):
    # Only OPT, outside this migration, retains its explicit legacy schedule.
    return getattr(settings, "host_training_mode", "frozen") == "frozen"


def read_manifest(checkpoint, *, require_converged=True):
    if checkpoint is None:
        raise ValueError("必须指定已通过收敛验收的 host_checkpoint；不会自动重新训练 Host。")
    directory = project_path(checkpoint).resolve()
    path = directory / "manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("role") != "shared_host":
        raise ValueError("不兼容的共享 Host checkpoint 格式")
    acceptance = metadata.get("acceptance", {})
    manually_accepted = (
        metadata.get("status") == "accepted_manual"
        and acceptance.get("method") == "manual"
        and acceptance.get("accepted_by") == "user"
        and isinstance(acceptance.get("reason"), str)
        and bool(acceptance["reason"].strip())
        and isinstance(acceptance.get("source_episode"), int)
        and acceptance["source_episode"] == metadata.get("training", {}).get("episode")
    )
    if require_converged and metadata.get("status") != "converged" and not manually_accepted:
        raise ValueError("Host checkpoint 尚未通过收敛验收或明确人工验收，不能用于调度器训练")
    expected = {f"hosts/{dc}.pt" for dc in metadata["structure"]["dc_ids"]}
    expected |= {"environment/hosts.csv", "environment/datacenters.csv"}
    if not expected.issubset(metadata["files"]):
        raise ValueError("Host checkpoint 文件声明不完整")
    for name, checksum in metadata["files"].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != checksum:
            raise ValueError(f"Host checkpoint 文件缺失或校验失败：{name}")
    return directory, metadata


def accept_host_bundle(checkpoint, destination, *, reason):
    """Publish the user's explicit acceptance without altering weights or metrics."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("人工验收必须记录明确的原因")
    directory, metadata = read_manifest(checkpoint, require_converged=False)
    episode = metadata.get("training", {}).get("episode")
    if not isinstance(episode, int) or episode < 1:
        raise ValueError("人工验收必须来自已完成训练轮次的 checkpoint")
    target = project_path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Host checkpoint 不允许覆盖：{target}")
    if target.is_relative_to(directory):
        raise ValueError("人工验收输出不能位于源 checkpoint 内")
    staging = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        for name in metadata["files"]:
            file = staging / name
            file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, file)
        accepted = copy.deepcopy(metadata)
        accepted.update(bundle_id=uuid.uuid4().hex, status="accepted_manual",
                        acceptance={"method": "manual", "accepted_by": "user",
                                    "reason": reason.strip(), "source_episode": episode,
                                    "source_checkpoint": str(directory),
                                    "source_bundle_id": metadata["bundle_id"],
                                    "source_manifest_sha256": sha256(directory / "manifest.json"),
                                    "source_status": metadata["status"],
                                    "automatic_convergence_passed": metadata["status"] == "converged",
                                    "accepted_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(accepted, staging / "manifest.json")
        read_manifest(staging)
        os.replace(staging, target)
    finally:
        if staging.exists():
            if staging.resolve().parent != target.parent:
                raise ValueError("拒绝清理超出目标目录的临时 checkpoint")
            shutil.rmtree(staging)
    return target


def prepare_scheduler_config(settings):
    if not shared_host_mode(settings):
        if type(settings).__module__ != "schedulers.OPT.train_opt" or settings.host_training_mode != "legacy":
            raise ValueError("这五个调度器只允许使用冻结的共享 Host")
        return settings
    if settings.host_pretrain_episodes != 0 or settings.joint_finetune_episodes != 0:
        raise ValueError("共享 Host 模式禁止 Host 预训练及联合微调；请仅配置 Routing 轮数。")
    directory, _ = read_manifest(settings.host_checkpoint)
    return replace(settings, old_env_path=str(project_path(settings.old_env_path).resolve())
                   if settings.old_env_path else str(directory / "environment"),
                   resume_checkpoint=str(project_path(settings.resume_checkpoint).resolve())
                   if settings.resume_checkpoint else None)


def host_structure(env):
    builder = HostObservationBuilder(env)
    dcs = {str(dc.dc_id): dc for dc in env.base_datacenters}
    return {
        "dc_ids": list(map(str, env.edge_dc_ids)),
        "hosts": {
            dc: [{"id": str(host.host_id), "cpu": float(host.cpu_num),
                  "gpu": float(host.gpu_capacity_num)} for host in dcs[dc].host_list]
            for dc in builder.edge_dc_ids
        },
        "dimensions": {dc: {"obs_dim": builder.get_obs_dim(dc),
                            "action_dim": builder.get_action_dim(dc)}
                       for dc in builder.edge_dc_ids},
        "observation_version": 1,
        "job_features": builder.JOB_FEAT_DIM,
        "dc_features": builder.LOCAL_DC_FEAT_DIM,
        "host_features": list(builder.HOST_FEATURE_NAMES),
        "normalization": {name: float(getattr(env, name)) for name in NORMALIZATION_FIELDS},
    }


def host_reference(agents):
    references = [agent.reference for agent in agents.values()]
    if not references or any(value != references[0] for value in references):
        raise ValueError("所有 DC 必须来自同一套共享 Host checkpoint")
    return references[0]


def parameter_digest(agent):
    digest = hashlib.sha256()
    for name in ("actor", "critic", "target_critic"):
        for key, tensor in getattr(agent, name).state_dict().items():
            digest.update(f"{name}/{key}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(agent.log_alpha.detach().cpu().numpy().tobytes())
    digest.update(str(agent.update_step).encode())
    return digest.hexdigest()


class FrozenHostPolicy:
    """Loads all learned states for auditing, but creates no optimizer."""

    def __init__(self, path, device, reference):
        self.path = Path(path)
        self.file_hash = sha256(self.path)
        self.reference = reference
        self.device = torch.device(device)
        state = torch.load(self.path, map_location=self.device, weights_only=False)
        if state.get("algorithm_role") != "local_host_sac":
            raise ValueError("Host 模型角色错误")
        options = {key: value for key, value in state["config"].items()
                   if key in {item.name for item in fields(HostSACConfig)}}
        options.update(device=str(device), allow_cpu=self.device.type == "cpu")
        self.config = HostSACConfig(**options)
        self.obs_dim, self.action_dim = int(state["obs_dim"]), int(state["action_dim"])
        self.actor = _agents.LocalHostDiscreteActor(self.obs_dim, self.action_dim,
                                                   self.config.actor_hidden_dim).to(self.device)
        self.critic = _agents.LocalHostTwinCritic(self.obs_dim, self.action_dim,
                                                 self.config.critic_hidden_dim).to(self.device)
        self.target_critic = _agents.LocalHostTwinCritic(self.obs_dim, self.action_dim,
                                                        self.config.critic_hidden_dim).to(self.device)
        for name in ("actor", "critic", "target_critic"):
            network = getattr(self, name)
            network.load_state_dict(state[f"{name}_state_dict"])
            network.requires_grad_(False)
            network.eval()
        self.log_alpha = state["log_alpha"].to(self.device).detach()
        self.update_step = int(state["update_step"])
        self._initial_digest = parameter_digest(self)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, host_obs, deterministic=True):
        if not deterministic:
            raise RuntimeError("冻结 Host 只允许确定性推理")
        return LocalHostSAC.select_action(self, host_obs, deterministic=True)

    def eval_mode(self):
        for name in ("actor", "critic", "target_critic"):
            getattr(self, name).eval()

    def train_mode(self):
        raise RuntimeError("冻结 Host 不允许进入训练模式")

    def update(self, *args, **kwargs):
        raise RuntimeError("冻结 Host 不允许更新")

    def assert_unchanged(self):
        if parameter_digest(self) != self._initial_digest:
            raise RuntimeError("共享 Host 参数或更新计数发生变化")

    def save(self, file_path):
        # Preserve existing scheduler checkpoint layout without serializing new Host states.
        self.assert_unchanged()
        if sha256(self.path) != self.file_hash:
            raise ValueError("源 Host checkpoint 已发生变化")
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(self.path, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, file_path, load_optimizers=False):
        # Existing full-scheduler resume checks the embedded copy; it cannot replace Host.
        if sha256(file_path) != self.file_hash:
            raise ValueError("续训 checkpoint 的 Host 与指定共享 Host 不一致")
        self.assert_unchanged()


class FrozenHostReplay:
    """A zero-storage sink; terminal traces and reward accounting still run."""
    capacity = 0

    def __init__(self, dc_id, **kwargs):
        self.dc_id = str(dc_id)

    def __len__(self):
        return 0

    def add(self, transition):
        if str(transition.dc_id) != self.dc_id:
            raise ValueError("Host 终局经验 DC 不一致")

    def can_sample(self, *args, **kwargs):
        return False


def load_frozen_hosts(env, checkpoint, device):
    directory, metadata = read_manifest(checkpoint)
    if host_structure(env) != metadata["structure"]:
        raise ValueError("共享 Host 的 DC、主机顺序/容量或观察量归一化与当前环境不兼容")
    reference = {"bundle_id": metadata["bundle_id"], "manifest_sha256": sha256(directory / "manifest.json")}
    # Network construction must not consume the Routing training random stream.
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        agents = {dc: FrozenHostPolicy(directory / "hosts" / f"{dc}.pt", device, reference)
                  for dc in metadata["structure"]["dc_ids"]}
    for dc, agent in agents.items():
        if {"obs_dim": agent.obs_dim, "action_dim": agent.action_dim} != metadata["structure"]["dimensions"][dc]:
            raise ValueError(f"Host 网络与 manifest 维度不一致：{dc}")
    return agents


def validate_scheduler_resume(resume, agents, settings):
    if not shared_host_mode(settings) or not resume:
        return
    model = project_path(resume)
    metadata = json.loads(model.with_suffix(".trainer.json").read_text(encoding="utf-8"))
    if metadata.get("shared_host") != host_reference(agents):
        raise ValueError("旧训练模式或不同 Host 的 checkpoint 不能直接续训")
    for dc, agent in agents.items():
        embedded = model.with_name(model.stem + "_hosts") / f"{dc}.pt"
        if not embedded.is_file() or sha256(embedded) != agent.file_hash:
            raise ValueError(f"续训 checkpoint 的 Host 文件损坏或已被替换：{dc}")
        agent.assert_unchanged()


def validate_frozen_host_context(env, agents, settings):
    if not shared_host_mode(settings):
        return
    directory, metadata = read_manifest(settings.host_checkpoint)
    if set(agents) != set(metadata["structure"]["dc_ids"]) or any(
            not isinstance(agent, FrozenHostPolicy) for agent in agents.values()):
        raise ValueError("评估必须使用统一加载的冻结 Host")
    reference = {"bundle_id": metadata["bundle_id"], "manifest_sha256": sha256(directory / "manifest.json")}
    if host_structure(env) != metadata["structure"] or host_reference(agents) != reference:
        raise ValueError("评估环境或 Host checkpoint 与训练使用的共享 Host 不一致")
    for agent in agents.values():
        agent.assert_unchanged()


def add_host_log_identity(row, agents):
    if agents and all(isinstance(agent, FrozenHostPolicy) for agent in agents.values()):
        reference = host_reference(agents)
        row.update(host_bundle_id=reference["bundle_id"],
                   host_manifest_sha256=reference["manifest_sha256"], host_frozen=True,
                   host_training_update_count=0)
    return row


def routing_guidance_lambda(settings, episode):
    """Warm up and decay over the Routing budget; Host is frozen throughout."""
    total = int(settings.routing_train_episodes)
    if total <= 1:
        return float(settings.guidance_lambda_stage3_end)
    progress = min(1.0, max(0.0, (int(episode) - 1) / (total - 1)))
    if progress <= 0.5:
        start, end, weight = settings.guidance_lambda_stage2_start, settings.guidance_lambda_stage2_peak, progress * 2
    else:
        start, end, weight = settings.guidance_lambda_stage2_peak, settings.guidance_lambda_stage3_end, (progress - 0.5) * 2
    return float(start + (end - start) * weight)


def export_environment(env, destination):
    """UseOldEnv-compatible snapshot with stable DC and Host ordering."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    graph = env.env_source.datacenter_graph
    dc_rows, host_rows = [], []
    for dc in env.base_datacenters:
        links = "; ".join(f"{neighbor}({float(graph[dc.dc_id][neighbor].get('weight', 0))}s)"
                          for neighbor in graph.neighbors(dc.dc_id))
        dc_rows.append({"datacenter_id": dc.dc_id, "cloud_latency_s": dc.cloud_latency,
                        "edge_topology_links": links})
        host_rows.extend({"host_id": host.host_id, "cpu_num": host.cpu_num,
                          "gpu_capacity_num": host.gpu_capacity_num, "datacenter_id": dc.dc_id}
                         for host in dc.host_list)
    for filename, rows in (("datacenters.csv", dc_rows), ("hosts.csv", host_rows)):
        with (destination / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_host_bundle(agents, env, destination, *, status, training, resume_state=None):
    """Commit an immutable directory; manifest is written after every file is complete."""
    target = project_path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Host checkpoint 不允许覆盖：{target}")
    staging = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    export_environment(env, staging / "environment")
    for dc, agent in agents.items():
        agent.save(staging / "hosts" / f"{dc}.pt")
    if resume_state is not None:
        _agents._atomic_torch_save(resume_state, staging / "training_state.pt")
    files = {str(file.relative_to(staging)).replace("\\", "/"): sha256(file)
             for file in staging.rglob("*") if file.is_file()}
    metadata = {"schema_version": SCHEMA_VERSION, "role": "shared_host",
                "bundle_id": uuid.uuid4().hex, "status": status,
                "structure": host_structure(env), "training": training, "files": files}
    atomic_json(metadata, staging / "manifest.json")
    read_manifest(staging, require_converged=False)
    os.replace(staging, target)
    return target
