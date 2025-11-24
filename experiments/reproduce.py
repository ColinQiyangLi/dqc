"""Generate launch scripts for reproducing parameter main experiments."""

import os

from generate import SbatchGenerator
from typing import NamedTuple

run_group = "dqc-reproduce"
dataset_root = ...  # TODO: fill in the root directory of your dataset

num_jobs_per_gpu = 1
gpu_limit = 16

domains = [
    "cube-triple-play-oraclerep-v0",
    "cube-quadruple-play-oraclerep-v0", 
    "cube-octuple-play-oraclerep-v0", 
    "humanoidmaze-giant-navigate-oraclerep-v0",
    "puzzle-4x5-play-oraclerep-v0", 
    "puzzle-4x6-play-oraclerep-v0", 
]

sizes = {
    "cube-triple-play-oraclerep-v0": "100m",
    "cube-quadruple-play-oraclerep-v0": "100m",
    "cube-octuple-play-oraclerep-v0": "1b",
    "humanoidmaze-giant-navigate-oraclerep-v0": None,
    "puzzle-4x5-play-oraclerep-v0": None,
    "puzzle-4x6-play-oraclerep-v0": "1b",
}

class HorizonConfig(NamedTuple):
    critic_chunking: bool
    backup_horizon: int
    policy_chunk_size: int

params = {
    # DQC (h=25, h_a=5)
    HorizonConfig(critic_chunking=True, backup_horizon=25, policy_chunk_size=5): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.93, kappa_d=0.8),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.93, kappa_d=0.8),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93, kappa_d=0.5),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5,  kappa_d=0.8),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.9,  kappa_d=0.5),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7,  kappa_d=0.5),
    },
    # QC-NS (h=25, h_a=5)
    HorizonConfig(critic_chunking=False, backup_horizon=25, policy_chunk_size=5): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.93),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.93),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.7),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.5),
    },
    # QC (h=25, h_a=25).
    HorizonConfig(critic_chunking=False, backup_horizon=25, policy_chunk_size=25): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.93),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.93),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.9),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7),
    },
    # DQC (h=25, h_a=1)
    HorizonConfig(critic_chunking=True, backup_horizon=25, policy_chunk_size=1): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.93, kappa_d=0.8),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.93, kappa_d=0.8),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93, kappa_d=0.5),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5,  kappa_d=0.8),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.9,  kappa_d=0.5),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7,  kappa_d=0.5),
    },
    # NS (n=25).
    HorizonConfig(critic_chunking=False, backup_horizon=25, policy_chunk_size=1): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.5),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.5),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.97),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.7),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7),
    },
    # DQC (h=5, h_a=1)
    HorizonConfig(critic_chunking=True, backup_horizon=5, policy_chunk_size=1): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.5, kappa_d=0.8),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.5, kappa_d=0.8),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93, kappa_d=0.5),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5,  kappa_d=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.5,  kappa_d=0.5),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.5,  kappa_d=0.5),
    },
    # QC (h=5, h_a=5).
    HorizonConfig(critic_chunking=False, backup_horizon=5, policy_chunk_size=5): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.93),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.93),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.93),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.9),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7),
    },
    # NS (n=5).
    HorizonConfig(critic_chunking=False, backup_horizon=5, policy_chunk_size=1): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.5),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.7),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.5),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.5),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.5),
    },
    # OS.
    HorizonConfig(critic_chunking=False, backup_horizon=1, policy_chunk_size=1): {
        "cube-triple-play-oraclerep-v0":               dict(kappa_b=0.5),
        "cube-quadruple-play-oraclerep-v0":            dict(kappa_b=0.7),
        "cube-octuple-play-oraclerep-v0":              dict(kappa_b=0.7),
        "humanoidmaze-giant-navigate-oraclerep-v0":    dict(kappa_b=0.5),
        "puzzle-4x5-play-oraclerep-v0":                dict(kappa_b=0.7),
        "puzzle-4x6-play-oraclerep-v0":                dict(kappa_b=0.7),
    },
}

for debug in [True, False]:
    gen = SbatchGenerator(j=num_jobs_per_gpu, limit=gpu_limit, prefix=("MUJOCO_GL=egl", "python main.py"), comment=run_group)
    if debug:
        gen.add_common_prefix({"run_group": run_group + "_debug", "offline_steps": 100, "eval_episodes": 0, 
                               "video_episodes": 0, "eval_interval": 20, "log_interval": 10, "dataset_replace_interval": 10})
    else:
        gen.add_common_prefix({"run_group": run_group, "offline_steps": 1000000, "eval_interval": 250000})

    for seed in [100001, 200002, 300003, 400004, 500005, 600006, 700007, 800008, 900009, 1000010]:  # 20002-60006
        for domain in domains:

            # environment-specific parameters
            if "humanoid" in domain:
                extra_kwargs = {"agent.q_agg": "mean"}
            elif "cube" in domain:
                extra_kwargs = {"agent.q_agg": "min"}
            elif "puzzle" in domain:
                extra_kwargs = {"agent.q_agg": "mean"}

            size = sizes[domain]
            if size is not None and not debug:
                if "puzzle-4x6" in domain:
                    extra_kwargs["dataset_dir"] = os.path.join(dataset_root, "puzzle-4x6-play-{size}-v0")
                if "cube-quadruple" in domain:
                    extra_kwargs["dataset_dir"] = os.path.join(dataset_root, "cube-quadruple-play-{size}-v0")
                if "cube-triple" in domain:
                    extra_kwargs["dataset_dir"] = os.path.join(dataset_root, "cube-triple-play-{size}-v0")
                if "cube-octuple" in domain:
                    extra_kwargs["dataset_dir"] = os.path.join(dataset_root, "cube-octuple-play-{size}-v0")

            # (h, ha) configurations
            for backup_horizon in [1, 5, 25]:
                for policy_chunk_size in [1, 5, 25]:
                    if policy_chunk_size > backup_horizon: continue
                    
                    for critic_chunking in [True, False]:
                        if policy_chunk_size == backup_horizon and critic_chunking: continue # can't dqc if they are the same chunk size
                        
                        kwargs = {
                            "seed": seed,
                            "agent": "agents/dqc.py",
                            "agent.num_qs": 2,
                            "agent.policy_chunk_size": policy_chunk_size,
                            "agent.backup_horizon": backup_horizon,
                            "agent.use_chunk_critic": critic_chunking,
                            "agent.distill_method": "expectile",
                            "agent.implicit_backup_type": "quantile",
                            "env_name": domain,
                            **extra_kwargs,
                        }

                        key = HorizonConfig(critic_chunking=critic_chunking, backup_horizon=backup_horizon, policy_chunk_size=policy_chunk_size)
                        configs = params[key]
                        
                        print(domain, key)
                        for k, v in configs[domain].items():
                            kwargs[f"agent.{k}"] = v
                            print("setting", k, "to", v)
                        
                        if debug:
                            kwargs["agent.batch_size"] = 8

                        if policy_chunk_size != backup_horizon and critic_chunking:
                            kwargs["tags"] = f'"DQC,h={backup_horizon},ha={policy_chunk_size}"'
                        elif policy_chunk_size != backup_horizon and not critic_chunking and policy_chunk_size != 1:
                            kwargs["tags"] = f'"QC-NS,h={backup_horizon},ha={policy_chunk_size}"'
                        elif backup_horizon == 1:
                            kwargs["tags"] = "OS"
                        elif policy_chunk_size == backup_horizon:
                            kwargs["tags"] = f'"QC,h={backup_horizon}"'
                            kwargs["action_chunk_eval_sizes"] = "0,1,5" # DQC-naive
                        else:
                            kwargs["tags"] = f'"NS,n={backup_horizon}"'
                        print(kwargs["tags"])
                        gen.add_run(kwargs)

    sbatch_str = gen.generate_str()
    if debug:
        with open(f"{run_group}_debug.sh", "w") as f:
            f.write(sbatch_str)
    else:
        with open(f"{run_group}.sh", "w") as f:
            f.write(sbatch_str)
