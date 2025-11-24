"""Generate launch scripts for reproducing parameter sensitivity analysis experiments."""

import os

from generate import SbatchGenerator
from typing import NamedTuple

run_group = "dqc-reproduce-sensitivity"
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

    for seed in [100001, 200002, 300003, 400004, 500005, 600006, 700007, 800008, 900009, 1000010]:
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
            for backup_horizon in [25]:
                
                if "human" in domain or "puzzle-4x6" in domain:
                    policy_chunk_size = 1
                else:
                    policy_chunk_size = 5
                critic_chunking = True

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


                for bs in [256, 1024]:
                    kwargs["agent.batch_size"] = bs
                    kwargs["tags"] = f'"DQC,sen-bs={bs},h={backup_horizon},ha={policy_chunk_size}"'
                    print(kwargs["tags"])
                    gen.add_run(kwargs)

                # set it back
                kwargs["agent.batch_size"] = 4096

                if "cube-quadruple" in domain:
                    assert kwargs["agent.kappa_d"] == 0.8 and kwargs["agent.kappa_b"] == 0.93
                    # tau sensitivity
                    for kappa_b in [0.5, 0.93]:
                        for kappa_d in [0.5, 0.8]:
                            if kappa_b == 0.93 and kappa_d == 0.8:
                                continue  # already covered
                            kwargs["agent.kappa_b"] = kappa_b
                            kwargs["agent.kappa_d"] = kappa_d
                    
                            kwargs["agent.backup_horizon"] = 25
                            kwargs["agent.policy_chunk_size"] = 5
                            kwargs["tags"] = f'"DQC,kappa-sen,h=25,ah=5,bt={kappa_b},dt={kappa_d}"'
                            gen.add_run(kwargs)

                    # expectile vs. quantile
                    kwargs["agent.kappa_d"] = 0.8
                    kwargs["agent.kappa_b"] = 0.93
                    
                    kwargs["agent.backup_horizon"] = 25
                    kwargs["agent.policy_chunk_size"] = 5

                    for bt in ["quantile", "expectile"]:
                        for dt in ["quantile", "expectile"]:
                            if bt == "quantile" and dt == "quantile":
                                continue  # already covered
                            kwargs["agent.implicit_backup_type"] = bt
                            if dt == "quantile":
                                kwargs["agent.distill_method"] = "quantile"
                            else:
                                kwargs["agent.distill_method"] = "expectile"
                            kwargs["tags"] = f'"DQC,qe-sen,h=25,ah=5,bt={bt},dt={dt}"'
                            gen.add_run(kwargs)

                    # best-of-n sensitivity
                    kwargs["agent.kappa_d"] = 0.8
                    kwargs["agent.kappa_b"] = 0.93
                    kwargs["agent.distill_method"] = "expectile"
                    kwargs["agent.implicit_backup_type"] = "quantile"
                    kwargs["agent.backup_horizon"] = 25
                    kwargs["agent.policy_chunk_size"] = 5
                    kwargs["best_of_n_eval_values"] = "4,8,16,64,128"
                    kwargs["tags"] = f'"DQC,bfn-sen,h=25,ah=5,bt=0.93,dt=0.8"'
                    gen.add_run(kwargs)

    sbatch_str = gen.generate_str()
    if debug:
        with open(f"{run_group}_debug.sh", "w") as f:
            f.write(sbatch_str)
    else:
        with open(f"{run_group}.sh", "w") as f:
            f.write(sbatch_str)
