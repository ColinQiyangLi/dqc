import glob
import os
import json
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_ogbench_env_and_datasets
from utils.datasets import Dataset, CGCDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'humanoidmaze-giant-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_integer('dataset_size', None, 'Size of the dataset.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 250000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/dqc.py', lock_config=False)

flags.DEFINE_string('tags', 'Default', 'Wandb tag.')

# overriding evaluation parameters
flags.DEFINE_string('action_chunk_eval_sizes', '0', 'separated by commas. 0 is the default. ' \
    'For example: "0,1,5" evaluates with the full policy chunk size first, and then with only ' \
    'the first action in the chunk, and finally with the first 5 actions in the chunk.')
flags.DEFINE_string('best_of_n_eval_values', '0',   'separated by commas. 0 is the default. ' \
    'For example: "0,8,128" evaluates with the agent.best_of_n first, and then with 8 action samples, ' \
    'and finally with 128 action samples.')

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='dqc', group=FLAGS.run_group, name=exp_name, tags=FLAGS.tags.split(","), mode="offline")

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    if FLAGS.dataset_dir is None:
        datasets = [None]
    elif FLAGS.dataset_dir.endswith('npz'):
        # Dataset file.
        datasets = [FLAGS.dataset_dir]
    else:
        # Dataset directory.
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    
    if FLAGS.num_datasets is not None:
        datasets = datasets[:FLAGS.num_datasets]
    
    dataset_idx = 0
    env, train_dataset, val_dataset = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=datasets[0], compact_dataset=True, add_info=True)
    eval_env = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=datasets[0], compact_dataset=True, env_only=True)
    if FLAGS.dataset_size is not None:
        train_dataset = Dataset.create(**{k: v[:FLAGS.dataset_size] for k, v in train_dataset.items()})
    else:
        train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    print(train_dataset.keys())
    print(train_dataset.size)

    env.reset()

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps)))
    if val_dataset is not None:
        val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))
    
    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class_dict = {'CGCDataset': CGCDataset,}
    example_transition = {k: v[0] for k, v in train_dataset.items()}
    example_transition.pop("oracle_reps")
    state = env.get_state()
    example_transition = {**example_transition, **state}
    print(example_transition.keys())

    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)

    example_batch = train_dataset.sample(1)


    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    metric_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    action_dim = example_batch["actions"].shape[-1]
    for i in tqdm.tqdm(range(0, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)
        
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            if hasattr(agent, 'compute_metrics'):
                metric_rng, key = jax.random.split(metric_rng)
                add_metrics = agent.compute_metrics(batch, rng=key)
                update_info.update({f'metrics/{k}': v for k, v in add_metrics.items()})
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = len(task_infos)
            
            action_chunk_eval_sizes = list(map(int, FLAGS.action_chunk_eval_sizes.split(",")))
            
            eval_metrics = {}
            for action_chunk_eval_size in action_chunk_eval_sizes:
                renders = []
                if action_chunk_eval_size == 0:
                    action_chunk_eval_size = None
                    suffix = ""
                else:
                    if action_chunk_eval_size >= config["policy_chunk_size"]: 
                        print("skipping", action_chunk_eval_size, f"because horizon_length is too short")
                        continue
                    suffix = f"-ac-{action_chunk_eval_size}"
                
                if action_chunk_eval_size is not None:
                    print(f"evaluating with action chunk size of {action_chunk_eval_size}")

                for eval_bfn in list(map(int, FLAGS.best_of_n_eval_values.split(","))):
                    overall_metrics = defaultdict(list)
                    for task_id in tqdm.trange(1, num_tasks + 1):
                        task_name = task_infos[task_id - 1]['task_name']

                        eval_info, _, cur_renders = evaluate(
                            agent=agent,
                            agent_name=config['agent_name'],
                            env=eval_env,
                            goal_conditioned=True,
                            task_id=task_id,
                            num_eval_episodes=FLAGS.eval_episodes,
                            num_video_episodes=FLAGS.video_episodes,
                            video_frame_skip=FLAGS.video_frame_skip,
                            action_dim=action_dim,
                            action_chunk_eval_size=action_chunk_eval_size,
                            best_of_n_override=eval_bfn if eval_bfn != 0 else None
                        )
                        renders.extend(cur_renders)
                        metric_names = ['success']

                        if eval_bfn == 0:
                            eval_metrics.update(
                                {f'evaluation{suffix}/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                            )
                        else:
                            eval_metrics.update(
                                {f'evaluation{suffix}-bfn{eval_bfn}/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                            )
                        for k, v in eval_info.items():
                            if k in metric_names:
                                overall_metrics[k].append(v)
                    
                    if eval_bfn == 0:
                        for k, v in overall_metrics.items():
                            eval_metrics[f'evaluation{suffix}/overall_{k}'] = np.mean(v)
                    else:
                        for k, v in overall_metrics.items():
                            eval_metrics[f'evaluation{suffix}-bfn{eval_bfn}/overall_{k}'] = np.mean(v)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            if len(datasets) > 1:
                print(f'Using new dataset: {datasets[dataset_idx]}', flush=True)
                dataset_idx = (dataset_idx + 1) % len(datasets)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name, dataset_path=datasets[dataset_idx], 
                    compact_dataset=True, dataset_only=True, cur_env=env)
                dataset_class = dataset_class_dict[config['dataset_class']]
                train_dataset = dataset_class(Dataset.create(**train_dataset), config)

    train_logger.close()
    eval_logger.close()

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
