from collections import defaultdict

import jax
import numpy as np
from tqdm import trange

from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    agent_name,
    env,
    goal_conditioned=False,
    task_id=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    action_dim=None,
    action_chunk_eval_size=None,
    best_of_n_override=None
    # max_episode_length=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        agent_name: Agent name.
        env: Environment.
        goal_conditioned: Whether to do goal-conditioned evaluation.
        task_id: Task ID to be passed to the environment (only used when goal_conditioned is True).
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        
    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if best_of_n_override is None:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    else:
        actor_fn = supply_rng(partial(agent.sample_actions, best_of_n_override=best_of_n_override), rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)


    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if goal_conditioned:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
        else:
            observation, info = env.reset()
            goal = None
            goal_frame = None
        done = False
        step = 0
        render = []
        action_queue = []

        while not done:
            if len(action_queue) == 0:
                v_observation = observation
                if agent_name.startswith('h'):
                    action, _ = actor_fn(observations=v_observation, goals=goal)
                else:
                    action = actor_fn(observations=v_observation, goals=goal)
                action = np.array(action)
                action = np.clip(action, -1, 1)

                action = np.array(action).reshape(-1, action_dim)
                for a in action:
                    action_queue.append(a)
                
                # only execute the first action_chunk_eval_size actions
                if action_chunk_eval_size is not None:
                    action_queue = action_queue[:action_chunk_eval_size]

            action = action_queue.pop(0)

            next_observation, reward, terminated, truncated, info = env.step(action)

                
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        # print(step)
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
