import os
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

# def construct_dataloader(dataset, **kwargs):
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
#     return dataloader

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += dataset['timeouts'][i] #+1]

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }

def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env
