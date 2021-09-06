import numpy as np
import torch
import pdb

from ..utils.arrays import to_torch

VALUE_PLACEHOLDER = 1e6

def make_prefix(discretizer, context, obs, prefix_context=True):
    observation_dim = obs.size
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(context, discretizer, observation, action, reward, max_context_transitions):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context