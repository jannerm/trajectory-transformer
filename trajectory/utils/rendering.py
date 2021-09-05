import time
import sys
import numpy as np
import torch
import gym
import mujoco_py as mjc
import pdb

from .arrays import to_np
from .video import save_video, save_videos

def make_renderer(args):
    render_str = getattr(args, 'renderer')
    render_class = getattr(sys.modules[__name__], render_str)
    return render_class(args.dataset)

def split(sequence, observation_dim, action_dim):
    assert sequence.shape[1] == observation_dim + action_dim + 2
    observations = sequence[:, :observation_dim]
    actions = sequence[:, observation_dim:observation_dim+action_dim]
    rewards = sequence[:, -2]
    values = sequence[:, -1]
    return observations, actions, rewards, values

class Renderer:

    def __init__(self, env):
        if type(env) is str:
            self.env = gym.make(env).unwrapped
        else:
            self.env = env

        self.qpos_dim = self.env.sim.data.qpos.size
        self.qvel_dim = self.env.sim.data.qvel.size
        self.qstate_dim = self.qpos_dim + self.qvel_dim

        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)

    def can_pad_observation(self, observation):
        return observation.size == self.qpos_dim - 1 or observation.size == self.qstate_dim - 1

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256, qvel=True, render_kwargs=None):
        observation = to_np(observation)

        if render_kwargs is None:
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [0, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        # self.viewer.cam.trackbodyid = render_kwargs['trackbodyid']
        # self.viewer.cam.distance = render_kwargs['distance']
        # self.viewer.cam.lookat[:] = render_kwargs['lookat'][:]
        # self.viewer.cam.elevation = render_kwargs['elevation']


        if self.can_pad_observation(observation):
            state = self.pad_observation(observation)
        else:
            state = observation

        if not qvel:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        if type(dim) == int:
            dim = (dim, dim)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

    def render_plan(self, savepath, discretizer, state, sequence):
        '''
            state : np.array[ observation_dim ]
            sequence : np.array[ horizon x transition_dim ]
                as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        '''

        if len(sequence) == 1:
            # raise RuntimeError(f'horizon is 1 in Renderer:render_plan: {sequence.shape}')
            return

        sequence = to_np(sequence)

        actions = sequence[:-1, self.observation_dim : self.observation_dim + self.action_dim]

        # actions_np = to_np(actions[:-1])
        actions_recon = discretizer.reconstruct(
            actions, subslice=(self.observation_dim, self.observation_dim + self.action_dim)
        )

        # actions_np = np.stack([
        #     discretizer.reconstruct(act, subslice=(self.observation_dim, self.observation_dim + self.action_dim))
        #     for act in actions_np
        # ])
        rollout_states = rollout_from_state(self.env, state, actions_recon)

        render_sequences(savepath, discretizer, self, self.action_dim, sequence, reference=rollout_states)

        # pdb.set_trace()

        # rollout_states = rollout_from_state(self.env, state, actions)
        # # render_sequences(savepath, dataset, self, action_dim)

        # actions_np = utils.to_np(sample_actions[argmax,:-1])
        # actions_np = np.stack([
        #     discretizer.reconstruct(act, subslice=(observation_dim, observation_dim + action_dim))
        #     for act in actions_np
        # ])
        # rollout_states = utils.rollout_from_state(env_vis, env.state_vector(), actions_np)
        # savepath = os.path.join(args.savepath, f'{t}_plan.mp4')
        # utils.render_sequences(savepath, dataset, renderer, action_dim, utils.to_np(best_resequences), reference=rollout_states)

    def render_rollout(self, savepath, states, **video_kwargs):
        images = self(states) #np.stack(states, axis=0))
        save_video(savepath, images, **video_kwargs)

class KitchenRenderer:

    def __init__(self, env):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env

        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)

    def set_obs(self, obs, goal_dim=30):
        robot_dim = self.env.n_jnt
        obj_dim = self.env.n_obj
        assert robot_dim + obj_dim + goal_dim == obs.size or robot_dim + obj_dim == obs.size
        self.env.sim.data.qpos[:robot_dim] = obs[:robot_dim]
        self.env.sim.data.qpos[robot_dim:robot_dim+obj_dim] = obs[robot_dim:robot_dim+obj_dim]
        self.env.sim.forward()

    def rollout(self, obs, actions):
        self.set_obs(obs)
        observations = [env._get_obs()]
        for act in actions:
            obs, rew, term, _ = env.step(act)
            observations.append(obs)
            if term:
                break
        for i in range(len(observations), len(actions)+1):
            ## if terminated early, pad with zeros
            observations.append( np.zeros(observations[-1].size) )
        return np.stack(observations)

    def render(self, observation, dim=512, onscreen=False):
        self.env.sim_robot.renderer._camera_settings.update({
            'distance': 4.5,
            'azimuth': 90,
            'elevation': -25,
            'lookat': [0, 1, 2],
        })
        self.set_obs(observation)
        if onscreen:
            self.env.render()
        return self.env.sim_robot.renderer.render_offscreen(dim, dim)

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    # def render_plan(self, savepath, discretizer, observation, sequence):
    #     '''
    #         state : np.array[ observation_dim ]
    #         sequence : np.array[ horizon x transition_dim ]
    #             as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
    #     '''

    #     if len(sequence) == 1:
    #         return

    #     sequence = to_np(sequence)

    #     actions = sequence[:-1, self.observation_dim : self.observation_dim + self.action_dim]

    #     actions_recon = discretizer.reconstruct(
    #         actions, subslice=(self.observation_dim, self.observation_dim + self.action_dim)
    #     )

    #     # rollout_states = rollout_from_state(self.env, state, actions_recon)
    #     rollout_observations = self.rollout(observation, actions_recon)

    #     render_sequences(savepath, discretizer, self, self.action_dim, sequence, reference=rollout_observations)

    def render_plan(self, *args, **kwargs):
        return self.render_rollout(*args, **kwargs)

    def render_rollout(self, savepath, states, **video_kwargs):
        images = self(states) #np.stack(states, axis=0))
        save_video(savepath, images, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

class DebugRenderer:

    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        return np.zeros((10, 10, 3))

    def render_plan(self, *args, **kwargs):
        pass

    def render_rollout(self, *args, **kwargs):
        pass

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size

    if state.size == qpos_dim:
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def get_body_pos(env, state, bodies=[]):
    set_state(env, state)
    bodies = bodies or env.sim.model.geom_names
    positions = {}
    for body in bodies:
        pos = env.sim.data.get_geom_xpos(body)
        positions[body] = pos.copy()
    return positions

def get_body_positions(env, states, bodies=[]):
    positions = []
    for state in states:
        pos_dict = get_body_pos(env, state, bodies=bodies)
        positions.append(pos_dict)
    positions = {
        k: np.stack([
            pos_dict[k] for pos_dict in positions
        ], axis=0)
        for k in positions[0].keys()
    }
    return positions

def to_np(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    else:
        return x

BOUNDS = {
    'antmaze-umaze-v0': (-3, 11),
    'antmaze-medium-play-v0': (-3, 23),
    'antmaze-medium-diverse-v0': (-3, 23),
    'antmaze-large-play-v0': (-3, 39),
    'antmaze-large-diverse-v0': (-3, 39),
}

import matplotlib.pyplot as plt

class AntMazeRenderer:

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)

    def renders(self, savepath, X):
        plt.clf()

        if X.ndim < 3:
            X = X[None]

        N, path_length, _ = X.shape
        if N > 4:
            fig, axes = plt.subplots(4, int(N/4))
            axes = axes.flatten()
            fig.set_size_inches(N/4,8)
        elif N > 1:
            fig, axes = plt.subplots(1, N)
            fig.set_size_inches(8,8)
        else:
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(8,8)

        # X = X.detach().cpu().numpy()

        colors = plt.cm.jet(np.linspace(0,1,path_length))
        for i in range(N):
            ax = axes if N == 1 else axes[i]
            xlim, ylim = self.plot_boundaries(ax=ax)
            x = X[i]
            # if i == N - 1:
            #     import d4rl
            #     dset = d4rl.qlearning_dataset(env)
            #     x = dset['observations']
            #     colors = 'red'
            ax.scatter(x[:,0], x[:,1], c=colors)
            # if i == N - 1:
            #     ax.scatter([20], [20], c='blue')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        plt.savefig(savepath + '.png')
        plt.close()
        print(f'[ attentive/utils/visualization ] Saved to: {savepath}')

    def plot_boundaries(self, N=100, ax=None):
        """
            plots the maze boundaries in the antmaze environments
        """
        ax = ax or plt.gca()

        xlim = BOUNDS[self.env_name] #(-3, 11)
        ylim = BOUNDS[self.env_name] # (-3, 11)

        X = np.linspace(*xlim, N)
        Y = np.linspace(*ylim, N)

        Z = np.zeros((N, N))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                collision = self.env.unwrapped._is_in_collision((x, y))
                Z[-j, i] = collision

        ax.imshow(Z, extent=(*xlim, *ylim), aspect='auto', cmap=plt.cm.binary)
        # ax.invert_yaxis()
        return xlim, ylim

    def render_plan(self, savepath, discretizer, state, sequence):
        '''
            state : np.array[ observation_dim ]
            sequence : np.array[ horizon x transition_dim ]
                as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        '''

        if len(sequence) == 1:
            # raise RuntimeError(f'horizon is 1 in Renderer:render_plan: {sequence.shape}')
            return

        sequence = to_np(sequence)

        sequence_recon = discretizer.reconstruct(sequence)

        observations, actions, *_ = split(sequence_recon, self.observation_dim, self.action_dim)

        rollout_states = rollout_from_state(self.env, state, actions[:-1])

        X = np.stack([observations, rollout_states], axis=0)

        self.renders(savepath, X)

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.stack(states, axis=0)[None]
        images = self.renders(savepath, states)

class Maze2dRenderer(AntMazeRenderer):

    def _is_in_collision(self, x, y):
        '''
            10 : wall
            11 : free
            12 : goal
        '''
        maze = self.env.maze_arr
        ind = maze[int(x), int(y)]
        return ind == 10

    def plot_boundaries(self, N=100, ax=None, eps=1e-6):
        """
            plots the maze boundaries in the antmaze environments
        """
        ax = ax or plt.gca()

        maze = self.env.maze_arr
        xlim = (0, maze.shape[1]-eps)
        ylim = (0, maze.shape[0]-eps)
        # xlim = BOUNDS[self.env_name] #(-3, 11)
        # ylim = BOUNDS[self.env_name] # (-3, 11)

        X = np.linspace(*xlim, N)
        Y = np.linspace(*ylim, N)

        Z = np.zeros((N, N))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                collision = self._is_in_collision(x, y)
                Z[-j, i] = collision

        ax.imshow(Z, extent=(*xlim, *ylim), aspect='auto', cmap=plt.cm.binary)
        # ax.invert_yaxis()
        return xlim, ylim

    def renders(self, savepath, X):
        return super().renders(savepath, X + 0.5)

#--------------------------------- planning callbacks ---------------------------------#

def render_sequences(savepath, discretizer, renderer, action_dim, *sequences, reference=None):
    images_l = []
    for sequence in sequences:
        recon = discretizer.reconstruct(sequence)
        images = renderer(recon[:,:-action_dim-2])
        images_l.append(images)
    if reference is not None:
        images = renderer(reference)
        images_l.append(images)

    save_videos(savepath, *images_l, fps=30)

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)

if __name__ == '__main__':
    import imageio
    import gym
    import d4rl

    env = gym.make('hopper-medium-expert-v2')
    obs = env.reset()

    renderer = Renderer('hopper-medium-expert-v2')
    img = renderer.render(obs)

    imageio.imsave('logs/render/hopper.png', img)
