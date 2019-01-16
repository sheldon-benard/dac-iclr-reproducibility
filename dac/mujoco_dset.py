"""
We modified the mujoco_dset.py file from openAI/baselines/gail/dataset and used the load_dataset() function
from openAI/imitation/scripts/imitate_mj.py
"""

import h5py
import numpy as np


def load_dataset(filename, limit_trajs, data_subsamp_freq=1):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by vis_mj.py
        full_dset_size = f['obs_B_T_Do'].shape[0]  # full dataset size
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size, ...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size, ...][...]
        exr_B_T = f['r_B_T'][:dset_size, ...][...]
        exlen_B = f['len_B'][:dset_size, ...][...]

    return exobs_B_T_Do, exa_B_T_Da, exr_B_T


class Dset(object):
    def __init__(self, obs, acs, num_traj, absorbing_state, absorbing_action):
        self.obs = obs
        self.acs = acs
        self.num_traj = num_traj
        assert len(self.obs) == len(self.acs)
        assert self.num_traj > 0
        self.steps_per_traj = int(len(self.obs) / num_traj)

        self.absorbing_state = absorbing_state
        self.absorbing_action = absorbing_action

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.obs)
        num_samples_per_traj = int(batch_size / self.num_traj)
        assert num_samples_per_traj * self.num_traj == batch_size
        N = self.steps_per_traj / num_samples_per_traj  # This is the importance weight for
        j = num_samples_per_traj
        num_samples_per_traj = num_samples_per_traj - 1  # make room for absorbing

        obs = None
        acs = None
        weights = [1 for i in range(batch_size)]
        while j <= batch_size:
            weights[j - 1] = 1.0 / N
            j = j + num_samples_per_traj + 1

        for i in range(self.num_traj):
            indicies = np.sort(
                np.random.choice(range(self.steps_per_traj * i, self.steps_per_traj * (i + 1)), num_samples_per_traj,
                                 replace=False))

            if obs is None:
                obs = np.concatenate((self.obs[indicies, :], self.absorbing_state), axis=0)
            else:
                obs = np.concatenate((obs, self.obs[indicies, :], self.absorbing_state), axis=0)

            if acs is None:
                acs = np.concatenate((self.acs[indicies, :], self.absorbing_action), axis=0)
            else:
                acs = np.concatenate((acs, self.acs[indicies, :], self.absorbing_action), axis=0)

        return obs, acs, weights


# This takes in 1 trajectory's observations and actions
# and adds absorbing state 0-vector with same dimension as observation_space
# and add absorbing action 0-vector with same dimension as action_space
# def WrapAbsorbingState(env, obs, acs):
#     obs_space = env.observation_space
#     acs_space = env.action_space
#
#     # Use zero matrix as generic absorbing state
#     absorbing_state = np.zeros((1,obs_space.shape[0]),dtype=np.float32)
#     zero_action = np.zeros_like(acs_space.sample(),dtype=np.float32).reshape(1, acs_space.shape[0])
#
#     # At terminal state obs[-1], under action acs[-1] we go to absorbing state.
#     new_obs = np.concatenate((obs, absorbing_state))
#     new_acs = np.concatenate((acs, random_action))
#
#     return new_obs, new_acs

class Mujoco_Dset(object):
    def __init__(self, env, expert_path, traj_limitation=-1):
        obs, acs, rets = load_dataset(expert_path, traj_limitation)

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = rets.sum(axis=1)
        self.avg_ret = sum(self.rets) / len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = len(rets)
        self.num_transition = len(self.obs)

        absorbing_state = np.zeros((1, env.observation_space.shape[0]), dtype=np.float32)
        zero_action = np.zeros_like(env.action_space.sample(), dtype=np.float32).reshape(1, env.action_space.shape[0])
        self.dset = Dset(self.obs, self.acs, traj_limitation, absorbing_state, zero_action)
        self.log_info()

    def log_info(self):
        print("Total trajs: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average returns: %f" % self.avg_ret)
        print("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size):
        return self.dset.get_next_batch(batch_size)
