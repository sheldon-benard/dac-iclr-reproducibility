'''
We modified the mujoco_dset.py file from openAI/baselines/gail/dataset.
'''



'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, weights, num_traj):
        self.inputs = inputs
        self.labels = labels
        self.num_traj = num_traj
        assert len(self.inputs) == len(self.labels)
        assert len(self.inputs) == len(weights)
        assert self.num_traj > 0
        #Calc probabilities from weights

        #If we have n samples (original states and newly added absorbing states)
        #and t trajectories -> t absorbing states

        # Thus, let p be probability of original sample
        # p*(n-t) + p*t/n = 1

        n = len(weights)
        t = self.num_traj

        p = 1.0/(n-t + t/n)
        self.probabilities = p * weights
        self.indicies = np.arange(len(inputs))

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.inputs)
        indicies = np.random.choice(self.indicies, batch_size, p=self.probabilities, replace=False)

        return self.inputs[indicies, :], self.labels[indicies, :]

def WrapAbsorbingState(env, obs, acs):
    obs_space = env.observation_space
    acs_space = env.action_space

    # Use zero matrix as generic absorbing state
    absorbing_state = np.zeros((1,obs_space.shape[0]),dtype=np.float32)
    random_action = np.zeros_like(acs_space.sample(),dtype=np.float32).reshape(1, acs_space.shape[0])

    # At terminal state obs[-1], under action acs[-1] we go to absorbing state.
    new_obs = np.concatenate((obs, absorbing_state))
    new_acs = np.concatenate((acs, random_action))

    return new_obs, new_acs

class Mujoco_Dset(object):
    def __init__(self, env, expert_path, train_fraction=0.7, traj_limitation=-1):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        temp_obs = traj_data['obs'][:traj_limitation]
        temp_acs = traj_data['acs'][:traj_limitation]

        obs = np.zeros((temp_obs.shape[0], temp_obs.shape[1] + 1, temp_obs.shape[2]))
        acs = np.zeros((temp_acs.shape[0], temp_acs.shape[1] + 1, temp_acs.shape[2]))
        n = traj_limitation * (temp_obs.shape[1] + 1)
        weights = [1 for i in range(n)]

        for i in range(temp_obs.shape[0]):
            obs[i], acs[i] = WrapAbsorbingState(env, temp_obs[i], temp_acs[i])
            weights[len(obs[i]) - 1] = 1/n

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.dset = Dset(self.obs, self.acs, weights, traj_limitation)
        # for behavior cloning
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
