import baselines
from dac.util.learning_rate import LearningRate
from dac.util.replay_buffer import ReplayBuffer
from dac.networks.adversary import Discriminator
from dac.networks.TD3 import TD3
from dac.dataset.mujoco_dset import Mujoco_Dset
import gym
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--traj_num', help='Number of Traj', type=int, default=4)
    return parser.parse_args()



def main(args):
	env = gym.make(args.env_id)
	expert_buffer = Mujoco_Dset(env, args.expert_path, args.traj_num)
	actor_replay_buffer = ReplayBuffer(env)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	lr = LearningRate.getInstance()
	lr.setLR(10e-3)
	lr.setDecay(1.0/2.0)

	td3 = TD3(state_dim, action_dim, max_action, 40, 10e5) #state_dim, action_dim, max_action, actor_clipping, decay_steps

	discriminator = Discriminator(state_dim + action_dim)

	batch_size = 100
	num_steps = 10e7 # 1 million timesteps
	T = 1000 # Trajectory length == T in the pseudo-code

	for i in range(num_steps / (batch_size*T)):
		# Sample from policy
		obs = env.reset()
		for j in range(T):
			action = td3.sample(obs)
			next_state, reward, done, _ = env.step(action)
			actor_replay_buffer.add((obs, action, next_state), done)
			if done:
				obs = env.reset()
			else:
				obs = next_state

		discriminator.train(actor_replay_buffer, expert_buffer, T, batch_size)

		td3.train(discriminator, actor_replay_buffer, T, batch_size) #discriminator, replay_buf, iterations, batch_size=100


if __name__ == '__main__':
	args = argsparser()
	main(args)