import numpy as np
import torch
import gym
import argparse
import os

import utils
#import TD3
#import OurDDPG
import DDPG

import gwr_replay


import sys
import resource
# limit the memory
def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.85, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			#why the np.array() ???
			#action = policy.select_action(np.array(state))

			#can this work? when only trained on latent representation?
			action = policy.select_action(state)

			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	#memory_limit() # Limitates maximun memory usage to 70%

	# added Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--a_t', type=float, help='Activation threshold', default=.95)
	#parser.add_argument('--batch_size', type=int, help='Batch size for training', default=8)
	parser.add_argument('--batch_size_ft', type=int, help='Batch size for fine tuning of feature extractors. '
															'E.g. BERT for NLP', default=32)
	parser.add_argument('--datasets', '-d', nargs='+', type=str, help='Datasets used for training', default=['IRIS'])
	parser.add_argument('--sem_a_t', type=float, help='Activation threshold for semantic memory in GDM model',
						default=.35)
	parser.add_argument('--del_freq', type=int, help='Deletion frequency in SOINN network', default=10)
	parser.add_argument('--delta_plus', type=float, help='Positive Label change rate', default=1)
	parser.add_argument('--delta_minus', type=float, help='Negative Label change rate', default=0.1)
	parser.add_argument('--dim', type=int, help='Dimension of growing memory and language model output', default=17)
	parser.add_argument('--eps_b', type=float, help='Learning rate for weight adaption (BMU)', default=.1)
	parser.add_argument('--eps_n', type=float, help='Learning rate for weight adaption of (sample)', default=.001)
	parser.add_argument('--h_t', type=float, help='Habituation/Firing threshold', default=.3)
	parser.add_argument('--gamma', type=float, help='Learning rate for label weight adaptation', default=.5)
	parser.add_argument('--kappa', type=float, help='Habituation controlling parameter', default=1.05)
	parser.add_argument('--learner', type=str, help='Learner method', default='SOINNPLUS')
	parser.add_argument('--load_latest', type=bool, help='Load latest pretrained model from file', default=False)
	parser.add_argument('--log_freq', type=int, help='Logging frequency of learning metrics', default=1)
	parser.add_argument('--lr', type=float, help='Learning rate (language model/feature extractor)', default=3e-5)
	parser.add_argument('--l_t', type=float, help='Label propagation threshold', default=.5)
	parser.add_argument('--max_age', type=int, help='Maximum age of a node connecting edge', default=5)
	parser.add_argument('--max_len', type=int, help='Maximum sequence length for the transformer input', default=20)
	parser.add_argument('--m_t', type=int, help='Misclassification threshold', default=0)
	parser.add_argument('--reduce', type=int, help='Maximum number of train/test samples per dataset', default=300)
	parser.add_argument('--reduce_test', type=int, help='Maximum number of test samples per dataset, if -1 reduce is'
														' used as maximum number of samples', default=-1)
	#parser.add_argument('--seed', type=int, help='Random state for reproducible output', default=42)
	parser.add_argument('--tau_b', type=float, help='Constant habituation controlling rate (BMU)', default=.3)
	parser.add_argument('--tau_n', type=float, help='Constant habituation controlling rate (sample)', default=.1)
	parser.add_argument('--beta', type=float, help='Regulate influence of context on merge vector', default=0.5)
	parser.add_argument('--n_context', type=int, help='Window size / number of context', default=4)
	parser.add_argument('--class_list', type=list, help='List of classes', default=None)
	parser.add_argument('--num_labels', type=int, help='Number of label classes', default=1)
	parser.add_argument('--num_sentences', type=int, help='Number of Sentences to Split TextClassificationDatasets into'
															', if -1 Dataset is not split into Sentences ', default=5)
	parser.add_argument('--tuning_share', type=float, help='Share of training data used during fine tuning',
						default=0.1)
	parser.add_argument('--bert_finetune', type=bool, help='finetune bert model on data ', default=False)
	#args = parser.parse_args()

	#original
	#parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	#my own arguments
	parser.add_argument("--replay_memory", default="gwr_replay")    # Choose the replay memory to use

	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	#choose replay memory
	if args.replay_memory.lower() == "original":
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	elif args.replay_memory.lower() == "gwr_replay":
		replay_buffer = gwr_replay.GWR_replay(**vars(args))

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			bmu = replay_buffer.get_bmu(state)
			action = (
				policy.select_action(np.array(bmu))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		if t%20 in [0,1,2,3,4]:
			replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if args.replay_memory.lower() == "original":
				policy.train(replay_buffer, args.batch_size)
			### changed line
			elif args.replay_memory.lower() == "gwr_replay":
				policy.train_on_gwr(replay_buffer, args.batch_size)
 		

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Memory Size: {replay_buffer.size}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
