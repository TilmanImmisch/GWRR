from tracemalloc import stop
import torch
import gym
import numpy as np
import os
from Helpers import custom_logging
import importlib
import psutil


from ray import tune


import gwr_replay.base_models.orig_replay
from gwr_replay.base_models.orig_replay import ReplayBuffer
ReplayBufferThe = importlib.reload(gwr_replay.base_models.orig_replay)
ReplayBuffer = ReplayBufferThe.ReplayBuffer

from gwr_replay.base_models.RL_agents.DDPG import DDPG
from gwr_replay.base_models.RL_agents.TD3 import TD3

from gwr_replay.gwr_replay import GWR_replay

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, logger, eval_episodes=10):
	eval_env = gym.make(env_name)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(state)

			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	logger.info("---------------------------------------")
	logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	logger.info("---------------------------------------")
	return avg_reward

	
def training(args, testing=False):
	logger, args['results_path'] = custom_logging.set_up_log()
	#for tracking RAM
	process = psutil.Process(os.getpid())
	#create env
	env = gym.make(args["env"])
	#env.action_space.seed(args["seed"])
	torch.manual_seed(args["seed"])
	np.random.seed(args["seed"])

	state_dim = env.observation_space.shape[0]
	#if using a discrete space
	if isinstance(env.action_space, gym.spaces.discrete.Discrete):
		action_dim= env.action_space.n 
	else:
		action_dim = env.action_space.shape[0]
		max_action = float(env.action_space.high[0])


	args["state_dim"] = state_dim
	args["dim"] = state_dim
	args["action_dim"] = action_dim
	args["max_action"] = max_action

	# Initialize policy
	policy_args = ["state_dim", "action_dim", "max_action", "discount","tau"]

	if args["policy"] == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		args["policy_noise"] = args["policy_noise"] * max_action
		args["noise_clip"] = args["noise_clip"] * max_action
		args["policy_freq"] = args["policy_freq"]
		policy = TD3(**{key: value for key, value in args.items() if key in policy_args})
	elif args["policy"] == "DDPG":
		policy = DDPG(**{key: value for key, value in args.items() if key in policy_args})

	if args["load_model"] != "":
		policy_file = args["model_path"] if args["load_model"] == "default" else args["load_model"]
		policy.load(f"./models/{policy_file}")

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	#choose replay memory
	if args["replay_memory"].lower() == "original":
		replay_buffer = ReplayBuffer(args["state_dim"], args["action_dim"], max_size=int(1e6))
		args["mem_batch_size"] = 1
	elif args["replay_memory"].lower() == "gwr_replay":
		replay_buffer = GWR_replay(**args)

	#short term memory for batch training gwrlo
	mem_batch_size = args["mem_batch_size"]
	short_term = ReplayBuffer(args["state_dim"], args["action_dim"], max_size=int(mem_batch_size*2))

	logger.info("---------------------------------------")
	logger.info(f'Policy: {args["policy"]}, Env: {args["env"]}, Replay Memory: {args["replay_memory"]}, Seed: {args["seed"]}')
	logger.info("---------------------------------------")

	#args to log
	gwr_args= ["a_t","h_t","max_age","tau_b","tau_n","eps_b","eps_n","kappa","delta_plus","delta_minus","n_context","beta"]
	ddpg_args = ["discount", "tau"]

	logger.info('Args for GWR: {}'.format({key: value for key, value in args.items() if key in gwr_args}))
	logger.info('Args for DDPG: {}'.format({key:value for key, value in args.items() if key in ddpg_args}))

	for t in range(int(args["max_timesteps"])):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args["start_timesteps"]:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(state)
				+ np.random.normal(0, args["max_action"] * args["expl_noise"], size=args["action_dim"])
			).clip(-args["max_action"], args["max_action"])

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		short_term.add(0, state, action, next_state, reward, done_bool)

		# Train agent after collecting sufficient data
		if t >= args["start_timesteps"]:
			policy.train(short_term, int(np.round(args["batch_size"]*args["amount_real_data"])))
		#so that there aren't some samples getting stuck in the upper registers. Some will not be learned at all
		# probably irrelevant
		short_term.size = min(mem_batch_size, short_term.size)
		
		# Store data in replay buffer
		#Batch Learning would not work yet, as done states would not be respected, prev_bmu = -1 wouldn't trigger
		if t % mem_batch_size == 0 and t != 0:
			if args['amount_real_data'] < 1.:
				replay_buffer.add(episode_num, short_term.state[:short_term.ep_size], short_term.action[:short_term.ep_size], short_term.next_state[:short_term.ep_size], short_term.reward[:short_term.ep_size],1.- short_term.not_done[:short_term.ep_size])
			short_term.ptr = 0
			short_term.ep_size = 0
	
		state = next_state
		episode_reward += reward

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			logger.info(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Memory Size: {replay_buffer.size}")
			state, done = env.reset(), False
			#abusing the short term memory to also store reset states in next states
			if args["replay_memory"].lower() == 'gwr_replay':
				short_term.add(0, next_state, 0 , state, 0, 0)
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args["eval_freq"] == 0:
			avg_reward = eval_policy(policy, args["env"], np.random.randint(100),logger, eval_episodes=6)
			if args["ray_tune"]:
				tune.report(episode_reward=avg_reward, memory_size=replay_buffer.size, episode_num=episode_num+1, total_t=t+1, CPU_load=psutil.getloadavg(), used_RAM=process.memory_info().rss)
			if args["save_model"]: policy.save(args["model_path"])
			if args["store_buffer"]:
				custom_logging.store_buffer(replay_buffer, episode_num+1, args["env"], args["batch_name"])
