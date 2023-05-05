import asynchat
from email.policy import default
import torch
import gym
import argparse
import os

import ray
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler

import train

from datetime import datetime

#switch to second gpu
#TODO: CHECK WHICH CUDA IS FREE
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def add_args() -> dict:
	# added Parse command line arguments
	parser = argparse.ArgumentParser()
	#Insertion/Deletion Behaviour
	parser.add_argument('--a_t',type=float,  help='Activation threshold', default=.95)
	parser.add_argument('--a_t_max',type=float,  help='Activation threshold', default=1.)
	parser.add_argument('--h_t', type=float,  help='Habituation/Firing threshold', default=.8)
	parser.add_argument('--h_t_max', type=float,  help='Habituation/Firing threshold', default=1.)
	parser.add_argument('--kappa',type=float,  help='Habituation controlling parameter', default=1.05)
	parser.add_argument('--tau_b', type=float, help='Constant habituation controlling rate (BMU)', default=.3) 
	parser.add_argument('--tau_n', type=float, help='Constant habituation controlling rate (neighbour)', default=.1)
	parser.add_argument('--max_age',type=int,   help='Maximum age of a node connecting edge', default=10)
	#Learning Rates
	parser.add_argument('--eps_b', type=float, help='Learning rate for weight adaption (BMU)', default=.1)
	parser.add_argument('--eps_n', type=float, help='Learning rate for weight adaption of (neighbour)', default=.001)
	#parser.add_argument('--eps_t', type=float, help='Learning rate for temporal edge adaption', default=.05)

	#Context Behaviour
	parser.add_argument('--beta', type=float, help='Regulate influence of context on merge vector', default=0.7)
	parser.add_argument('--n_context',type=int,  help='Window size / number of context', default=4)
	
	parser.add_argument("--phi", type=float, default=.7) #regulate the influence of the current and past on the moving temporal average 
	parser.add_argument("--neighborhood", default=True, type=bool) #Turn Neighborhood (and node deletion) on or off

	#Run specific 
	parser.add_argument('--log_freq', type=int, help='Logging frequency of learning metrics', default=1)
	parser.add_argument('--load_latest', type=bool, help='Load latest pretrained model from file', default=False)

	#ddpg
	parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="InvertedPendulum-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=42, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=10000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=2500, type=int)  		 # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e5, type=int)       # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise--ray
	parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	
	#Logging
	parser.add_argument("--replay_memory", default="gwr_replay")    # Choose the replay memory to use
	parser.add_argument("--logging", default=True, type=bool)
	parser.add_argument("--run_name",default=str(datetime.now().strftime('%y%m%d_%H-%M')), type=str)
	parser.add_argument("--batch_name",default="default_batch_name", type=str)
	parser.add_argument("--log_path", default = os.getcwd())
	parser.add_argument("--save_network", default=False, type=bool)        # Store network nodes
	parser.add_argument("--store_buffer", type=str, default=False) 

	#Ray Args
	parser.add_argument("--ray_tune", default=False, type=bool )
	parser.add_argument("--resume_ray", default=False, type=bool )
	parser.add_argument("--num_ray_runs", default=20, type=int) 
	parser.add_argument("--run_schedule", type=str)
	parser.add_argument("--scheduler", type=str, default=None) 
	parser.add_argument("--search_alg", type=str, default=None) 

	args = vars(parser.parse_args())

	if args['ray_tune']:
		args["ray_dir"] = os.path.join(args['log_path'], f"../results/{args['env']}/")

		args["max_timesteps"] = 10e4
		args["start_timesteps"] = 15e3

		args["a_t"] = .95
		args["h_t"] = .8
		args["phi"] = .7

		if args["run_schedule"] == 'modulate_at':
			args["h_t"] = 1.
			try:
				if args["env"] == "InvertedPendulum-v2":
					args["a_t"]= tune.grid_search([.98, .96,.92,.88])
				elif args["env"] == "Reacher-v2":
					args["a_t"]= tune.grid_search([.91,.83,.68,.55])
				elif args["env"] == "HalfCheetah-v2":
					args["a_t"]= tune.grid_search([.55, .29,.15,.07])
				elif args["env"] == "Walker2d-v2":
					args["a_t"]= tune.grid_search([.55, .29,.15,.07])
			except:
				print("Unknown a_t-config for this gym env.")

		if args["run_schedule"] == 'modulate_ht':
			args["h_t"]= tune.grid_search([.9,.65,.45,.3])
			args["a_t"] = 1.

		if args["run_schedule"] == 'modulate_phi':
			args["phi"]= tune.grid_search([1.,.7,.4,.1])

	return args


def main():
	args = add_args()

	model_path = os.path.join(args["log_path"],"models/")
	if args["save_model"] and not os.path.exists(model_path):
		os.makedirs(model_path)
		model_path = os.path.join(model_path, f'{args["policy"]}_{args["env"]}_{args["seed"]}')
		args["model_path"] = model_path

	#ray tuning setup
	if args["ray_tune"]:
		ray.init()
		analysis = tune.run(train.training,config=args,num_samples=args["num_ray_runs"], resources_per_trial={"cpu":6, "gpu":1}, name=args["batch_name"],local_dir=args["ray_dir"], stop={"total_t":args["max_timesteps"]}, resume=args["resume_ray"], search_alg=args['search_alg'], scheduler=args['scheduler'])
		dfs = analysis.trial_dataframes
	else:
		train.training(args)	


if __name__ == "__main__":
	main()
