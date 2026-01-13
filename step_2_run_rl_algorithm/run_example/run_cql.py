import argparse
import random

import gym
import d4rl

import numpy as np
import pickle
import torch

import sys
sys.path.append('.')

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy import CQLPolicy

from offlinerlkit_modified.policy_trainer import MFPolicyTrainer
from offlinerlkit_modified.buffer import ReplayBuffer
from offlinerlkit_modified.S_A_NS_SCM import TransitionSCM


"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_scm_path", type=str, default="to_write")
    parser.add_argument("--original_data_path", type=str, default="to_write")

    parser.add_argument("--vars_dim_list", nargs="+", type=int)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--sample_range", type=float, default=1.0)

    parser.add_argument("--prob_l", type=float, default=0.0)
    parser.add_argument("--prob_r", type=float, default=1.0)

    parser.add_argument("--index_1", type=int, default=1)
    parser.add_argument("--index_2", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--weight_aug", type=float, default=1.0)

    parser.add_argument("--tag", type=str, default="to_write")
    
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=1.0)

    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

#-----------------------------------------------------------------------------------------

def train(args=get_args()):

    #----------------------------------------------------------------------------------
    # vars: (state, action, reward, next state)
    scm_model = TransitionSCM(ckpt_path = args.trained_scm_path, vars_dims=args.vars_dim_list) 
    #----------------------------------------------------------------------------------

    # create env and dataset
    env = gym.make(args.task)
    #----------------------------------------------------------------------------------
    with open(args.original_data_path, 'rb') as f:
        tmp_data = pickle.load(f)

    # dataset = qlearning_dataset(env)
    dataset = {}
    dataset['observations'] = tmp_data[0]
    dataset['actions'] = tmp_data[1]
    dataset['rewards'] = tmp_data[2]
    dataset['next_observations'] = tmp_data[3]
    dataset['terminals'] = tmp_data[4]
    #----------------------------------------------------------------------------------
    # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    if 'antmaze' in args.task:
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 4.0
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)


    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=args.step_size, gamma=args.lr_gamma)
    critic1_scheduler = torch.optim.lr_scheduler.StepLR(critic1_optim, step_size=args.step_size, gamma=args.lr_gamma)
    critic2_scheduler = torch.optim.lr_scheduler.StepLR(critic2_optim, step_size=args.step_size, gamma=args.lr_gamma)
    scheduler_list = [actor_scheduler, critic1_scheduler, critic2_scheduler]

    print("----------------------------------------auto alpha------------------------------",args.auto_alpha)
    print("----------------------------------------alpha lr value------------------------------",args.alpha_lr)
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    #-----------------------------------------------------------------------------
    buffer.load_dataset_and_augment(dataset, scm_model, args.num_samples, args.sample_range, args.prob_l, args.prob_r, args.index_1, args.index_2, args.topk, args.weight_aug)
    #-----------------------------------------------------------------------------

    # log
    log_dirs = make_log_dirs(args.tag, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=scheduler_list
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()