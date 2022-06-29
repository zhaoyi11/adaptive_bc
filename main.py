import random
import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import tqdm
import time
import wandb

import utils
import redq_bc
import torch.nn.functional as F

# These values are used to replace the target_R
R_MAX_CHEETAH = 15.743
R_MAX_WALKER = 10.271
R_MAX_HOPPER = 6.918

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="REDQ_BC")               # Policy name
    parser.add_argument("--env", default="hopper-medium-replay-v0")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5000, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=250_000, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_policy_id", default="")
    parser.add_argument("--episode_length", default=1000, type=int)
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    
    parser.add_argument("--pretrain_timesteps", default=1_000_000, type=int)
    parser.add_argument("--num_updates", type=int, default=10, help='Num of update steps per data point.')
    # ablation, dataset
    parser.add_argument('--sample_method', choices=['random', 'best'], default='random')
    parser.add_argument('--sample_ratio', type=float, default=0.05)
    # ablation, q_update
    parser.add_argument('--use_q_min', action='store_true')
    # adaptive alpha
    parser.add_argument('--Kp', type=float, default=0.00003)
    parser.add_argument('--Kd', type=float, default=0.0001)
    # ablation, avoid using the target_R
    parser.add_argument("--use_r_max", action='store_true') # if True, the return will normalized by the R_MAX * T

    args = parser.parse_args()
    # get task-specific args
    if args.env == "walker2d-random-v0":
        args.alpha = 100
        args.alpha_finetune = 0.4
    elif args.env in ['pen-expert-v0', 'door-expert-v0', 'hammer-expert-v0', 'relocate-expert-v0']:
        # these tasks need larger constraints during pretraining.
        args.alpha = 8.0
        args.alpha_finetune = 8.0
        # change the Kp and Kd accordingly
        args.Kp = (args.alpha / 0.4) * args.Kp # 0.4 is the default value for locomotion tasks
        args.Kd = (args.alpha / 0.4) * args.Kd
        # change the num_updates -- fix it to 1
        args.num_updates = 1
    else: # default value for all locomotion tasks except the 'walker2d-random-v0'
        args.alpha = 0.4
        args.alpha_finetune = 0.4
    print(args)    
    

    # use R_MAX * T to normalize the reward, and as target return
    # This is only needed when no target_R is available.
    if args.use_r_max:
        domain = str(args.env).split('-')[0]
        if domain == "halfcheetah":
            max_R = R_MAX_CHEETAH * args.episode_length
        elif domain == "walker2d":
            max_R = R_MAX_WALKER * args.episode_length
        elif domain == "hopper":
            max_R = R_MAX_HOPPER * args.episode_length
        else:
            raise ValueError


    run_id = int(time.time())
    run_name = f"{args.policy}_{args.env}_{args.seed}_{run_id}"

    if args.save_model:
        if not os.path.exists(f"./models/{args.env}/pretrain"):
            os.makedirs(f"./models/{args.env}/pretrain")
        if not os.path.exists(f"./models/{args.env}/finetune"):
            os.makedirs(f"./models/{args.env}/finetune")
        else:
            print(f"The pretrained model will be stored under path: ./models/{args.env}/pretrain; The finetuned model will be stored under path: ./models/{args.env}/finetune")

    # seeding and setup env
    env = gym.make(args.env)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    # init d4rl_buffer
    d4rl_replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    d4rl_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env)) # fill buffer


    # init policy
    kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau, 
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq,
        "alpha": args.alpha,
        "use_q_min": args.use_q_min,
        "pretrain": True,
    }
    policy = redq_bc.REDQ_BC(**kwargs)


    ########## Load or Pretrain the model with d4rl dataset. ##########
    # load/pretrain policy
    if args.load_policy_id != "":
        policy_file = f'{args.policy}_{args.env}' if args.load_policy_id == "default" else f'{args.policy}_{args.env}_{0}_{args.load_policy_id}'
        policy.load(f"./models/{args.env}/pretrain/{policy_file}")
        print(f"------ Load policy model {policy_file} ------")
    else: # finetune
        with wandb.init(project='adaptive_bc', group=args.env, job_type="pretrain", name=run_name):
            wandb.config.update(args)
            for i in tqdm.tqdm(range(args.pretrain_timesteps)):
                batch = d4rl_replay_buffer.sample(args.batch_size)
                pretrain_info = policy.train(batch)
                wandb.log({"pretrain_training/": pretrain_info})
            
                if i % args.eval_freq == 0:
                    pretrain_eval = utils.eval_policy(policy, args.env, args.seed)
                    wandb.log({'pretrain_evaluation/': pretrain_eval})
            # save policy
            if args.save_model: policy.save(f'./models/{args.env}/pretrain/{run_name}')    
    

    ########## Begin to finetune the model ##########
    state, done = env.reset(), False
    episode_timesteps = 0
    update_info, eval_info = {}, {}

    # distill dataset
    buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(args.max_timesteps))
    buffer.distill(d4rl.qlearning_dataset(env), args.env, args.sample_method, args.sample_ratio)
    del d4rl_replay_buffer # to save memory


    policy.alpha = args.alpha_finetune
    policy.pretrain = False # set flag of pretrain
 

    # finetune
    with wandb.init(project='adaptive_bc', group=args.env, job_type="finetune", name=run_name):
        wandb.config.update(args)

        # init last_R, current_R and target_R
        # last_R ranges from 0 to 1
        if args.use_r_max:
            last_R = utils.eval_policy(policy, args.env, args.seed, eval_episodes=1)['evaluation'] / max_R
        else: 
            last_R  = utils.eval_policy(policy, args.env, args.seed, eval_episodes=1)['d4rl']*0.01 

        current_R = last_R
        target_R = 1.05


        episode_return = 0.
        for t in tqdm.tqdm(range(args.max_timesteps)):
            episode_timesteps += 1

            action = (policy.select_action(state) 
                    + np.random.normal(0, scale=args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step(action)

            episode_return += reward  # recore episode return

            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0. # important!
            buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            
            for _ in range(args.num_updates):
                update_info = policy.train(buffer.sample(args.batch_size))

            update_info.update({'current_R': current_R, 'last_R': last_R})

            wandb.log({'finetune_training/': update_info,
                        'finetune_training/alpha': policy.alpha})

            if done:
                state, done = env.reset(), False
                
                # update current_R and the alpha of policy
                if args.use_r_max:
                    current_R = episode_return / max_R
                else:
                    current_R = env.get_normalized_score(episode_return)

                policy.alpha += episode_timesteps * (- args.Kp * (target_R - last_R)
                                + args.Kd * max(0, last_R - current_R))
                # clip the alpha value between 0.0 and 0.4
                policy.alpha = max(0., min(policy.alpha, args.alpha_finetune))

                # moving average
                last_R = 0.05 * current_R + 0.95 * last_R
                
                episode_timesteps = 0
                episode_return = 0.
            
            # Evaluate episode
            if t % args.eval_freq == 0:
                eval_info = utils.eval_policy(policy, args.env, args.seed, eval_episodes=10)
                wandb.log({'finetune_evaluation/': eval_info})

                if args.save_model:
                    policy.save(f"./models/{args.env}/finetune/{run_name}")
