import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from sac_baseline import SAC_baseline
from sac_fourier import SAC_fourier
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from utils import soft_update, hard_update

import roboschool # MACR

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--Qapproximation', default="baseline",
                    help='baseline, fourier, byactiondim')
parser.add_argument('--filter', default="none",
                    help='Q filter for policy optimization: none, rec_inside, rec_outside')
parser.add_argument('--TDfilter', default="none",
                    help='Q filter for TD step: none, rec_inside, rec_outside')
parser.add_argument('--noise', default="none",
                    help='none, twgn, swgn')
parser.add_argument('--rnoise', type=float, default=0., metavar='G',
                    help='std of rewards multiplicative noise term (default = 0)')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
args.env_name = "RoboschoolHopper-v1" # MACR
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
assert ( args.Qapproximation in ('baseline','fourier','byactiondim') )
if args.Qapproximation == 'baseline':
    agent = SAC_baseline(env.observation_space.shape[0], env.action_space, args)
    print('\nusing baseline\n')
else:
    agent = SAC_fourier(env.observation_space.shape[0], env.action_space, args)
    print('\nusing {}\n'.format(args.Qapproximation))

if False:
    print('\n\n\n LOADING MODEL')
    agent.load_model(
                     actor_path = "./models/sac_actor_{}_{}_{}_{}_{}_{}_{}".format('miguelca_test01', 
                        args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise),str(args.num_steps)),
                     critic_path = "./models/sac_critic_{}_{}_{}_{}_{}_{}_{}".format('miguelca_test01', 
                        args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise),str(args.num_steps))
                     )
    hard_update(agent.critic_target, agent.critic)

#TesnorboardX
writer = SummaryWriter(logdir='./runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

# prep swgn noise?
if args.noise == 'swgn':
    swgn = list([])
    swgn.append( (1/np.sqrt(3)) * np.random.normal(0, args.rnoise, size=201) )
    swgn.append( (1/np.sqrt(3)) * np.random.normal(0, args.rnoise, size=201) )
    swgn.append( (1/np.sqrt(3)) * np.random.normal(0, args.rnoise, size=201) )

if args.noise == 'awgn':
    awgn_baseline = list([])
    awgn_baseline.append(.20)
    awgn_baseline.append(.30)
    awgn_baseline.append(.60)

best_eval_avg_reward = 0
for i_episode in itertools.count(1):

    if args.noise == 'awgn':
        awgn = list([])
        for aidx in range(env.action_space.shape[0]):
            awgn.append( np.random.normal(0, args.rnoise, size=201) )

    episode_reward = 0
    episode_steps = 0
    episode_std = 0
    episode_std_count = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, \
                    std = agent.update_parameters(memory, args.batch_size, updates)
                episode_std += std
                episode_std_count += 1

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        if args.noise == 'awgn':
            action_w_noise = action
            for aidx in range(env.action_space.shape[0]):
                action_w_noise[aidx] += awgn_baseline[aidx]*awgn[aidx][int(100+100*action[aidx])]
            next_state, reward, done, _ = env.step(action_w_noise) # Step
        else:
            next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if args.rnoise > 0:
            if args.noise == 'twgn':
                reward += np.random.normal(0, args.rnoise)
            if args.noise == 'swgn':
                reward += sum([swgn[i][int(100+100*action[i])] for i in range(3)])
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    if episode_std_count >= 1:
        episode_std_mean = episode_std/episode_std_count
    else:
        episode_std_mean = np.nan
    print("Ep.: {:6.0f}, tot. steps: {:8.0f}, ep. steps: {:4.0f}, re.: {:4.0f}, nstd {:4.2f}".format(i_episode,
                                                                                               total_numsteps,
                                                                                               episode_steps,
                                                                                               round(episode_reward, 2),
                                                                                               episode_std_mean
                                                                                               )
          )

    if i_episode % 100 == 0 and args.eval == True:
        
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            
            if args.noise == 'awgn':
                awgn = list([])
                for aidx in range(env.action_space.shape[0]):
                    awgn.append( np.random.normal(0, args.rnoise, size=201) )

            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                if args.noise == 'awgn':
                    action_w_noise = action
                    for aidx in range(env.action_space.shape[0]):
                        action_w_noise[aidx] += awgn_baseline[aidx]*awgn[aidx][int(100+100*action[aidx])]
                    next_state, reward, done, _ = env.step(action_w_noise) # Step
                else:
                    next_state, reward, done, _ = env.step(action) # Step

                episode_reward += reward

                # still_open = env.render("human") # MACR

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("--------------------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Max. Reward {}".format(episodes, round(avg_reward, 2), round(best_eval_avg_reward,2) ))
        print("--------------------------------------------------")

        if avg_reward > best_eval_avg_reward:
            agent.save_model(
                env_name = 'miguelca_test01', 
                suffix = "{}_{}_{}_{}_{}_{}".format(
                    args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise).replace('.','_'),str(args.num_steps))
                )
            best_eval_avg_reward = avg_reward

# final eval

avg_reward = 0.
episodes = 10
for _  in range(episodes):
    
    if args.noise == 'awgn':
        awgn = list([])
        for aidx in range(env.action_space.shape[0]):
            awgn.append( np.random.normal(0, args.rnoise, size=201) )

    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, eval=True)

        if args.noise == 'awgn':
            action_w_noise = action
            for aidx in range(env.action_space.shape[0]):
                action_w_noise[aidx] += awgn_baseline[aidx]*awgn[aidx][int(100+100*action[aidx])]
            next_state, reward, done, _ = env.step(action_w_noise) # Step
        else:
            next_state, reward, done, _ = env.step(action) # Step
        
        episode_reward += reward
                
        # still_open = env.render("human") # MACR
                
        state = next_state
    avg_reward += episode_reward
avg_reward /= episodes
        
writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        
print("--------------------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}, Max. Reward {}".format(episodes, round(avg_reward, 2), round(best_eval_avg_reward,2) ))
print("--------------------------------------------------")

if avg_reward > best_eval_avg_reward:
    agent.save_model(
        env_name = 'miguelca_test01', 
        suffix = "{}_{}_{}_{}_{}_{}".format(
            args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise).replace('.','_'),str(args.num_steps))
    )
    best_eval_avg_reward = avg_reward

env.close()

