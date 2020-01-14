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
from replay_memory_big import ReplayMemoryBig
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
parser.add_argument('--fouriermodes', type=int, default=10, metavar='N',
                    help='N of fourier modes (default: 10)')
parser.add_argument('--batches', type=int, default=10, metavar='N',
                    help='N of batches (default: 10)')
parser.add_argument('--To', type=int, default=2, metavar='N',
                    help='To (default: 2)')
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
if args.Qapproximation == 'baseline':
    agent = SAC_baseline(env.observation_space.shape[0], env.action_space, args)
else:
    agent = SAC_fourier(env.observation_space.shape[0], env.action_space, args)

agent.load_model(
    actor_path = "./models/sac_actor_{}_{}_{}_{}_{}_{}_{}".format('miguelca_test01',
        args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise).replace('.','_'),str(args.num_steps)),
    critic_path = "./models/sac_critic_{}_{}_{}_{}_{}_{}_{}".format('miguelca_test01',
        args.Qapproximation,args.filter,args.TDfilter,str(args.noise),str(args.rnoise).replace('.','_'),str(args.num_steps))
    )
hard_update(agent.critic_target, agent.critic)

#TesnorboardX
writer = SummaryWriter(logdir='./runs/{}_SAC_eval_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Training Loop
total_numsteps = 0
updates = 0
action_history = list([])
action_history_w_noise = list([])

if args.noise == 'awgn':
    awgn_baseline = list([])
    awgn_baseline.append(.20)
    awgn_baseline.append(.30)
    awgn_baseline.append(.60)

for i_episode in range(5):
    episode_reward = 0
    episode_steps = 0
    episode_std = 0
    episode_std_count = 0
    done = False
    state = env.reset()

    if True:
        
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
                action_history.append(action)
                if args.noise == 'awgn':
                    action_w_noise = action
                    for aidx in range(env.action_space.shape[0]):
                        action_w_noise[aidx] += awgn_baseline[aidx]*awgn[aidx][int(100+100*action[aidx])]
                    next_state, reward, done, _ = env.step(action_w_noise) # Step
                else:
                    next_state, reward, done, _ = env.step(action) # Step
                action_history_w_noise.append(action)

                episode_reward += reward

                # still_open = env.render("human") # MACR

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("------------------------------------------------------------------------------------------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {:6.1f}, a1: {:5.2f},{:5.2f}, a2: {:5.2f},{:5.2f}, a3: {:5.2f},{:5.2f}".format(
            episodes, 
            round(avg_reward, 2),
            np.mean(np.array(action_history),0)[0],
            np.std(np.array(action_history),0)[0],
            np.mean(np.array(action_history),0)[1],
            np.std(np.array(action_history),0)[1],
            np.mean(np.array(action_history),0)[2],
            np.std(np.array(action_history),0)[2],
            ))
        print("------------------------------------------------------------------------------------------------------------------------")

# Q specrum
total_numsteps = 0
updates = 0
memory = ReplayMemoryBig(args.replay_size)

while updates<1:
    episode_reward = 0
    episode_steps = 0
    episode_std = 0
    episode_std_count = 0
    done = False
    state = env.reset()
    
    while updates<1:

#        action = agent.select_action(state, eval=True)  # Sample action from policy
#        _     , log_prob, action, std, _ = sample_for_spectrogram(state)
#        action, log_prob, mean  , std, _ = agent.policy.sample_for_spectrogram(torch.FloatTensor(state).unsqueeze(0))
        _, log_prob, action  , std, _ = agent.policy.sample_for_spectrogram(torch.FloatTensor(state).unsqueeze(0))
        action = action.detach().cpu().numpy()[0]
        
        if len(memory) > args.batches*args.batch_size:
            # Number of updates per step in environment
            for i in range(1):
                # Update parameters of all the networks
                
                agent.spectrum(memory,
                               args.batches*args.batch_size,
                               env.action_space,
                               To = args.To,
                               modes = args.fouriermodes)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        
        memory.push(state, action, reward, next_state, mask, log_prob.detach().numpy(), std.detach().numpy()) # Append transition to memory
        
        state = next_state

env.close()

