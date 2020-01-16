import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, Qfourier, Qbyactiondim


class SAC_fourier(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.Qapproximation = args.Qapproximation
        try:
            self.filter = args.filter
        except:
            self.filter = 'none'
        try:
            self.TDfilter = args.TDfilter
        except:
            self.TDfilter = 'none'

        self.cutoffX = args.cutoffX
        
        if args.Qapproximation == 'fourier':
            self.critic = Qfourier(num_inputs,
                action_space.shape[0],
                256,
                action_space,
                gridsize = 20).to(device=self.device)
                # target doesn't need to filter Q in high frequencies
            self.critic_target = Qfourier(num_inputs,
                action_space.shape[0],
                256,
                action_space,
                gridsize = 20).to(device=self.device)

        if args.Qapproximation == 'byactiondim':
            self.critic = Qbyactiondim(num_inputs,
                action_space.shape[0],
                256,8,5,
                action_space).to(device=self.device)
            self.critic_target = Qbyactiondim(num_inputs,
                action_space.shape[0],
                256,8,5,
                action_space).to(device=self.device)

#        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

#        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            
            if self.Qapproximation == 'byactiondim':
                qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            if self.Qapproximation == 'fourier':
                if self.TDfilter == 'none':
                    qf1_next_target = self.critic_target(next_state_batch, next_state_action)
                if self.TDfilter == 'rec_inside':
                    with torch.no_grad():
                        _, _, _, std = self.policy.sample(state_batch)
                    qf1_next_target = self.critic_target(next_state_batch, next_state_action,
                                         std     = std, # detached
                                         logprob = None,
                                         mu      = None,
                                         filter  = self.TDfilter, cutoffXX = self.cutoffX)

            min_qf_next_target = qf1_next_target - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1  = self.critic(state_batch, action_batch) # before
#        qf1  = self.critic(state_batch, action_batch,
#                           std     = std, # detached
#                           logprob = None,
#                           mu      = None,
#                           filter  = self.TDfilter) # this is to learn bandwidth

        qf1_loss = F.mse_loss(qf1, next_q_value)

        if self.Qapproximation == 'byactiondim':
            pi, log_pi, _, _ = self.policy.sample(state_batch)
            qf1_pi = self.critic(state_batch, pi)

        if self.Qapproximation == 'fourier':
            if self.filter == 'none':
#                with torch.no_grad():
#                    _, _, _, std = self.policy.sample(state_batch)
                pi, log_pi, _, std = self.policy.sample(state_batch)
                qf1_pi = self.critic(state_batch, pi,
                                     std     = None,
                                     logprob = None,
                                     mu      = None,
                                     filter  = 'none')
            if self.filter == 'rec_inside':
                with torch.no_grad():
                    _, _, _, std = self.policy.sample(state_batch)
                pi, log_pi, _, _ = self.policy.sample(state_batch)
                qf1_pi = self.critic(state_batch, pi,
                                     std     = std, # detached
                                     logprob = None,
                                     mu      = None,
                                     filter  = self.filter, cutoffXX = self.cutoffX)
            if self.filter == 'rec_outside':
                with torch.no_grad():
                    _, _, mu, std, log_pi_ = self.policy.sample_for_spectrogram(state_batch)
                pi, log_pi, _, _, _ = self.policy.sample_for_spectrogram(state_batch)
                qf1_pi = self.critic(state_batch, pi,
                                     std     = std, # detached
                                     logprob = log_pi_, # sum of logprobs w/o tanh correction
                                     mu      = mu,
                                     filter  = self.filter, cutoffXX = self.cutoffX)
#                print(((std*torch.exp(log_pi_)).sum(1, keepdim=True)).shape,
#                      std.mean(0),
#                      ((std*torch.exp(log_pi_)).sum(1, keepdim=True)).mean(0),
#                      torch.exp((log_pi).mean(0)))

        min_qf_pi = qf1_pi

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

#        print('{:5.2f} {:5.2f} {:5.2f}'.format(std[:,0].std().item(),std[:,1].std().item(),std[:,2].std().item()))
        return qf1_loss.item(), 0, policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), \
            std.mean().item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "./models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "./models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    def spectrum(self, memory, batch_size, action_space, To = 2, modes = 10):

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, \
            log_prob_batch, std_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        std_batch = torch.FloatTensor(std_batch).to(self.device).squeeze(1)
        log_prob_batch = torch.FloatTensor(log_prob_batch).to(self.device).squeeze(1)
        prob_batch = torch.exp(log_prob_batch)

        with torch.no_grad():
        
            qf1  = self.critic.spectrum(state_batch, action_batch, std_batch, prob_batch, action_space, To, modes)
