import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        #        torch.nn.init.constant_(m.weight, 0) # convergence issues with this init.
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def spectrum(self, state, action, std_batch, prob_batch, action_space = None, To = 2, modes = 10):
        
        self_num_actions = 3 # hacky alert!!!!!!!!!!!!!!!!!!
        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        
        action_m1p1 = (action - self.action_bias) / self.action_scale
        
        spectrogram = list([])
        spectrogram_sin = list([])
        spectrogram_cos = list([])

        for i in range(self_num_actions):
            
            print('spectrogram, action dim',i)
            spectrogram.append(list([]))
            spectrogram_sin.append(list([]))
            spectrogram_cos.append(list([]))

            for k in range(modes):
                
                #uniform grid for integration in 'a' dimension
                gsize = modes*5 # if < modes, then strobelight effect
                agrid = -1
                action_m1p1[:,i] = agrid
                tempQvalue = self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
                                                                                             self.action_bias + action_m1p1*self.action_scale], 1))))))

                spectrogram_sin[i].append( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                spectrogram_cos[i].append( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )

                for uidx in range(gsize):
                    
                    agrid += (2/gsize) # ... will go from -1 to +1 in gsize steps
                    action_m1p1[:,i] = agrid
                    tempQvalue = self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
                                                                                                 self.action_bias + action_m1p1*self.action_scale], 1))))))
                    spectrogram_sin[i][k] += ( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                    spectrogram_cos[i][k] += ( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )

                spectrogram_sin[i][k] = torch.pow(spectrogram_sin[i][k],2)
                spectrogram_cos[i][k] = torch.pow(spectrogram_cos[i][k],2)
                spectrogram[i].append(torch.pow(spectrogram_sin[i][k] + spectrogram_cos[i][k],1/2))
                spectrogram[i][k] = torch.sum(spectrogram[i][k]).item() # MC sum across random states
        
            spectrogram[i] = np.array(spectrogram[i]) # list --> array
                    
        print('\n>> Q spectrum\n')
        
        energies = [sum(spectrogram[i]) for i in range(self_num_actions)]
        spectrogram = [spectrogram[i]/spectrogram[i][0] for i in range(self_num_actions)]
        
        print('energies , a0, {:13.2f} , a1, {:13.2f} , a2, {:13.2f}'.format(energies[0]/energies[0],energies[1]/energies[0],energies[2]/energies[0]))
        print('\n')
        
        for k in range(modes):

            print('w, {:5.2f} , a0, {:5.2f} , {:5.2f} , a1, {:5.2f} , {:5.2f} , a2, {:5.2f} , {:5.2f}'.format(
                                                                                                              (k+1)*2*3.14/To,
                                                                                                              spectrogram[0][k], torch.mean(1/std_batch[:,0]).item(),
                                                                                                              spectrogram[1][k], torch.mean(1/std_batch[:,1]).item(),
                                                                                                              spectrogram[2][k], torch.mean(1/std_batch[:,2]).item()
                                                                                                              ))
        
        if True:
            
            t = np.arange(0, modes, 1)
            splot = [np.array(np.log(spectrogram[i])) for i in range(self_num_actions)]
            
            ax1 = plt.subplot(311)
            plt.plot(t, splot[0])
            plt.setp(ax1.get_xticklabels(), fontsize=6)
            
            # share x only
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(t, splot[2])
            # make these tick labels invisible
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            # share x and y
            ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
            plt.plot(t, splot[2])
            plt.xlim(0, modes)
            plt.show()


class Qfourier(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, gridsize = 20):
        super(Qfourier, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
#        self.filterbw_0p1 = nn.Linear(num_inputs, 3) # torch.sigmoid(nn.Parameter(torch.tensor(1.)))

        self.apply(weights_init_)
        self.gsize = gridsize
        self.num_actions = num_actions
        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

                
        self.integration_grid = [round(i*2*.75/4-.75,2) for i in range(4+1)] # ..0001
#        self.integration_grid = [round(i*2*1.95/10-1.95,2) for i in range(10+1)] # ..0002
#        self.integration_grid = [round(i*2*.75/2-.75,2) for i in range(2+1)] # ..0003
        print('\nintegration grid')
        print(self.integration_grid)


    def forward(self, state, action, std = None, logprob = None, mu = None, filter = 'none', cutoffXX = 2):

        if filter == 'none':
            xu = torch.cat([state, action], 1)

            x1 = F.relu(self.linear1(xu))
            x1 = F.relu(self.linear2(x1))
            x1 = self.linear3(x1)
            return x1

        if filter == 'rec_outside':
            
            xu = torch.cat([state, action], 1)
            
            x1 = F.relu(self.linear1(xu))
            x1 = F.relu(self.linear2(x1))
            x1 = self.linear3(x1)

            action_m1p1 = (action - self.action_bias) / self.action_scale

            # argument of sin function
            cutoffW = 1 * (1 / (F.relu(std) + 0.1))
            WxA = cutoffW * (mu - action_m1p1) + 0.001 # = -1.5*pi, 0, +1.5*pi
            # denominator sinc function if we wanted the filter to be H=1 in frequency domain
            # piA = 3.14 * (mu - action_m1p1) + 0.001

            x1 = x1 * ((
                       (torch.sin(WxA[:,0])/WxA[:,0]).view(-1,1) * \
                       (torch.sin(WxA[:,1])/WxA[:,1]).view(-1,1) * \
                       (torch.sin(WxA[:,2])/WxA[:,2]).view(-1,1)
                        ) / ((torch.exp(logprob)).sum(1, keepdim=True)) ) # don't need sum(1...), it is already [:,1]

                       # note logprob = sum logprob_i
                       # exp(logprod) = prod prob_i


                       
            return x1

        if filter == 'rec_inside':

            action_m1p1 = (action - self.action_bias) / self.action_scale

            # initialize Q value = 0
            tempQvalue = torch.zeros((state.shape[0],1))
            filterpower = torch.zeros((state.shape[0],1))
            Qpower = torch.zeros((state.shape[0],1))

            cutoffW = cutoffXX * (1 / (F.relu(std) + 0.1)) # ..0001 and ..0002
#            bw = torch.sigmoid(self.filterbw_0p1(state)) # ..0010
#            cutoffW = bw * 3 * (1 / (F.relu(std) + 0.1)) # ..0010 and ..0011
            delta_a = 3.14 / cutoffW # first zero of sincWx/pix
            
            # cutoffW = 1 ... [-0.1,0,0.1] ... main.py --Qapproximation fourier --num_steps 100000 --TDfilter rec_inside --alpha 0.05 ~ 1100
            
#            dVolume = 1/(len(integration_grid)**3)
            dVolume = 1/(len(self.integration_grid)*3)

#            for i0 in integration_grid:
#                for i1 in integration_grid:
#                    for i2 in integration_grid:
            idic = {0:[1,0,0],
                    1:[0,1,0],
                    2:[0,0,1]}
            for actionid in [0,1,2]:

                for i_ in self.integration_grid:
                        i0 = i_*idic[actionid][0]
                        i1 = i_*idic[actionid][1]
                        i2 = i_*idic[actionid][2]

                        # new point in the grid
                        action_m1p1_grid = torch.zeros(action_m1p1.shape)
                        action_m1p1_grid[:,0] = torch.tanh(action_m1p1[:,0] + i0 * delta_a[:,0])
                        action_m1p1_grid[:,1] = torch.tanh(action_m1p1[:,1] + i1 * delta_a[:,1])
                        action_m1p1_grid[:,2] = torch.tanh(action_m1p1[:,2] + i2 * delta_a[:,2])

                        # argument of sin function
                        WxA = cutoffW * (action_m1p1_grid - action_m1p1) + 0.001 # = -1.5*pi, 0, +1.5*pi
                        # denominator sinc function if we wanted the filter to be H=1 in frequency domain
                        piA = 3.14 * (action_m1p1_grid - action_m1p1) + 0.001
                        
                        filter = (
                                  (torch.sin(WxA[:,0])/WxA[:,0]).view(-1,1) * \
                                  (torch.sin(WxA[:,1])/WxA[:,1]).view(-1,1) * \
                                  (torch.sin(WxA[:,2])/WxA[:,2]).view(-1,1)
                                  )
                                  
#                        tempQvalue += self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
#                                (self.action_bias + action_m1p1_grid*self.action_scale)], 1)))))) * \
#                                filter * dVolume

                        Q            = self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
                                        (self.action_bias + action_m1p1_grid*self.action_scale)], 1))))))

                        tempQvalue  += ( Q * filter ) * dVolume
#                        Qpower      += np.power(Q,2) * dVolume
                        filterpower += filter * dVolume # np.power(filter,2) * dVolume

#            filterpower = np.power(filterpower,1/2)
#            Qpower      = np.power(Qpower,1/2)
            tempQvalue  = tempQvalue / filterpower
            
            return tempQvalue


    def spectrum(self, state, action, std_batch, prob_batch, action_space = None, To = 2, modes = 10):
        
        self_num_actions = 3 # hacky alert!!!!!!!!!!!!!!!!!!
            
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
            
            action_m1p1 = (action - self.action_bias) / self.action_scale
            
            spectrogram = list([])
            spectrogram_sin = list([])
            spectrogram_cos = list([])
            
            for i in range(self_num_actions):
                
                print('spectrogram, action dim',i)
                spectrogram.append(list([]))
                spectrogram_sin.append(list([]))
                spectrogram_cos.append(list([]))
                
                for k in range(modes):
                    
                    #uniform grid for integration in 'a' dimension
                    gsize = modes*5 # if < modes, then strobelight effect
                    agrid = -1
                    action_m1p1[:,i] = agrid
                    tempQvalue = self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
                                                                                                 self.action_bias + action_m1p1*self.action_scale], 1))))))
                        
                    spectrogram_sin[i].append( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                    spectrogram_cos[i].append( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                                                                                                 
                    for uidx in range(gsize):
                        
                        agrid += (2/gsize) # ... will go from -1 to +1 in gsize steps
                        action_m1p1[:,i] = agrid
                        tempQvalue = self.linear3(F.relu(self.linear2(F.relu(self.linear1(torch.cat([state,
                                                                                                     self.action_bias + action_m1p1*self.action_scale],
                                                                                                    1))))))
                        spectrogram_sin[i][k] += ( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                        spectrogram_cos[i][k] += ( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )

                    spectrogram_sin[i][k] = torch.pow(spectrogram_sin[i][k],2)
                    spectrogram_cos[i][k] = torch.pow(spectrogram_cos[i][k],2)
                    spectrogram[i].append(torch.pow(spectrogram_sin[i][k] + spectrogram_cos[i][k],1/2))
                    spectrogram[i][k] = torch.sum(spectrogram[i][k]).item() # MC sum across random states
                        
                spectrogram[i] = np.array(spectrogram[i]) # list --> array
            
            print('\n>> Q spectrum\n')

            energies = [sum(spectrogram[i]) for i in range(self_num_actions)]
            spectrogram = [spectrogram[i]/spectrogram[i][0] for i in range(self_num_actions)]

            print('energies , a0, {:13.2f} , a1, {:13.2f} , a2, {:13.2f}'.format(energies[0]/energies[0],energies[1]/energies[0],energies[2]/energies[0]))
            print('\n')

            for k in range(modes):
                print('w, {:5.2f} , a0, {:5.2f} , {:5.2f} , a1, {:5.2f} , {:5.2f} , a2, {:5.2f} , {:5.2f}'.format(
                                                                                                                  (k+1)*2*3.14/To,
                                                                                                                  spectrogram[0][k], torch.mean(1/std_batch[:,0]).item(),
                                                                                                                  spectrogram[1][k], torch.mean(1/std_batch[:,1]).item(),
                                                                                                                  spectrogram[2][k], torch.mean(1/std_batch[:,2]).item()
                                                                                                                  ))

        if True:
        
            t = np.arange(0, modes, 1)
            splot = [np.array(np.log(spectrogram[i])) for i in range(self_num_actions)]
        
            ax1 = plt.subplot(311)
            plt.plot(t, splot[0])
            plt.setp(ax1.get_xticklabels(), fontsize=6)
            
            # share x only
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(t, splot[2])
            # make these tick labels invisible
            plt.setp(ax2.get_xticklabels(), visible=False)
                
            # share x and y
            ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
            plt.plot(t, splot[2])
            plt.xlim(0, modes)
            plt.show()


class fouriercoeffnet(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_fourier):
        super(fouriercoeffnet, self).__init__()
        self.fouriercoeff = nn.Sequential(
                                          nn.Linear(hidden_dim, hidden_dim_fourier),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim_fourier, 1)
                                          )
        self.apply(weights_init_)

    def forward(self, state):
        return self.fouriercoeff(state)


class Qfourier_synthesis(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, hidden_dim_fourier,
                 dimensionality = 10, action_space=None):
        super(Qfourier, self).__init__()
        
        print('num_inputs        ',num_inputs)
        print('num_actions       ',num_actions)
        print('hidden_dim        ',hidden_dim)
        print('hidden_dim_fourier',hidden_dim_fourier)
        print('dimensionality    ',dimensionality)

        # dimensionality
        self.dimensionality = dimensionality

        # state effect
        self.lineara = nn.Linear(num_inputs, hidden_dim)
        self.lineara2 = nn.Linear(hidden_dim, hidden_dim)
        self.linearb = nn.Linear(hidden_dim, 1)

        # action effect
#        self.linearc = nn.Linear(num_actions, hidden_dim)
#        self.linearc2 = nn.Linear(hidden_dim, hidden_dim)
#        self.lineard = nn.Linear(hidden_dim, 1)

        # state action effect
        # fourier coefficients, common
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # fourier coefficients, one per fourier mode
        self.fouriercoeffs_sin = nn.ModuleList()
        self.fouriercoeffs_cos = nn.ModuleList()
        
        for k in range(dimensionality):
            self.fouriercoeffs_sin.append(fouriercoeffnet(hidden_dim, hidden_dim_fourier))
            self.fouriercoeffs_cos.append(fouriercoeffnet(hidden_dim, hidden_dim_fourier))

        # fourier coefficients, one per fourier mode per action dimension
        self.fouriercoeffs_sin_list2 = nn.ModuleList()
        self.fouriercoeffs_cos_list2 = nn.ModuleList()

        self.num_actions = num_actions

        for k in range(self.num_actions):
            self.fouriercoeffs_sin_list2.append(self.fouriercoeffs_sin)
            self.fouriercoeffs_cos_list2.append(self.fouriercoeffs_cos)
        
        self.apply(weights_init_)

        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                                                  (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                                                 (action_space.high + action_space.low) / 2.)


    def forward(self, state, action):

        # fourier coefficients
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))

        x1_fouriercoeffs_sin = []
        x1_fouriercoeffs_cos = []
        
        for dimA in range(self.num_actions):

            x1_fouriercoeffs_sin.append([self.fouriercoeffs_sin_list2[dimA][k](x1) for k in range(self.dimensionality)])
            x1_fouriercoeffs_cos.append([self.fouriercoeffs_cos_list2[dimA][k](x1) for k in range(self.dimensionality)])

        # x1_fouriercoeffs_sin is list_actions(list_coeffs(tensor_batchX1))
        # i is list with fourier coeffs for a given action dimension
        # i[k] is list of energy in each fourier mode, for each instance in batchX1
        # i[k][0] is energy in fourier mode 0, for each instance in batch
        # i[k][0] + i[k][1] + ... is total energy for each instance in batch; is a tensor batchX1
        # mean(i[k][0] + i[k][1] + ...) is avg. energy in action dimension 'k' for a sample instance in batch
        
#        energies = []
#        for k in range(self.dimensionality):
#            energies.append(
#                  (k>0)*round(sum([(
#                    torch.mean(i[k][0]*i[k][0]+i[k][1]*i[k][1]).item()
#                    ) for i in x1_fouriercoeffs_sin]),2) +
#                  round(sum([(
#                    torch.mean(i[k][0]*i[k][0]+i[k][1]*i[k][1]).item()
#                    ) for i in x1_fouriercoeffs_cos]),2)
#                  )
#        print('Q energy spectrum',[(' {:3.1f}'.format(i)) for i in energies])

        Qreconstructed = torch.zeros(x1_fouriercoeffs_sin[0][0].shape)
        
        action_m1p1 = (action - self.action_bias) / self.action_scale
        
#        for dimA in range(self.num_actions):
        if True:
            for k in range(self.dimensionality):

                Freconstructed_dim0 = (
                                    x1_fouriercoeffs_sin[0][k] * torch.sin(((k+1)*3.14)*action_m1p1[:,0].view(-1,1))
                                    +
                                    x1_fouriercoeffs_cos[0][k] * torch.cos(((k+1)*3.14)*action_m1p1[:,0].view(-1,1))
                                    )
                Freconstructed_dim1 = (
                                   x1_fouriercoeffs_sin[1][k] * torch.sin(((k+1)*3.14)*action_m1p1[:,1].view(-1,1))
                                   +
                                   x1_fouriercoeffs_cos[1][k] * torch.cos(((k+1)*3.14)*action_m1p1[:,1].view(-1,1))
                                   )

                Freconstructed_dim2 = (
                                    x1_fouriercoeffs_sin[2][k] * torch.sin(((k+1)*3.14)*action_m1p1[:,2].view(-1,1))
                                    +
                                    x1_fouriercoeffs_cos[2][k] * torch.cos(((k+1)*3.14)*action_m1p1[:,2].view(-1,1))
                                    )
                        
        Qreconstructed = (
#                          (Freconstructed_dim0) +
#                          (Freconstructed_dim1) +
#                          (Freconstructed_dim2)
                          torch.sigmoid(Freconstructed_dim0) +
                          torch.sigmoid(Freconstructed_dim1) +
                          torch.sigmoid(Freconstructed_dim2)
#                          torch.sigmoid(Freconstructed_dim0) *
#                          torch.sigmoid(Freconstructed_dim1) *
#                          torch.sigmoid(Freconstructed_dim2)
                          ) / self.num_actions
                          
        statevalue = F.relu(self.lineara(state))
        statevalue = F.relu(self.lineara2(statevalue))
        statevalue = self.linearb(statevalue)

#        actionvalue = F.relu(self.linearc(action))
#        actionvalue = F.relu(self.linearc2(actionvalue))
#        actionvalue = F.relu(self.lineard(actionvalue))

#        return ( statevalue ) + Qreconstructed
        return ( statevalue ) * Qreconstructed


class QNetworkQ1p1(nn.Module):
    def __init__(self, num_inputs, hidden_dim, action_space=None):
        super(QNetworkQ1p1, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state, actiond1):
        xu = torch.cat([state, actiond1], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        return x1


class Qbyactiondim(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, hidden_dim_fourier,
                 dimensionality = 10, action_space=None):
        super(Qbyactiondim, self).__init__()
        
        print('num_inputs        ',num_inputs)
        print('num_actions       ',num_actions)
        print('hidden_dim        ',hidden_dim)
        print('hidden_dim_fourier',hidden_dim_fourier)
        print('dimensionality    ',dimensionality)
        
        # dimensionality
        self.dimensionality = dimensionality
        
        # fourier coefficients, one per fourier mode per action dimension
        self.actiondim_list = nn.ModuleList()
        
        self.num_actions = num_actions
        
        for k in range(self.num_actions):
            self.actiondim_list.append(QNetworkQ1p1(num_inputs, hidden_dim))
        
        self.apply(weights_init_)


    def forward(self, state, action):
        
        return (
                sum([
                     self.actiondim_list[i](state, action[:,i].view(-1,1)) for i in range(self.num_actions)
                     ])
                     )


    def spectrum(self, state, action, std_batch, prob_batch, action_space = None, To = 2, modes = 10):

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        action_m1p1 = (action - self.action_bias) / self.action_scale

        spectrogram = list([])
        spectrogram_sin = list([])
        spectrogram_cos = list([])

        for i in range(self.num_actions):
            
            print('spectrogram, action dim',i)
            spectrogram.append(list([]))
            spectrogram_sin.append(list([]))
            spectrogram_cos.append(list([]))

            for k in range(modes):

                #uniform grid for integration in 'a' dimension
                gsize = modes*5 # if < modes, then strobelight effect
                agrid = -1
                action_m1p1[:,i] = agrid

                tempQvalue = self.actiondim_list[i](state, (self.action_bias[i] + action_m1p1[:,i]*self.action_scale[i]).view(-1,1) )
                spectrogram_sin[i].append( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                spectrogram_cos[i].append( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )

                for uidx in range(gsize):
                    agrid += (2/gsize) # ... will go from -1 to +1 in gsize steps
                    action_m1p1[:,i] = agrid
                    tempQvalue = self.actiondim_list[i](state, (self.action_bias[i] + action_m1p1[:,i]*self.action_scale[i]).view(-1,1) )
                    spectrogram_sin[i][k] += ( (1/gsize) * tempQvalue * torch.sin(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )
                    spectrogram_cos[i][k] += ( (1/gsize) * tempQvalue * torch.cos(((k+1)*2*3.14/To)*action_m1p1[:,i].view(-1,1)) )

                spectrogram_sin[i][k] = torch.pow(spectrogram_sin[i][k],2)
                spectrogram_cos[i][k] = torch.pow(spectrogram_cos[i][k],2)
                spectrogram[i].append(torch.pow(spectrogram_sin[i][k] + spectrogram_cos[i][k],1/2))
                spectrogram[i][k] = torch.sum(spectrogram[i][k]).item() # MC sum across random states
            spectrogram[i] = np.array(spectrogram[i]) # list --> array

        print('\n>> Q spectrum\n')

        energies = [sum(spectrogram[i]) for i in range(self.num_actions)]
        spectrogram = [spectrogram[i]/spectrogram[i][0] for i in range(self.num_actions)]

        print('energies , a0, {:13.2f} , a1, {:13.2f} , a2, {:13.2f}'.format(energies[0]/energies[0],energies[1]/energies[0],energies[2]/energies[0]))
        print('\n')

        for k in range(modes):
            print('w, {:5.2f} , a0, {:5.2f} , {:5.2f} , a1, {:5.2f} , {:5.2f} , a2, {:5.2f} , {:5.2f}'.format(
                                                                                                    (k+1)*2*3.14/To,
                                                                                                    spectrogram[0][k], torch.mean(1/std_batch[:,0]).item(),
                                                                                                    spectrogram[1][k], torch.mean(1/std_batch[:,1]).item(),
                                                                                                    spectrogram[2][k], torch.mean(1/std_batch[:,2]).item()
                                                                                                    ))

        if True:

            t = np.arange(0, modes, 1)
            splot = [np.array(spectrogram[i]) for i in range(self.num_actions)]

            ax1 = plt.subplot(311)
            plt.plot(t, splot[0])
            plt.setp(ax1.get_xticklabels(), fontsize=6)

            # share x only
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(t, splot[2])
            # make these tick labels invisible
            plt.setp(ax2.get_xticklabels(), visible=False)

            # share x and y
            ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
            plt.plot(t, splot[2])
            plt.xlim(0, modes)
            plt.show()

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # between -1 and +1
        action = y_t * self.action_scale + self.action_bias # between low and high
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std

    def sample_for_spectrogram(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # between -1 and +1
        action = y_t * self.action_scale + self.action_bias # between low and high
        log_prob_ = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob_ - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std, log_prob_

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
