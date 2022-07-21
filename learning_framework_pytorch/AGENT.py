import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from MLP import MLP

class Agent():
    def __init__(self,N,lr,hidden_dims,len_buffer=100):
        # network input dim:N+1 
        # input is observations
        # network output dim:2
        # output is action probability for match or non-match
        self.lr=lr
        self.net=MLP(in_dim=N+1, out_dim=2, hidden_dims=hidden_dims)
        print(self.net)
        self.optimizer=optim.Adam(self.net.parameters(), lr=self.lr)
        self.len_buffer=len_buffer

    def init_replay_buffer(self):
        buffer=np.ones((1,5))
        return buffer

    def save_experience(self,obs,act,reward,buffer):
        # obs is a list of three stimuli
        # buffer has 5 columns
    
        sample=np.append(obs,(act,reward))
        buffer=np.vstack([buffer,sample])
        return buffer

    def  choose_action(self,X,if_single=False):

        X=torch.tensor(X,dtype=torch.float)
        action=self.net.forward(X)
        # action_prob_np=action_prob_tensor.detach().numpy()
        # print(action_prob_np)
        # prob_range=np.append([-1e-5],np.cumsum(action_prob_np))
        # action=sum(prob_range<np.random.uniform())-1
        # action=np.argmax(action_prob_np,axis=1)
        # print(action)
        if if_single:
            action=action[0]
        return action

    def sample_from_buffer(self,buffer):
        # to extract 50% random samples
        length=np.size(buffer,0)
        samples_idx=np.random.choice(length,int(length*0.5))
        samples=buffer[samples_idx,:]
        return samples
        
    def cost(self,y_true,y_predict):
        # count number of wrong action as loss  
        # loss=sum(y_true==y_predict)/len(y_true)
        # binary cross entropy loss
        loss_fun=nn.BCELoss()
        loss=loss_fun(y_predict,y_true)
        return loss

    def learn_from_sample(self,sample_batch):
        self.optimizer.zero_grad()
        # clear gradient
        X_input=np.array(sample_batch[:,:3])
        # print(X_input)
        # X is stimuli, y is action
        y_predict=self.choose_action(X_input)
        # y_true= sample_batch[:,0]==sample_batch[:,2]
        # this is the reward function and only known to environment
        # agent can only get y_true compare action and reward
        # but the result should be the same 
        y_true= sample_batch[:,-1]==sample_batch[:,-2]
        y_true=np.multiply(y_true, 1)
        y_true_tensor=torch.tensor(y_true,dtype=torch.float)
        # bool to int 0 or 1
        loss=self.cost(y_true_tensor,y_predict)
        loss_tensor=torch.tensor(loss,requires_grad=True)
        loss_tensor.backward()
        # self.optimizer=optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.step()
        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # print(list(self.net.parameters())[0].grad)
        return loss
