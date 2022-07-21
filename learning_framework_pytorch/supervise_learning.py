import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ENVIRONMENT import NbackEnv
from MLP import MLP
import torch
import torch.nn as nn
import torch.optim as optim
# supervised learning solve n-back problem like XOR
# input: [s_t-2, s_t-1, s_t]
# y_true=1 if s_t-2==s_t
N=2
# for n-back
epoch=10
epoch_steps=100
stimuli_choices=[1,2,3,4,5,6]
lr=0.001
# learning rate of optimizer
env=NbackEnv(N,epoch_steps,stimuli_choices)
epoch_loss=np.zeros(epoch)
net=MLP(in_dim=N+1, out_dim=2, hidden_dims=[6,3])
print(net)
optimizer=optim.Adam(net.parameters(), lr=lr)

for e in tqdm(range(epoch)):
    stimuli=env.generate_stimuli()
    sti_mat=np.vstack([stimuli[:-3],stimuli[1:-2],stimuli[2:-1]])
    X_input= sti_mat.T
    X_tensor=torch.tensor(X_input,dtype=torch.float)
    y_true= stimuli[:-3]==stimuli[2:-1]
    y_true=np.multiply(y_true, 1)
    y_true_tensor=torch.tensor(y_true,dtype=torch.float)
    y_predict=net.forward(X_tensor)
    loss_fun=nn.BCELoss()
    loss=loss_fun(y_predict,y_true_tensor)
    epoch_loss[e]=loss
    loss_tensor=torch.tensor(loss,requires_grad=True)
    loss_tensor.backward()
    optimizer.step()

plt.plot(epoch_loss)
plt.show()
