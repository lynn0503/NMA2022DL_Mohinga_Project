import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ENVIRONMENT import NbackEnv
from AGENT import Agent

N=2
num_episode=11
episode_steps=100
# len_buffer=1000
stimuli_choices=[1,2,3,4,5,6]
lr=0.001
env=NbackEnv(N,episode_steps,stimuli_choices)
agent=Agent(N,lr,hidden_dims=[5,5])

# training
return_train=np.zeros((num_episode,episode_steps))
running_loss=np.zeros((num_episode,episode_steps))
for e in tqdm(range(num_episode)):
    stimuli=env.generate_stimuli()
    # print(stimuli)
    buffer=agent.init_replay_buffer()
    for t in range(episode_steps):
        # print(e," episode ",t," trial")
        if t<2:
            continue
        obs=env.generate_observation(stimuli,t)
        act=agent.choose_action([obs],if_single=True)
        reward=env.give_reward(obs,act)
        return_train[e,t]=reward
        buffer=agent.save_experience(obs,act,reward,buffer)
        if t>2:
            sample_batch=agent.sample_from_buffer(buffer)
            loss=agent.learn_from_sample(sample_batch)
            running_loss[e,t]=loss

return_cumsum=np.cumsum(return_train,axis=1)
return_sum=np.sum(return_train,axis=1)
ax1=plt.subplot(121)
ax1.plot(return_sum)
ax1.set_title("cumulative reward by step")

running_loss_mean=np.mean(running_loss,axis=1)
ax2=plt.subplot(122)
ax2.plot(running_loss_mean)
ax2.set_title("loss by episode")
plt.show()

# testing
# return_test=np.zeros(num_episode,episode_steps)
# for e in num_episode:
#     stimuli=env.generate_stimuli(episode_steps)
#     for t in episode_steps:
#         obs=env.generate_observation(stimuli,t)
#         act=agent.choose_action(obs)
#         reward=env.give_reward(act)
#         return_train[e,t]=reward

