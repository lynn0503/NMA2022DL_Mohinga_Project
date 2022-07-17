import numpy as np

class NbackEnv():
    def __init__(self,N,episode_steps,stimuli_choices,seed=1):
        self.N=N
        self.episode_steps=episode_steps
        self.stimuli_choices=stimuli_choices

    def generate_stimuli(self):
        # self.stimuli=np.random.choice(self.stimuli_choices, self.episode_steps)
        stimuli=np.ones(self.episode_steps)
        # 50% match and 50% non-match
        for i in range(self.episode_steps):
            if i<self.N:
                continue
            rand=np.random.random()
            if rand <0.5:
                stimuli[i]=stimuli[i-self.N]
            else:
                stimuli[i]=np.random.choice(len(self.stimuli_choices))
        return stimuli
        
    def generate_observation(self,stimuli,t):
        return stimuli[t-self.N:t+1]

    def give_reward(self,obs,act):
        expected_action= obs[-1]==obs[0]
        return expected_action==bool(act)
        