import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.game import *

# 需要一个策略模型，一个新的agent

class Policy_net(torch.nn.Module):
    '''
    input_size: 2, 一个是对手的牌值，另一个是当前自己的牌值
    output_size: 1, 0代表STICK，1代表HIT
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy_net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class agent:
    def __init__(self,input_size,hidden_size,output_size, device, learning_rate, gamma):
        self.policy_net = Policy_net(input_size, hidden_size, output_size).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def take_action(self, state:State):
        state_tensor = torch.tensor([state.dealer_sum, state.agent_sum], dtype=torch.float32).to(self.device)
        state_tensor = state_tensor.view(1,-1)
        action_probs = self.policy_net(state_tensor)
        action = torch.distributions.Categorical(action_probs).sample()
        return Action.STICK if action == 0 else Action.HIT

    def update(self, episode_list):
        state_list = episode_list['states']
        action_list = episode_list['actions']
        reward_list = episode_list['rewards']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i].dealer_sum, state_list[i].agent_sum], dtype=torch.float32).view(1,-1).to(self.device)
            action_convert = 0 if action_list[i] == Action.STICK else 1
            action = torch.tensor(action_convert).view(-1, 1).to(self.device)
            G += reward + self.gamma * G
            log_prob = torch.log(self.policy_net(state).gather(1,action)).to(self.device)
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    input_size = 2
    hidden_size = 64
    output_size = 2
    learning_rate = 0.001
    gamma = 0.98

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    player = agent(input_size, hidden_size, output_size, device, learning_rate, gamma)

    iteration_num = 4
    episode_num = 1000

    return_list = []

    for i in range(iteration_num):
        with tqdm(total=int(episode_num / 10), desc="Iteration %d" % i) as pbar:
            for e in range(int(episode_num / 10)):
                episode_return = 0
                episode_list = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'terminates': []
                }
                game = Game()
                terminate = False
                s:State = State(game.initial_card(), game.initial_card(), False)
                a:Action = Action.HIT
                T = 0
                while not terminate and T < 200:
                    # action = player.take_action(s)
                    next_s, r, terminate = game.step(s, a)
                    episode_list['states'].append(s)
                    episode_list['actions'].append(a)
                    episode_list['rewards'].append(r)
                    episode_list['next_states'].append(next_s)
                    s = next_s
                    a = player.take_action(s)
                    episode_return += r
                    T += 1
                return_list.append(episode_return)
                player.update(episode_list)
                if (e + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (episode_num / 10 * i + e + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
