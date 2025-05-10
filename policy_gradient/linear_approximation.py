import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.game import Action, State, Params, Game, Agent, Nums_Data
from utils.plot import plot_mse, plot_mse_group, plot_v
from tqdm import tqdm
from utils.compute_fn import compute_MSE, feature_vector_pos, MapW2Q
import numpy as np
import copy


def linear_approximation(data: Nums_Data, params:Params, q_star=None):
    dealer_max_nums = data.dealer_max_nums
    agent_max_nums = data.agent_max_nums
    episodes = data.episodes
    epsilon = data.eps
    step_size = data.step_size

    lambda_value = params.lambda_value
    gamma_value = params.gamma_value
    alpha_value = params.alpha_value

    dealer_cubio = [np.arange(1,5),np.arange(4,8),np.arange(7,11)]
    player_cubio = [np.arange(1,7),np.arange(4,10),np.arange(7,13),np.arange(10,16),np.arange(13,19),np.arange(16,22)]

    game = Game()
    # 在linear approximation中，使用feature vector和一个可优化参数向量w 相乘用于表示
    # 随机一个w向量，在这个设定里是36维的，然后开始进行更新
    w = np.random.rand(36)
    player = Agent()
    Q = np.zeros((dealer_max_nums+1, agent_max_nums+1, 2))
    mse = []
    for e in tqdm(range(episodes),desc="Linear_Approximation"):
        terminated = False
        E = np.zeros(36)
        s:State = State(game.initial_card(), game.initial_card(), False)
        a:Action = Action.HIT

        while not terminated:
            s_prime = copy.copy(s)
            a_prime:Action = copy.copy(a)
            s_prime, r, terminated = game.step(s, a)

            
            x = feature_vector_pos(s=s, a=a, dealer_cubio=dealer_cubio, player_cubio=player_cubio)
            if not terminated:
                # Q: q在这个时候只是一个值要怎么进行贪心策略的更新呢？
                # A: 传两个q进去让它比就好了，选择的新策略是一个状态下，Q值更高的那个，因此只需要把两个动作的Q值都给它就好了
                stick_q = np.dot(feature_vector_pos(s=s_prime, a=Action.STICK, dealer_cubio=dealer_cubio, player_cubio=player_cubio), w)
                hit_q = np.dot(feature_vector_pos(s=s_prime, a=Action.HIT, dealer_cubio=dealer_cubio, player_cubio=player_cubio), w)
                a_prime = player.linear_policy(flag=0, eps=epsilon) if stick_q > hit_q else player.linear_policy(flag=1, eps=epsilon)
                # print(a_prime)
                TD_Error = r + gamma_value * np.dot(feature_vector_pos(s=s_prime, a=a_prime, dealer_cubio=dealer_cubio, player_cubio=player_cubio), w) - np.dot(x, w)
            else: TD_Error = r - np.dot(x, w)
            
            E = gamma_value * lambda_value * E
            E[x == 1] += 1
            gradient_w = step_size * TD_Error * E
            
            w += gradient_w
            
            s = s_prime
            a = a_prime

        if e % 1000 == 0:
            Q = MapW2Q(w, Q,dealer_cubio, player_cubio)
            mse.append(compute_MSE(Q, q_star))
    return Q[1:,1:,:], mse

if __name__ == '__main__':
    params = Params(lambda_value=1, gamma_value=1, alpha_value=0.03)
    data = Nums_Data(dealer_max_nums=10, agent_max_nums=21, episodes=10, eps=0.05, step_size=0.01)
    q_star = np.load("q_star.npy")
    _lambda = np.arange(0, 1.0, 0.1)
    t_mse = []
    for l in _lambda:
        print(f"lambda is {round(l, 4)}")
        params = Params(lambda_value=l, gamma_value=1, alpha_value=0.03)
        q, mse = linear_approximation(data, params, q_star)
        t_mse.append(mse)
    v = np.max(q,axis=-1)
    plot_mse_group(t_mse)
    plot_v(data.dealer_max_nums, data.agent_max_nums, v)
    
    
    