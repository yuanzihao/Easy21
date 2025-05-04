import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plot_v(dealer_max_nums, agent_max_nums,V):
    X, Y = np.meshgrid(
        np.arange(0, agent_max_nums),
        np.arange(0, dealer_max_nums)
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Agent State')
    ax.set_ylabel('Dealer State')
    ax.set_zlabel('Value (V)')
    ax.set_title('Plot of Value')

    plt.show()


def plot_mse(mse_list):
    x = np.arange(0, len(mse_list))
    y = mse_list
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
    plt.title("Simple Line Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def plot_mse_group(mse_group):
    color = sns.color_palette("husl", n_colors=len(mse_group))
    for i, mse_list in enumerate(mse_group):
        x = np.arange(0, len(mse_list))
        y = mse_list
        plt.plot(x, y, label=f"lambda{round(i+1/10., 4)}", color=color[i], linewidth=2)
    plt.title("Simple Line Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()