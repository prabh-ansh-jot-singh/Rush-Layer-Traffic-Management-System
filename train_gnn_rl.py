import matplotlib.pyplot as plt
import numpy as np
from sumo_env import SumoEnvironment
from gnn_rl_agent import GNN_DQNAgent

# ----------------------------
# Hyperparameters
# ----------------------------
num_node_features = 3
hidden_dim = 64
num_actions = 4
episodes = 20               # more episodes
batch_size = 64

epsilon = 1.0
epsilon_min = 0.1            # never fully greedy
epsilon_decay = 0.995

sumo_cfg = "data/net.sumocfg"
sumo_binary = "sumo"

# ----------------------------
# Init environment & agent
# ----------------------------
env = SumoEnvironment(
    sumo_cfg=sumo_cfg,
    sumo_binary=sumo_binary,
    num_node_features=num_node_features,
    num_actions=num_actions,
    max_steps=500,
    fixed_num_nodes=None,
    topology="fully_connected"
)

agent = GNN_DQNAgent(num_node_features, hidden_dim, num_actions)

# ----------------------------
# Training Loop
# ----------------------------
rewards_per_episode = []

def compute_reward(env, next_state, reward_from_env):
    features, _ = next_state
    halting = np.sum(features[:, 0])  # halting vehicles
    count = np.sum(features[:, 1])    # total vehicles
    avg_speed = np.mean(features[:, 2]) if len(features) > 0 else 0

    reward = (
        reward_from_env
        - 0.5 * halting
        - 0.2 * count
        + 1.0 * avg_speed
    )
    return reward

for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        state_x, edge_index = state

        # Agent chooses actions
        actions = agent.act(state_x, edge_index, epsilon=epsilon)

        # Step SUMO
        next_state, reward_from_env, done = env.step(actions)

        # Enhanced reward
        reward = compute_reward(env, next_state, reward_from_env)

        # Store transition
        agent.remember(state, actions, reward, next_state, done)

        # Train only after warm-up
        if len(agent.memory) > batch_size * 5:
            agent.replay(batch_size=batch_size)

        state = next_state
        total_reward += reward

    # Update target network every 5 episodes
    if (ep + 1) % 5 == 0:
        agent.update_target_model()

    rewards_per_episode.append(total_reward)

    # Decay epsilon but never below 0.1
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# ----------------------------
# Save model
# ----------------------------
agent.save("data/gnn_dqn_best.pth")
env.close()
print("✅ Training finished and model saved.")

# ----------------------------
# Plot rewards
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(rewards_per_episode, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve (Improved Reward + Epsilon)")
plt.legend()
plt.grid(True)
plt.show()
