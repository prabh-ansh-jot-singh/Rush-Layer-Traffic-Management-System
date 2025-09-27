import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import GCNConv


# ----------------------------
# GNN Q-Network (per-node actions)
# ----------------------------
class GNN_QNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_actions):
        super(GNN_QNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, edge_index):
        """
        Input:
            x: [num_nodes, num_node_features]
            edge_index: [2, num_edges]
        Output:
            Q-values per node: [num_nodes, num_actions]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.fc(x)  # shape [num_nodes, num_actions]


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ----------------------------
# DQN Agent with GNN
# ----------------------------
class GNN_DQNAgent:
    def __init__(self, num_node_features, hidden_dim, num_actions, device="cpu"):
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device

        self.model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.target_model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=50000)
        self.gamma = 0.95

    # ----------------------------
    # Choose action (per-node)
    # ----------------------------
    def act(self, x, edge_index, epsilon=0.1):
        """
        Returns an array of actions, one per node.
        """
        num_nodes = x.shape[0]

        # Normalize features before feeding into GNN
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-6)

        if np.random.rand() <= epsilon:
            return np.random.randint(self.num_actions, size=num_nodes)
        else:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            q_values = self.model(x, edge_index)  # shape [num_nodes, num_actions]
            return torch.argmax(q_values, dim=1).cpu().numpy()  # shape [num_nodes]

    # ----------------------------
    # Store transition
    # ----------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    # ----------------------------
    # Vectorized replay
    # ----------------------------
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        state_x = torch.tensor(np.concatenate([s[0][None] for s in states]),
                               dtype=torch.float).to(self.device)  # [B, num_nodes, num_features]
        state_edge = states[0][1]  # assume same graph structure
        state_edge = torch.tensor(state_edge, dtype=torch.long).to(self.device)

        next_state_x = torch.tensor(np.concatenate([ns[0][None] for ns in next_states]),
                                    dtype=torch.float).to(self.device)
        next_edge = torch.tensor(next_states[0][1], dtype=torch.long).to(self.device)

        actions = torch.tensor(np.stack(actions), dtype=torch.long).to(self.device)  # [B, num_nodes]
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)           # [B]
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)               # [B]

        # Normalize features
        state_x = (state_x - state_x.mean(dim=1, keepdim=True)) / (state_x.std(dim=1, keepdim=True) + 1e-6)
        next_state_x = (next_state_x - next_state_x.mean(dim=1, keepdim=True)) / (next_state_x.std(dim=1, keepdim=True) + 1e-6)

        # Q-values for current states
        B, N, F = state_x.shape
        q_values = self.model(state_x.view(-1, F), state_edge)  # [B*N, num_actions]
        q_values = q_values.view(B, N, -1)

        # Q-values for actions taken
        chosen_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_state_x.view(-1, F), next_edge)  # [B*N, num_actions]
            next_q = next_q.view(B, N, -1).max(dim=2)[0]  # [B, N]
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # Loss
        loss = self.criterion(chosen_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ----------------------------
    # Sync target network
    # ----------------------------
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ----------------------------
    # Save & Load
    # ----------------------------
    def save(self, path="data/gnn_dqn.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="data/gnn_dqn.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.update_target_model()
            print(f"✅ Model loaded from {path}")
