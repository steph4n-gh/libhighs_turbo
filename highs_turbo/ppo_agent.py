import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple, Dict
from highs_turbo.surrogate_model import extract_topological_features, EquivariantMessagePassingLayer
from highs_turbo.graph_generator import GraphInstance
import itertools

class GATPolicyNetwork(nn.Module):
    """
    GNN-based Actor-Critic network for PPO.
    Uses EquivariantMessagePassingLayer from surrogate_model.py.
    Actor outputs mean and std for the continuous action space (multipliers).
    Critic outputs the value of the state.
    """
    def __init__(self, node_in_dim: int = 5, edge_in_dim: int = 8, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EquivariantMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Actor heads for cliques
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Mean multiplier should be >= 0
        )
        
        self.actor_logstd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Critic head (pools all nodes/edges to a single value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph: GraphInstance, k5_cliques: List[Tuple[int, int, int, int, int]]):
        node_feats, edge_feats, edge_to_idx = extract_topological_features(graph)
        
        if graph.num_edges == 0:
            return None, None, None, None

        edge_index = torch.tensor([[e[0] for e in graph.edges], [e[1] for e in graph.edges]], dtype=torch.long)

        h_n = self.node_encoder(node_feats)
        h_e = self.edge_encoder(edge_feats)

        for layer in self.layers:
            h_n, h_e = layer(h_n, h_e, edge_index)

        # Value prediction (Critic)
        global_n = h_n.mean(dim=0)
        global_e = h_e.mean(dim=0)
        state_repr = torch.cat([global_n, global_e], dim=-1)
        value = self.critic(state_repr)

        # Action prediction (Actor)
        valid_cliques = []
        e_indices_list = []
        for clq in k5_cliques:
            e_indices = []
            for u, v in itertools.combinations(clq, 2):
                e = (min(u, v), max(u, v))
                if e in edge_to_idx:
                    e_indices.append(edge_to_idx[e])
            if len(e_indices) == 10:
                e_indices_list.append(e_indices)
                valid_cliques.append(clq)
                
        if not valid_cliques:
            return value, torch.zeros(0), torch.zeros(0), valid_cliques
            
        e_indices_tensor = torch.tensor(e_indices_list, dtype=torch.long, device=h_e.device)
        clq_e_emb = h_e[e_indices_tensor].mean(dim=1)
        
        action_means = self.actor_mean(clq_e_emb).squeeze(-1)
        action_logstds = self.actor_logstd(clq_e_emb).squeeze(-1)
        
        # Clamp logstd to prevent numerical instability
        action_logstds = torch.clamp(action_logstds, -20.0, 2.0)
        action_std = action_logstds.exp()
        
        return value, action_means, action_std, valid_cliques


class PPOAgent:
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = GATPolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = GATPolicyNetwork()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.MseLoss = nn.MSELoss()
        
        self.memory = []

    def select_action(self, graph, k5_cliques):
        with torch.no_grad():
            value, action_mean, action_std, valid_cliques = self.policy_old(graph, k5_cliques)
            
            if len(valid_cliques) == 0:
                return {}, 0.0, None, None, None, None
                
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, min=0.0) # multipliers >= 0
            action_logprob = dist.log_prob(action)
            
        action_dict = {clq: float(act) for clq, act in zip(valid_cliques, action)}
        
        return action_dict, float(value), action, action_logprob, action_mean, action_std

    def store_transition(self, graph, k5_cliques, action, action_logprob, reward, value):
        self.memory.append((graph, k5_cliques, action, action_logprob, reward, value))
        
    def update(self):
        if not self.memory:
            return
            
        # Compute returns and advantages
        rewards = []
        discounted_reward = 0
        for transition in reversed(self.memory):
            reward = transition[4]
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            for (graph, k5_cliques, old_action, old_logprob, reward, old_value), ret in zip(self.memory, rewards):
                if old_action is None or len(old_action) == 0:
                    continue
                    
                value, action_mean, action_std, valid_cliques = self.policy(graph, k5_cliques)
                
                dist = Normal(action_mean, action_std)
                logprob = dist.log_prob(old_action)
                dist_entropy = dist.entropy()
                
                # Advantage
                advantage = ret - value.detach()
                
                # Ratio
                ratios = torch.exp(logprob - old_logprob)
                
                # Surrogate Loss
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(value.squeeze(), ret) - 0.01 * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()
