import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Normal

from neural_surrogate.surrogate_model import EdgeEquivariantSurrogateGNN
from neural_surrogate.large_scale_benchmarks import LargeScaleBenchmarkSuite, generate_pegasus_instance, generate_chimera_instance, generate_gset_instance
from neural_surrogate.graph_generator import generate_exp91_family

def ppo_train_curriculum():
    print("Starting PPO RL Curriculum Training...")
    # Initialize model
    model = EdgeEquivariantSurrogateGNN(hidden_dim=32, num_layers=2)
    
    # (Removed degenerate initialization)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    
    # Generate curriculum
    print("Generating curriculum graphs...")
    graphs = []
    
    # 1. Exp 91/92 cluster families
    try:
        graphs.extend(generate_exp91_family().values())
    except Exception as e:
        print(f"Skipping exp91: {e}")
        
    # 2. D-Wave Pegasus & Chimera subgraphs
    try:
        graphs.append(generate_pegasus_instance(m=4, seed=42))
        graphs.append(generate_chimera_instance(m=4, n=4, l=4, seed=42, embed_cliques=True, num_embedded_cliques=5))
    except Exception as e:
        print(f"Skipping quantum hardware graphs: {e}")
        
    # 3. Random regular graphs (if available, else fallback)
    print(f"Total graphs in curriculum: {len(graphs)}")
    
    suite = LargeScaleBenchmarkSuite(seed=42)
    
    epochs = 5
    gamma = 0.99
    eps_clip = 0.2
    
    model.train()
    for epoch in range(epochs):
        epoch_reward = 0.0
        for graph in graphs:
            if graph.num_edges == 0:
                continue
                
            # Forward pass
            out = model(graph)
            tensor_k5 = out.get("k5_tensor_multipliers", {})
            if not tensor_k5:
                continue
                
            cliques = list(tensor_k5.keys())
            means = torch.stack([tensor_k5[c] for c in cliques])
            
            # PPO-like exploration (REINFORCE with clipped surrogate for simplicity)
            std = torch.ones_like(means) * 1e-8
            dist = Normal(means, std)
            action = dist.sample()
            action_clamped = torch.clamp(action, min=1e-4) # strict positivity
            log_prob = dist.log_prob(action).sum()
            
            # Evaluate reward via exact verification
            action_dict = {c: float(action_clamped[i].item()) for i, c in enumerate(cliques)}
            
            try:
                cert = suite.verifier.verify_clique_conic_combination(graph, action_dict)
                if not cert.is_valid:
                    reward = -10.0
                else:
                    from neural_surrogate.exact_solver import ExactMaxCutSolver
                    from scipy.optimize import linprog
                    import numpy as np
                    
                    ex_solver = ExactMaxCutSolver()
                    c, A, b, _, _ = ex_solver.build_relaxation_matrices(graph, include_k5=False)
                    
                    res_base = linprog(c, A_ub=A, b_ub=b, method="highs")
                    
                    n_vars = len(c)
                    a_surr = np.zeros(n_vars, dtype=np.float64)
                    for i, e in enumerate(graph.edges):
                        if e in cert.exact_coefficients:
                            coeff = float(cert.exact_coefficients[e])
                            if i < n_vars:
                                a_surr[i] = coeff
                                
                    A_aug = np.vstack([np.asarray(A, dtype=np.float64), a_surr.reshape(1, -1)])
                    b_aug = np.append(np.asarray(b, dtype=np.float64).flatten(), float(cert.exact_rhs))
                    
                    res_cut = linprog(c, A_ub=A_aug, b_ub=b_aug, method="highs")
                    
                    iters_base = res_base.nit if hasattr(res_base, 'nit') else 0
                    iters_cut = res_cut.nit if hasattr(res_cut, 'nit') else 0
                    
                    simplex_reduction = float(iters_base - iters_cut)
                    gap_improvement = float(res_base.fun - res_cut.fun) if res_base.success and res_cut.success else 0.0
                    
                    reward = simplex_reduction + gap_improvement
            except Exception as e:
                reward = -10.0
                
            advantage = reward - 0.0
            loss = -log_prob * advantage
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_reward += reward
            
        print(f"Epoch {epoch+1}/{epochs} - Avg Reward: {epoch_reward/len(graphs):.4f}")
        
    # Save default weights
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_surrogate", "default_weights.pt")
    if not os.path.exists(os.path.dirname(weights_path)):
        weights_path = os.path.join(os.getcwd(), "neural_surrogate", "default_weights.pt")
            
    torch.save(model.state_dict(), weights_path)
    print(f"Saved default weights to {weights_path}")

if __name__ == "__main__":
    ppo_train_curriculum()
