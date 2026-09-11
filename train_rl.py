import sys
import os

# Ensure libhighs_turbo is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from highs_turbo.api import TurboSolver
from highs_turbo.graph_generator import generate_exp91_family, generate_k5_cluster_graph
from highs_turbo.rl_environment import HiGHSTreeEnv
from highs_turbo.ppo_agent import PPOAgent

def main():
    print("Initializing RL environment and generating graphs...")
    
    # Generate some training graphs (e.g., Exp91 family with K5 clusters)
    # generate_exp91_family returns a dict, we want a list of GraphInstances
    graphs_dict = generate_exp91_family(num_graphs=5, seed=42)
    graphs = list(graphs_dict.values())
    
    solver = TurboSolver()
    env = HiGHSTreeEnv(solver=solver, graphs=graphs, max_cuts=50)
    agent = PPOAgent(lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4)
    
    num_episodes = 20
    update_timestep = 5
    timestep = 0
    
    print("Starting PPO Training Loop...")
    for ep in range(1, num_episodes + 1):
        graph, info = env.reset()
        k5_cliques = graph.find_all_k5_cliques()
        
        # Select action
        action_dict, value, action_tensor, action_logprob, _, _ = agent.select_action(graph, k5_cliques)
        
        # Step environment
        _, reward, done, step_info = env.step(action_dict)
        
        # Store transition
        agent.store_transition(graph, k5_cliques, action_tensor, action_logprob, reward, value)
        
        print(f"Episode {ep:03d} | Graph: {graph.name} | Reward: {reward:.4f} | "
              f"Pivots: {step_info['baseline_pivots']} -> {step_info['accel_pivots']} "
              f"(Reduction: {step_info['pivot_reduction']}) | "
              f"Gap Closure: {step_info['gap_closure']:.4f}")
              
        timestep += 1
        
        if timestep % update_timestep == 0:
            print("Updating PPO Policy...")
            agent.update()
            
    print("Training complete! The RL agent is ready for production use.")

if __name__ == "__main__":
    main()
