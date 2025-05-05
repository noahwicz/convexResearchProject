import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

def visualize_network_topologies(N=4, save_dir='./'):
    """Create visualizations of different network topologies
    
    Args:
        N: Number of nodes
        save_dir: Directory to save images
    """
    topologies = ['ring', 'star', 'complete']
    
    plt.figure(figsize=(15, 5))
    
    for idx, topology in enumerate(topologies):
        plt.subplot(1, 3, idx+1)
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        if topology == 'ring':
            # Ring topology
            for i in range(N):
                G.add_edge(i, (i+1) % N)
            pos = nx.circular_layout(G)
            title = 'Ring Topology'
        
        elif topology == 'star':
            # Star topology
            for i in range(1, N):
                G.add_edge(0, i)
            pos = nx.spring_layout(G, seed=42)
            title = 'Star Topology'
        
        elif topology == 'complete':
            # Complete topology
            for i in range(N):
                for j in range(i+1, N):
                    G.add_edge(i, j)
            pos = nx.circular_layout(G)
            title = 'Complete Topology'
        
        # Draw network
        nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                node_size=700, font_size=15, font_weight='bold')
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}network_topologies.pdf')
    plt.close()

def visualize_communication_cost(N_range=range(4, 21, 4), save_dir='./'):
    """
    Visualize communication cost scaling for different algorithms and topologies
    
    Args:
        N_range: Range of network sizes to visualize
        save_dir: Directory to save visualization
    """
    # Total communication cost per iteration (messages sent/received)
    dsgt_ring_cost = [2*N for N in N_range]  # Each node sends/receives to/from 2 neighbors
    dsgt_star_cost = [N*(N-1) for N in N_range]  # Hub sends/receives from N-1 nodes, leaves send/receive from hub
    dsgt_complete_cost = [N*(N-1) for N in N_range]  # Each node sends/receives to/from all other N-1 nodes
    fedavg_cost = [2*N for N in N_range]  # Each node sends/receives to/from server
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_range, dsgt_ring_cost, 'o-', label='DSGT - Ring')
    plt.plot(N_range, dsgt_star_cost, 's-', label='DSGT - Star')
    plt.plot(N_range, dsgt_complete_cost, '^-', label='DSGT - Complete')
    plt.plot(N_range, fedavg_cost, 'x-', label='FedAvg')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Total Messages per Iteration')
    plt.title('Communication Cost Scaling')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}communication_cost.pdf')
    plt.close()

def compare_convergence_rates(N_range=range(4, 21, 4), save_dir='./'):
    """
    Compare theoretical convergence rates vs. network size
    
    Args:
        N_range: Range of network sizes to visualize
        save_dir: Directory to save visualization
    """
    # Theoretical spectral gaps (1-ρ) for different topologies
    ring_gaps = [1 - np.cos(2*np.pi/N) for N in N_range]  # Approximation for ring
    star_gaps = [0.5 for _ in N_range]  # Constant for star with proper weights
    complete_gaps = [0.75 for _ in N_range]  # Constant for complete with proper weights
    
    # Theoretical convergence rates (relative to optimal, lower is better)
    dsgt_ring_rate = [1/gap for gap in ring_gaps]
    dsgt_star_rate = [1/gap for gap in star_gaps]
    dsgt_complete_rate = [1/gap for gap in complete_gaps]
    fedavg_rate = [1 for _ in N_range]  # Independent of topology
    
    # Normalize by the rate for N=4
    norm_factor = dsgt_complete_rate[0]
    dsgt_ring_rate = [r/norm_factor for r in dsgt_ring_rate]
    dsgt_star_rate = [r/norm_factor for r in dsgt_star_rate] 
    dsgt_complete_rate = [r/norm_factor for r in dsgt_complete_rate]
    fedavg_rate = [r/norm_factor for r in fedavg_rate]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_range, dsgt_ring_rate, 'o-', label='DSGT - Ring')
    plt.plot(N_range, dsgt_star_rate, 's-', label='DSGT - Star')
    plt.plot(N_range, dsgt_complete_rate, '^-', label='DSGT - Complete')
    plt.plot(N_range, fedavg_rate, 'x-', label='FedAvg')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Relative Convergence Time (lower is better)')
    plt.title('Theoretical Convergence Scaling with Network Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}convergence_scaling.pdf')
    plt.close()

if __name__ == "__main__":
    # Run the main experiment first to get the results
    from distributed_optimization import run_experiment
    
    results = run_experiment(N=4, d=5, m=20, max_iter=100)
    
    # Create visualizations
    visualize_network_topologies()
    
    # Communication cost analysis
    visualize_communication_cost()
    
    # Convergence rate analysis
    compare_convergence_rates()