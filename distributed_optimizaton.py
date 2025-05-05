import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(N, d, m, heterogeneity=0.5):
    """
    Generate synthetic data for linear regression
    Args:
        N: Number of agents
        d: Dimension of feature vectors
        m: Number of samples per agent
        heterogeneity: Factor controlling data heterogeneity among agents
    
    Returns:
        A_list: List of feature matrices for each agent
        b_list: List of target vectors for each agent
        w_true: True parameter vector
    """
    # Generate true parameter vector
    w_true = np.random.randn(d)
    
    A_list = []
    b_list = []
    
    for i in range(N):
        # Create heterogeneous mean for each agent
        mean_i = heterogeneity * (i - (N-1)/2) * np.ones(d)
        
        # Generate features
        A_i = np.random.randn(m, d) + mean_i
        
        # Generate targets with noise
        noise = 0.1 * np.random.randn(m)
        b_i = A_i @ w_true + noise
        
        A_list.append(A_i)
        b_list.append(b_i)
    
    return A_list, b_list, w_true

def compute_global_loss(w, A_list, b_list):
    """Compute global loss function value"""
    total_loss = 0
    total_samples = 0
    
    for A_i, b_i in zip(A_list, b_list):
        m_i = A_i.shape[0]
        loss_i = 0.5 * np.linalg.norm(A_i @ w - b_i)**2 / m_i
        total_loss += loss_i
        total_samples += m_i
        
    return total_loss

def compute_optimal_solution(A_list, b_list):
    """Compute the optimal solution w* using normal equations"""
    d = A_list[0].shape[1]
    AtA = np.zeros((d, d))
    Atb = np.zeros(d)
    
    for A_i, b_i in zip(A_list, b_list):
        m_i = A_i.shape[0]
        AtA += A_i.T @ A_i / m_i
        Atb += A_i.T @ b_i / m_i
    
    w_star = np.linalg.solve(AtA, Atb)
    return w_star

def compute_consensus_error(x_list):
    """Compute consensus error: sum of squared distances from average"""
    x_avg = np.mean(x_list, axis=0)
    error = 0
    for x_i in x_list:
        error += np.linalg.norm(x_i - x_avg)**2
    return error

def create_mixing_matrix(topology, N, beta=0.6):
    """
    Create mixing matrix W for different topologies
    Args:
        topology: 'ring', 'star', or 'complete'
        N: Number of agents
        beta: Self-weight parameter
    
    Returns:
        W: Mixing matrix
    """
    W = np.zeros((N, N))
    
    if topology == 'ring':
        for i in range(N):
            W[i, i] = beta
            W[i, (i-1) % N] = (1-beta)/2
            W[i, (i+1) % N] = (1-beta)/2
    
    elif topology == 'star':
        # Node 0 is the hub
        hub_neighbors = N - 1
        W[0, 0] = beta
        for i in range(1, N):
            W[0, i] = (1-beta)/hub_neighbors  # Hub to leaf
            W[i, 0] = 1-beta                  # Leaf to hub
            W[i, i] = beta                    # Self-weight for leaf
    
    elif topology == 'complete':
        for i in range(N):
            W[i, i] = beta
            for j in range(N):
                if j != i:
                    W[i, j] = (1-beta)/(N-1)
    
    return W

def estimate_spectral_gap(W, N):
    """Estimate spectral gap (1-ρ) of the mixing matrix"""
    # Compute eigenvalues of W - (1/N)11^T
    correction = np.ones((N, N)) / N
    adjusted_W = W - correction
    eigenvalues = np.linalg.eigvals(adjusted_W)
    spectral_radius = max(abs(eigenvalues))
    return 1 - spectral_radius

def dsgt(A_list, b_list, W, alpha, max_iter, w_star=None):
    """
    Distributed Stochastic Gradient Tracking
    
    Args:
        A_list: List of feature matrices for each agent
        b_list: List of target vectors for each agent
        W: Mixing matrix
        alpha: Step size
        max_iter: Maximum number of iterations
        w_star: Optional optimal solution for computing optimality gap
    
    Returns:
        loss_history: History of global loss function values
        consensus_history: History of consensus errors
        x_history: History of agent parameter vectors
    """
    N = len(A_list)
    d = A_list[0].shape[1]
    
    # Initialize parameter vectors and gradient trackers
    x_list = [np.zeros(d) for _ in range(N)]
    
    # Compute initial gradients
    y_list = []
    for i in range(N):
        m_i = A_list[i].shape[0]
        grad_i = A_list[i].T @ (A_list[i] @ x_list[i] - b_list[i]) / m_i
        y_list.append(grad_i)
    
    # History for logging
    loss_history = []
    consensus_history = []
    x_history = []
    
    # Compute initial metrics
    x_avg = np.mean(x_list, axis=0)
    loss = compute_global_loss(x_avg, A_list, b_list)
    consensus_error = compute_consensus_error(x_list)
    
    loss_history.append(loss)
    consensus_history.append(consensus_error)
    x_history.append(x_list.copy())
    
    # Main loop
    for t in range(max_iter):
        # Store old parameters and gradients
        x_list_old = x_list.copy()
        y_list_old = y_list.copy()
        
        # Update x
        x_list_new = []
        for i in range(N):
            # Consensus step
            x_i_consensus = sum(W[i, j] * x_list_old[j] for j in range(N))
            # Gradient descent step
            x_i_new = x_i_consensus - alpha * y_list[i]
            x_list_new.append(x_i_new)
        
        x_list = x_list_new
        
        # Update y
        y_list_new = []
        for i in range(N):
            # Compute new gradient
            m_i = A_list[i].shape[0]
            grad_i_new = A_list[i].T @ (A_list[i] @ x_list[i] - b_list[i]) / m_i
            grad_i_old = A_list[i].T @ (A_list[i] @ x_list_old[i] - b_list[i]) / m_i
            
            # Consensus step
            y_i_consensus = sum(W[i, j] * y_list_old[j] for j in range(N))
            
            # Gradient tracking
            y_i_new = y_i_consensus + (grad_i_new - grad_i_old)
            y_list_new.append(y_i_new)
        
        y_list = y_list_new
        
        # Compute metrics
        x_avg = np.mean(x_list, axis=0)
        loss = compute_global_loss(x_avg, A_list, b_list)
        consensus_error = compute_consensus_error(x_list)
        
        loss_history.append(loss)
        consensus_history.append(consensus_error)
        x_history.append(x_list.copy())
    
    return loss_history, consensus_history, x_history

def fedavg(A_list, b_list, K, eta, max_rounds, server_failure=None, w_star=None):
    """
    Federated Averaging
    
    Args:
        A_list: List of feature matrices for each agent
        b_list: List of target vectors for each agent
        K: Number of local gradient steps
        eta: Local step size
        max_rounds: Maximum number of communication rounds
        server_failure: Tuple (start, end) for server failure period, or None
        w_star: Optional optimal solution for computing optimality gap
    
    Returns:
        loss_history: History of global loss function values
        x_history: History of global parameter vectors
    """
    N = len(A_list)
    d = A_list[0].shape[1]
    
    # Initialize global parameter vector
    w = np.zeros(d)
    
    # History for logging
    loss_history = []
    x_history = []
    
    # Compute initial loss
    loss = compute_global_loss(w, A_list, b_list)
    loss_history.append(loss)
    x_history.append(w.copy())
    
    # Main loop
    for t in range(max_rounds):
        # Check for server failure
        if server_failure and server_failure[0] <= t < server_failure[1]:
            # During server failure, just append previous metrics
            loss_history.append(loss_history[-1])
            x_history.append(x_history[-1])
            continue
        
        # Initialize local parameters
        w_local_list = [w.copy() for _ in range(N)]
        
        # Local updates
        for i in range(N):
            w_i = w_local_list[i]
            A_i = A_list[i]
            b_i = b_list[i]
            m_i = A_i.shape[0]
            
            # K steps of gradient descent
            for k in range(K):
                grad_i = A_i.T @ (A_i @ w_i - b_i) / m_i
                w_i = w_i - eta * grad_i
            
            w_local_list[i] = w_i
        
        # Aggregate updates
        m_values = [A_i.shape[0] for A_i in A_list]
        M = sum(m_values)
        
        w = sum(m_i / M * w_i for m_i, w_i in zip(m_values, w_local_list))
        
        # Compute metrics
        loss = compute_global_loss(w, A_list, b_list)
        
        loss_history.append(loss)
        x_history.append(w.copy())
    
    return loss_history, x_history

def run_experiment(N=4, d=5, m=20, max_iter=100, topologies=None):
    """
    Run experiments for DSGT and FedAvg across different topologies
    
    Args:
        N: Number of agents
        d: Dimension of feature vectors
        m: Number of samples per agent
        max_iter: Maximum number of iterations/rounds
        topologies: List of topologies to test, defaults to ['ring', 'star', 'complete']
    
    Returns:
        results: Dictionary containing experimental results
    """
    if topologies is None:
        topologies = ['ring', 'star', 'complete']
    
    # Generate data
    A_list, b_list, w_true = generate_synthetic_data(N, d, m)
    
    # Compute optimal solution
    w_star = compute_optimal_solution(A_list, b_list)
    opt_loss = compute_global_loss(w_star, A_list, b_list)
    
    print(f"Optimal loss: {opt_loss:.6f}")
    
    # Results dictionary
    results = {
        'dsgt': {},
        'fedavg': {},
        'opt_loss': opt_loss,
        'w_star': w_star
    }
    
    # DSGT parameters
    alpha = 0.05
    
    # FedAvg parameters
    K = 5
    eta = 0.05
    server_failure = (max_iter // 4, max_iter // 2)  # Simulate server failure
    
    # Run DSGT for different topologies
    for topology in topologies:
        print(f"\nRunning DSGT with {topology} topology...")
        W = create_mixing_matrix(topology, N)
        spectral_gap = estimate_spectral_gap(W, N)
        print(f"Spectral gap (1-ρ): {spectral_gap:.6f}")
        
        start_time = time.time()
        loss_history, consensus_history, x_history = dsgt(A_list, b_list, W, alpha, max_iter, w_star)
        elapsed_time = time.time() - start_time
        
        # Find iterations to reach target loss and consensus
        loss_target = 1e-4
        consensus_target = 1e-6
        
        loss_iters = max_iter
        for i, loss in enumerate(loss_history):
            if loss - opt_loss < loss_target:
                loss_iters = i
                break
        
        consensus_iters = max_iter
        for i, cons in enumerate(consensus_history):
            if cons < consensus_target:
                consensus_iters = i
                break
        
        results['dsgt'][topology] = {
            'loss_history': loss_history,
            'consensus_history': consensus_history,
            'x_history': x_history,  # Store the parameter history
            'time': elapsed_time,
            'loss_iters': loss_iters,
            'consensus_iters': consensus_iters,
            'spectral_gap': spectral_gap
        }
        
        print(f"Time: {elapsed_time:.2f}s")
        print(f"Iterations to loss target: {loss_iters}")
        print(f"Iterations to consensus target: {consensus_iters}")
    
    # Run FedAvg
    print("\nRunning FedAvg...")
    start_time = time.time()
    loss_history, x_history = fedavg(A_list, b_list, K, eta, max_iter, server_failure, w_star)
    elapsed_time = time.time() - start_time
    
    # Find iterations to reach target loss
    loss_target = 1e-4
    loss_iters = max_iter
    for i, loss in enumerate(loss_history):
        if loss - opt_loss < loss_target:
            loss_iters = i
            break
    
    results['fedavg'] = {
        'loss_history': loss_history,
        'x_history': x_history,  # Store the parameter history
        'time': elapsed_time,
        'loss_iters': loss_iters,
        'server_failure': server_failure
    }
    
    print(f"Time: {elapsed_time:.2f}s")
    print(f"Iterations to loss target: {loss_iters}")
    
    return results

def plot_results(results, save_dir='./'):
    """
    Plot experimental results
    
    Args:
        results: Dictionary containing experimental results
        save_dir: Directory to save plot files
    """
    opt_loss = results['opt_loss']
    
    # Plot 1: Loss convergence
    plt.figure(figsize=(10, 6))
    
    # Plot DSGT for different topologies
    for topology, data in results['dsgt'].items():
        plt.semilogy(np.array(data['loss_history']) - opt_loss, 
                    label=f'DSGT - {topology.capitalize()}')
    
    # Plot FedAvg
    fedavg_data = results['fedavg']
    plt.semilogy(np.array(fedavg_data['loss_history']) - opt_loss, 
                label='FedAvg')
    
    # Mark server failure period
    if 'server_failure' in fedavg_data and fedavg_data['server_failure'] is not None:
        start, end = fedavg_data['server_failure']
        plt.axvspan(start, end, color='lightgray', alpha=0.3, label='Server Failure')
    
    plt.xlabel('Iteration')
    plt.ylabel('Optimality Gap: F(x) - F*')
    plt.title('Convergence of Global Objective Function')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}loss_curves.pdf')
    
    # Plot 2: Consensus error
    plt.figure(figsize=(10, 6))
    
    # Plot DSGT for different topologies
    for topology, data in results['dsgt'].items():
        plt.semilogy(data['consensus_history'], 
                    label=f'DSGT - {topology.capitalize()}')
    
    # Plot horizontal line for FedAvg (always 0)
    plt.semilogy(np.zeros(len(fedavg_data['loss_history'])), 
                label='FedAvg (centralized)', linestyle='--')
    
    plt.xlabel('Iteration')
    plt.ylabel('Consensus Error: ||x - 1̄x̄||²')
    plt.title('Consensus Error over Iterations')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}consensus_error.pdf')
    
    # Plot 3: Robustness to failures
    plt.figure(figsize=(10, 6))
    
    # Plot normal DSGT for complete topology
    plt.semilogy(np.array(results['dsgt']['complete']['loss_history']) - opt_loss, 
                label='DSGT - Complete')
    
    # Plot FedAvg with server failure
    plt.semilogy(np.array(fedavg_data['loss_history']) - opt_loss, 
                label='FedAvg with Server Failure')
    
    # Mark server failure period
    if 'server_failure' in fedavg_data and fedavg_data['server_failure'] is not None:
        start, end = fedavg_data['server_failure']
        plt.axvspan(start, end, color='lightgray', alpha=0.3, label='Server Failure Period')
    
    plt.xlabel('Iteration')
    plt.ylabel('Optimality Gap: F(x) - F*')
    plt.title('Algorithm Performance under Network Failures')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}robustness.pdf')

if __name__ == "__main__":
    # Run experiments
    results = run_experiment(N=4, d=5, m=20, max_iter=200)
    
    # Plot results
    plot_results(results)
    
    # Print key metrics
    print("\n--- Summary of Results ---")
    print(f"Optimal loss: {results['opt_loss']:.6f}")
    
    # DSGT metrics
    complete_iters = results['dsgt']['complete']['loss_iters']
    print(f"\nDSGT (Complete): {complete_iters} iterations to target loss")
    
    for topology in ['star', 'ring']:
        topo_iters = results['dsgt'][topology]['loss_iters']
        slowdown = topo_iters / complete_iters if complete_iters > 0 else float('inf')
        print(f"DSGT ({topology.capitalize()}): {topo_iters} iterations (slowdown: {slowdown:.1f}x)")
    
    # FedAvg metrics
    fedavg_iters = results['fedavg']['loss_iters']
    print(f"\nFedAvg: {fedavg_iters} iterations to target loss")
    
    # Spectral gap analysis
    print("\nSpectral gap analysis:")
    for topology, data in results['dsgt'].items():
        print(f"{topology.capitalize()}: {data['spectral_gap']:.6f}")