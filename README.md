# convexResearchProject# Topology-Aware Distributed Optimization in Satellite Mesh Networks

This repository contains code and experiments comparing Distributed Stochastic Gradient Tracking (DSGT) and Federated Averaging (FedAvg) algorithms across different network topologies.

## Overview

The project investigates how network topology affects distributed optimization algorithms in satellite constellations. It implements and compares:

1. **Distributed Stochastic Gradient Tracking (DSGT)** - A decentralized algorithm where nodes exchange information only with immediate neighbors
2. **Federated Averaging (FedAvg)** - A centralized algorithm where nodes communicate with a central server

These are evaluated across three representative network topologies:
- Ring (sparse connections)
- Star (centralized hub structure)
- Complete (fully connected mesh)

## Requirements

The code requires the following Python packages:
```
numpy
matplotlib
networkx
```

Install them using pip:
```bash
pip install numpy matplotlib networkx
```

## File Structure

- `distributed_optimization.py` - Core implementation of DSGT and FedAvg algorithms
- `network_visualization.py` - Visualizations for network topologies and convergence animations
- `report.tex` - LaTeX source file for the technical report

## Running the Experiments

### Basic Experiment

To run the basic experiment with default parameters:

```bash
python distributed_optimization.py
```

This will:
1. Generate synthetic linear regression data
2. Implement DSGT and FedAvg on different topologies
3. Produce convergence plots
4. Print summary statistics

### Network Visualizations

To generate network topology visualizations and animations:

```bash
python network_visualization.py
```

This will create:
1. Static visualizations of ring, star, and complete topologies
2. Animated GIFs showing DSGT convergence on each topology
3. Communication cost analysis plots
4. Theoretical convergence rate comparisons

## Experiment Parameters

You can modify the following parameters in the scripts:

- `N` - Number of agents/nodes (default: 4)
- `d` - Dimension of the parameter vector (default: 5)
- `m` - Number of data samples per agent (default: 20)
- `max_iter` - Maximum number of iterations (default: 100)
- `alpha` - Step size for DSGT (default: 0.05)
- `K` - Number of local steps for FedAvg (default: 5)
- `eta` - Local step size for FedAvg (default: 0.05)

## Understanding the Results

### Key Metrics

The experiments track several key metrics:

1. **Global Loss**: The value of the global objective function, measuring algorithm progress
2. **Optimality Gap**: The difference between current loss and optimal loss (F(x) - F*)
3. **Consensus Error**: Measures how well the nodes agree on a common solution
4. **Convergence Rate**: How quickly algorithms reach a target accuracy
5. **Communication Cost**: Number of messages exchanged per iteration

### Expected Findings

- **DSGT on Complete Graph**: Fastest convergence among DSGT variants
- **DSGT on Star**: Moderately slower than complete graph (~1.4x)
- **DSGT on Ring**: Slowest convergence (~2x slower than complete)
- **FedAvg**: Similar performance across topologies, but vulnerable to server failures
- **Robustness**: DSGT continues learning during server failures, FedAvg halts

## Customizing the Experiments

### Testing Different Topologies

To implement a custom topology, modify the `create_mixing_matrix` function in `distributed_optimization.py`. You'll need to define how agents connect and the appropriate weight matrix W.

### Simulating Network Failures

The code includes server failure simulation for FedAvg. You can customize the failure period by modifying:

```python
server_failure = (max_iter // 4, max_iter // 2)  # Start and end of failure period
```

To simulate link failures in DSGT, you can modify the mixing matrix W during iterations.

## Report Generation

To compile the LaTeX report:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

## Additional Notes

- The spectral gap analysis in `network_visualization.py` provides theoretical convergence rates
- Animations help visualize the consensus process in DSGT
- The communication cost analysis highlights the tradeoff between convergence speed and message overhead

## References

For more information on the algorithms and theory, refer to:
- S. Pu and A. Nedić, "Distributed Stochastic Gradient Tracking Methods," Mathematical Programming, vol. 187, pp. 409-457, 2021.
- H. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," in Proc. AISTATS, 2017, pp. 1273-1282.