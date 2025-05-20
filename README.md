# BRACE: Byzantine-Resilient Aggregation via Consensus Enforcement

**BRACE** is a research-oriented simulation framework for exploring **Byzantine-resilient Federated Learning (FL)** systems through consensus-based strategies. The project implements and analyzes different configurations of FL systems to evaluate their robustness under adversarial conditions, such as data poisoning and malicious client behavior.

## Overview

This project provides a simulation environment where:

- A central **aggregator** coordinates training across multiple **clients**.
- A configurable number of **Byzantine clients** introduce faulty updates to simulate real-world adversarial threats.
- Experiments can be conducted with varying numbers of clients, rounds, selection strategies, and datasets.

The system has been tested on two benchmark datasets:

- **MNIST**
- **FashionMNIST**

Two models, one for each dataset, are used and optimized via hyperparameter tuning.

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can launch the simulation using the `federated_learning.py` python script.

### Available Arguments:

```python
-t | --task : Task to run (mnist | fashionmnist)
-n | --n_clients : Number of clients (default: 5)
-r | --rounds : Number of rounds (default: 2)
-p | --percentage : Percentage of clients selected per round (default: 1, i.e. all)
-b | --byzantine : Number of Byzantine clients (default: 1)
-rtime | --realtime : Enable real-time training (different local datasets each round)
```

### Example:

`python federated_learning.py -t mnist -n 10 -r 5 -p 0.5`

runs the simulation on the MNIST task, with:

- 10 total clients,

- 5 training rounds,

- 50% of clients randomly selected per round.

### Automation Tools

Two Bash scripts are included to facilitate large-scale experimentation:

- `collect.sh`: Automates simulations and collects results for standard FL settings.

- `collect_rtime.sh`: Automates simulations under real-time training conditions.

## Analysis and Results

Some results have been collected and plotted for visualization into the notebook `analysis.ipynb`.