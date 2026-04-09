# Reinforcement Learning — MDP Implementation
## Overview
The goal of this project is to understand and analyze how different RL algorithms behave under:
   -Deterministic environments
   -Stochastic environments (with uncertainty)
## Algorithms Implemented
- Value Iteration
- Policy Iteration
- Q-Learning (off-policy)
- SARSA (on-policy)
## Key Features
- Implementation of classic RL algorithms from scratch
- Simulation of deterministic and stochastic environments
- Introduced action stochasticity:
   -20% probability of taking a random action (action slip)
-Generated learning curves to analyze:
  -Convergence speed
  -Stability of learning
-Comparative study of algorithm performance
## Results
- Q-Learning converges faster in deterministic environments
- SARSA performs more cautiously in stochastic environments
- Value Iteration converges in just 6 iterations
## Visualizations
<p align="center">
  <img src="https://github.com/user-attachments/assets/cab45169-541d-4a31-bfc8-757269b98cf8" width="300"/>
  <img src="https://github.com/user-attachments/assets/7133e4a1-caa3-4076-a515-d00ef70af2e3" width="500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ccd28829-c1dc-4ab3-9110-d403adf797eb" width="500"/>
</p>
  
## How to Run
```bash
python value_iteration.py
python q_learning.py
python sarsa.py
python final_comparison.py
```
