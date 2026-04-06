# Reinforcement Learning — MDP Implementation

## Algorithms Implemented
- Value Iteration
- Policy Iteration
- Q-Learning (off-policy)
- SARSA (on-policy)

## My Contribution
- Compared SARSA vs Q-Learning on deterministic and stochastic environments
- Added stochastic transitions (20% random action slip)
- Generated learning curve plots showing convergence behavior

## Results
- Q-Learning converges faster in deterministic environments
- SARSA performs more cautiously in stochastic environments
- Value Iteration converges in just 6 iterations

## How to Run
```bash
python value_iteration.py
python q_learning.py
python sarsa.py
python final_comparison.py
```