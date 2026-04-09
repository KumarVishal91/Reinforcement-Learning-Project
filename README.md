
# Reinforcement Learning — MDP Implementation

This project demonstrates the implementation and comparison of core Reinforcement Learning (RL) algorithms using a Markov Decision Process (MDP) framework.

📌 Overview

The goal of this project is to understand and analyze how different RL algorithms behave under:

Deterministic environments
Stochastic environments (with uncertainty)

We implemented both model-based and model-free approaches and compared their performance using learning curves and convergence analysis.

🧠 Algorithms Implemented
🔹 Model-Based Methods
Value Iteration
Policy Iteration
🔹 Model-Free Methods
Q-Learning (Off-Policy)
SARSA (On-Policy)
⚙️ Key Features
✅ Implementation of classic RL algorithms from scratch
✅ Simulation of deterministic and stochastic environments
✅ Introduced action stochasticity:
20% probability of taking a random action (action slip)
✅ Generated learning curves to analyze:
Convergence speed
Stability of learning
✅ Comparative study of algorithm performance
🧪 Experimental Setup
Environment modeled as an MDP
States, actions, rewards, and transitions defined explicitly

Stochastic behavior introduced via:

20% chance → random action
80% chance → intended action
📊 Results & Observations
🔹 Q-Learning
Faster convergence in deterministic environments
Learns optimal policy aggressively
🔹 SARSA
More stable and cautious in stochastic environments
Accounts for actual policy being followed
🔹 Value Iteration
Converged in just 6 iterations
Efficient for small state spaces
📈 Visualizations

The project includes:

📉 Learning curves (Reward vs Episodes)
📊 Convergence comparisons
📌 Policy behavior differences
  <img width="600" height="600" alt="Figure_" src="https://github.com/user-attachments/assets/cab45169-541d-4a31-bfc8-757269b98cf8" />

  <img width="1400" height="600" alt="Figure" src="https://github.com/user-attachments/assets/7133e4a1-caa3-4076-a515-d00ef70af2e3" />

  <img width="900" height="500" alt="Figure_7" src="https://github.com/user-attachments/assets/ccd28829-c1dc-4ab3-9110-d403adf797eb" />

🗂️ Project Structure
├── value_iteration.py        # Value Iteration implementation
├── policy_iteration.py       # Policy Iteration implementation
├── q_learning.py             # Q-Learning implementation
├── sarsa.py                  # SARSA implementation
├── final_comparison.py       # Performance comparison + plots
├── utils/                    # Helper functions (if any)
└── README.md                 # Project documentation
▶️ How to Run

Make sure you have Python installed (preferably Python 3.x).

python value_iteration.py
python q_learning.py
python sarsa.py
python final_comparison.py
📦 Requirements

Install required dependencies (if any):

pip install numpy matplotlib
🎯 Learning Outcomes
Understanding difference between on-policy vs off-policy learning
Impact of environment stochasticity
Trade-offs between exploration and exploitation
Practical implementation of MDP-based RL algorithms
🔮 Future Improvements
Add Deep Q-Networks (DQN)
Extend to larger and continuous state spaces
Introduce more complex environments (e.g., GridWorld variations)
Hyperparameter tuning visualization
