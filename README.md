# MDP-Visualization
Web-based visualization of Value Iteration and Policy Iteration
Project Overview

This project demonstrates the concept of Markov Decision Processes (MDPs) using a Grid World environment.
It provides an interactive web-based visualization of two fundamental dynamic programming algorithms:
1-Value Iteration
2-Policy Iteration

The goal is to help understand how optimal policies and state values evolve over time in a decision-making environment under uncertainty.

**Objectives**

Model a Grid World problem as a Markov Decision Process

Implement Value Iteration and Policy Iteration algorithms

Visualize state values and optimal policies

Compare the behavior and convergence of both algorithms

Provide an interactive interface for experimentation

**MDP Components**

States: Each cell in a 5×5 grid
Actions: Up, Down, Left, Right
Rewards:
Goal State: +10
Negative State: −10
Step Cost: −0.1
Obstacles: Blocked states that cannot be entered
Discount Factor (γ): User-controlled via slider

 **Technologies Used**
Python
Streamlit (Web Interface)
NumPy (Numerical Computation)
Matplotlib (Visualization)

**How to Run the Project**
1-Install Required Libraries
pip install streamlit numpy matplotlib

2-Run the Application
streamlit run app.py
(Replace app.py with your actual Python file name if different)

**Application Features**
Algorithm selection (Value Iteration / Policy Iteration)
Adjustable discount factor and iteration count
Step-by-step iteration visualization
Heatmap showing state values
Arrow indicators showing optimal policy directions
Clear representation of goal, obstacles, and negative states

**Observations**
Value Iteration shows gradual propagation of rewards across states.
Policy Iteration converges faster to an optimal policy.
Both algorithms reach the same optimal solution.
Visualization helps in understanding the learning process clearly.

**Team Members**\n
Member 1: Eman Rashid\n
Member 2: Zoya Khalid\n
Member 3: Sarwat Naveed\n

**Instructor**\n
Instructor Name: Zuhaib Hussain Butt
