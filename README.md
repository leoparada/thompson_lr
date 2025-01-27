# Contextual Thompson Sampling with Logistic Regression

This project implements a **contextual bandit algorithm** combining **Thompson Sampling** with **regularized logistic regression** using batch updates. The implementation follows the algorithm from the paper _"An Empirical Evaluation of Thompson Sampling"_ by Chapelle and Li.

## Project Structure

```markdown
.
├── bandit/                
│   ├── contextual_bandit.py       # Implementation of contextual bandit algorithms
├── utils.py                        # Utility functions for data generation and plotting
├── main.py                         # Main script to evaluate bandit strategies
├── bandit_evaluation.ipynb          # Jupyter notebook for further analysis and visualization
└── README.md                        # Project documentation
```
## Setup

Ensure Python 3 and required dependencies are installed:

```bash
pip install numpy matplotlib scipy jupyter
```

## Running the Project

### 1. Evaluate Bandit Strategies
Run the main script to evaluate strategies and visualize results:

```bash
python main.py
```

### 2. Detailed Analysis with Jupyter
Open the notebook for further evaluation and insights:

```bash
jupyter notebook bandit_evaluation.ipynb
```

## Implemented Policies

- **Thompson Sampling** – Bayesian posterior sampling for exploration-exploitation.
- **LinUCB** – Confidence-based action selection.
- **Epsilon-Greedy** – Random exploration with probability $\varepsilon$.
- **Exploit-Only** – Always selects the best-known action.
- **Random** – Uniform action selection.

## Evaluation Metrics

- **Total Reward:** Measures overall performance.
- **Cumulative Regret:** Tracks suboptimal actions over time.

## Development Environment and Tools

This project was developed using the following tools and resources:

- **IDE:**  
  - PyCharm: Used for code development, debugging, and project organization.

- **Libraries and Frameworks:**  
  - `numpy`: For numerical operations and data handling.  
  - `scipy`: Specifically, the `minimize` function for updating the posterior in logistic regression.  
  - `matplotlib`: For visualization of bandit performance and cumulative regret analysis.

- **AI Assistants:**  
  - GitHub Copilot: Assisted with code completion and function documentation within the IDE.  
  - ChatGPT: Used to structure the project framework, brainstorm experiment ideas, and ensure conceptual clarity.

- **References and Documentation:**  
  - Sutton and Barto’s *Reinforcement Learning: An Introduction* – Consulted to understand theoretical concepts.  
  - Online documentation and resources such as NumPy, SciPy, and Stack Overflow for technical guidance.
