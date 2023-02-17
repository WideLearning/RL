Just a repository with reinforcement learning experiments.

To use environments from here you will want to install it as a module. 
To do so run `pip install -e implementations` from the root of the repository.

## Multi-armed bandit problem
Here several algorithms were compared in the same conditions:
$5$ arms, $300$ steps, action value for each arm is sampled from standard normal distribution,
on each step observed value is sampled from $\mathcal{N}(\text{action value}, 3)$. For other details see `benchmark.py`.

Greedy algorithm tries each arm ones, and then selects the one with maximal mean value observed so far.
![greedy](images/greedy.svg)

Its modification, $\varepsilon$-greedy, does the same, but with chance $\varepsilon$ it selects arm randomly.
![eps_greedy](images/eps_greedy.svg)

Upper Confidence Bound method calculates for each arm an optimistic estimate of its value which gets closer to the real value with more tries.
Then selects the action with the highest estimate.
![ucb](images/ucb.svg)

This one isn't really an algorithm for bandit problem because it uses the knowledge of true values. But it provides a useful upper-bound: no actual algorithm can score higher.
![optimal](images/optimal.svg)