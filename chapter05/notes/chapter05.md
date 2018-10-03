# Chapter 05

## 5.1 Monte Carlo Prediction
##### First-visit MC method
- estimates $v_{\pi}(s)$ as the average of returns following first visits to $s$
-

##### Every-visit MC method
- averages returns following all visits to $s$

##### Advantage over DP
- ability to learn from actual experience and from simulated experience (no need for complete knowledge of the environment's dynamics)
- computational expense of estimating the value of a single state is independent of the number of states


## 5.2 Monte Carlo Estimation of Action Values
- With a model: estimate state values $v_{\pi}(s)$ to determine a policy
  - looks ahead one step and chooses whichever  action leads to the best combination of reward and next state
- Without a model: estimate value of each action $q_{\pi}(s, a)$

##### Policy Evaluation Problem for action values
A state-action pair $s, a$ is visited in an episode if the state $s$ is visited and action $a$ is taken in it
- Problem: many state-action pairs may never be visited, deterministic $\pi$ may never visit some state-action pairs $\rightarrow$ **maintaining exploration**

##### Exploring starts
- specifying that the episodes start in a state-action pair, and that every pair has a nonzero probability of being selected as the start
- not reliable in general when learning directly from actual interaction with the environnment
- alternative approach: only consider stochastic policies


## 5.3 Monte Carlo Control
GPI: maintains both an approximate policy and an approximate value function
- value function repeatedly altered to more closely approximate the value function for the current policy
- policy repeatedly improved with respect to the current value function

## 5.4 Monte Carlo Control without Exploring Starts
How can we avoid the unlikely assumption of exploring starts?
- ensure that all actions are selected infinitely often
- agent to continue to select them

1. on-policy methods
    - attempt to evaluate/improve the policy that is used to make decisions
    - i.e. Monte Carlo ES
2. off-policy methods
    - evaluate or improve a policy different from that used to generate the data

##### On-Policy first-visit MC control (for $\epsilon$-soft policies)
- all nongreedy actions are given the minimal probability of selection $\frac{\epsilon}{|A(s)|}$
- the remaining greedy actions the probability $1 - \epsilon + \frac{\epsilon}{|A(s)|}$

## 5.5 Off-policy Prediction via Importance Sampling
Use 2 policies
- one learned about that becomes the optimal policy --> **target policy**
- one that is explorartory, used to generate behavior --> **behavior policy**

##### Off-policy prediction problem
- We wish to estimate $v_{\pi}$ or $q_{\pi}$
- episodes followng another policy $b$
- $\pi$: target policy
- $b$: behavior policy

##### Importance Sampling
General technique for estimating expected values under one distribution given samples from another
- weighting returns according to the relative probability of their trajectories occurring under the target and behavior policies **importance-sampling ratio**
$$p_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

## 5.6 Incremental Implementation
