# Chapter 02: Multi-armed Bandits

##### Reinforcement Learning (RL) v.s. Supervised Learning (SL)
- RL: **evaluates** actions taken rather than **instructs** by giving correct actions
- pure evaluative feedback indicates how good the action taken was, not whether it was the best or the worst, **dependent** on action taken
- RL: need for active exploration
- SL: pure instructive feedback, **independent** on action taken

##### Nonassociative Setting ($k$-armed Bandit Problem)
- does not involve learning to act in more than one situation
- **associative**: when actions are taken in more than one situation

## 2.1 A $k$-armed Bandit Problem

![bandit](figures/multiarmedbandit.jpg)

You are faced repeatedly with a choice among $k$ different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period (i.e. over 1000 action selections, or *time steps*)

##### Action Value
###### Expected Reward given that $a$ is selected
$$q^*(a) = \mathbb{E}[R_t | A_t = a]$$


###### Estimated value of action $a$ at time step $t$
$$Q_t(a)$$

- **greedy action**: action whose estimated value is greatest $\rightarrow$ *exploitation*

$$EXPLORATION\ v.s. EXPLOITATION$$

## 2.2 Action-value Methods

###### Sample-Average Method
$$Q_t(a) = \frac{\text{sum of rewards when }a \text{ taken prior to }t}{\text{number of times }a \text{ taken prior to }t}$$

###### Greedy action selection method
$$A_t = \operatorname*{argmax}_a Q_t(a)$$

- **$\epsilon$-greedy** method for exploration

## 2.3 The 10-armed Testbed

## 2.4 Incremental Implementation

- $R_i$: reward received after the $i$th selection of this action
- $Q_n$: estimate of its action value after it has been selected $n-1$ times

$$Q_n = \frac{R_1 + R_2 + ... + R_{n-1}}{n-1}$$

- Keeping record of all the rewards and then perform this computation whenever the estimated value is needed $\rightarrow$ memory + computation overhead!

##### Incremental Update Rule
$$Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n]$$

$$NewEstimate \leftarrow OldEstimate + StepSize [Target - OldEstimate]$$

## 2.5 Tracking a Nonstationary Problem
- give more weight to recent rewards than to long-past rewards
###### Exponential Recency-Weighted Average
$$Q_{n+1} = Q_n + \alpha [R_n - Q_n]$$

##### Vary step-sze parameter from step to step
- $\alpha_n(a)$: steps-zie parameter used to process the reward received after $n$th selection of action $a$
- Conditions required to assure convergence with probability 1:
$$\sum_{n=1}^{\infty} \alpha_n(a) = \infty$$
> guarantee that steps are large enough to eventually overcome any initial conditions or random fluctuations
$$\sum_{n=1}^{\infty} \alpha^2_n(a) < \infty$$
> guarantees that eventually the steps become small enough to assure convergence

TODO: Exercise 2.5

## 2.6 Optimistic Initial Values




















---
