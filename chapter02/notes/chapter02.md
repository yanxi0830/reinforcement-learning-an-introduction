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
- $\alpha_n(a)$: steps-size parameter used to process the reward received after $n$th selection of action $a$
- Conditions required to assure convergence with probability 1:
$$\sum_{n=1}^{\infty} \alpha_n(a) = \infty$$
> guarantee that steps are large enough to eventually overcome any initial conditions or random fluctuations

$$\sum_{n=1}^{\infty} \alpha^2_n(a) < \infty$$

> guarantees that eventually the steps become small enough to assure convergence

TODO: Exercise 2.5

## 2.6 Optimistic Initial Values
##### Methods *biased* by the initial action-value estimates $Q_1(a)$
- *sample-average*: bias disapeears once all actions have been selected at least once
- *constant $\alpha$*: bias is permanent, decreasing over time
- initial estimates become a set of hyperparameters to be picked by user
- easy way to supply some prior knowledge about what level of rewards can be expected

##### Initial action values encourage exploration
- Example: set initial values to +5 with 10-armed bandit
  - $q^*(a) \sim N(0, 1)$
  - initial estimate of +5 is too optimistic
  - whichever actions are initially selected, the reward is less than the starting estimates, the learner swiches to other actions, being "disappointed" with the rewards it is receiving
  - all actions are tried several times before value estimates change
- Not well suited to nonstationary problems because its drive to exploration is inherently temporary
  - if the task changes, creating a renewed need for exploration
  - the beginning of time occurs only once

## 2.7 Upper-Confidence-Bound Action Selection
- Select among the non-greedy actions according to their potential for actually being optimal, taking into account both:
  - how close their estimated are to being maximal
  - uncertainties in those estimates

$$A_t = \operatorname*{argmax}_a \left[Q_t(a) + c \sqrt\frac{ln(t)}{N_t(a)} \right]$$

- $N_t(a)$: number of times that action $a$ has been selected prior to time $t$
  - if $N_t(a)=0$: $a$ is a maximizing action
- $c > 0$: controls the degree of exploration

##### Upper Confidence Bound Action
- square-root term is a measure of the uncertainty or variance in the estimate of $a$'s value
- the quantity being maxed over is upper bound on the possible true value of action $a$, with $c$ determining the confidence level
- each time $a$ is selected, the uncertainty is reduced
- each time an action other than $a$ is selected, $t$ increases but $N_t(a)$ does not, the uncertainty estimates increases

##### Difficulty
- dealing with non-stationary problems
- dealing with large state spaces, when using function approximation

## 2.8 Gradient Bandit Algorithms

##### Learning a numerical *preference* for each action $a$: $H_t(a)$
- the larger the preference, the more often that action is taken
- preference has no interpretation in terms of reward
- only the relative preference of one action over another is important
- **soft-max distribution**
$$Pr\left\{ A_t = a\right\} = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{h_t(b)}} = \pi_t(a)$$

- $\pi_t(a)$: probability of taking action $a$ at time $t$

##### Stochastic Gradient Ascent
On each step, after selection action $A_t$ and receiving the reward $R_t$, the action preferences are updated by:

$$H_{t+1}(A_t) = H_t(A_t) + \alpha (R_t - \bar{R_t})(1 - \pi_t(A_t))$$

$$H_{t+1}(a) = H_t(a) - \alpha (R_t - \bar{R_t})\pi_t(a), \text{for all $a \neq A_t$}$$

$\bar{R_t}$: average of all the rewards up through and including time $t$
- baselin with which the reward is compared
- if reward higher than baseline, probability of taking $A_t$ in the furture is increased
- if reward is below baseline, probability is decreased
- non-selected actions move in the opposite direction

## 2.9 Associative Search
- Facing an actual slot machine that changes the color of its display as it changed its action values
- Learn a policy associating each task, signaled by the color you see, with the best action to take when facing that task
  - if red, select arm 1, if green select arm 2

##### Associative Search Task
- trial-and-error learning to search for best actions
- association of these actions with situations in which they are best
- **contextual bandits**

TODO: Exercise 2.11



---
