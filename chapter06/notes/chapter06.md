# Temporal-Difference Learning

TD learning: combination of Monte Carlo + DP ideas
- like Monte Carlo: TD learn directly from raw experience, model-free
- like DP: update estimates based in part on other learned estimates, **bootstrap**

## 6.1. TD Prediction
##### $constant-\alpha$ MC
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
- Monte Carlo methods must wait until the end of the episode to determine the increment to $V(S_t)$. pm;u yjem os $G_t$ known
- target = $G_t$

##### TD method update (one-step TD)
$$V(S_t) \leftarrow + \alpha [R_{t+1} + \gamma V(S_{t+1} - V(S_t))]$$
- TD methods wait only until the next time step, at time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$ and the estimate $V(S_{t+1})$
- target = $R_{t+1} + \gamma V(S_{t+1})$
-

## 6.2 Advantages of TD Prediction Methods
- over DP: do not require model of the environment of its reward, next-state probability distributions
- over MC: natually implemented in an online, fully incremental fashin, do not wait until the end of an episode
## 6.3. Optimality of TD(0)

##### batch updating
- all the available experience is processed again with the new value function to produce a new overall increment
- updates are made only after processing each complete batch of training data
-

## 6.4 Sarsa: On-policy TD Control
- learn an action-value function $q_{\pi}(s, a)$ rather than a state-value function
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

## 6.5 Q-Learning: Off-policy TD Control
$$Q(S_t, A_t) \leftarrow Q(S_t + A_t) + \alpha [R_{t+1} + \gamma \operatorname*{max}_a Q(S_{t+1}, a) - Q(S_t, A_t) ]$$

- learned action-value function $Q$ approximates $q*$, the optimal action-value function, independent of the policy being followed

## 6.6 Expected Sarsa
Just like Q-learning except that instead of the maximum over next state-action pairs it uses the expected value, taking into account how likely each action is under the current policy
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \mathbb{E}_{\pi} [Q(S_{t+1}, A_{t+1}) | S_{t+1}] - Q(S_t, A_t)\right]$$
$$\leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi(a | S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]$$

- unlike Q-learning: Given the next state $S_{t+1}$, moves deterministically in the same direction as Sarsa moves in expectation

## 6.7 Maximization Bias and Double Learning
- control algorithms involve maximization in the construction of their target policies --> *bias*

## 6.8 Games, Afterstates, and Other Special Cases

## 6.9 Summary
