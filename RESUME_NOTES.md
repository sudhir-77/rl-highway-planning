# Resume notes

Skills/tools: Python, PyTorch, stable-baselines3, Gymnasium, highway-env, NumPy, pytest, reinforcement learning (PPO), behavior cloning, ego-attention.

This project collects IDM and MOBIL demonstrations in highway-env and trains MLP and ego-attention policies with behavior cloning and PPO. Held-out traffic seeds measure collision rate, speed, and jerk on highway-v0 and merge-v0.

- Trained behavior-cloning and PPO driving policies in highway-env with an ego-attention network over nearby vehicles, reaching a collision rate of 1.50% on 200 held-out highway traffic seeds
- Initialized PPO from behavior cloning to match the from-scratch PPO final return in 0.00% of the environment steps
- Built an evaluation harness for collision rate, speed, and jerk across highway and merge scenarios, with a failure-mode analysis of remaining collisions
- Matched 99.14% of the rule-based expert's mean speed on highway-v0 while the merge-v0 collision rate of the selected learned policy was 0.00%

GitHub description: Behavior cloning and PPO ego-attention policies for highway and merge driving in highway-env.
