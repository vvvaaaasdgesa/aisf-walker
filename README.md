# AISF Bipedal Walker (PPO)

This project trains a multi-layer perceptron (MLP) policy using Proximal Policy Optimization (PPO) to solve the Gymnasium BipedalWalker-v3 environment. The goal is to learn a stable walking policy that moves the robot forward without the hull touching the ground.

The implementation uses Stable-Baselines3 and Gymnasium, with observation normalization and evaluation callbacks for tracking performance.

---

## Environment
- Gymnasium: BipedalWalker-v3
- Action space: Continuous (4 joints)
- Observation space: 24-dimensional state vector

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

To train the agent, run:

```bash
python src/train.py
```
---

## Results

### Training Metrics

The following metrics were tracked during training and evaluation:

- Evaluation mean reward
- Evaluation episode length
- Rollout mean reward
- Rollout episode length

Screenshots of these plots are stored in:

```text
results/plots/
```

### Best Policy Performance

A GIF showing the best-performing policy after training is available at:

```text
results/gifs/best.gif
```

The final model was trained for **200,000 timesteps** and demonstrates stable forward walking behavior.

---

## Techniques Used

- Proximal Policy Optimization (PPO)
- Observation normalization (VecNormalize)
- Action clipping for continuous control stability
- Evaluation callbacks for model checkpointing
- Manual hyperparameter tuning
- TensorBoard logging for performance analysis

---

## Ablation Studies

Ablation experiments were conducted to evaluate the impact of individual design choices on training stability and final performance.

The following factors were examined:

- Observation normalization vs. no normalization
- Different rollout lengths
- Different entropy coefficients
In particular, an ablation study on the PPO entropy coefficient (ent_coef ∈ {0.0, 0.01, 0.03}) was conducted to analyze the exploration–exploitation tradeoff. Moderate entropy (0.01) resulted in faster convergence and higher final evaluation reward, while higher entropy introduced instability.

Performance was compared using evaluation mean reward and episode length curves. Observation normalization was found to significantly improve learning stability and convergence speed.

---

## Discussion

During initial experiments, training was unstable and the agent frequently fell early in episodes. This behavior was especially prominent when observation normalization was disabled, leading to high variance in learning signals.

Introducing observation normalization significantly improved training stability and allowed the agent to learn a consistent walking gait. Several hyperparameter configurations were tested, including different rollout lengths and entropy coefficients. Some configurations resulted in slower convergence or overly cautious behavior, requiring iterative adjustment.

Overall, the final configuration represents a balance between exploration and stability, informed by empirical testing and ablation results.

---

## Conclusion

The PPO-based agent successfully learned a stable walking policy in the BipedalWalker-v3 environment. Through observation normalization, careful hyperparameter tuning, and iterative experimentation, training stability and final performance were significantly improved.

With additional time and computational resources, future work could explore curriculum learning, recurrent policy architectures, or training on the more challenging BipedalWalkerHardcore-v3 environment to further improve robustness and generalization.


