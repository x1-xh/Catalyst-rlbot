# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `6b91b2b5d9e34ecd721820f229a8748b33e109dbe71db59fd027729649607722`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 RLBotPack Tournament Distillation
The policy was trained via behavioral cloning and policy distillation utilizing tournament algorithms from top bots in the official RLBotPack (`VirxEB` & `Necto`):
- **Dataset Version:** `catalyst-rlbotpack-distillation-v3`
- **Splits:** 30 training matches (48,000 timesteps) and 6 validation matches (9,600 timesteps) partitioned by whole continuous matches (83/17 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics:** 
  - **VirxEB Cubic Steer PD:** Uses the cubic steering curve $\text{clamp}\left(\frac{(35 \cdot \text{angle})^3}{10}, -1, 1\right)$ to deliver dead-straight driving without angle oscillation while providing agile turning response when needed.
  - **Goal-Directed Approach Offset:** Aims along the vector from the ball to the opponent's goal `(0, 5120)` to set up clean, powerful shots through the ball rather than orbiting its center.
  - **VirxEB 3-Stage Timed Flip:** Accurate timing structure: `Mode 2 (Jump)` for 0.075s $\to$ `Mode 0 (Release)` for 0.025s $\to$ `Mode 3 (Front Dodge Flip)` for 0.30s.
  - **Car-Ball Collisions:** Full momentum transfer into the ball with kickoff resets upon scoring.
- **Random Seed:** 42

### 2.2 Optimization & Loss Objectives (Exact Tournament Specification)
- **Objective Function:** Standard tournament composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}) + 1.0 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** Adam (`lr = 1.0e-3`) with gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 256 with truncated backpropagation through time over 25 epochs. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ was preserved across chunks within matches and detached at chunk boundaries, only being reset at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9498 | 0.6643 | 0.8179 | 0.5395 | 68.5% | Warmup |
| 5 | 0.6039 | 0.2298 | 0.5555 | 0.2289 | 81.6% | SteerPD convergence |
| 10 | 0.4269 | 0.1225 | 0.4935 | 0.2338 | 83.7% | Approach alignment |
| 15 | 0.3103 | 0.0664 | 0.2859 | 0.0419 | 89.6% | Strike timing learned |
| 20 | 0.2156 | 0.0544 | 0.2265 | 0.0230 | 90.6% | High precision |
| 24 | 0.1878 | 0.0505 | 0.1673 | 0.0261 | **93.8%** | **Best Checkpoint** |
| 25 | 0.1803 | 0.0506 | 0.2005 | 0.0343 | 92.4% | Final epoch |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **93.8%** (Best Checkpoint)
- **Validation Mode Loss:** **0.1673**
- **Validation Steering MSE:** **0.0261**
- **Lowest Total Validation Loss:** **0.1934**

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `6b91b2b5d9e34ecd721820f229a8748b33e109dbe71db59fd027729649607722`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to verify direct ball strikes and clean straight driving.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
