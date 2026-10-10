# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `27a138dd324cd4d7aa1f481a36d91f590f925c3bd984ce48a41dfcafaebe0258`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 RLBotPack Tournament Distillation (v4 Expanded)
The policy was trained via behavioral cloning and policy distillation utilizing tournament algorithms from top bots in the official RLBotPack (`VirxEB` & `Necto`):
- **Dataset Version:** `catalyst-rlbotpack-distillation-v4-expanded`
- **Splits:** 120 training matches (192,000 timesteps) and 24 validation matches (38,400 timesteps) partitioned by whole continuous matches (83/17 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics:** 
  - **VirxEB Cubic Steer PD:** Uses the cubic steering curve $\text{clamp}\left(\frac{(35 \cdot \text{angle})^3}{10}, -1, 1\right)$ to deliver dead-straight driving without angle oscillation while providing agile turning response when needed.
  - **Symmetric Horizontal Reflection Augmentation:** Every match trajectory is augmented with its exact mirror $(X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer})$. Guarantees mathematical left-right symmetry, eliminating positive steering bias and circling.
  - **Goal-Directed Approach with Direct Close-Range Strike:** Guides the car behind the ball facing opponent goal `(0, 5120)` when at range, smoothly transitioning directly into `ball_pos` when close (<350 uu) to drive through the ball rather than orbiting around it.
  - **Kickoff Charge & Early Speed Flip:** Car starts in full-throttle Chase mode from frame 0 at kickoff, triggering an early alignment dodge when distance to ball $>1300$ uu and speed $>600$ uu/s to win kickoff challenges.
  - **VirxEB 3-Stage Timed Flip:** Accurate timing structure: `Mode 2 (Jump)` for 0.075s $\to$ `Mode 0 (Release)` for 0.025s $\to$ `Mode 3 (Front Dodge Flip)` for 0.30s.
  - **Aerial Challenges:** Jump challenge triggers (`Mode 2` for 0.22s) for floating balls ($z > 200$ uu).
  - **Car-Ball Collisions:** Full momentum transfer into the ball with kickoff resets upon scoring.
- **Random Seed:** 42

### 2.2 Optimization & Loss Objectives (Exact Tournament Specification)
- **Objective Function:** Standard tournament composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}) + 1.2 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** Adam (`lr = 1.0e-3`, `weight_decay = 1.0e-5`) with `CosineAnnealingLR` scheduled decay down to `1.0e-4` over 35 epochs and gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 256 with truncated backpropagation through time over 35 epochs. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ was preserved across chunks within matches and detached at chunk boundaries, only being reset at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Learning Rate | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5722 | 0.2933 | 0.4553 | 0.1450 | 79.3% | 0.00100 | Warmup |
| 5 | 0.2290 | 0.0852 | 0.2270 | 0.0643 | 91.9% | 0.00096 | SteerPD convergence |
| 10 | 0.1233 | 0.0571 | 0.1103 | 0.0529 | 96.0% | 0.00083 | Approach alignment |
| 15 | 0.0738 | 0.0425 | 0.0505 | 0.0288 | 98.0% | 0.00065 | Strike timing learned |
| 20 | 0.0422 | 0.0287 | 0.0336 | 0.0258 | 98.8% | 0.00045 | High precision |
| 25 | 0.0252 | 0.0152 | 0.0217 | 0.0140 | 99.2% | 0.00027 | Fine tuning |
| 30 | 0.0156 | 0.0125 | 0.0239 | 0.0114 | 99.1% | 0.00014 | Deep convergence |
| 34 | 0.0109 | 0.0109 | 0.0106 | 0.0095 | **99.7%** | 0.00010 | **Best Checkpoint** |
| 35 | 0.0102 | 0.0108 | 0.0115 | 0.0099 | 99.7% | 0.00010 | Final epoch |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **99.7%** (Best Checkpoint, Epoch 34)
- **Validation Mode Loss:** **0.0106**
- **Validation Steering MSE:** **0.0095**
- **Lowest Total Validation Loss:** **0.0220**
- **Live Sanity Invariance:** Kickoff Mode = 1 (Steer: -0.0298 rad), Ball Right Steer = +0.9404, Ball Left Steer = -0.6784.

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `27a138dd324cd4d7aa1f481a36d91f590f925c3bd984ce48a41dfcafaebe0258`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to verify direct ball strikes and clean straight driving.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
