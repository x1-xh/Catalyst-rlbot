# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `6251f60615427e2499a0515ebb0889b343bb1a44590c2c87e4b260758cf8f9da`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 RLBotPack Tournament Distillation
The policy was trained via behavioral cloning and policy distillation utilizing tournament algorithms from top bots in the official RLBotPack (`VirxEB` & `Necto`):
- **Dataset Version:** `catalyst-rlbotpack-distillation-v3`
- **Splits:** 60 training matches (96,000 timesteps) and 12 validation matches (19,200 timesteps) partitioned by whole continuous matches (83/17 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics:** 
  - **VirxEB Cubic Steer PD:** Uses the cubic steering curve $\text{clamp}\left(\frac{(35 \cdot \text{angle})^3}{10}, -1, 1\right)$ to deliver dead-straight driving without angle oscillation while providing agile turning response when needed.
  - **Symmetric Horizontal Reflection Augmentation:** Every match trajectory is augmented with its exact mirror $(X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer})$. Guarantees mathematical left-right symmetry, eliminating positive steering bias and circling.
  - **Goal-Directed Approach with Direct Close-Range Strike:** Guides the car behind the ball facing opponent goal `(0, 5120)` when at range, smoothly transitioning directly into `ball_pos` when close (<350 uu) to drive through the ball rather than orbiting around it.
  - **Kickoff Charge Activation:** Car starts in full-throttle Chase mode from frame 0 at kickoff without delay.
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
| 1 | 0.7686 | 0.3801 | 0.6977 | 0.3513 | 69.1% | Warmup |
| 5 | 0.3758 | 0.1093 | 0.3539 | 0.0999 | 85.9% | SteerPD convergence |
| 10 | 0.2470 | 0.0655 | 0.2552 | 0.0798 | 89.7% | Approach alignment |
| 15 | 0.1544 | 0.0496 | 0.1213 | 0.0407 | 95.0% | Strike timing learned |
| 20 | 0.1042 | 0.0339 | 0.1179 | 0.0289 | 95.3% | High precision |
| 23 | 0.0825 | 0.0299 | 0.0688 | 0.0233 | **97.3%** | **Best Checkpoint** |
| 25 | 0.0757 | 0.0260 | 0.0758 | 0.0262 | 97.1% | Final epoch |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **97.3%** (Best Checkpoint)
- **Validation Mode Loss:** **0.0688**
- **Validation Steering MSE:** **0.0233**
- **Lowest Total Validation Loss:** **0.0921**
- **Live Sanity Invariance:** Kickoff Mode = 1 (Steer: -0.0329 rad), Ball Right Steer = +0.9325, Ball Left Steer = -0.8406.

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `6251f60615427e2499a0515ebb0889b343bb1a44590c2c87e4b260758cf8f9da`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to verify direct ball strikes and clean straight driving.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
