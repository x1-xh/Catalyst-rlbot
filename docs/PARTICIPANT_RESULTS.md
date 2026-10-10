# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `d46eee89a797222d904b9a167d3767f2808b6aa85a4d324d76aac3cc58e7e836`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 Dataset & Simulation Environment
The policy was trained using behavioral cloning (imitation learning) on simulated Rocket League 1v1 Soccar matches with full car-ball collisions:
- **Dataset Version:** `catalyst-strike-anti-circling-v2`
- **Splits:** 28 training matches (44,800 timesteps) and 6 validation matches (9,600 timesteps) partitioned by whole continuous matches (82/18 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Strike & Anti-Circling Dynamics:** 
  - Aiming targets a contact offset behind the ball oriented directly towards the opponent's goal `(0, 5120)`.
  - Close-range approach steering uses smooth proportional gain to prevent the car from orbiting or circling the ball.
  - Front-flip strikes are generated via the proper 3-stage sequence: `Mode 2 (Jump)` for 0.05s $\to$ `Mode 0 (Neutral release)` for 0.033s $\to$ `Mode 3 (Front Dodge)` for 0.18s.
  - Car-ball collisions transfer linear momentum into the ball towards the shot direction, resetting to kickoff upon goal conversion.
- **Random Seed:** 2026

### 2.2 Optimization & Loss Objectives
- **Objective Function:** Weighted composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}, w=[1.0, 1.0, 3.5, 3.5]) + 0.5 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** AdamW (`lr = 1.5e-3`, `weight_decay = 1e-4`) with gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 64 with truncated backpropagation through time over 25 epochs. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ was preserved across chunks within matches and detached at chunk boundaries, only being reset at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.1110 | 0.2283 | 1.2602 | 0.1742 | 54.4% | Weighted mode warmup |
| 5 | 0.4632 | 0.0471 | 0.3956 | 0.0359 | 82.6% | Strike triggers learned |
| 10 | 0.2330 | 0.0178 | 0.1590 | 0.0155 | 92.4% | Anti-circling alignment |
| 15 | 0.1372 | 0.0127 | 0.0546 | 0.0164 | 96.7% | Flip timing sharpened |
| 21 | 0.1086 | 0.0087 | 0.0415 | 0.0060 | **97.8%** | **Best Checkpoint** |
| 25 | 0.0900 | 0.0095 | 0.0850 | 0.0095 | 96.9% | Convergence reached |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **97.8%** (Best Checkpoint)
- **Validation Mode Loss:** **0.0415**
- **Validation Steering MSE:** **0.0060**
- **Lowest Total Validation Loss:** **0.0445**

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `d46eee89a797222d904b9a167d3767f2808b6aa85a4d324d76aac3cc58e7e836`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to verify direct ball strikes and elimination of circular orbiting.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
