# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `21878b0325bec643552f62e555c35f2cac8cf6cc366182776d804f92411fc8cb`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 Dataset & Simulation Environment
The policy was trained using behavioral cloning (imitation learning) on simulated Rocket League 1v1 Soccar matches:
- **Dataset Version:** `catalyst-synthetic-soccar-v1`
- **Splits:** 25 training matches (37,500 timesteps) and 5 validation matches (7,500 timesteps) partitioned by whole continuous matches (80/20 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Teacher Strategy:** An expert kinematic teacher incorporating ball-prediction lead targeting for distances > 1500 UU, continuous proportional steering towards the target (`steer = clamp(angle * 5, -1, 1)`), aerial single jumps for balls above 180 UU, and front-flip strikes for close-range balls at speed > 700 UU/s.
- **Random Seed:** 1337

### 2.2 Optimization & Loss Objectives
- **Objective Function:** Composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}) + 0.5 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** AdamW (`lr = 1.5e-3`, `weight_decay = 1e-4`) with gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 64 with truncated backpropagation through time over 25 epochs. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ was preserved across chunks within matches and detached at chunk boundaries, only being reset at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3490 | 0.6231 | 0.1550 | 0.6849 | 94.1% | Initial convergence |
| 5 | 0.0407 | 0.0345 | 0.0733 | 0.0567 | 96.8% | Steering stabilized |
| 10 | 0.0269 | 0.0235 | 0.0393 | 0.0928 | 98.7% | Mode distinction sharp |
| 13 | 0.0248 | 0.0194 | 0.0299 | 0.0198 | **98.9%** | **Best Checkpoint** |
| 20 | 0.0195 | 0.0176 | 0.0653 | 0.0291 | 97.8% | Fine-tuning flips |
| 25 | 0.0175 | 0.0165 | 0.0394 | 0.0283 | 98.6% | Training complete |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **98.9%** (Best Checkpoint)
- **Validation Mode Loss:** **0.0299**
- **Validation Steering MSE:** **0.0198**
- **Lowest Total Validation Loss:** **0.0398**

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `21878b0325bec643552f62e555c35f2cac8cf6cc366182776d804f92411fc8cb`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to observe ball tracking, kickoff transitions, and front-flip striking.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
