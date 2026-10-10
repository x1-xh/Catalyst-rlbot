# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `8b53f6e066f5afa2ef877018affbe0a95365144f5164c7ad1779e254da7cdc39`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 Dataset & Simulation Environment
The policy was trained using behavioral cloning (imitation learning) on simulated Rocket League 1v1 Soccar matches:
- **Dataset Version:** `catalyst-synthetic-soccar-v1`
- **Splits:** 25 training matches (37,500 timesteps) and 5 validation matches (7,500 timesteps) partitioned by whole continuous matches (80/20 train/validation split).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Teacher Strategy:** An expert kinematic teacher incorporating ball-prediction lead targeting for distances > 1500 UU, continuous proportional steering towards the target (`steer = clamp(angle * 5, -1, 1)`), aerial single jumps for balls above 180 UU, and front-flip strikes for close-range balls at speed > 700 UU/s.

### 2.2 Optimization & Loss Objectives
- **Objective Function:** Composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}) + 0.5 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** AdamW (`lr = 1.5e-3`, `weight_decay = 1e-4`) with gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 64 with truncated backpropagation through time. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ was preserved across chunks within matches and detached at chunk boundaries, only being reset at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3592 | 0.7573 | 0.2601 | 0.2248 | 86.0% | Initial convergence |
| 5 | 0.0499 | 0.0473 | 0.0664 | 0.0887 | 97.3% | Steering stabilized |
| 10 | 0.0324 | 0.0327 | 0.0479 | 0.0431 | 98.6% | Mode distinction sharp |
| 15 | 0.0294 | 0.0200 | 0.0507 | 0.0513 | 98.1% | Fine-tuning flips |
| 20 | 0.0274 | 0.0183 | 0.0348 | 0.0262 | **98.6%** | **Best Checkpoint** |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **98.6%**
- **Validation Mode Loss:** **0.0348**
- **Validation Steering MSE:** **0.0262**
- **Total Validation Loss:** **0.0479**

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `8b53f6e066f5afa2ef877018affbe0a95365144f5164c7ad1779e254da7cdc39`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to observe ball tracking, kickoff transitions, and front-flip striking.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
