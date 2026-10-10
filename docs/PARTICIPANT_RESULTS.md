# Catalyst (Catalyst/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** Catalyst
- **Agent Identifier:** `Catalyst/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `90529e565bf199c499da2282a45b3662a8f5c53a2a452ebec73b3eea224c98df`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 Dataset & Simulation Environment
The policy was trained using behavioral cloning (imitation learning) on simulated Rocket League 1v1 Soccar matches strictly conforming to the official tournament contract:
- **Dataset Version:** `catalyst-strike-anti-circling-v2`
- **Splits:** 28 training matches (44,800 timesteps) and 6 validation matches (9,600 timesteps) partitioned by whole continuous matches (82/18 train/validation split).
- **Observation Space:** Exact 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Strike & Anti-Circling Dynamics:** 
  - Aiming targets a contact offset behind the ball oriented directly towards the opponent's goal `(0, 5120)`.
  - Close-range approach steering uses smooth proportional gain to prevent the car from orbiting or circling the ball.
  - Front-flip strikes are generated via the proper 3-stage sequence: `Mode 2 (Jump)` for 0.05s $\to$ `Mode 0 (Neutral release)` for 0.033s $\to$ `Mode 3 (Front Dodge)` for 0.18s.
  - Car-ball collisions transfer linear momentum into the ball towards the shot direction, resetting to kickoff upon goal conversion.
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
| 1 | 1.2286 | 0.3137 | 1.1376 | 0.2432 | 48.2% | Initial convergence |
| 5 | 0.7459 | 0.0563 | 0.7316 | 0.0597 | 68.8% | Steering aligned |
| 10 | 0.5418 | 0.0258 | 0.5206 | 0.0188 | 78.8% | Mode distinction clear |
| 15 | 0.3684 | 0.0174 | 0.3534 | 0.0134 | 85.5% | Strike timing learned |
| 20 | 0.2794 | 0.0127 | 0.2888 | 0.0093 | 90.7% | High precision |
| 25 | 0.1845 | 0.0106 | 0.1431 | 0.0059 | **94.5%** | **Best Checkpoint** |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **94.51%**
- **Validation Mode Loss:** **0.1431**
- **Validation Steering MSE:** **0.0059**
- **Lowest Total Validation Loss:** **0.1490**

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `90529e565bf199c499da2282a45b3662a8f5c53a2a452ebec73b3eea224c98df`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing Catalyst in a 1v1 match on standard DFH Stadium against baseline opponents to verify direct ball strikes and elimination of circular orbiting.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
