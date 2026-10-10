# IEEE (IEEE/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** IEEE
- **Agent Identifier:** `IEEE/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `08d4741f25e9a85330ece0a178881dcb7dd12739d9bfd3f85f051744add784aa`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 IEEE Championship Tactical Engine (v7 Scaled to 768k Steps)
The policy was trained via behavioral cloning across extensive full-match simulated game states:
- **Dataset Version:** `ieee-championship-v7`
- **Dataset Scale:** 400 training matches (640,000 timesteps) and 80 validation matches (128,000 timesteps), total **768,000 timesteps** partitioned into continuous full matches with symmetric horizontal reflection mirroring ($X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer}$).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics & Tactical Motivations:** 
  - **Shadow Defense & Retreat Motivation:** If the car gets caught ahead of the ball ($Y_{\text{car}} > Y_{\text{ball}} + 100$), it avoids turning directly into the ball to eliminate self-goals. Instead, it motivates a retreat arc into defensive positioning ($Y < Y_{\text{ball}} - 800$) to regain goal-side clearance before re-engaging.
  - **Offensive Goal Strike Arc:** Post-clamped shot line ($X \in [-700, 700]$ at $Y = 5120$) with tangential arc adjustment to swing behind the ball and drive through it towards the net.
  - **Shoot-Through Strike Alignment:** When aligned behind the ball ($\text{dot\_align} > 0.65$), the targeting point drives through the ball directly toward the opponent's net (`ball_pos + shot_dir * 120`).
  - **Goal-Strike Powershot Trigger:** 3-stage front flip (`Jump 0.075s -> Release 0.025s -> Front Dodge 0.30s`) strictly gated by net alignment ($\text{dot\_align} > 0.60$), delivering high-speed powershots right into the net.
  - **Aerial Challenge & Jump Pops:** Jump challenges (`Mode 2` for 0.22s) to contest and block floating balls ($Z > 180$ uu).
  - **Kickoff Speed Flip:** High-speed kickoff rush with early alignment dodges when $Y < -1500$, speed $>600$ uu/s, and distance $>1300$ uu to win kickoff challenges.
  - **Cubic Steer PD Curve:** Dead-straight driving without twitch or wobble ($\text{clamp}(((35 \cdot \text{angle})^3) / 10, -1, 1)$).
  - **Callback Clock Tolerances:** Seamless handling of duplicate ticks and match resets without exceptions.
- **Random Seed:** 42

### 2.2 Optimization & Loss Objectives (Exact Tournament Specification)
- **Objective Function:** Standard tournament composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{mode logits}, \text{target mode}) + 1.2 \times \mathcal{L}_{\text{MSE}}(\text{steer}, \text{target steer})_{\text{mode=Chase}}$$
- **Optimizer:** Adam (`lr = 1.0e-3`, `weight_decay = 1.0e-5`) with `CosineAnnealingLR` scheduled decay down to `1.0e-4` over 35 epochs and gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Processed in sequence chunks of length 256 with truncated backpropagation through time over 35 epochs. Recurrent hidden state $h \in \mathbb{R}^{1 \times 1 \times 64}$ preserved across chunks within matches and reset only at whole-match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Learning Rate | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.4313 | 0.2952 | 0.3137 | 0.2314 | 88.7% | 0.00100 | Warmup |
| 5 | 0.0555 | 0.0928 | 0.0482 | 0.0825 | 98.1% | 0.00096 | SteerPD convergence |
| 10 | 0.0271 | 0.0478 | 0.0237 | 0.0489 | 99.1% | 0.00083 | Shadow defense learned |
| 15 | 0.0158 | 0.0330 | 0.0142 | 0.0397 | 99.5% | 0.00065 | Powershot timing |
| 20 | 0.0103 | 0.0252 | 0.0130 | 0.0226 | 99.6% | 0.00045 | High precision |
| 25 | 0.0064 | 0.0206 | 0.0070 | 0.0178 | 99.7% | 0.00027 | Fine tuning |
| 30 | 0.0036 | 0.0181 | 0.0029 | 0.0151 | 99.9% | 0.00014 | Deep convergence |
| 33 | 0.0028 | 0.0171 | **0.0017** | **0.0144** | **100.0%** | 0.00011 | **Best Checkpoint** |
| 35 | 0.0027 | 0.0166 | 0.0024 | 0.0144 | 99.9% | 0.00010 | Final epoch |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **100.0%** (Best Checkpoint, Epoch 33)
- **Validation Mode Loss:** **0.0017**
- **Validation Steering MSE:** **0.0144**
- **Lowest Total Validation Loss:** **0.0190**
- **Live Sanity Invariance:** Kickoff Mode = 1 (Steer: +0.0025 rad), Ball Right Steer = +0.9655, Ball Left Steer = -0.8831.

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `08d4741f25e9a85330ece0a178881dcb7dd12739d9bfd3f85f051744add784aa`.
3. **Tournament Validator Passed:** `tools/validate_submission.py .` confirmed 100% compliant with zero AST violations, exact 26,181 parameter count, and SHA-256 match.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground, shadow defense, and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
