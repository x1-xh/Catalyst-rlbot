# IEEE (IEEE/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** IEEE
- **Agent Identifier:** `IEEE/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `985fa35c4390cb99ef19efce8c85e8806f9f843a3cac606096e65c21d2447b4b`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 RLBotPack Full-Arsenal Distillation (v6 Scaled to 768k Steps)
The policy was distilled directly from the top bots in the official local `RLBotPack` (`VirxEB`, `Necto`, and `Nexto`):
- **Dataset Version:** `ieee-rlbotpack-distillation-v6-full-arsenal`
- **Dataset Scale:** 400 training matches (640,000 timesteps) and 80 validation matches (128,000 timesteps), total **768,000 timesteps** partitioned into continuous full matches with symmetric horizontal mirroring ($X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer}$).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics & Motivations:** 
  - **Shadow Defense & Retreat Motivation (from `VirxEB` `shadow` & `retreat`):** If the car gets caught ahead of the ball ($Y_{\text{car}} > Y_{\text{ball}} + 150$), it avoids cutting into the ball towards its own net. Instead, it motivates a retreat arc into defensive positioning ($Y < Y_{\text{ball}} - 900$) to regain goal-side clearance before re-engaging.
  - **Offensive Goal Strike Arc (from `VirxEB ground_shot`):** Post-clamped shot line with tangential arc adjustment to swing behind the ball and drive through it towards the net.
  - **Shoot-Through Strike Alignment:** When aligned behind the ball ($\text{dot\_align} > 0.75$), targeting point drives through the ball directly toward the opponent's net (`ball_pos + shot_dir * 90`).
  - **Goal-Strike Powershot Trigger:** VirxEB 3-stage front flip (`Jump 0.075s -> Release 0.025s -> Front Dodge 0.30s`) strictly gated by net alignment ($\text{dot\_align} > 0.65$), delivering high-speed powershots right into the net.
  - **Aerial Challenge & Jump Pops:** Jump challenges (`Mode 2` for 0.22s) to block floating balls ($Z > 200$ uu).
  - **Kickoff Speed Flip (from `Nexto` & `VirxEB`):** High-speed kickoff rush with early alignment dodges when $Y < -1500$, speed $>600$ uu/s, and distance $>1300$ uu.
  - **VirxEB Cubic Steer PD:** Dead-straight driving without twitch or wobble ($\text{clamp}(((35 \cdot \text{angle})^3) / 10, -1, 1)$).
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
| 1 | 0.4196 | 0.2353 | 0.2550 | 0.1501 | 90.9% | 0.00100 | Warmup |
| 5 | 0.0640 | 0.0795 | 0.0950 | 0.0766 | 96.9% | 0.00096 | SteerPD convergence |
| 10 | 0.0296 | 0.0440 | 0.0204 | 0.0349 | 99.2% | 0.00083 | Shadow defense learned |
| 15 | 0.0163 | 0.0344 | 0.0138 | 0.0291 | 99.5% | 0.00065 | Powershot timing |
| 20 | 0.0102 | 0.0198 | 0.0101 | 0.0119 | 99.6% | 0.00045 | High precision |
| 25 | 0.0060 | 0.0123 | 0.0050 | 0.0090 | 99.8% | 0.00027 | Fine tuning |
| 30 | 0.0036 | 0.0088 | 0.0031 | 0.0083 | 99.9% | 0.00014 | Deep convergence |
| 34 | 0.0027 | 0.0076 | 0.0020 | 0.0071 | 99.9% | 0.00010 | Optimal steering MSE |
| 35 | 0.0029 | 0.0080 | **0.0018** | **0.0072** | **100.0%** | 0.00010 | **Best Checkpoint** |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **100.0%** (Best Checkpoint, Epoch 35)
- **Validation Mode Loss:** **0.0018**
- **Validation Steering MSE:** **0.0072**
- **Lowest Total Validation Loss:** **0.0104**
- **Live Sanity Invariance:** Kickoff Mode = 1 (Steer: -0.0143 rad), Ball Right Steer = +0.9808, Ball Left Steer = -0.9972.

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `985fa35c4390cb99ef19efce8c85e8806f9f843a3cac606096e65c21d2447b4b`.
3. **Tournament Validator Passed:** `tools/validate_submission.py .` confirmed 100% compliant with zero AST violations, exact 26,181 parameter count, and SHA-256 match.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground, shadow defense, and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
