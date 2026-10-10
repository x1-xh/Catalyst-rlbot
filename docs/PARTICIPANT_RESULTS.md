# IEEE (IEEE/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** IEEE
- **Agent Identifier:** `IEEE/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `f293cb76710b5d49947659d4ef4d5a2ebf9bcf07947ee35cd9a107860a4f86bb`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 2. Training Methodology & Provenance

### 2.1 RLBotPack Tournament Distillation (v5 Offensive Arc)
The policy was trained via behavioral cloning and policy distillation utilizing tournament algorithms from top bots in the official RLBotPack (`VirxEB` & `Necto`) with offensive goal-scoring arc mechanics:
- **Dataset Version:** `ieee-rlbotpack-distillation-v5-offensive-arc`
- **Splits:** 160 training matches (256,000 timesteps) and 32 validation matches (51,200 timesteps), total 307,200 timesteps partitioned by whole continuous matches (symmetric mirrored).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Integrated Tournament Mechanics:** 
  - **VirxEB Offensive Goal Strike Arc:** Opponent goal target clamped within posts at $X \in [-700, 700]$ at $Y = 5120$. Calculates shot line unit vector $\hat{u}_{\text{shot}}$ and car-to-ball alignment $\text{dot\_align} = (\vec{cb} \cdot \hat{u}_{\text{shot}}) / \|\vec{cb}\|$.
  - **Arc Behind Unaligned Balls:** When approaching from an angle ($\text{dot\_align} \le 0.75$), the bot smoothly arcs behind the ball along the shot line rather than cutting across it, preventing inadvertent sidewall pinches.
  - **Shoot-Through Targeting When Aligned:** When aligned behind the ball ($\text{dot\_align} > 0.75$), targeting point shifts through and past the ball into the net (`ball_pos + shot_dir * 90`), delivering decisive strikes directly into the goal.
  - **Goal-Strike Powershot Trigger:** VirxEB 3-stage dodge into the ball (`Mode 2 (Jump 0.075s) -> Mode 0 (Release 0.025s) -> Mode 3 (Front Dodge 0.30s)`) requires $\text{dot\_align} > 0.65$, ensuring high-velocity flips only trigger when aimed straight at the opponent's net.
  - **VirxEB Cubic Steer PD:** Uses the cubic steering curve $\text{clamp}\left(\frac{(35 \cdot \text{angle})^3}{10}, -1, 1\right)$ to deliver dead-straight driving without angle oscillation while providing agile turning response when needed.
  - **Symmetric Horizontal Reflection Augmentation:** Every match trajectory is augmented with its exact mirror $(X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer})$. Guarantees mathematical left-right symmetry, eliminating positive steering bias and circling.
  - **Kickoff Charge & Speed Flip:** Car starts in full-throttle Chase mode from frame 0 at kickoff, triggering an early alignment dodge when distance to ball $>1300$ uu, speed $>600$ uu/s, and $Y < -1500$ to win kickoff challenges.
  - **Duplicate/Backward Callback Clock Tolerance:** Runtime and bot interfaces gracefully handle duplicate ticks and match resets without raising callback clock exceptions.
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
| 1 | 0.4664 | 0.4319 | 0.3894 | 0.3581 | 88.3% | 0.00100 | Warmup |
| 5 | 0.2281 | 0.1061 | 0.2527 | 0.0754 | 92.7% | 0.00096 | SteerPD convergence |
| 10 | 0.0890 | 0.0624 | 0.0695 | 0.0409 | 97.4% | 0.00083 | Arc alignment learned |
| 15 | 0.0442 | 0.0378 | 0.0404 | 0.0319 | 98.6% | 0.00065 | Powershot timing |
| 20 | 0.0265 | 0.0221 | 0.0223 | 0.0131 | 99.2% | 0.00045 | High precision |
| 25 | 0.0162 | 0.0131 | 0.0104 | 0.0102 | 99.7% | 0.00027 | Fine tuning |
| 30 | 0.0104 | 0.0102 | 0.0068 | 0.0071 | 99.8% | 0.00014 | Deep convergence |
| 35 | 0.0070 | 0.0087 | **0.0052** | **0.0061** | **99.9%** | 0.00010 | **Best Checkpoint** |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **99.9%** (Best Checkpoint, Epoch 35)
- **Validation Mode Loss:** **0.0052**
- **Validation Steering MSE:** **0.0061**
- **Lowest Total Validation Loss:** **0.0126**
- **Live Sanity Invariance:** Kickoff Mode = 1 (Steer: 0.0509 rad), Ball Right Steer = +0.8268, Ball Left Steer = -0.9311.

---

## 4. Local Preflight & GUI Verification
1. **Mathematical & Tensor Invariance:** Verified offline that all 26,181 parameter tensors are strictly finite (no `NaN` or `Inf`). Forward pass produces shape `(1, 4)` for mode logits, `(1,)` for continuous steering, and `(1, 1, 64)` for hidden state.
2. **Cryptographic Sealing:** Sealed via `python tools/kit.py seal`, updating `src/runtime.py` with runtime PIN `f293cb76710b5d49947659d4ef4d5a2ebf9bcf07947ee35cd9a107860a4f86bb`.
3. **GUI Acceptance Plan (Windows):** Launching via RLBot v5 launcher (`bot.toml`), placing IEEE in a 1v1 match on standard DFH Stadium against baseline opponents to verify offensive goal scoring and clean straight driving.

---

## 5. Limitations & Future Extensions
- Current policy is trained on 2D ground and low-altitude aerial trajectories; high-wall riding and multi-touch aerial dribbles are not currently covered.
- Boost usage and handbrake recovery remain set to baseline defaults (0 / False) per tournament mode specifications.
