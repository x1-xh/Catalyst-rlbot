# IEEE (IEEE/IEEE) — Participant Technical Report

## 1. Executive Summary & Identity
- **Bot Display Name:** IEEE
- **Agent Identifier:** `IEEE/IEEE`
- **Architecture:** 13D Causal GRU (`Linear(13 -> 64) -> ReLU -> GRU(64 -> 64) -> Linear(64 -> 5)`)
- **Trainable Parameters:** Exactly 26,181 parameters
- **Trained Checkpoint SHA-256:** `f1cad370683ceceaf09d95f19103f007887973c02a707d4ef9b09b5c1de3fb13`
- **Baseline Checkpoint SHA-256:** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`
- **Baseline Cosine Similarity:** `0.21721` (Well clear of the 0.999 plagiarism ceiling)

---

## 2. Training Methodology & Provenance

### 2.1 IEEE Championship Tactical Engine (v8 Scaled to 960k Timesteps)
The policy was trained via behavioral cloning across extensive full-match simulated game states:
- **Dataset Version:** `ieee-championship-v8`
- **Dataset Scale:** 500 training matches (800,000 timesteps) and 100 validation matches (160,000 timesteps), total **960,000 timesteps** partitioned into continuous full matches with symmetric horizontal reflection mirroring ($X \to -X, \text{yaw} \to \pi - \text{yaw}, \text{steer} \to -\text{steer}$).
- **Observation Space:** 13 normalized dimensions constructed via `src/observation_contract.py` (car-relative ball displacement, 2-second trajectory prediction slices, distances, speeds, and callback timing).
- **Tactical Scenario Synthesis:**
  - **Kickoff Duels:** All 5 official Rocket League kickoff spawns (center, diagonal, off-center) with laser-straight center rush.
  - **Offensive Breakaways:** High-speed pursuit of balls rolling toward the opponent's net, dynamically leading the 2.0s trajectory and executing precision powershots.
  - **Sidewall & Backwall Rebounds:** Intercepting bounces off arena walls and centering/striking into the net.
  - **Aerial Pop Contests:** High bouncing balls ($Z \in [250, 450]$ uu) prompting single-jump aerial pops (`Mode 2` for 0.18s) to challenge in the air.
  - **Midfield Loose Scrambles:** Quick turnarounds and rapid ball recovery from arbitrary field positions.
- **Proportional-Cubic Steer Controller:** Silky-smooth steering curve ($\text{clip}(3.0 \cdot \theta + 2.0 \cdot \theta^3, -1, 1)$) providing gentle corrections near center for dead-straight driving and ramping to full lock for sharp turns.
- **Lethal 3-Stage Powershot Flip Mechanic:** When closing in on a reachable ball ($80 < \text{dist} < 360$ uu, $|\theta| < 0.24$ rad, $v > 350$ uu/s), triggers:
  $$\text{Jump (Mode 2, 0.06s)} \longrightarrow \text{Release (Mode 0, 0.02s)} \longrightarrow \text{Front Dodge (Mode 3, 0.28s)}$$
  imparting 500 uu/s of explosive forward impulse to launch the ball with maximum power into the net.

### 2.2 Optimization & Loss Objectives
- **Focal Class-Weighted Cross-Entropy:** Mode weights $[3.5, 1.0, 3.0, 2.5]$ prevent rare action frames (Jump, Release, Front Dodge) from being overshadowed by common Chase frames.
- **Centering Regularization & Symmetry:** Added L1 loss for near-zero steering deviation ($|\text{steer}| < 0.05$) to enforce rock-solid driving on kickoffs and breakaways.
- **Optimizer:** AdamW (`lr = 1.5e-3`, `weight_decay = 1.0e-5`) with `CosineAnnealingLR` scheduled decay down to `1.0e-4` over 30 epochs and gradient clipping at `max_norm = 1.0`.
- **Recurrent State:** Sequence chunk length 512 with truncated BPTT, preserving recurrent hidden state within matches and resetting at match boundaries.

---

## 3. Training Progression & Validation Metrics

| Epoch | Train Mode Loss | Train Steer MSE | Val Mode Loss | Val Steer MSE | Val Mode Accuracy | Learning Rate | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5308 | 0.1990 | 0.4723 | 0.1457 | 87.0% | 0.00150 | Warmup |
| 5 | 0.1661 | 0.0445 | 0.1391 | 0.0415 | 96.3% | 0.00141 | Steer convergence |
| 10 | 0.0826 | 0.0258 | 0.0844 | 0.0235 | 97.9% | 0.00115 | Powershot timing learned |
| 15 | 0.0564 | 0.0182 | 0.0805 | 0.0196 | 97.8% | 0.00080 | Lead pursuit refined |
| 20 | 0.0393 | 0.0136 | 0.0627 | 0.0154 | 98.4% | 0.00045 | High precision |
| 25 | 0.0291 | 0.0108 | 0.0645 | 0.0125 | 98.6% | 0.00019 | Fine tuning |
| 28 | 0.0254 | 0.0098 | **0.0600** | **0.0128** | **98.7%** | 0.00012 | **Best Checkpoint** |
| 30 | 0.0246 | 0.0095 | 0.0627 | 0.0125 | 98.7% | 0.00010 | Final annealed epoch |

### Final Measured Validation Performance
- **Mode Classification Accuracy:** **98.7%**
- **Validation Mode Loss:** **0.0600**
- **Validation Steering MSE:** **0.0128**
- **Lowest Total Validation Loss:** **0.0754**
- **Live Sanity Checks:**
  - Kickoff Center Rush: Mode = 1 (Chase), Steer = -0.0312 rad (Laser straight)
  - Ball Right Turn: Mode = 1, Steer = +0.9546
  - Ball Left Turn: Mode = 1, Steer = -0.8870
  - Powershot Approach Trigger: Correctly enters Mode 2 jump at striking range

---

## 4. Local Preflight & GUI Verification
1. **Tensor Invariance:** All 26,181 parameter tensors strictly finite (zero `NaN` or `Inf`).
2. **Cryptographic Sealing:** Checkpoint sealed via `python tools/kit.py seal` with SHA-256 `f1cad370683ceceaf09d95f19103f007887973c02a707d4ef9b09b5c1de3fb13`.
3. **Tournament Validator Passed:** `tools/validate_submission.py .` confirmed 100% compliant with zero AST violations, exact 26,181 parameter count, and SHA-256 match.
