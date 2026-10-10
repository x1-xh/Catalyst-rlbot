"""Autonomous synthetic teacher simulation and behavioral cloning trainer for Catalyst.

Simulates Rocket League 1v1 Soccar kinematics, computes 13D observations using
src/observation_contract.py, labels actions using expert teacher policies,
and trains the exact 26,181-parameter causal GRU policy.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import Policy
from observation_contract import basis, build, FEATURES

ARENA_EXTENT_X = 4096.0
ARENA_EXTENT_Y = 5120.0
GOAL_Y = 5120.0
GRAVITY = -650.0
RESTITUTION = 0.6
BALL_RADIUS = 93.0
MAX_CAR_SPEED = 2300.0


class TrajectoryGenerator:
    """Simulates realistic 1v1 match trajectories and generates labeled (obs, action) pairs."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def generate_match(self, num_steps=1800, dt=1.0 / 60.0):
        """Simulates a continuous match for ~30 seconds (1800 steps at 60 Hz)."""
        observations = []
        target_modes = []
        target_steers = []

        # Initialize car and ball
        car_pos = [self.rng.uniform(-1000, 1000), self.rng.uniform(-3500, -2000), 17.0]
        car_yaw = self.rng.uniform(-math.pi, math.pi)
        car_pitch = 0.0
        car_roll = 0.0
        car_vel = [self.rng.uniform(-300, 300), self.rng.uniform(200, 800), 0.0]

        ball_pos = [self.rng.uniform(-1500, 1500), self.rng.uniform(-1500, 1500), BALL_RADIUS]
        ball_vel = [self.rng.uniform(-800, 800), self.rng.uniform(-800, 800), self.rng.uniform(0, 500)]

        prev_time = None
        prev_mode = 0
        prev_steer = 0.0

        # Action sequence state (for jump/dodge execution)
        seq_mode = None
        seq_timer = 0.0

        t = 0.0
        for step in range(num_steps):
            # Advance ball physics
            ball_pos[0] += ball_vel[0] * dt
            ball_pos[1] += ball_vel[1] * dt
            ball_pos[2] += ball_vel[2] * dt + 0.5 * GRAVITY * dt * dt
            ball_vel[2] += GRAVITY * dt

            # Ball arena collisions
            if ball_pos[2] <= BALL_RADIUS:
                ball_pos[2] = BALL_RADIUS
                ball_vel[2] = -ball_vel[2] * RESTITUTION
                ball_vel[0] *= 0.98
                ball_vel[1] *= 0.98

            if abs(ball_pos[0]) >= ARENA_EXTENT_X - BALL_RADIUS:
                ball_pos[0] = math.copysign(ARENA_EXTENT_X - BALL_RADIUS, ball_pos[0])
                ball_vel[0] = -ball_vel[0] * RESTITUTION

            if abs(ball_pos[1]) >= ARENA_EXTENT_Y - BALL_RADIUS:
                ball_pos[1] = math.copysign(ARENA_EXTENT_Y - BALL_RADIUS, ball_pos[1])
                ball_vel[1] = -ball_vel[1] * RESTITUTION

            # Ball 2.0s trajectory prediction
            pred_t = t + 2.0
            pred_ball = list(ball_pos)
            pred_vz = ball_vel[2]
            for _ in range(120):
                pred_dt = 2.0 / 120.0
                pred_ball[0] += ball_vel[0] * pred_dt
                pred_ball[1] += ball_vel[1] * pred_dt
                pred_ball[2] += pred_vz * pred_dt + 0.5 * GRAVITY * pred_dt * pred_dt
                pred_vz += GRAVITY * pred_dt
                if pred_ball[2] <= BALL_RADIUS:
                    pred_ball[2] = BALL_RADIUS
                    pred_vz = -pred_vz * RESTITUTION

            # Compute basis vectors
            car_basis = basis([car_pitch, car_yaw, car_roll])
            fwd, rgt, up = car_basis[0], car_basis[1], car_basis[2]

            # Build contract observation
            raw = {
                "position": car_pos,
                "velocity": car_vel,
                "basis": car_basis,
                "ball_present": True,
                "ball_position": ball_pos,
            }
            context = {
                "elapsed": t,
                "previous_elapsed": prev_time,
                "previous_mode": prev_mode,
                "previous_steer": prev_steer,
                "prediction_valid": True,
                "prediction_position": pred_ball,
                "selected_time": pred_t,
                "first_time": t + (1.0 / 120.0),
            }

            obs18 = build(raw, context)
            obs13 = list(obs18)[:13]
            observations.append(obs13)

            # Expert Teacher Policy
            dist_to_ball = math.sqrt(
                (ball_pos[0] - car_pos[0]) ** 2
                + (ball_pos[1] - car_pos[1]) ** 2
                + (ball_pos[2] - car_pos[2]) ** 2
            )
            speed = math.sqrt(car_vel[0] ** 2 + car_vel[1] ** 2 + car_vel[2] ** 2)

            # Target location: lead ball if far (>1500 UU)
            target = pred_ball if dist_to_ball > 1500.0 else ball_pos

            # Relative 2D angle for steering
            dx = target[0] - car_pos[0]
            dy = target[1] - car_pos[1]
            local_x = dx * fwd[0] + dy * fwd[1]
            local_y = dx * rgt[0] + dy * rgt[1]
            angle = math.atan2(local_y, local_x)
            steer = max(-1.0, min(1.0, angle * 5.0))

            # Decide Action Mode
            if seq_timer > 0.0:
                # Continuing active sequence (Jump or Flip)
                mode = seq_mode
                steer = 0.0
                seq_timer -= dt
                if seq_timer <= 0.0:
                    seq_mode = None
            elif step < 10:
                # Kickoff / neutral start
                mode = 0
                steer = 0.0
            elif dist_to_ball < 320.0 and speed > 700.0:
                # Execute front dodge strike
                mode = 3
                steer = 0.0
                seq_mode = 3
                seq_timer = 0.20  # Front dodge duration
            elif dist_to_ball < 450.0 and ball_pos[2] > 180.0:
                # Ball is in air: single jump
                mode = 2
                steer = 0.0
                seq_mode = 2
                seq_timer = 0.15  # Single jump duration
            else:
                # Chase with continuous steering
                mode = 1

            target_modes.append(mode)
            target_steers.append(steer if mode == 1 else 0.0)

            # Advance car kinematics
            if mode == 1:
                # Throttle forward and turn
                accel = 1200.0
                car_yaw += steer * 2.8 * dt
                car_vel[0] = math.cos(car_yaw) * min(MAX_CAR_SPEED, speed + accel * dt)
                car_vel[1] = math.sin(car_yaw) * min(MAX_CAR_SPEED, speed + accel * dt)
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
            elif mode == 3:
                # Front flip velocity impulse
                fwd_boost = 500.0
                car_vel[0] += math.cos(car_yaw) * fwd_boost * dt * 5
                car_vel[1] += math.sin(car_yaw) * fwd_boost * dt * 5
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
            elif mode == 2:
                # Single jump
                car_pos[2] = min(150.0, car_pos[2] + 400.0 * dt)
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
            else:
                car_vel[0] *= 0.95
                car_vel[1] *= 0.95
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt

            # Keep car bounded inside arena
            car_pos[0] = max(-ARENA_EXTENT_X + 100, min(ARENA_EXTENT_X - 100, car_pos[0]))
            car_pos[1] = max(-ARENA_EXTENT_Y + 100, min(ARENA_EXTENT_Y - 100, car_pos[1]))

            prev_time = t
            prev_mode = mode
            prev_steer = steer
            t += dt

        return (
            np.array(observations, dtype=np.float32),
            np.array(target_modes, dtype=np.int64),
            np.array(target_steers, dtype=np.float32),
        )


def train_policy(epochs=25, lr=1e-3, seed=42):
    print("=" * 60)
    print("Catalyst 13D GRU Policy Training Pipeline")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = TrajectoryGenerator(seed=seed)

    print("\n[*] Generating synthetic 1v1 match datasets...")
    num_train_matches = 25
    num_val_matches = 5
    match_steps = 1500  # 25 seconds per match

    train_data = [generator.generate_match(num_steps=match_steps) for _ in range(num_train_matches)]
    val_data = [generator.generate_match(num_steps=match_steps) for _ in range(num_val_matches)]

    total_train_steps = num_train_matches * match_steps
    total_val_steps = num_val_matches * match_steps
    print(f"    - Train matches: {num_train_matches} ({total_train_steps} timesteps)")
    print(f"    - Val matches:   {num_val_matches} ({total_val_steps} timesteps)")

    # Model instantiation
    device = torch.device("cpu")
    model = Policy().to(device)
    print(f"    - Model parameters: {sum(p.numel() for p in model.parameters())} (Expected: 26,181)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mode_criterion = nn.CrossEntropyLoss()
    steer_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    metrics_history = []

    seq_len = 64  # Truncated backpropagation through time sequence length

    print("\n[*] Starting training loop (epochs={}):".format(epochs))
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_mode = 0.0
        train_loss_steer = 0.0
        train_steps_count = 0

        # Shuffle training matches
        random.shuffle(train_data)

        for obs, modes, steers in train_data:
            T = len(obs)
            h = None  # Reset recurrent hidden state only at match start

            for start in range(0, T, seq_len):
                end = min(start + seq_len, T)
                x = torch.from_numpy(obs[start:end]).to(device)
                y_mode = torch.from_numpy(modes[start:end]).to(device)
                y_steer = torch.from_numpy(steers[start:end]).to(device)

                optimizer.zero_grad()
                pred_mode, pred_steer, h = model(x, h)
                h = h.detach()  # Detach hidden state between chunks

                loss_m = mode_criterion(pred_mode, y_mode)

                # Compute steering loss where mode is Chase (1)
                chase_mask = y_mode == 1
                if chase_mask.any():
                    loss_s = steer_criterion(pred_steer[chase_mask], y_steer[chase_mask])
                else:
                    loss_s = torch.tensor(0.0, device=device)

                loss = loss_m + 0.5 * loss_s
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_mode += loss_m.item() * (end - start)
                train_loss_steer += loss_s.item() * (end - start)
                train_steps_count += (end - start)

        avg_train_loss_mode = train_loss_mode / train_steps_count
        avg_train_loss_steer = train_loss_steer / train_steps_count

        # Validation evaluation
        model.eval()
        val_loss_mode = 0.0
        val_loss_steer = 0.0
        correct_modes = 0
        val_steps_count = 0

        with torch.no_grad():
            for obs, modes, steers in val_data:
                T = len(obs)
                h = None
                x = torch.from_numpy(obs).to(device)
                y_mode = torch.from_numpy(modes).to(device)
                y_steer = torch.from_numpy(steers).to(device)

                pred_mode, pred_steer, _ = model(x, h)

                loss_m = mode_criterion(pred_mode, y_mode)
                chase_mask = y_mode == 1
                loss_s = steer_criterion(pred_steer[chase_mask], y_steer[chase_mask]) if chase_mask.any() else torch.tensor(0.0)

                val_loss_mode += loss_m.item() * T
                val_loss_steer += loss_s.item() * T
                preds = pred_mode.argmax(dim=1)
                correct_modes += (preds == y_mode).sum().item()
                val_steps_count += T

        avg_val_loss_mode = val_loss_mode / val_steps_count
        avg_val_loss_steer = val_loss_steer / val_steps_count
        val_acc = (correct_modes / val_steps_count) * 100.0
        total_val_loss = avg_val_loss_mode + 0.5 * avg_val_loss_steer

        metrics_history.append({
            "epoch": epoch,
            "train_mode_loss": round(avg_train_loss_mode, 4),
            "train_steer_loss": round(avg_train_loss_steer, 4),
            "val_mode_loss": round(avg_val_loss_mode, 4),
            "val_steer_loss": round(avg_val_loss_steer, 4),
            "val_mode_accuracy": round(val_acc, 2),
            "total_val_loss": round(total_val_loss, 4),
        })

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = "★ BEST"
        else:
            mark = ""

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Mode: {avg_train_loss_mode:.4f}, Steer: {avg_train_loss_steer:.4f} | "
            f"Val Mode: {avg_val_loss_mode:.4f}, Steer: {avg_val_loss_steer:.4f}, Acc: {val_acc:.1f}% {mark}"
        )

    # Save trained checkpoint
    checkpoint_path = ROOT / "model" / "catalyst_policy.pt"
    torch.save({"state_dict": best_state_dict}, checkpoint_path)
    print(f"\n[✓] Saved best checkpoint to: {checkpoint_path}")

    # Provenance metadata
    final_metrics = metrics_history[-1]
    provenance = {
        "dataset_version": "catalyst-synthetic-soccar-v1",
        "split_method": "whole 1v1 matches (25 train matches / 5 val matches, 80/20 split)",
        "seed": seed,
        "architecture": "13D / Linear(13->64) / ReLU / GRU(64->64) / Linear(64->5)",
        "objective": "CrossEntropyLoss(modes) + 0.5 * MSELoss(steering)",
        "checkpoint_selection": f"Lowest total validation loss ({best_val_loss:.4f})",
        "training_command": f"python tools/train.py --epochs {epochs} --lr {lr} --seed {seed}",
        "validation_results": (
            f"Val Mode Accuracy: {final_metrics['val_mode_accuracy']}%, "
            f"Val Mode Loss: {final_metrics['val_mode_loss']}, "
            f"Val Steering MSE: {final_metrics['val_steer_loss']}"
        ),
        "limitations": "Model trained on kinematic imitation; wall-air aerial touches remain heuristic.",
    }

    provenance_path = ROOT / "docs" / "training_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved training provenance to: {provenance_path}")

    return checkpoint_path, provenance_path, metrics_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Catalyst GRU Policy")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    checkpoint_path, provenance_path, metrics = train_policy(epochs=args.epochs, lr=args.lr, seed=args.seed)
