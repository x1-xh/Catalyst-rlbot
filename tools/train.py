"""Autonomous realistic 1v1 Soccar simulation and behavioral cloning trainer for Catalyst.

Includes:
- Realistic car-ball collisions and goal scoring
- Precision strike targeting (aiming through the ball towards the opponent goal)
- Proper 3-stage flip strike sequence: Jump (Mode 2) -> Release (Mode 0) -> Front Dodge (Mode 3)
- Kickoff fast approach and aerial touch handling
- Anti-circling proportional steering
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import Policy
from observation_contract import basis, build

ARENA_EXTENT_X = 4096.0
ARENA_EXTENT_Y = 5120.0
GOAL_WIDTH_HALF = 892.0
GOAL_HEIGHT = 642.0
GRAVITY = -650.0
RESTITUTION = 0.6
BALL_RADIUS = 93.0
CAR_RADIUS = 75.0
MAX_CAR_SPEED = 2300.0


class TrajectoryGenerator:
    """Simulates realistic 1v1 match trajectories with real collisions, goals, and strikes."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def generate_match(self, num_steps=2000, dt=1.0 / 60.0):
        observations = []
        target_modes = []
        target_steers = []

        def reset_kickoff():
            # Standard kickoff positions
            kickoff_types = [
                ([0.0, -4608.0, 17.0], math.pi / 2),           # Straight back
                ([-2048.0, -2560.0, 17.0], math.pi / 4),       # Diagonal left
                ([2048.0, -2560.0, 17.0], 3 * math.pi / 4),    # Diagonal right
                ([-256.0, -3840.0, 17.0], math.pi / 2),        # Offset left
                ([256.0, -3840.0, 17.0], math.pi / 2),         # Offset right
            ]
            c_pos, c_yaw = self.rng.choice(kickoff_types)
            c_pos = list(c_pos)
            c_vel = [0.0, 0.0, 0.0]
            b_pos = [0.0, 0.0, BALL_RADIUS]
            b_vel = [0.0, 0.0, 0.0]
            return c_pos, c_vel, c_yaw, b_pos, b_vel

        car_pos, car_vel, car_yaw, ball_pos, ball_vel = reset_kickoff()
        car_pitch = 0.0
        car_roll = 0.0

        prev_time = None
        prev_mode = 0
        prev_steer = 0.0

        # Multi-stage sequence: list of (mode, remaining_time)
        active_sequence = []

        t = 0.0
        kickoff_neutral_timer = 0.1  # Brief pause on kickoff
        opponent_goal = [0.0, 5120.0, 100.0]

        for step in range(num_steps):
            # Ball physics with gravity and bounce
            ball_pos[0] += ball_vel[0] * dt
            ball_pos[1] += ball_vel[1] * dt
            ball_pos[2] += ball_vel[2] * dt + 0.5 * GRAVITY * dt * dt
            ball_vel[2] += GRAVITY * dt

            # Ground bounce
            if ball_pos[2] <= BALL_RADIUS:
                ball_pos[2] = BALL_RADIUS
                ball_vel[2] = -ball_vel[2] * RESTITUTION
                ball_vel[0] *= 0.985
                ball_vel[1] *= 0.985

            # Goal detection (into opponent goal or own goal)
            if abs(ball_pos[1]) >= ARENA_EXTENT_Y:
                if abs(ball_pos[0]) <= GOAL_WIDTH_HALF and ball_pos[2] <= GOAL_HEIGHT:
                    # GOAL SCORED! Reset to next kickoff
                    car_pos, car_vel, car_yaw, ball_pos, ball_vel = reset_kickoff()
                    active_sequence.clear()
                    kickoff_neutral_timer = 0.1
                else:
                    ball_pos[1] = math.copysign(ARENA_EXTENT_Y - BALL_RADIUS, ball_pos[1])
                    ball_vel[1] = -ball_vel[1] * RESTITUTION

            # Sidewall bounce
            if abs(ball_pos[0]) >= ARENA_EXTENT_X - BALL_RADIUS:
                ball_pos[0] = math.copysign(ARENA_EXTENT_X - BALL_RADIUS, ball_pos[0])
                ball_vel[0] = -ball_vel[0] * RESTITUTION

            # 2.0s Trajectory prediction
            pred_t = t + 2.0
            pred_ball = list(ball_pos)
            pred_vz = ball_vel[2]
            for _ in range(120):
                p_dt = 2.0 / 120.0
                pred_ball[0] += ball_vel[0] * p_dt
                pred_ball[1] += ball_vel[1] * p_dt
                pred_ball[2] += pred_vz * p_dt + 0.5 * GRAVITY * p_dt * p_dt
                pred_vz += GRAVITY * p_dt
                if pred_ball[2] <= BALL_RADIUS:
                    pred_ball[2] = BALL_RADIUS
                    pred_vz = -pred_vz * RESTITUTION

            # Basis vectors
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

            # --- EXPERT TEACHER DECISION LOGIC ---
            dist_to_ball = math.sqrt(
                (ball_pos[0] - car_pos[0]) ** 2
                + (ball_pos[1] - car_pos[1]) ** 2
                + (ball_pos[2] - car_pos[2]) ** 2
            )
            speed = math.sqrt(car_vel[0] ** 2 + car_vel[1] ** 2 + car_vel[2] ** 2)

            # Goal-oriented targeting:
            # We want to strike the ball TOWARDS the opponent's goal!
            # Calculate unit vector from ball to goal:
            bg_x = opponent_goal[0] - ball_pos[0]
            bg_y = opponent_goal[1] - ball_pos[1]
            bg_dist = math.sqrt(bg_x ** 2 + bg_y ** 2) + 1e-5
            shot_dir_x = bg_x / bg_dist
            shot_dir_y = bg_y / bg_dist

            # Strike offset behind the ball:
            # Car should aim through this point to smash the ball into the goal
            offset_dist = 110.0
            strike_target = [
                ball_pos[0] - shot_dir_x * offset_dist,
                ball_pos[1] - shot_dir_y * offset_dist,
                BALL_RADIUS
            ]

            if dist_to_ball > 1600.0:
                # Far away: lead the predicted ball position
                target = pred_ball
            else:
                # Close: drive through the strike target towards goal
                target = strike_target

            # Calculate steering angle
            dx = target[0] - car_pos[0]
            dy = target[1] - car_pos[1]
            local_x = dx * fwd[0] + dy * fwd[1]
            local_y = dx * rgt[0] + dy * rgt[1]
            angle = math.atan2(local_y, local_x)

            # Anti-circling steering:
            # When aligned and closing in, steer smoothly rather than over-locking
            if dist_to_ball < 500.0 and abs(angle) < 0.4:
                steer = max(-1.0, min(1.0, angle * 2.8))
            else:
                steer = max(-1.0, min(1.0, angle * 4.2))

            # Triggering strike sequences
            if kickoff_neutral_timer > 0.0:
                mode = 0
                steer = 0.0
                kickoff_neutral_timer -= dt
            elif len(active_sequence) > 0:
                # Continue active sequence
                mode, remaining = active_sequence[0]
                steer = 0.0
                remaining -= dt
                if remaining <= 0.0:
                    active_sequence.pop(0)
                else:
                    active_sequence[0] = (mode, remaining)
            elif (
                # STRIKE FLIP TRIGGER:
                # In front of car (local_x > 0), within strike range (<380 UU),
                # roughly facing the ball (abs(angle) < 0.55 rad / ~30 deg)
                dist_to_ball < 380.0
                and local_x > 0
                and abs(angle) < 0.55
                and ball_pos[2] < 220.0
            ):
                # 3-Stage Rocket League Flip Sequence:
                # 1. Mode 2 (Single Jump) to leave ground (0.05s)
                # 2. Mode 0 (Neutral - release jump in air) (0.033s)
                # 3. Mode 3 (Front Dodge Flip into ball) (0.18s)
                active_sequence = [
                    (2, 0.05),
                    (0, 0.033),
                    (3, 0.18)
                ]
                mode = 2
                steer = 0.0
            elif (
                # AERIAL CHALLENGE: ball is high, car is close
                dist_to_ball < 450.0
                and ball_pos[2] > 220.0
                and local_x > 0
            ):
                active_sequence = [
                    (2, 0.20)
                ]
                mode = 2
                steer = 0.0
            elif (
                # KICKOFF SPEED FLIP: speeding straight at the kickoff ball
                dist_to_ball > 1200.0
                and speed > 750.0
                and abs(angle) < 0.12
            ):
                active_sequence = [
                    (2, 0.05),
                    (0, 0.033),
                    (3, 0.18)
                ]
                mode = 2
                steer = 0.0
            else:
                # Chase with continuous steering
                mode = 1

            target_modes.append(mode)
            target_steers.append(steer if mode == 1 else 0.0)

            # Advance car kinematics
            if mode == 1:
                accel = 1200.0
                car_yaw += steer * 2.8 * dt
                car_vel[0] = math.cos(car_yaw) * min(MAX_CAR_SPEED, speed + accel * dt)
                car_vel[1] = math.sin(car_yaw) * min(MAX_CAR_SPEED, speed + accel * dt)
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
                car_pos[2] = 17.0
            elif mode == 3:
                # Front flip velocity impulse forward
                car_vel[0] += math.cos(car_yaw) * 450.0 * dt * 5
                car_vel[1] += math.sin(car_yaw) * 450.0 * dt * 5
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
                car_pos[2] = max(17.0, car_pos[2] - 50.0 * dt)
            elif mode == 2:
                # Jump upward
                car_pos[2] = min(180.0, car_pos[2] + 450.0 * dt)
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
            else:
                # Neutral
                car_vel[0] *= 0.96
                car_vel[1] *= 0.96
                car_pos[0] += car_vel[0] * dt
                car_pos[1] += car_vel[1] * dt
                if car_pos[2] > 17.0:
                    car_pos[2] = max(17.0, car_pos[2] - 300.0 * dt)

            # --- REAL CAR-BALL COLLISION PHYSICS ---
            c_dist = math.sqrt(
                (ball_pos[0] - car_pos[0]) ** 2
                + (ball_pos[1] - car_pos[1]) ** 2
                + (ball_pos[2] - car_pos[2]) ** 2
            )
            if c_dist <= (BALL_RADIUS + CAR_RADIUS):
                # HIT! Transfer car momentum to the ball towards the shot direction
                hit_normal_x = (ball_pos[0] - car_pos[0]) / (c_dist + 1e-5)
                hit_normal_y = (ball_pos[1] - car_pos[1]) / (c_dist + 1e-5)

                if mode == 3:
                    # Powerful front-flip strike!
                    hit_power = 1600.0
                    ball_vel[0] = hit_normal_x * hit_power + fwd[0] * 500.0
                    ball_vel[1] = hit_normal_y * hit_power + fwd[1] * 500.0
                    ball_vel[2] = 450.0
                elif mode == 2:
                    # Aerial / jump hit
                    ball_vel[0] = hit_normal_x * 900.0
                    ball_vel[1] = hit_normal_y * 900.0
                    ball_vel[2] = 650.0
                else:
                    # Ground bumper touch
                    ball_vel[0] = hit_normal_x * 850.0 + car_vel[0] * 0.8
                    ball_vel[1] = hit_normal_y * 850.0 + car_vel[1] * 0.8
                    ball_vel[2] = 200.0

                # Recoil car slightly
                car_vel[0] *= 0.85
                car_vel[1] *= 0.85

            # Arena boundaries for car
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


def train_policy(epochs=25, lr=1.5e-3, seed=42):
    print("=" * 60)
    print("Catalyst 13D GRU Policy: Anti-Circling & Strike Training")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = TrajectoryGenerator(seed=seed)

    print("\n[*] Generating 1v1 match datasets with collision & strike sequences...")
    num_train_matches = 28
    num_val_matches = 6
    match_steps = 1600

    train_data = [generator.generate_match(num_steps=match_steps) for _ in range(num_train_matches)]
    val_data = [generator.generate_match(num_steps=match_steps) for _ in range(num_val_matches)]

    total_train_steps = num_train_matches * match_steps
    total_val_steps = num_val_matches * match_steps
    print(f"    - Train matches: {num_train_matches} ({total_train_steps} timesteps)")
    print(f"    - Val matches:   {num_val_matches} ({total_val_steps} timesteps)")

    device = torch.device("cpu")
    model = Policy().to(device)
    print(f"    - Model parameters: {sum(p.numel() for p in model.parameters())} (Expected: 26,181)")

    # Exact tournament specification: Adam lr=0.001, unweighted CE + Chase MSE with coefficient 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mode_criterion = nn.CrossEntropyLoss()
    steer_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    metrics_history = []
    seq_len = 256

    print(f"\n[*] Starting training loop (epochs={epochs}, chunk_callbacks={seq_len}):")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_mode = 0.0
        train_loss_steer = 0.0
        train_steps_count = 0

        random.shuffle(train_data)

        for obs, modes, steers in train_data:
            T = len(obs)
            h = None

            for start in range(0, T, seq_len):
                end = min(start + seq_len, T)
                x = torch.from_numpy(obs[start:end]).to(device)
                y_mode = torch.from_numpy(modes[start:end]).to(device)
                y_steer = torch.from_numpy(steers[start:end]).to(device)

                optimizer.zero_grad()
                pred_mode, pred_steer, h = model(x, h)
                h = h.detach()

                loss_m = mode_criterion(pred_mode, y_mode)
                chase_mask = y_mode == 1
                loss_s = steer_criterion(pred_steer[chase_mask], y_steer[chase_mask]) if chase_mask.any() else torch.tensor(0.0)

                loss = loss_m + 1.0 * loss_s
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
        total_val_loss = avg_val_loss_mode + 1.0 * avg_val_loss_steer

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

    final_metrics = metrics_history[-1]
    provenance = {
        "dataset_version": "catalyst-strike-anti-circling-v2",
        "split_method": "whole 1v1 matches (28 train matches / 6 val matches, 82/18 split)",
        "seed": seed,
        "architecture": "13D / Linear(13->64) / ReLU / GRU(64->64) / Linear(64->5)",
        "objective": "CrossEntropyLoss(modes) + 1.0 * MSELoss(steering)",
        "checkpoint_selection": f"Lowest total validation loss ({best_val_loss:.4f})",
        "training_command": f"python tools/train.py --epochs {epochs} --lr {lr} --seed {seed}",
        "validation_results": (
            f"Val Mode Accuracy: {final_metrics['val_mode_accuracy']}%, "
            f"Val Mode Loss: {final_metrics['val_mode_loss']}, "
            f"Val Steering MSE: {final_metrics['val_steer_loss']}"
        ),
        "limitations": "Model trained on goal-directed strike imitation and multi-stage flip sequences.",
    }

    provenance_path = ROOT / "docs" / "training_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved training provenance to: {provenance_path}")

    return checkpoint_path, provenance_path, metrics_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Catalyst GRU Policy")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    checkpoint_path, provenance_path, metrics = train_policy(epochs=args.epochs, lr=args.lr, seed=args.seed)
