"""Command-tracking success and fall rate on the training and unseen terrains."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="FPM-G1-Validate-Locomotion-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--num_steps", type=int, default=1100)
parser.add_argument("--save_results", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

import torch
from common import make_env_and_policy, terrain_column

from isaaclab_tasks.utils.hydra import hydra_task_config

from fpm.terrains import LOCOMOTION_TERRAIN_CATEGORY, LOCOMOTION_TERRAIN_NAMES, SPARSE_TERRAINS

STANDSTILL_THRESH = 0.1
MIN_FALL_TIME_S = 3.0


def _summary(acc, fell, names):
    a = [x for i in names for x in acc[i]]
    f = [x for i in names for x in fell[i]]
    return (100 * sum(a) / len(a), 100 * sum(f) / len(f), len(a)) if a else (float("nan"), float("nan"), 0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env, policy, run_dir = make_env_and_policy(args_cli.task, env_cfg, agent_cfg, args_cli.checkpoint, args_cli.device)
    u = env.unwrapped
    n, dt, ep_len = u.num_envs, u.step_dt, u.max_episode_length_s

    obs = env.get_observations()
    terrain = terrain_column(env).cpu()
    cmd = u.command_manager.get_command("base_velocity")[:, :2].clone()
    dist = torch.zeros(n, device=u.device)
    time = torch.zeros(n, device=u.device)
    acc = {t: [] for t in LOCOMOTION_TERRAIN_NAMES}
    fell = {t: [] for t in LOCOMOTION_TERRAIN_NAMES}
    early = {t: 0 for t in LOCOMOTION_TERRAIN_NAMES}

    origin = u.scene.env_origins
    spawn = (u.scene["robot"].data.root_pos_w[:, :2] - origin[:, :2]).clone()
    rec = []
    for step in range(args_cli.num_steps):
        vel_b = u.scene["robot"].data.root_lin_vel_b[:, :2]
        cmd_mag = torch.norm(cmd, dim=-1)
        dist += (vel_b * cmd / cmd_mag.unsqueeze(-1).clamp(min=1e-6)).sum(dim=-1) * dt
        time += dt

        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))

        done = dones.bool()
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            t_ep, mag = time[i].item(), cmd_mag[i].item()
            is_fall = t_ep < ep_len - 2 * dt
            if mag >= STANDSTILL_THRESH:
                rec.append((int(terrain[i]), spawn[i, 0].item(), spawn[i, 1].item(), t_ep, bool(is_fall)))
            if mag >= STANDSTILL_THRESH and is_fall and t_ep < MIN_FALL_TIME_S:
                early[int(terrain[i])] += 1
            if mag < STANDSTILL_THRESH or (is_fall and t_ep < MIN_FALL_TIME_S):
                continue
            acc[int(terrain[i])].append(min(1.0, max(0.0, dist[i].item() / (mag * ep_len))))
            fell[int(terrain[i])].append(is_fall)
        spawn[done] = u.scene["robot"].data.root_pos_w[done, :2] - u.scene.env_origins[done, :2]
        dist[done] = 0.0
        time[done] = 0.0
        cmd[done] = u.command_manager.get_command("base_velocity")[done, :2]
        if (step + 1) % 100 == 0:
            print(f"[INFO] step {step + 1}/{args_cli.num_steps}  episodes {sum(len(v) for v in acc.values())}")

    print(f"\n{'terrain':<24}{'cat':<7}{'success %':>10}{'falls %':>9}{'n':>7}")
    for t, name in LOCOMOTION_TERRAIN_NAMES.items():
        s, f, k = _summary(acc, fell, [t])
        print(f"{name:<24}{LOCOMOTION_TERRAIN_CATEGORY[name]:<7}{s:>10.1f}{f:>9.1f}{k:>7}")
    groups = {
        "train (incl. flat)": [t for t, nm in LOCOMOTION_TERRAIN_NAMES.items() if LOCOMOTION_TERRAIN_CATEGORY[nm] != "ood"],
        "ood": [t for t, nm in LOCOMOTION_TERRAIN_NAMES.items() if LOCOMOTION_TERRAIN_CATEGORY[nm] == "ood"],
        "sparse": [t for t, nm in LOCOMOTION_TERRAIN_NAMES.items() if nm in SPARSE_TERRAINS],
        "all": list(LOCOMOTION_TERRAIN_NAMES),
    }
    print("-" * 57)
    for label, ids in groups.items():
        s, f, k = _summary(acc, fell, ids)
        print(f"{'[' + label + ']':<31}{s:>10.1f}{f:>9.1f}{k:>7}")

    print("\nEARLY (falls before 3 s, excluded above): terrain early counted falls%_incl_early")
    for t, name in LOCOMOTION_TERRAIN_NAMES.items():
        k, nf = len(fell[t]), sum(fell[t])
        print(f"EARLY {name:<24}{early[t]:>6}{k:>7}{100 * (nf + early[t]) / max(1, k + early[t]):>9.1f}")
    e, k, nf = sum(early.values()), sum(len(v) for v in fell.values()), sum(sum(v) for v in fell.values())
    print(f"EARLY {'[all]':<24}{e:>6}{k:>7}{100 * (nf + e) / max(1, k + e):>9.1f}")

    torch.save(rec, os.environ["SPAWN_OUT"])
    for t in (1, 7, 0, 11, 6):
        r = [x for x in rec if x[0] == t]
        ef = [x for x in r if x[4] and x[3] < MIN_FALL_TIME_S]
        ts = sorted(x[3] for x in ef)
        print(f"SPAWN {LOCOMOTION_TERRAIN_NAMES[t]}: early {len(ef)}, median fall time {ts[len(ts)//2] if ts else -1:.2f}s, under 1s {sum(v < 1 for v in ts)}")
        for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.25), (1.25, 1.51)):
            b = [x for x in r if lo <= max(abs(x[1]), abs(x[2])) < hi]
            k = sum(x[4] and x[3] < MIN_FALL_TIME_S for x in b)
            print(f"SPAWN    max|dx,dy| {lo}-{hi} m: {k}/{len(b)} early falls ({100 * k / max(1, len(b)):.0f}%)")

    if args_cli.save_results:
        path = os.path.join(run_dir, "locomotion_validation.pt")
        torch.save({"success": acc, "fell": fell, "terrain_names": LOCOMOTION_TERRAIN_NAMES}, path)
        print(f"[INFO] saved {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
