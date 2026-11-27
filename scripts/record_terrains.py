"""Record one first-try episode per validation terrain: centre spawn, constant forward command."""
import argparse, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", required=True)
p.add_argument("--speed", type=float, default=0.8)
AppLauncher.add_app_launcher_args(p)
args, hydra_args = p.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import os
import imageio.v2 as imageio
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw
from rsl_rl.runners import OnPolicyRunner
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config
import fpm  # noqa: F401
from common import terrain_column
from fpm.terrains import LOCOMOTION_TERRAIN_NAMES
from fpm.vecenv import FiniteRslRlVecEnvWrapper


def footholds(u, obs, i, patch=7):
    robot, sensor = u.scene["robot"], u.scene["height_scanner"]
    stride = 2 + patch**2
    scan = obs[:, -2 * stride:]
    pts = []
    for k in range(2):
        f = scan[:, k * stride:(k + 1) * stride]
        pb = torch.cat([f[:, :2], torch.zeros_like(f[:, :1])], -1)
        pw = robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, pb)
        pw[:, 2] = sensor.data.pos_w[:, 2] - 0.5 - f[:, 2 + patch**2 // 2]
        pts.append(pw[i:i + 1])
    return torch.cat(pts)


@hydra_task_config("FPM-G1-Validate-Locomotion-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    env_cfg.log_dir = env_cfg.load_run_dir = run_dir
    env_cfg.seed = agent_cfg.seed = 1
    env_cfg.scene.num_envs = 13
    env_cfg.viewer.resolution = (960, 540)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y = (args.speed, args.speed), (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    cmd.rel_turning_envs = 0.0
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env = FiniteRslRlVecEnvWrapper(gym.make("FPM-G1-Validate-Locomotion-v0", cfg=env_cfg, render_mode="rgb_array"),
                                   clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/PredictedFootholds", markers={
        "f": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)))}))
    os.makedirs(args.out, exist_ok=True)
    env.reset()
    column = terrain_column(env).cpu().tolist()
    print("COLUMNS", column)

    def look(i):
        pos = robot.data.root_pos_w[i].cpu().numpy()
        u.sim.set_camera_view((pos + np.array([-1.8, -2.4, 1.3])).tolist(), pos.tolist())

    everything = imageio.get_writer(os.path.join(args.out, "all_terrains.mp4"), fps=50, macro_block_size=2)
    for t, name in LOCOMOTION_TERRAIN_NAMES.items():
        i = column.index(t)
        with torch.inference_mode():
            env.reset()
        obs = env.get_observations()
        look(i)
        for _ in range(20):
            u.render()
        writer = imageio.get_writer(os.path.join(args.out, f"{t:02d}_{name}.mp4"), fps=50, macro_block_size=2)
        start = robot.data.root_pos_w[i, :2].clone()
        fell, dist, steps = False, 0.0, int(u.max_episode_length) 
        for step in range(steps - 1):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            if dones[i]:
                fell = step < steps - 3
                break
            dist = torch.norm(robot.data.root_pos_w[i, :2] - start).item()
            markers.visualize(translations=footholds(u, obs["policy"], i))
            look(i)
            img = Image.fromarray(u.render()[..., :3])
            ImageDraw.Draw(img).text((16, 12), f"{name}   t={(step + 1) * u.step_dt:4.1f}s   cmd {args.speed} m/s   moved {dist:.1f} m",
                                     fill=(255, 255, 0), font_size=24)
            frame = np.asarray(img)
            writer.append_data(frame)
            everything.append_data(frame)
        if fell:
            d = img.copy()
            ImageDraw.Draw(d).text((16, 48), "FELL", fill=(255, 40, 40), font_size=40)
            for _ in range(60):
                writer.append_data(np.asarray(d)); everything.append_data(np.asarray(d))
        writer.close()
        imageio.imwrite(os.path.join(args.out, f"{t:02d}_{name}.png"), frame)
        print(f"CLIP {name} cmd={u.command_manager.get_command('base_velocity')[i].tolist()} fell={fell} t={(step + 1) * u.step_dt:.1f}s moved={dist:.2f}m", flush=True)
    everything.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
