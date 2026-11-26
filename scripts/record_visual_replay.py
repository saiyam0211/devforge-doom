#!/usr/bin/env python
"""Visual Telemetry Replay Generator for DevForge Track 1 DoomAgent.

Records agent gameplay frames and generates:
1. An annotated animated GIF with a side-by-side 360-degree spatial radar HUD.
2. A structured JSON replay package for the interactive generative UI dashboard.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import time
from pathlib import Path

# Add site-packages if PIL not directly in venv
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from env.doom_env import make_env
from submission.agent import DoomAgent


HIT_TYPE_COLORS = {
    0: (60, 60, 70),       # None / open space
    1: (180, 180, 190),   # Solid Wall
    2: (40, 180, 255),    # Door (Cyan)
    3: (255, 200, 30),     # Locked Door (Gold)
    4: (255, 60, 60),      # Enemy (Red)
    5: (60, 240, 100),     # Item / Pickup (Green)
}

ENTITY_TYPE_COLORS = {
    1: (255, 80, 80),      # Melee Enemy
    2: (255, 120, 40),     # Hitscan Enemy
    3: (255, 40, 160),     # Projectile Enemy
    4: (255, 220, 0),      # Keycard / Skull
    5: (80, 255, 120),     # Health / Armor / Ammo
}


def render_radar(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    radius: int,
    ray_dist: np.ndarray,
    ray_types: np.ndarray,
    entities_feat: np.ndarray,
    entities_type: np.ndarray,
    entities_mask: np.ndarray,
) -> None:
    """Draws a 360-degree ego-centric circular radar."""
    # Outer radar boundary
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        outline=(60, 80, 100),
        width=2,
    )
    # Range rings
    for r in [radius * 0.33, radius * 0.66]:
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], outline=(40, 50, 60), width=1)

    # Crosshair axes
    draw.line([center_x - radius, center_y, center_x + radius, center_y], fill=(40, 50, 60), width=1)
    draw.line([center_x, center_y - radius, center_x, center_y + radius], fill=(40, 50, 60), width=1)

    # Draw 64 rays
    num_rays = len(ray_dist)
    for i in range(num_rays):
        angle = (2.0 * math.pi * i) / num_rays - (math.pi / 2.0)  # 0 is Forward (Up)
        dist = float(ray_dist[i])
        ht = int(ray_types[i])
        ray_len = min(dist, 1.0) * radius
        end_x = center_x + ray_len * math.cos(angle)
        end_y = center_y + ray_len * math.sin(angle)

        color = HIT_TYPE_COLORS.get(ht, (150, 150, 150))
        draw.line([center_x, center_y, end_x, end_y], fill=color, width=1)
        if ht in (2, 3, 4, 5):  # Highlight special hitpoints
            draw.ellipse([end_x - 2, end_y - 2, end_x + 2, end_y + 2], fill=color)

    # Draw visible entities
    if entities_mask is not None:
        for idx in range(len(entities_mask)):
            if entities_mask[idx] == 1:
                e_dist = float(entities_feat[idx, 0])
                sin_a = float(entities_feat[idx, 1])
                cos_a = float(entities_feat[idx, 2])
                e_type = int(entities_type[idx])
                e_r = min(e_dist, 1.0) * radius
                # In ego coords: forward is cos_a, right is sin_a
                ex = center_x + e_r * sin_a
                ey = center_y - e_r * cos_a
                e_col = ENTITY_TYPE_COLORS.get(e_type, (255, 255, 255))
                draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=e_col, outline=(255, 255, 255))

    # Player position marker (triangle pointing UP)
    draw.polygon(
        [
            (center_x, center_y - 5),
            (center_x - 4, center_y + 4),
            (center_x + 4, center_y + 4),
        ],
        fill=(0, 255, 200),
    )


def create_composite_frame(
    rgb_frame: np.ndarray,
    obs: dict,
    action: np.ndarray,
    step_num: int,
    latency_ms: float,
    canvas_w: int = 580,
    canvas_h: int = 280,
) -> Image.Image:
    """Builds a frame combining game viewport with live spatial telemetry HUD."""
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # 1. DOOM RGB viewport (160x120 -> 320x240)
    game_img = Image.fromarray(rgb_frame).resize((320, 240), resample=Image.Resampling.NEAREST)
    canvas.paste(game_img, (12, 20))
    draw.rectangle([11, 19, 332, 260], outline=(70, 80, 95), width=1)

    # Viewport header
    draw.text((14, 6), f"DOOM POV · STEP {step_num:03d} · {latency_ms:.1f}ms", fill=(160, 180, 200))

    # 2. Ego-Radar Canvas
    radar_cx = 455
    radar_cy = 90
    radar_rad = 65
    render_radar(
        draw=draw,
        center_x=radar_cx,
        center_y=radar_cy,
        radius=radar_rad,
        ray_dist=obs["rays"]["distance"],
        ray_types=obs["rays"]["hit_type"],
        entities_feat=obs["visible_entities"]["features"],
        entities_type=obs["visible_entities"]["entity_type"],
        entities_mask=obs["visible_entities"]["mask"],
    )
    draw.text((radar_cx - 50, 10), "360° EGO-RADAR HUD", fill=(140, 200, 240))

    # 3. Telemetry Readouts
    vitals = obs["vitals"]
    hp = max(0, int(vitals[0]))
    armor = max(0, int(vitals[1]))
    keys = int(np.sum(vitals[8:14]))
    vel_fwd = float(vitals[14])

    hud_y = 175
    draw.text((345, hud_y), f"HEALTH: {hp:3d}%", fill=(80, 240, 120) if hp > 30 else (255, 80, 80))
    draw.text((465, hud_y), f"ARMOR: {armor:3d}%", fill=(80, 180, 240))

    draw.text((345, hud_y + 18), f"KEYS: {keys}", fill=(255, 210, 40))
    draw.text((465, hud_y + 18), f"VEL: {vel_fwd:+.1f}", fill=(200, 200, 210))

    # 4. Action Indicators
    act_y = hud_y + 42
    draw.text((345, act_y), "ACTIONS:", fill=(160, 170, 180))

    # Actions: [0:Fwd, 1:Back, 2:Left, 3:Right, 4:TurnL, 5:TurnR, 6:Attack, 7:Use]
    badges = []
    if action[0]:
        badges.append(("FWD", (60, 220, 100)))
    elif action[1]:
        badges.append(("BACK", (220, 140, 60)))
    if action[2]:
        badges.append(("STR-L", (100, 180, 255)))
    elif action[3]:
        badges.append(("STR-R", (100, 180, 255)))
    if action[4]:
        badges.append(("TRN-L", (180, 140, 255)))
    elif action[5]:
        badges.append(("TRN-R", (180, 140, 255)))
    if action[6]:
        badges.append(("FIRE", (255, 60, 60)))
    if action[7]:
        badges.append(("USE", (255, 200, 40)))

    badge_x = 345
    badge_y = act_y + 18
    for label, col in badges[:4]:
        draw.rectangle([badge_x, badge_y, badge_x + 48, badge_y + 16], fill=col)
        draw.text((badge_x + 6, badge_y + 2), label, fill=(10, 10, 15))
        badge_x += 54

    return canvas


def record_replay(
    wad_name: str = "maps/two_rooms.wad",
    map_name: str = "MAP01",
    max_steps: int = 200,
    step_skip: int = 2,
    seed: int | None = 0,
    output_gif: Path | None = None,
    output_json: Path | None = None,
) -> None:
    """Runs one episode, saves an animated GIF and JSON telemetry data."""
    print(f"\n[Recording Replay] Loading {wad_name}:{map_name} (Seed: {seed}) ...")
    env = make_env(wad_name, map_name, episode_ticks=2100)
    agent = DoomAgent(env.observation_space, env.action_space, device="cpu")

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    agent.reset()

    frames: list[Image.Image] = []
    replay_data: list[dict] = []

    print(f"[Recording Replay] Agent playing {max_steps} steps ...")
    for step in range(max_steps):
        t0 = time.perf_counter()
        action = agent.act(obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if step % step_skip == 0:
            frame = create_composite_frame(
                rgb_frame=obs["rgb"],
                obs=obs,
                action=action,
                step_num=step,
                latency_ms=dt_ms,
            )
            frames.append(frame)

            # Downscaled thumbnail for JSON dashboard
            thumb = Image.fromarray(obs["rgb"]).resize((80, 60), resample=Image.Resampling.NEAREST)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=65)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            replay_data.append({
                "step": step,
                "latency_ms": round(dt_ms, 2),
                "hp": int(obs["vitals"][0]),
                "armor": int(obs["vitals"][1]),
                "keys": int(np.sum(obs["vitals"][8:14])),
                "vel_fwd": round(float(obs["vitals"][14]), 2),
                "action": [int(x) for x in action],
                "rays_dist": [round(float(d), 2) for d in obs["rays"]["distance"][::4]],  # downsample to 16 for JSON
                "rays_type": [int(t) for t in obs["rays"]["hit_type"][::4]],
                "image": f"data:image/jpeg;base64,{b64_img}",
            })

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"[Recording Replay] Episode terminated at step {step}.")
            break

    env.close()

    if output_gif and frames:
        print(f"[Recording Replay] Saving animated GIF ({len(frames)} frames) to {output_gif} ...")
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame = 10 FPS
            loop=0,
            optimize=True,
        )
        print(f"[Recording Replay] GIF saved successfully ({output_gif.stat().st_size / 1024:.1f} KB).")

    if output_json and replay_data:
        print(f"[Recording Replay] Saving JSON telemetry ({len(replay_data)} frames) to {output_json} ...")
        output_json.write_text(json.dumps(replay_data, indent=2))
        print(f"[Recording Replay] JSON saved successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record Doom Visual Telemetry Replay")
    parser.add_argument("--wad", type=str, default="maps/two_rooms.wad", help="WAD file")
    parser.add_argument("--map", type=str, default="MAP01", help="Map name")
    parser.add_argument("--steps", type=int, default=200, help="Max steps")
    parser.add_argument("--skip", type=int, default=2, help="Frame step skip")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    record_replay(
        wad_name=args.wad,
        map_name=args.map,
        max_steps=args.steps,
        step_skip=args.skip,
        seed=args.seed,
        output_gif=Path(args.gif) if args.gif else None,
        output_json=Path(args.json) if args.json else None,
    )
