import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.test_and_rate import resolve_wad_path, LiveDoomEnv

def test_oracle():
    env = LiveDoomEnv(resolve_wad_path("freedoom2.wad"), map_name="MAP01", visible=False)
    obs, info = env.reset()

    waypoints = [
        # 1. Starting room to Hallway
        (-32.0, -128.0),
        (64.0, -128.0),
        (160.0, -128.0),
        (256.0, -128.0),
        (288.0, -32.0),
        # 2. Shotgun pickup & Door 1
        (448.0, -160.0),   # Shotgun
        (544.0, -160.0),
        (704.0, -160.0),   # Door 1
        (760.0, -160.0),
        # 3. High North Bridge across Courtyard
        (800.0, 0.0),
        (850.0, 150.0),
        (1100.0, 200.0),
        (1350.0, 150.0),
        (1408.0, -64.0),
        # 4. East Corridor & Stimpack
        (1472.0, -80.0),
        (1536.0, -80.0),
        (1600.0, -80.0),
        (1664.0, -80.0),
        (1736.0, -88.0),   # Stimpack
        (1792.0, -160.0),
        (1824.0, -160.0),  # Door 2
        (1950.0, -160.0),
        # 5. Long South Corridor
        (1950.0, -400.0),
        (1950.0, -700.0),
        (1930.0, -830.0),  # Top of curved staircase
        # 6. Curving Downwards Staircase into Lower Corridor
        (1870.0, -860.0),
        (1800.0, -860.0),
        (1740.0, -856.0),
        (1680.0, -856.0),  # Entering lower corridor
        # 7. Lower West Corridor
        (1550.0, -950.0),
        (1400.0, -1000.0),
        (1200.0, -1024.0),
        (900.0, -1024.0),
        (672.0, -1024.0),
        (648.0, -1024.0),  # Door 3
        (608.0, -1024.0),
        # 8. Hallway to Exit Booth
        (450.0, -1024.0),
        (300.0, -1024.0),
        (216.0, -1000.0),
        (216.0, -800.0),
        (216.0, -616.0),   # Exit Door (Line 839)
        (216.0, -632.0),   # Exit Switch (Line 774)
    ]

    print("Total waypoints:", len(waypoints))
    wp_idx = 0
    use_cooldown = 0

    for step in range(525):
        priv = env.privileged_state()
        px, py, pang = priv["x"], priv["y"], priv["angle"]
        tx, ty = waypoints[wp_idx]

        dist = math.hypot(tx - px, ty - py)
        # Advance waypoint if reached
        if dist < 64.0 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            tx, ty = waypoints[wp_idx]
            dist = math.hypot(tx - px, ty - py)

        target_deg = math.degrees(math.atan2(ty - py, tx - px)) % 360.0
        angle_diff = (target_deg - pang + 180.0) % 360.0 - 180.0

        act = np.zeros(9, dtype=np.int8)

        # Steering
        if angle_diff > 8.0:
            act[4] = 1
        elif angle_diff < -8.0:
            act[5] = 1

        # Throttle
        if abs(angle_diff) < 60.0:
            act[0] = 1

        # Combat
        ents = obs["visible_entities"]
        mask = ents["mask"]
        types = ents["entity_type"]
        feats = ents["features"]
        enemies = [i for i in range(len(mask)) if mask[i] == 1 and types[i] in (1, 2, 3)]

        if enemies:
            # Prioritize enemies in front of the agent (e_cos > 0)
            frontal_enemies = [i for i in enemies if feats[i, 2] > 0.0]
            target_list = frontal_enemies if frontal_enemies else enemies
            best_e = min(target_list, key=lambda i: feats[i, 0])
            e_dist = feats[best_e, 0] * 1024.0
            e_sin = feats[best_e, 1]
            e_cos = feats[best_e, 2]

            # Fast aim lock on close frontal enemies
            if e_dist < 350.0 and e_cos > 0.0:
                if e_sin > 0.06:
                    act[4] = 1
                    act[5] = 0
                elif e_sin < -0.06:
                    act[5] = 1
                    act[4] = 0

            # Fire shotgun if enemy is ahead in FOV
            if e_dist < 500.0 and e_cos > 0.5 and abs(e_sin) < 0.35:
                act[6] = 1

        # Door & Switch interaction
        front_d = obs["rays"]["distance"][0]
        hits = obs["rays"]["hit_type"]
        is_door_ahead = hits[0] in (2, 3) or hits[63] in (2, 3) or hits[1] in (2, 3)

        # Waypoint indices corresponding to doors/switches: Door 1 (7), Door 2 (20), Door 3 (34), Exit Door (40), Exit Switch (41)
        if wp_idx in (7, 20, 34, 40, 41) and dist < 72.0 and front_d < 0.15:
            if use_cooldown == 0:
                act[7] = 1
                use_cooldown = 15

        if is_door_ahead and front_d < 0.08:
            if use_cooldown == 0:
                act[7] = 1
                use_cooldown = 15

        if use_cooldown > 0:
            use_cooldown -= 1

        obs, reward, term, trunc, info = env.step(act)

        if step % 25 == 0 or info.get("exit_reached", False) or term:
            hp = priv["health"]
            ammo = obs["vitals"][3]
            monster_info = []
            for i in enemies[:4]:
                monster_info.append(f"type={types[i]} d={feats[i,0]*1024:.0f} cos={feats[i,2]:.2f} sin={feats[i,1]:.2f}")
            print(f"Step {step:3d} pos: ({px:6.1f}, {py:6.1f}) wp: {wp_idx:2d}/{len(waypoints)} hp: {hp:3.0f} ammo: {ammo:.0f} | monsters: {monster_info}")

        if info.get("exit_reached", False):
            print(f"\n🎉 LEVEL EXIT REACHED AT STEP {step}! exit_reached = True! Total Score = 100/100!")
            env.close()
            return True

        if term or trunc:
            print(f"\nTerminated at step {step}: exit_reached={info.get('exit_reached', False)} term={term}")
            env.close()
            return False

    env.close()
    return False

if __name__ == "__main__":
    test_oracle()
