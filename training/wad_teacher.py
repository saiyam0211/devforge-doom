"""Training-only route labels for Doom-format WADs."""

from __future__ import annotations

from collections import deque
import math
import struct

import numpy as np

from env.doom_env.wad import map_lumps, read_wad

STRIDE = 24
RADIUS = 19
KEY_NAMES = ("BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull")
EXIT_SPECIALS = {11, 51, 52, 124}
OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


class RouteGrid:
    def __init__(self, geometry, keys):
        vertices = np.vstack((geometry.p1, geometry.p2))
        self.origin = np.floor(vertices.min(axis=0) / STRIDE) * STRIDE - STRIDE
        shape = tuple((np.ceil((vertices.max(axis=0) - self.origin) / STRIDE) + 2).astype(int))
        self.free = np.ones(shape, dtype=bool)
        self.free[[0, -1], :] = self.free[:, [0, -1]] = False
        closed = geometry.move_block.copy()
        for lock, held in enumerate((True, keys[2] or keys[5], keys[0] or keys[3], keys[1] or keys[4])):
            if held:
                doors = (geometry.door_lock == lock)
                closed &= ~((geometry.sec_b >= 0) & (doors[geometry.sec_a] | doors[geometry.sec_b]))
        for a, b in zip(geometry.p1[closed], geometry.p2[closed]):
            lo = np.maximum(0, np.floor((np.minimum(a, b) - RADIUS - self.origin) / STRIDE).astype(int))
            hi = np.minimum(shape, np.ceil((np.maximum(a, b) + RADIUS - self.origin) / STRIDE).astype(int) + 1)
            x, y = np.mgrid[lo[0]:hi[0], lo[1]:hi[1]]
            points = np.stack((x, y), axis=-1) * STRIDE + self.origin - a
            edge = b - a
            fraction = np.clip((points @ edge) / (edge @ edge), 0.0, 1.0)
            squared = np.sum((points - fraction[..., None] * edge) ** 2, axis=-1)
            self.free[lo[0]:hi[0], lo[1]:hi[1]] &= squared >= RADIUS**2
        self.routes = {}

    def cells_near(self, point):
        center = np.rint((point - self.origin) / STRIDE).astype(int)
        cells = [
            (x, y) for x in range(max(0, center[0] - 2), min(self.free.shape[0], center[0] + 3))
            for y in range(max(0, center[1] - 2), min(self.free.shape[1], center[1] + 3))
            if self.free[x, y]
        ]
        return sorted(cells, key=lambda c: np.sum((self.origin + np.asarray(c) * STRIDE - point) ** 2))

    def route(self, position, goal):
        starts = self.cells_near(position)
        targets = self.cells_near(goal)
        if not starts or not targets:
            return None
        target = targets[0]
        if target not in self.routes:
            pending = deque([target])
            predecessor = {target: None}
            while pending:
                x, y = pending.popleft()
                for dx, dy in OFFSETS:
                    node = (x + dx, y + dy)
                    if (node not in predecessor and self.free[node] and
                            (not dx or not dy or (self.free[x + dx, y] and self.free[x, y + dy]))):
                        predecessor[node] = (x, y)
                        pending.append(node)
            self.routes[target] = predecessor
        predecessor = self.routes[target]
        current = next((node for node in starts if node in predecessor), None)
        if current is None:
            return None
        route = []
        while current is not None:
            route.append(self.origin + np.asarray(current) * STRIDE)
            current = predecessor[current]
        return route

    def straight(self, start, end):
        steps = max(2, math.ceil(np.linalg.norm(end - start) / (STRIDE / 3)))
        samples = np.linspace(start, end, steps + 1)[1:]
        cells = np.rint((samples - self.origin) / STRIDE).astype(int)
        return self.free[cells[:, 0], cells[:, 1]].all()


class WadTeacher:
    def __init__(self, env, exit_only=False):
        self.env = env
        lumps = map_lumps(read_wad(env.wad), env.map_name)
        vertices = np.array(list(struct.iter_unpack("<hh", lumps["VERTEXES"])), dtype=float)
        self.exits = []
        for ia, ib, _, special, *_ in struct.iter_unpack("<7H", lumps["LINEDEFS"]):
            if special in EXIT_SPECIALS:
                a, b = vertices[ia], vertices[ib]
                midpoint = (a + b) / 2
                dx, dy = b - a
                normal = np.array([dy, -dx]) / max(1, math.hypot(dx, dy))
                use = special in (11, 51)
                self.exits.append((midpoint + normal * 32, midpoint if use else midpoint - normal * 48, use))
        self.grid = None
        self.keys = None
        self.targets = None
        self.step = 0
        self.last_use = -15
        self.fallback = 0
        self.exit_only = exit_only

    def act(self, obs, s):
        self.step += 1
        position = np.array([s["x"], s["y"]])
        keys = tuple(bool(n) for n in s["keys"])
        if self.grid is None or keys != self.keys:
            self.grid = RouteGrid(self.env._geometry, keys)
            self.keys = keys
            available = {
                o.name: np.array([o.position_x, o.position_y])
                for o in self.env._game.get_state().objects if o.name in KEY_NAMES
            }
            self.targets = [(available[name], None, False) for i, name in enumerate(KEY_NAMES)
                            if not self.exit_only and not keys[i] and name in available]
            self.targets += self.exits

        options = []
        for goal, trigger, use in self.targets:
            route = self.grid.route(position, goal)
            if route is not None:
                options.append((trigger is not None, len(route), route, goal, trigger, use))
        action = np.zeros(9, dtype=np.int8)
        if options:
            _, _, route, goal, trigger, use = min(options, key=lambda entry: entry[:2])
            near = np.linalg.norm(goal - position) < 48
            waypoint = route[0]
            for point in route[1:6]:
                if not self.grid.straight(position, point):
                    break
                waypoint = point
            if near:
                waypoint = trigger if trigger is not None else goal
        else:
            # Recover a label for active switch/lift rooms the static grid cannot solve.
            self.fallback += 1
            waypoint = position + np.array([math.cos(math.radians(s["angle"])), math.sin(math.radians(s["angle"]))])
            if obs["rays"]["distance"][0] < 0.12:
                action[4 if self.fallback % 24 < 12 else 5] = 1

        dx, dy = waypoint - position
        delta = (math.degrees(math.atan2(dy, dx)) - s["angle"] + 180) % 360 - 180
        if not action.any():
            if abs(delta) > 11:
                action[4 if delta > 0 else 5] = 1
            if abs(delta) < 26:
                action[0] = 1
        rays = obs["rays"]
        front = float(rays["distance"][0]) * 1024
        use_exit = options and near and use and abs(delta) < 18
        door = rays["hit_type"][0] in (2, 3) and front < 57 and abs(delta) < 24
        if (use_exit or door) and self.step - self.last_use >= 13:
            action[7] = 1
            self.last_use = self.step
        ents = obs["visible_entities"]
        enemies = np.flatnonzero(ents["mask"] & np.isin(ents["entity_type"], (1, 2, 3)))
        if len(enemies):
            distance, sine, cosine, *_ = ents["features"][enemies[0]]
            if distance < 0.31 and cosine > 0.56 and not use_exit:
                action[:7] = 0
                if abs(sine) > 0.12:
                    action[4 if sine > 0 else 5] = 1
                if abs(sine) < 0.23:
                    action[6] = 1
                if distance < 0.09 and rays["distance"][32] > 0.10:
                    action[1] = 1
                elif distance > 0.18:
                    action[0] = 1
        if obs["vitals"][3] == 0:
            action[8] = 1
        return action
