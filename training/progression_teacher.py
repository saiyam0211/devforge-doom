"""Privileged route labels used only while training on generated courses."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from training.progression_maps import CELL, DIRECTIONS, Cell, Course


class CourseTeacher:
    def __init__(self, course: Course) -> None:
        self.course = course
        self.inspected = course.inspect_cell is None

    def next_cell(self, origin: Cell, goal: Cell) -> Cell:
        queue = deque([(origin, origin)])
        seen = {origin}
        while queue:
            cell, first = queue.popleft()
            if cell == goal:
                return first
            for dx, dy in DIRECTIONS:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in self.course.cells and neighbor not in seen:
                    queue.append((neighbor, neighbor if cell == origin else first))
                    seen.add(neighbor)
        return origin

    def act(self, obs, state, previous_use: bool) -> np.ndarray:
        action = np.zeros(9, dtype=np.int8)
        x, y = state["x"], state["y"]
        cell = (math.floor(x / CELL), math.floor(y / CELL))
        if cell == self.course.inspect_cell:
            self.inspected = True
        missing_key = next((point for point, index in self.course.objectives if obs["vitals"][index] == 0), None)
        if not self.inspected and missing_key is not None:
            goal = self.course.inspect_cell
        else:
            goal = missing_key or self.course.exit_cell
        target = self.next_cell(cell, goal)
        tx, ty = ((target[0] + 0.5) * CELL, (target[1] + 0.5) * CELL)
        exiting = cell == self.course.exit_cell and goal == cell
        if exiting:
            tx = (cell[0]+1)*CELL

        bearing = math.degrees(math.atan2(ty - y, tx - x))
        delta = (bearing - state["angle"] + 180) % 360 - 180
        if abs(delta) > 10:
            action[4 if delta > 0 else 5] = 1
        if abs(delta) < 22:
            action[0] = 1

        rays = obs["rays"]
        # Door actions are only labeled if the planned route goes through that door.
        if not previous_use and abs(delta) < 18:
            front_close = float(rays["distance"][0]) * 1024 < 55
            opening = target != cell and self.course.cells.get(target, 0) and rays["hit_type"][0] in (2, 3)
            if (front_close and opening) or (exiting and tx - x < 55):
                action[7] = 1

        ents = obs["visible_entities"]
        monsters = np.flatnonzero(ents["mask"] & np.isin(ents["entity_type"], (1, 2, 3)))
        if len(monsters):
            distance, sine, cosine, *_ = ents["features"][monsters[0]]
            if distance < 0.40 and cosine > 0.60:
                action[:] = 0
                if abs(sine) > 0.09:
                    action[4 if sine > 0 else 5] = 1
                if abs(sine) < 0.22:
                    action[6] = 1
                if distance < 0.07 and float(rays["distance"][32]) > 0.08:
                    action[1] = 1
                elif distance > 0.20:
                    action[0] = 1
        if obs["vitals"][3] == 0:
            action[8] = 1
        return action
