"""Original Doom-format skill courses for training and held-out evaluation."""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from pathlib import Path

from env.doom_env.wad import Lump, write_wad

CELL = 96
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
KEYS = ((5, 26, 8), (6, 27, 9), (13, 28, 10))
Cell = tuple[int, int]


@dataclass
class Course:
    path: Path
    phase: str
    cells: dict[Cell, int]
    objectives: list[tuple[Cell, int]]
    exit_cell: Cell
    lock_count: int
    inspect_cell: Cell | None


def make_course(directory: Path, seed: int, phase: str, *, long_route: bool = False) -> Course:
    """Build a winding corridor with randomized turns, starts, and side branches."""
    rng = random.Random(seed)
    length = rng.randrange(12, 15) if phase == "gauntlet" else rng.randrange(9, 13)
    bend = rng.randrange(4, 6) if phase == "gauntlet" else rng.randrange(4, 7)
    shift = rng.choice((-2, -1, 1, 2))
    spine = [(x, 0) for x in range(bend + 1)]
    sign = 1 if shift > 0 else -1
    spine += [(bend, y) for y in range(sign, shift + sign, sign)]
    lane = shift
    if long_route:
        length += rng.randrange(7, 12)
        second_bend = bend + rng.randrange(4, 7)
        lane += sign * rng.randrange(2, 5)
        spine += [(x, shift) for x in range(bend + 1, second_bend + 1)]
        spine += [(second_bend, y) for y in range(shift + sign, lane + sign, sign)]
        spine += [(x, lane) for x in range(second_bend + 1, length)]
    else:
        spine += [(x, shift) for x in range(bend + 1, length)]
    cells = dict.fromkeys(spine, 0)
    direction = rng.choice((-1, 1))
    start = spine[rng.randrange(2)]
    exit_cell = spine[-1]
    lock = (bend + 1, shift) if phase == "gauntlet" else (length - 3, lane)
    objectives: list[tuple[Cell, int]] = []
    things: list[tuple[int, int, int, int, int]] = []

    def place(cell: Cell, kind: int, heading: int = 0) -> None:
        x, y = cell
        things.append((x * CELL + CELL // 2, y * CELL + CELL // 2, heading, kind, 7))

    def branch(x: int, y: int, sign: int) -> Cell:
        end = y + sign * rng.randrange(2, 4)
        for row in range(y + sign, end + sign, sign):
            cells[x, row] = 0
        return x, end

    place(start, 1, rng.choice((0, 45, 90, 180, 270)))
    branch_end = branch(rng.randrange(2, 4), 0, direction)
    branch(1, 0, -direction)
    if long_route:
        branch(second_bend - 1, shift, -sign)
        branch(length - 5, lane, -sign)
    if phase in ("key", "gauntlet"):
        key_kind, special, vitals_index = rng.choice(KEYS)
        place(branch_end, key_kind)
        objectives.append((branch_end, vitals_index))
        cells[lock] = special
        if phase == "gauntlet":
            second = branch(length - 4, lane, sign)
            kind2, special2, index2 = rng.choice([key for key in KEYS if key[0] != key_kind])
            place(second, kind2)
            objectives.append((second, index2))
            cells[length - 2, lane] = special2
    elif phase == "navigation":
        cells[lock] = 1
    if phase in ("combat", "gauntlet"):
        place((bend, shift), rng.choice((3004, 3001)), 180)
        place((length - 4, lane), 3004, 180)
        place((bend - 1, 0), 2001)
        place(branch_end if phase == "combat" else (length - 2, lane), 2011)

    path = directory / f"{'long-' if long_route else ''}{phase}-{seed}.wad"
    write_course(path, cells, things, exit_cell)
    inspect_cell = spine[spine.index(lock) - 1] if phase == "key" and seed % 2 else None
    return Course(path, phase, cells, objectives, exit_cell, len(objectives), inspect_cell)


def write_course(path: Path, cells: dict[Cell, int], things, exit_cell: Cell) -> None:
    vertices: dict[Cell, int] = {}
    sides, lines, sectors = [], [], []
    edges = {}
    for sector, ((x, y), special) in enumerate(cells.items()):
        left, bottom = x * CELL, y * CELL
        right, top = left + CELL, bottom + CELL
        corners = ((left, bottom), (left, top), (right, top), (right, bottom))
        sectors.append(struct.pack("<hh8s8shhh", 0, 0 if special else 128, b"FLOOR0_1", b"CEIL1_1", 160, 0, 0))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            ia = vertices.setdefault(a, len(vertices))
            ib = vertices.setdefault(b, len(vertices))
            previous = edges.pop((ib, ia), None)
            if previous is not None:
                side, neighbor, neighbor_special = previous
                # A manual door moves the sector on the back side of the trigger.
                if special and not neighbor_special:
                    front, back, door = side, len(sides), special
                    start, end = ib, ia
                else:
                    front, back, door = len(sides), side, neighbor_special
                    start, end = ia, ib
                sides.append((sector, b"-"))
                sides[side] = (neighbor, b"-")
                lines.append((start, end, 4, door, 0, front, back))
            else:
                exit_line = (x, y) == exit_cell and a[0] == right and b[0] == right
                edges[ia, ib] = (len(sides), sector, special)
                sides.append((sector, b"SW1BRCOM" if exit_line else b"STARTAN3"))
    for (a, b), (side, _, _) in edges.items():
        special = 11 if sides[side][1] == b"SW1BRCOM" else 0
        lines.append((a, b, 1, special, 0, side, 65535))

    # ViZDoom builds nodes for these small maps from the original geometry.
    lumps = [
        Lump("MAP01", b""),
        Lump("THINGS", b"".join(struct.pack("<hhhhh", *t) for t in things)),
        Lump("LINEDEFS", b"".join(struct.pack("<7H", *line) for line in lines)),
        Lump("SIDEDEFS", b"".join(struct.pack("<hh8s8s8sH", 0, 0, b"BIGDOOR2", b"-", tex, sec) for sec, tex in sides)),
        Lump("VERTEXES", b"".join(struct.pack("<hh", *v) for v in vertices)),
        Lump("SEGS", b""), Lump("SSECTORS", b""), Lump("NODES", b""),
        Lump("SECTORS", b"".join(sectors)), Lump("REJECT", b""), Lump("BLOCKMAP", b""),
    ]
    write_wad(path, lumps)
