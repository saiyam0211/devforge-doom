# Training

Put your training pipeline here: environment setup, reward shaping, curriculum, model code, logs.
Judges review this folder during manual review.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.

## Expanded behavioral cloning

From the repository root, fine-tune an existing checkpoint into a separate output:

```sh
python training/train_hybrid_bc_ppo.py --expanded --device cpu --threads 4 --epochs 8 --seed 73 \
  --warm-start submission/weights/agent_weights.pt --output /tmp/devforge-candidate.pt
```

The expanded curriculum includes the starter map, 19 Freedoom layouts, and the red-key
fixture. It reflects sequences left-to-right during training and samples key and
locked-door windows more often. The trainer reports accepted demonstrations, exits, and
key pickups by map. Its checkpoint records the configured curriculum and seed.

Use Freedoom MAP19, MAP23, MAP26, MAP29, and MAP30 as the frozen local holdout. Evaluate
with the neural actions, seeds 0, 7, and 42, and a 4,200-tic cap at skill 3 and
Ultra-Violence. Compare exits, key pickups, deaths, and timeouts against the baseline.

## Progression curriculum

The progression trainer generates original ViZDoom courses for navigation, combat,
key retrieval with backtracking, and a two-lock gauntlet. It first clones a route
teacher, then labels mistakes on mixed teacher/policy rollouts, then fine-tunes the
network with PPO. PPO rewards first visits to areas, key pickups, actual lock
openings, and exits. The teacher and privileged engine state are used in training.

```sh
uv run --no-sync python training/train_progression.py \
  --warm-start /absolute/path/to/agent_weights.pt --work-dir /tmp/doom-progression

uv run --no-sync python training/train_progression.py --extend-freedoom \
  --warm-start /tmp/doom-progression/ppo.pt --work-dir /tmp/doom-progression

uv run --no-sync python training/train_progression.py --extend-freedoom --freeze-encoder --threads 2 \
  --warm-start /tmp/doom-progression/corrections.pt --work-dir /tmp/doom-progression

uv run --no-sync python scripts/evaluate_progression.py \
  --checkpoint /tmp/doom-progression/corrections.pt \
  --checkpoint /tmp/doom-progression/ppo.pt \
  --checkpoint /tmp/doom-progression/ppo_freedoom.pt \
  --checkpoint /tmp/doom-progression/ppo_frozen.pt \
  --training-courses /tmp/doom-progression --output /tmp/doom-progression/short.csv

uv run --no-sync python scripts/evaluate_progression.py --full \
  --checkpoint /tmp/doom-progression/ppo.pt \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD" \
  --output /tmp/doom-progression/public.csv
```

The short-course promotion gate is at least eight successful key, lock, and exit
chains on ten held-out single-key courses at skill 3. The public suite measures
transfer on prior benchmarks and four additional public layouts. It also runs
Scythe MAP28 as a timed stress test.

## Full-WAD exit training

This pass adds a lightweight RGB encoder to the spatial model so the policy can
learn from textures and switch markings. A training-only planner proposes real WAD
routes. The collector checks whether ViZDoom reaches the exit and keeps key pickups
as partial progression. Most exit demonstrations start from relocated positions
along the final route. Freedoom MAP01 and the starter map provide original-start
demonstrations, and procedural courses rehearse keys, locks, and combat.
The planner's grid does not model remote switches or timed lifts, so the
collector may fail to find original-start routes through those maps.

```sh
uv run --no-sync python training/train_wad_exits.py \
  --warm-start /absolute/path/to/imitation.pt --work-dir /tmp/doom-wad-exits \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"

uv run --no-sync python scripts/evaluate_progression.py --fresh-full \
  --checkpoint /tmp/doom-wad-exits/visual_corrected.pt --output /tmp/doom-wad-exits/fresh.csv \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"
```

Use `scripts/test_and_rate.py --checkpoint /absolute/path/to/visual_corrected.pt`
with the usual `--wad`, `--map`, and `--window` flags to watch a checkpoint directly.

## Long-route rehearsal

The long-route pass uses 48-decision training windows. It mixes longer
backtracking courses with shorter skill and WAD rehearsal. Padding on short
rehearsal clips is masked out of the loss.

```sh
uv run --no-sync python training/train_long_routes.py \
  --warm-start /absolute/path/to/visual_routes.pt --work-dir /tmp/doom-long-routes \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"

uv run --no-sync python scripts/evaluate_progression.py --long-courses \
  --checkpoint /tmp/doom-long-routes/long_imitation.pt \
  --checkpoint /tmp/doom-long-routes/long_corrected.pt --training-courses /tmp/doom-long-routes \
  --output /tmp/doom-long-routes/heldout-long.csv

uv run --no-sync python scripts/evaluate_progression.py --long-full \
  --checkpoint /tmp/doom-long-routes/long_corrected.pt --output /tmp/doom-long-routes/full.csv \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"
```

The local scorecard in `test_and_rate.py` uses measured outcomes, keys,
door sectors that became passable, visited grid cells, and timing. Windowed
play uses the same 160×120 agent render settings as the headless benchmark.

## Recurrent PPO

Recurrent PPO runs six ViZDoom episodes in parallel. It unrolls recurrent memory
across contiguous decision sequences and clears memory at episode boundaries.
The vision and spatial encoders start from the imitation checkpoint and stay
fixed. A KL penalty anchors the learned policy while PPO trains the recurrent
controller and action heads. The critic uses its own gradients so value losses
cannot overwrite the shared controller early in training.

The reward pays once for a new area and actual door openings, and rewards kills,
keys, and exits. It penalizes damage, death, time, and staying within a small
area for 24 decisions. Absolute position and course routes provide training
rewards. The submitted agent receives the official observation stream.

```sh
uv run --no-sync python training/train_recurrent_rl.py --updates 96 \
  --warm-start /absolute/path/to/long_imitation.pt --work-dir /tmp/doom-rl \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"

uv run --no-sync python training/train_recurrent_rl.py --updates 72 \
  --warm-start /tmp/doom-rl/rl-096.pt --work-dir /tmp/doom-rl/phase-two \
  --exit-practice /tmp/doom-rl/exit-practice \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"

uv run --no-sync python scripts/evaluate_progression.py --rl-gate \
  --checkpoint /tmp/doom-rl/phase-two/rl-072.pt --training-courses /tmp/doom-rl \
  --output /tmp/doom-rl/gate.csv

uv run --no-sync python scripts/evaluate_progression.py --rl-public \
  --checkpoint /tmp/doom-rl/phase-two/rl-072.pt --output /tmp/doom-rl/public.csv \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"

uv run --no-sync python scripts/evaluate_progression.py --rl-freedoom \
  --checkpoint /tmp/doom-rl/phase-two/rl-072.pt --output /tmp/doom-rl/freedoom.csv
```

In the second phase, the original-spawn public WAD lane uses a 4,200-tic cap.
The other public-WAD lane practices exit approaches from teacher-verified
relocated starts on training maps. The public assessment uses original spawns.

### Full-map PPO experiment

Four lanes play original-spawn Freedoom, Scythe and Requiem maps for up to 4,200 tics. Two lanes rehearse key retrieval and two-lock combat routes. The sampled controller draws buttons from the learned distributions used in PPO. Reset restores a private random generator for reproducible episodes.

The second phase backpropagates through 96 decisions at a time. Its training-only reward estimates distance to reachable keys and exit triggers on a coarse grid and pays for reductions in that distance. This estimate can miss switches, lifts, and teleporters. The deployed network consumes only the official observations.

```sh
uv run --no-sync python training/train_recurrent_rl.py --full-map-focus --no-route-reward \
  --updates 60 --threads 3 --warm-start /absolute/path/to/track1-recurrent-rl.pt \
  --work-dir /tmp/doom-focus --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"
uv run --no-sync python training/train_recurrent_rl.py --full-map-focus --route-reward --keep-critic \
  --sequence-length 96 --updates 36 --threads 3 --warm-start /tmp/doom-focus/rl-060.pt \
  --work-dir /tmp/doom-focus/continuation --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD"
uv run --no-sync python scripts/evaluate_progression.py --full-map-audit \
  --checkpoint submission/weights/agent_weights.pt --checkpoint /tmp/doom-focus/continuation/rl-036.pt \
  --output /tmp/doom-focus/known-and-unseen.csv
```

For the route-imitation comparison after PPO, run the following command. It trains the encoders and recurrent controller on complete Freedoom MAP01 runs with expert corrections, alongside generated key and two-lock courses.

```sh
uv run --no-sync python training/refine_full_routes.py \
  --checkpoint /tmp/doom-focus/continuation/rl-036.pt --output /tmp/doom-focus/refined
```

### Route handoffs, recovery, and door memory

The handoff probe replays a verified teacher prefix while warming the agent's
recurrent state on the same observations. It measures what the agent achieves
after control transfers. A reference can verify a key pickup without verifying
an exit. Route coverage counts reference grid cells that the policy approaches
within 48 map units after handoff.

```sh
uv run --no-sync python scripts/route_handoff.py \
  --checkpoint submission/weights/agent_weights.pt --output /tmp/handoff.csv \
  --map freedoom2.wad:MAP01 --map "$HOME/Downloads/SCYTHE.WAD:MAP02"

uv run --no-sync python training/train_route_recovery.py \
  --checkpoint submission/weights/agent_weights.pt --scythe "$HOME/Downloads/SCYTHE.WAD" \
  --output /tmp/doom-recovery

uv run --no-sync python training/train_door_memory.py \
  --checkpoint /tmp/doom-recovery/recovery.pt --output /tmp/doom-memory

uv run --no-sync python scripts/evaluate_progression.py --route-gate \
  --checkpoint /tmp/doom-recovery/recovery.pt --checkpoint /tmp/doom-memory/memory.pt \
  --scythe "$HOME/Downloads/SCYTHE.WAD" --requiem "$HOME/Downloads/REQUIEM.WAD" \
  --output /tmp/route-gate.csv
```

Recovery examples use brief blocks of policy control followed by the teacher.
Training retains episodes where the mixed controller reaches a key or exit.
Door-memory training uses teacher trajectories with a locked door visible
before the key pickup. It trains a bearing prediction while the door is out of
view and anchors the recurrent activations to the recovery checkpoint. The
auxiliary head and training labels are excluded from the deployed checkpoint.
