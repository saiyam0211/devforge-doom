# Spatial-EgoNet

Yugaantar 2026 · DevForge Track 1 · Can It Play DOOM?

Spatial-EgoNet controls a ViZDoom player using the official ego-centric observations. It encodes 64 directional rays with circular convolutions, pools visible entities with masked attention, and fuses these features with vitals and a small RGB branch. A two-layer, 256-unit GRU carries state between decisions. Separate learned heads choose movement, turning, attack, use, and weapon switching. The agent clears its memory at each episode reset.

The submitted checkpoint warm-starts from imitation training and fine-tunes the recurrent controller with custom PPO. Generated navigation, combat, and key-and-lock courses complement original-spawn training WADs. Training rewards track newly visited areas, key pickups, doors becoming passable, kills, and exits, with penalties for damage, deaths, and stalls. The route teachers and engine telemetry used during training are outside the inference path. See the [training pipeline](training/README.md) and [two-page project brief](docs/Spatial-EgoNet-Project-Brief.pdf).

## Run locally

Clone the repo with Git LFS installed. From the repository root, run:

```sh
git lfs pull
uv sync --locked
uv run --no-sync python validate.py
uv run --no-sync python scripts/test_and_rate.py \
  --wad maps/two_rooms.wad --map MAP01 --skill 3 --ticks 2100 --seed 0
```

Add `--window` to watch the episode. The checkpoint is under `submission/weights/`. The repository uses the organizers' runtime for container evaluation.

## Results and limitations

| Local evaluation | Result |
| --- | --- |
| Held-out generated courses at skill 3 | 22/22 exits, including 10 single-key and six two-lock courses |
| Held-out original-spawn Scythe MAP19 and Requiem MAP20 at skill 3 | 0/6 exits and 24 kills across seeds 0, 7, and 42 |
| Local validator on the starter map and a generated two-lock map | Both exits reached |

The main limitation is long-horizon progression on full levels. The agent can retrieve keys and open locks on generated layouts, but can lose the route between objectives or die before reaching a lock on larger maps. Local macOS validation covered the CPU inference interface and episode timing. The organizers' Linux CPU affinity, peak RAM, and RTX 3090 memory limits require an environment that exposes those checks.
