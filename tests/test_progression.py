import numpy as np
import pytest
import torch

from env.doom_env.wrapper import DoomEnv
from training.progression_maps import make_course
from training.progression_teacher import CourseTeacher
from training.rewards import ProgressReward
from scripts.test_and_rate import local_scorecard
from training.train_hybrid_bc_ppo import DoomTrajectoryDataset
from training.train_recurrent_rl import Progress, recurrent_features


def test_local_scorecard_uses_measured_progress_and_reports_total():
    result = dict(exit_reached=False, ticks=4200, keys_collected=1,
                  doors_opened=1, visited_cells=9, timing_pass=True)
    points, total, grade = local_scorecard(result, 4200)
    assert points == pytest.approx((15, 15, 7.2, 20))
    assert total == pytest.approx(57.2)
    assert grade == "B+"


def test_short_rehearsal_window_masks_padding():
    steps = 12
    sequence = dict(
        ray_dist=np.zeros((steps, 64)), ray_hit=np.zeros((steps, 64)),
        ent_feat=np.zeros((steps, 32, 5)), ent_type=np.zeros((steps, 32)),
        ent_mask=np.zeros((steps, 32)), vitals=np.zeros((steps, 17)),
        actions=np.zeros((steps, 9)), rgb=np.zeros((steps, 30, 40, 3), np.uint8),
    )
    dataset = DoomTrajectoryDataset([sequence], seq_len=48, include_short=True, diversify=True)
    assert all(start >= 0 for _, start in dataset.indices)
    item = dataset[0]
    assert item["rgb"].shape == (48, 30, 40, 3)
    assert item["valid"].tolist() == [True] * steps + [False] * 36


def test_procedural_gauntlet_requires_two_keys_and_locks(tmp_path):
    course = make_course(tmp_path, 1301, "gauntlet")
    env = DoomEnv(course.path, episode_ticks=1000)
    try:
        obs, _ = env.reset(seed=0)
        progress = ProgressReward()
        progress.reset(env.privileged_state())
        teacher = CourseTeacher(course)
        cooldown = 0
        for _ in range(250):
            action = teacher.act(obs, env.privileged_state(), cooldown > 0)
            cooldown = 12 if action[7] else max(0, cooldown - 1)
            obs, _, ended, timed_out, info = env.step(action)
            state = env.privileged_state()
            progress.step(state)
            if ended or timed_out:
                break
        assert info["exit_reached"]
        assert int(state["keys"].sum()) == len(progress.opened) == 2
        assert state["kills"] >= 1
    finally:
        env.close()


def test_progress_reward_pays_each_cell_and_lock_once():
    state = dict(x=0, y=0, keys=np.zeros(6), door_lock=np.array([1]),
                 sector_ceiling=np.array([0]), sector_floor=np.array([0]), kills=0,
                 health=100, exit_reached=False, dead=False)
    reward = ProgressReward()
    reward.reset(state)
    assert reward.step(state) < 0
    new = dict(state, x=128, keys=np.array([1, 0, 0, 0, 0, 0]), sector_ceiling=np.array([128]))
    assert reward.step(new) > 4
    assert reward.step(dict(state, keys=new["keys"])) < 0
    assert reward.step(new) < 0


def test_recurrent_chunks_match_stepwise_resets():
    rng = torch.Generator().manual_seed(341)
    gru = torch.nn.GRU(5, 7, num_layers=2, batch_first=True)
    features = torch.randn(3, 8, 5, generator=rng)
    initial = torch.randn(2, 3, 7, generator=rng)
    resets = torch.zeros(3, 8, dtype=torch.bool)
    resets[0, 0] = resets[1, 3] = resets[2, 5] = True
    memory = initial
    expected = []
    for step in range(features.shape[1]):
        memory = memory * (~resets[:, step])[None, :, None]
        piece, memory = gru(features[:, step:step+1], memory)
        expected.append(piece)
    torch.testing.assert_close(recurrent_features(gru, features, initial, resets), torch.cat(expected, 1))


def test_rl_penalizes_long_stationary_spans_and_does_not_pay_for_reopening():
    start = dict(x=0, y=0, keys=np.zeros(6), door_lock=np.array([0, 1]), sector_ceiling=np.zeros(2),
                 sector_floor=np.zeros(2), kills=0, health=100, exit_reached=False, dead=False)
    reward = Progress(start, None)
    for _ in range(23):
        _, stalled = reward.step(start)
        assert not stalled
    value, stalled = reward.step(start)
    assert stalled and value < -0.05
    opened = dict(start, x=140, sector_ceiling=np.full(2, 128))
    value, stalled = reward.step(opened)
    assert value > 3 and not stalled
    reward.step(start)
    assert reward.step(opened)[0] < 0
