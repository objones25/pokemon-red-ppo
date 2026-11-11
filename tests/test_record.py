"""Recording an episode of a policy to a video."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
from pokered.record import FfmpegWriter, VideoError, record_episode, scale_frame, stop_step, write_timeline
from support import ROM, requires_rom

from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.model import ActorCritic

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _screen() -> np.ndarray:
    frame = np.zeros((144, 160, 3), np.uint8)
    frame[0, 0] = (10, 20, 30)
    frame[143, 159] = (200, 100, 50)
    return frame


def test_scaling_repeats_each_pixel_into_a_square():
    scaled = scale_frame(_screen(), 3)
    assert scaled.shape == (432, 480, 3)
    assert (scaled[:3, :3] == (10, 20, 30)).all()
    assert (scaled[-3:, -3:] == (200, 100, 50)).all()


def test_scaling_by_one_keeps_the_frame():
    assert np.array_equal(scale_frame(_screen(), 1), _screen())


def test_scaling_rejects_a_frame_that_is_not_the_game_screen():
    with pytest.raises(AssertionError, match="shape"):
        scale_frame(np.zeros((144, 160, 4), np.uint8), 3)


def test_scaling_rejects_a_scale_below_one():
    with pytest.raises(AssertionError, match="scale"):
        scale_frame(_screen(), 0)


def test_without_a_stop_milestone_the_video_runs_to_the_end_of_the_episode():
    assert stop_step({"got_hm01": 100}, stop_after=None, tail_steps=50) is None


def test_the_video_stops_the_tail_after_the_stop_milestone():
    assert stop_step({"got_hm01": 100}, stop_after="got_hm01", tail_steps=50) == 150


def test_the_video_keeps_going_until_the_stop_milestone_is_reached():
    assert stop_step({"beat_misty": 90}, stop_after="got_hm01", tail_steps=50) is None


def test_an_unknown_stop_milestone_is_refused():
    with pytest.raises(ValueError, match="no milestone named 'got_hm1'"):
        stop_step({}, stop_after="got_hm1", tail_steps=50)


def test_the_timeline_gives_each_milestone_its_step_and_video_time(tmp_path):
    path = tmp_path / "run.json"
    write_timeline(path, {"got_hm01": 120}, fps=60, steps=600, seed=3, env_steps=400)
    timeline = json.loads(path.read_text())
    assert timeline["milestones"] == {"got_hm01": {"step": 120, "time_s": 2.0}}
    assert (timeline["steps"], timeline["seed"], timeline["env_steps"], timeline["fps"]) == (600, 3, 400, 60)


@needs_ffmpeg
@pytest.mark.slow
def test_the_writer_makes_a_video_with_one_frame_per_write(tmp_path):
    path = tmp_path / "clip.mp4"
    with FfmpegWriter(path, width=160, height=144, fps=30) as writer:
        for _ in range(12):
            writer.write(_screen())
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "ffprobe ships with ffmpeg"
    probe = subprocess.run(  # noqa: S603 a fixed argument list with the test's own file
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )  # fmt: skip
    stream = json.loads(probe.stdout)["streams"][0]
    assert (int(stream["nb_read_frames"]), stream["width"], stream["height"]) == (12, 160, 144)


def test_a_missing_ffmpeg_raises_a_video_error(tmp_path):
    writer = FfmpegWriter(tmp_path / "clip.mp4", width=160, height=144, fps=30, ffmpeg="ffmpeg-not-installed-here")
    with pytest.raises(VideoError, match="ffmpeg"):
        writer.__enter__()


@needs_ffmpeg
@pytest.mark.slow
def test_the_writer_rejects_a_frame_of_another_size(tmp_path):
    with pytest.raises(AssertionError, match="shape"), FfmpegWriter(tmp_path / "clip.mp4", width=160, height=144, fps=30) as writer:
        writer.write(np.zeros((10, 10, 3), np.uint8))


class _Frames:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


@requires_rom
@pytest.mark.slow
def test_recording_writes_one_scaled_frame_per_step_until_the_step_limit():
    env = PokemonRedEnv(EnvConfig(rom=ROM, stagger_first_episode=False))
    frames = _Frames()
    try:
        torch.manual_seed(0)
        result = record_episode(env, ActorCritic(), frames, seed=0, scale=2, max_steps=30, stop_after=None, tail_steps=0)
    finally:
        env.close()
    assert result.steps == 30
    assert len(frames.frames) == 30
    assert frames.frames[0].shape == (288, 320, 3)
