"""Record one episode of a checkpoint's policy to a video, with a timeline of the milestones it reached.

One video frame per agent step. A step is 24 game frames, so at 60 fps the video runs at 24x game speed. Actions are
sampled, as in the evaluator: the emulator is deterministic, so the seed picks the episode. Encoding pipes raw frames
to ffmpeg, which must be installed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, Protocol

import numpy as np
import torch
import tyro
from torch import nn

from pokered.checkpoints import CheckpointStore
from pokered.env import PokemonRedEnv
from pokered.evaluate import env_config_from_checkpoint
from pokered.frame import read_frame
from pokered.milestones import MILESTONES, MilestoneTracker
from pokered.model import ActorCritic
from pokered.negative_space import check_shape, expect, require
from pokered.start_state import START_STATE
from pokered.train import pick_device

log = logging.getLogger(__name__)

SCREEN_SHAPE: Final[tuple[int, int, int]] = (144, 160, 3)
_MILESTONE_NAMES: Final[frozenset[str]] = frozenset(milestone.name for milestone in MILESTONES)
_ENCODE_TIMEOUT_S: Final[int] = 600


class VideoError(RuntimeError):
    """ffmpeg is missing or failed to encode."""


class FrameSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...


def scale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upscale of the game screen, so each Game Boy pixel stays a sharp square."""
    check_shape(frame, SCREEN_SHAPE, "screen frame")
    require(scale >= 1, f"scale must be at least 1, got {scale}")
    scaled = frame.repeat(scale, axis=0).repeat(scale, axis=1)
    require(scaled.shape == (SCREEN_SHAPE[0] * scale, SCREEN_SHAPE[1] * scale, 3), "scaling changed the channels")
    return scaled


def stop_step(reached: dict[str, int], *, stop_after: str | None, tail_steps: int) -> int | None:
    """The step the video should end on: ``tail_steps`` after ``stop_after`` was first reached, or None to play on."""
    if stop_after is not None and stop_after not in _MILESTONE_NAMES:
        raise ValueError(f"no milestone named {stop_after!r}; choose one of {sorted(_MILESTONE_NAMES)}")
    require(tail_steps >= 0, f"tail steps cannot be negative, got {tail_steps}")
    if stop_after is None or stop_after not in reached:
        return None
    return reached[stop_after] + tail_steps


def write_timeline(path: Path, reached: dict[str, int], *, fps: int, steps: int, seed: int, env_steps: int) -> None:
    """Each milestone's first step and where it is in the video, for finding the moments worth showing."""
    require(fps >= 1, f"fps must be positive, got {fps}")
    require(all(0 <= step <= steps for step in reached.values()), f"a milestone step is outside the episode: {reached}")
    timeline = {
        "seed": seed,
        "env_steps": env_steps,
        "fps": fps,
        "steps": steps,
        "milestones": {name: {"step": step, "time_s": step / fps} for name, step in sorted(reached.items(), key=lambda item: item[1])},
    }
    path.write_text(json.dumps(timeline, indent=2))


class FfmpegWriter:
    """Pipes raw RGB frames to ffmpeg, which writes an H.264 MP4 any player opens."""

    def __init__(self, path: Path, *, width: int, height: int, fps: int, ffmpeg: str = "ffmpeg") -> None:
        require(width > 0, f"video width must be positive, got {width}")
        require(height > 0, f"video height must be positive, got {height}")
        require(fps > 0, f"fps must be positive, got {fps}")
        self._path = path
        self._shape = (height, width, 3)
        self._fps = fps
        self._ffmpeg = ffmpeg
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> FfmpegWriter:
        executable = shutil.which(self._ffmpeg)
        if executable is None:
            raise VideoError(f"{self._ffmpeg} was not found; install ffmpeg (on a Mac: brew install ffmpeg)")
        height, width, _ = self._shape
        command = [
            executable, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(self._path),
        ]  # fmt: skip
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603 a fixed argument list; the only input is our own path
        return self

    def write(self, frame: np.ndarray) -> None:
        check_shape(frame, self._shape, "video frame")
        stdin = self._stdin()
        stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def _stdin(self) -> IO[bytes]:
        process = expect(self._process, "write called outside the writer's with block")
        return expect(process.stdin, "ffmpeg was started without a stdin pipe")

    def __exit__(self, *_: object) -> None:
        process = expect(self._process, "the writer was never entered")
        try:
            _stdout, raw_stderr = process.communicate(timeout=_ENCODE_TIMEOUT_S)  # closes stdin, waits, closes stderr
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise VideoError(f"ffmpeg did not finish {self._path} within {_ENCODE_TIMEOUT_S} s") from error
        stderr = raw_stderr.decode(errors="replace") if raw_stderr else ""
        if process.returncode != 0:
            raise VideoError(f"ffmpeg failed on {self._path} with exit status {process.returncode}: {stderr.strip()}")


def _reached(tracker: MilestoneTracker) -> dict[str, int]:
    """Milestone name to the first step it held, for the milestones reached so far."""
    summary = tracker.summary()
    reached = {
        milestone.name: int(summary[f"steps_to_{milestone.name}"])
        for milestone in MILESTONES
        if summary[f"milestone_{milestone.name}"] == 1.0
    }
    require(all(step >= 0 for step in reached.values()), f"a milestone was reached at a negative step: {reached}")
    return reached


@dataclass(frozen=True, slots=True)
class Recording:
    steps: int
    reached: dict[str, int]  # milestone name to the first step it held


def record_episode(
    env: PokemonRedEnv,
    model: nn.Module,
    sink: FrameSink,
    *,
    seed: int,
    scale: int,
    max_steps: int,
    stop_after: str | None,
    tail_steps: int,
) -> Recording:
    """Play one episode from the env's start state with sampled actions, writing each step's screen to ``sink``.

    Ends at the episode's end, ``max_steps``, or ``tail_steps`` after ``stop_after`` is first reached."""
    require(max_steps >= 1, f"max_steps must be positive, got {max_steps}")
    stop_step({}, stop_after=stop_after, tail_steps=tail_steps)  # refuse a bad milestone name before playing
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    tracker = MilestoneTracker()
    obs, _ = env.reset(seed=seed)
    tracker.reset(read_frame(env.pyboy.memory))
    steps = 0
    for steps in range(1, max_steps + 1):
        with torch.inference_mode():
            logits, _ = model({key: torch.from_numpy(value[None]).to(device) for key, value in obs.items()})
        action = int(torch.multinomial(logits.float().softmax(1).cpu(), 1, generator=generator).item())
        obs, _, terminated, truncated, _ = env.step(action)
        tracker.update(read_frame(env.pyboy.memory), step=steps)
        sink.write(scale_frame(env.pyboy.screen.ndarray[:, :, :3], scale))
        end = stop_step(_reached(tracker), stop_after=stop_after, tail_steps=tail_steps)
        if terminated or truncated or (end is not None and steps >= end):
            break
    require(1 <= steps <= max_steps, f"recorded {steps} steps, outside 1-{max_steps}")
    return Recording(steps=steps, reached=_reached(tracker))


@dataclass(frozen=True, slots=True)
class RecordConfig:
    rom: Path
    checkpoints: Path  # a run's checkpoint directory; its newest intact checkpoint is used
    output: Path  # the .mp4 to write; the timeline goes next to it as .json
    seed: int = 0
    fps: int = 60
    scale: int = 3
    max_steps: int = 200_000
    stop_after: str | None = None  # a milestone name, such as got_hm01 or knows_cut
    tail_steps: int = 600
    device: str = "cpu"
    state: Path = START_STATE


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(RecordConfig)
    payload = expect(CheckpointStore(config.checkpoints).load_latest(), f"no intact checkpoint in {config.checkpoints}")
    env_config = env_config_from_checkpoint(payload["config"]["env"], rom=config.rom, state=config.state)
    model = ActorCritic()
    model.load_state_dict(payload["model"])
    model.to(pick_device(config.device))
    env = PokemonRedEnv(env_config)
    height, width, _ = SCREEN_SHAPE
    config.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FfmpegWriter(config.output, width=width * config.scale, height=height * config.scale, fps=config.fps) as writer:
            recording = record_episode(
                env, model, writer, seed=config.seed, scale=config.scale, max_steps=config.max_steps,
                stop_after=config.stop_after, tail_steps=config.tail_steps,
            )  # fmt: skip
    finally:
        env.close()
    timeline = config.output.with_suffix(".json")
    write_timeline(timeline, recording.reached, fps=config.fps, steps=recording.steps, seed=config.seed, env_steps=payload["env_steps"])
    log.info("wrote %s (%d steps) and %s; milestones: %s", config.output, recording.steps, timeline, sorted(recording.reached))


if __name__ == "__main__":
    main()
