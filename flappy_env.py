"""Thin wrapper around the packaged flappy-bird-gym env.

Lives inside the repo, so `flappy_bird_gym` imports directly -- no path hacks.

This file is plumbing, not RL -- it is fully implemented. It does three things
the raw env does not:

  1. Adds `player_vel_y` to the observation. The packaged env yields only
     (h_dist, v_dist), which is NOT Markov: the same position falling at +10
     and rising at -9 needs opposite actions but looks identical. No agent can
     learn well without velocity.
  2. Replaces the reward. The packaged env returns `reward = 1` on every step
     (see flappy_bird_gym/envs/flappy_bird_env_simple.py:141) despite its
     docstring claiming otherwise. We use the shaping you asked for:
     +0.1/frame alive, +1/pipe passed, -1 on death.
  3. Separates `terminated` (the bird died) from `truncated` (we hit a step
     cap). These must be handled differently in the TD target -- see dqn.py.
  4. Adds a ceiling. The packaged game's check_crash() only tests the ground
     and the pipes, so the bird can fly off the top of the screen and stay
     there. Measured: with ~50% random flapping it pins at y = -92 and EVERY
     episode ends at exactly frame 101 when the first pipe arrives. Identical
     returns across every episode means zero learning signal, and the agent
     sits flat for a very long time. Killing on ceiling contact fixes it.
"""

import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless: no display needed

from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple  # noqa: E402
from flappy_bird_gym.envs.game_logic import PLAYER_MAX_VEL_Y  # noqa: E402

OBS_DIM = 4
N_ACTIONS = 2  # 0 = do nothing, 1 = flap

REWARD_ALIVE = 0.1
REWARD_PIPE = 1.0
REWARD_DEATH = -1.0

# Optional extra shaping, OFF by default -- see `align_penalty` below.
ALIGN_PENALTY = 0.05


def _ensure_audio_driver():
    """Fall back to a dummy audio device if the real one will not open.

    The packaged renderer always builds with audio_on=True and calls
    pygame.mixer.init() unconditionally, which raises on machines with no
    working sound device (headless boxes, containers, WSL). Sound is not part
    of the task, so silently going quiet beats not rendering at all.
    """
    import pygame

    try:
        pygame.mixer.init()
    except pygame.error:
        os.environ["SDL_AUDIODRIVER"] = "dummy"


class FlappyEnv:
    """Gymnasium-style API: reset() -> obs, step(a) -> (obs, r, terminated, truncated, info)."""

    def __init__(
        self,
        max_steps: int | None = None,
        render: bool = False,
        ceiling_kills: bool = True,
        align_penalty: float = 0.0,
    ):
        """align_penalty: subtract `align_penalty * |v_dist|` each frame.

        Off by default, because the requested reward (+0.1/frame, +1/pipe,
        -1/death) is the honest specification of the task. Turn it on if
        training stalls at return ~9.0.

        Why it helps, measured on this setup: with the plain reward, DQN
        converges by ~60k steps to "hover near the top, survive to the first
        pipe, die" -- bird_y ~98 at death with v_dist always positive, i.e.
        always above the gap. Return is exactly 9.0 on every episode, so there
        is no variance to learn from, and it had passed a pipe once in 1341
        episodes. It never sees the +1 that would teach it to descend.

        This term gives a gradient toward the gap centre before the agent has
        ever passed a pipe. Cost: it biases toward centring, which is not
        actually required, so it slightly caps the ceiling on final play.
        Consider annealing it to 0 once scores take off.
        """
        if render:  # SDL reads this when the display is first initialised
            os.environ.pop("SDL_VIDEODRIVER", None)
            _ensure_audio_driver()
        self.env = FlappyBirdEnvSimple(normalize_obs=True)
        self.max_steps = max_steps  # None = unbounded (what we want for training)
        self.render_mode = render
        self.ceiling_kills = ceiling_kills  # see point 4 in the module docstring
        self.align_penalty = align_penalty
        self._steps = 0
        self._score = 0

    def _obs(self, raw):
        """raw is the packaged env's [h_dist, v_dist], already normalised by screen size."""
        game = self.env._game
        h_dist, v_dist = float(raw[0]), float(raw[1])
        player_y = game.player_y / self.env._screen_size[1]
        vel_y = game.player_vel_y / PLAYER_MAX_VEL_Y  # roughly [-0.9, 1.0]
        return np.array([h_dist, v_dist, player_y, vel_y], dtype=np.float32)
        # Extension once this works: append the vertical distance to the SECOND
        # pipe's gap. Lets the agent pre-position for a big gap-to-gap jump.

    def render(self):
        """Draw one frame. Also what first creates the pygame window.

        The packaged env builds its renderer lazily inside render(), so nothing
        pygame-related (display, event queue) exists until this is called once.
        """
        self.env.render()

    def reset(self):
        raw = self.env.reset()
        self._steps = 0
        self._score = 0
        if self.render_mode:
            self.render()  # open the window now, before any caller pumps events
        return self._obs(raw)

    def step(self, action: int):
        raw, _, done, info = self.env.step(int(action))  # packaged env uses the old 4-tuple API
        self._steps += 1

        if self.ceiling_kills and self.env._game.player_y < 0:
            done = True  # the packaged game has no ceiling; we add one

        obs = self._obs(raw)

        reward = REWARD_ALIVE
        if self.align_penalty:
            reward -= self.align_penalty * abs(float(obs[1]))  # |v_dist| to gap centre
        if info["score"] > self._score:  # passed a pipe this frame
            reward += REWARD_PIPE * (info["score"] - self._score)
            self._score = info["score"]
        if done:
            reward = REWARD_DEATH

        terminated = bool(done)
        truncated = bool(
            self.max_steps is not None and self._steps >= self.max_steps and not terminated
        )

        if self.render_mode:
            self.render()

        return (
            obs,
            reward,
            terminated,
            truncated,
            {"score": self._score, "steps": self._steps},
        )

    def close(self):
        self.env.close()
