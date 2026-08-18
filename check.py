"""Self-check for your four TODOs. Fully implemented -- just run it.

    uv run --project . python check.py

It tests each function independently and tells you which hint you missed.
Everything passing does not guarantee the agent learns, but every bug it
catches is one you would otherwise have spent an hour staring at a flat
reward curve to find.
"""

import numpy as np
import torch
import torch.nn.functional as F

from dqn import DQNAgent, ReplayBuffer
from flappy_env import N_ACTIONS, OBS_DIM, FlappyEnv

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
results = []


def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append(True)
    except NotImplementedError:
        print(f"  \033[33mTODO\033[0m  {name} -- not implemented yet")
        results.append(False)
    except AssertionError as e:
        print(f"  {FAIL}  {name}\n         {e}")
        results.append(False)
    except Exception as e:
        print(f"  {FAIL}  {name}\n         {type(e).__name__}: {e}")
        results.append(False)


# ---------------------------------------------------------------- environment
def test_env():
    env = FlappyEnv(max_steps=50)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,), f"reset() gave shape {obs.shape}, expected ({OBS_DIM},)"
    assert obs.dtype == np.float32, f"obs dtype is {obs.dtype}, expected float32"
    obs, r, term, trunc, info = env.step(1)
    assert abs(r - 0.1) < 1e-6, f"a survived frame should give +0.1, got {r}"
    # fly into the ground: never flap
    total, term = 0.0, False
    env.reset()
    for _ in range(200):
        _, r, term, trunc, info = env.step(0)
        total += r
        if term or trunc:
            break
    assert term, "never flapping should kill the bird within 200 frames"
    assert abs(r - (-1.0)) < 1e-6, f"death should give -1.0, got {r}"

    # The packaged game has no ceiling, so we add one. Without it, ~50% random
    # flapping pins the bird off-screen and every episode ends at exactly frame
    # 101 -- identical returns, so nothing to learn from.
    env.reset()
    lengths = set()
    for _ in range(20):
        env.reset()
        n, done = 0, False
        while not done:
            _, _, term, trunc, _ = env.step(1)  # flap every frame -> straight up
            n += 1
            done = term or trunc
        lengths.add(n)
    assert min(lengths) < 60, (
        f"flapping every frame should hit the ceiling quickly, but episodes ran "
        f"{sorted(lengths)} frames -- is ceiling_kills disabled?")
    env.close()


# ------------------------------------------------------------- TODO 1: push
def test_push():
    buf = ReplayBuffer(capacity=4)
    for i in range(3):
        buf.push(np.full(OBS_DIM, i, np.float32), i % N_ACTIONS, float(i),
                 np.full(OBS_DIM, i + 1, np.float32), 0.0)
    assert len(buf) == 3, f"after 3 pushes len(buffer) should be 3, got {len(buf)} (hint c)"
    assert buf.ptr == 3, f"ptr should be 3, got {buf.ptr} (hint b)"
    assert np.allclose(buf.obs[1], 1.0), "obs was not stored at index ptr (hint a)"
    assert buf.rewards[2] == 2.0, "rewards were not stored (hint a)"
    assert np.allclose(buf.next_obs[2], 3.0), "next_obs was not stored (hint a)"
    # overflow must wrap, not grow
    for i in range(3, 7):
        buf.push(np.full(OBS_DIM, i, np.float32), 0, float(i),
                 np.full(OBS_DIM, i + 1, np.float32), 0.0)
    assert len(buf) == 4, f"size must cap at capacity=4, got {len(buf)} (hint c)"
    assert buf.ptr == 3, f"ptr must wrap with %capacity, got {buf.ptr} (hint b)"
    assert buf.rewards[0] == 4.0, "old transitions must be overwritten circularly (hint b)"


# ----------------------------------------------------------- TODO 2: sample
def test_sample():
    buf = ReplayBuffer(capacity=1000)
    for i in range(10):  # only 10 of 1000 slots filled
        buf.push(np.full(OBS_DIM, i + 1, np.float32), i % N_ACTIONS, 1.0,
                 np.full(OBS_DIM, i + 1, np.float32), 0.0)
    obs, actions, rewards, next_obs, dones = buf.sample(32, torch.device("cpu"))

    assert obs.shape == (32, OBS_DIM), f"obs batch shape {tuple(obs.shape)}, expected (32, {OBS_DIM})"
    assert actions.shape == (32,), f"actions shape {tuple(actions.shape)}, expected (32,) (hint d)"
    assert rewards.shape == (32,), f"rewards shape {tuple(rewards.shape)}, expected (32,)"
    assert dones.shape == (32,), f"dones shape {tuple(dones.shape)}, expected (32,)"
    assert actions.dtype == torch.int64, (
        f"actions must be int64 for .gather(), got {actions.dtype}")
    assert obs.dtype == torch.float32, f"obs must be float32, got {obs.dtype}"
    assert (obs != 0).all(), (
        "sampled a zero-filled row -- you drew indices from self.capacity "
        "instead of self.size (hint a). This is the classic silent bug.")
    assert torch.is_tensor(obs), "sample() must return torch tensors, not numpy arrays (hint c)"


# ---------------------------------------------------- TODO 3: select_action
def test_select_action():
    agent = DQNAgent(device="cpu")
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    a = agent.select_action(obs, epsilon=0.0)
    assert isinstance(a, (int, np.integer)), f"must return a python int, got {type(a)}"
    assert 0 <= a < N_ACTIONS, f"action {a} outside [0, {N_ACTIONS})"

    with torch.no_grad():
        expected = int(agent.q(torch.as_tensor(obs).unsqueeze(0)).argmax(dim=1).item())
    for _ in range(20):
        assert agent.select_action(obs, epsilon=0.0) == expected, (
            "with epsilon=0 the action must be the deterministic argmax of Q (hint b)")

    seen = {agent.select_action(obs, epsilon=1.0) for _ in range(200)}
    assert seen == set(range(N_ACTIONS)), (
        f"with epsilon=1.0 both actions should appear, only saw {seen} (hint a)")


# ------------------------------------------------------ TODO 4: compute_loss
def _reference_loss(agent, batch):
    obs, actions, rewards, next_obs, dones = batch
    q = agent.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if agent.double_dqn:
            next_a = agent.q(next_obs).argmax(dim=1)
            next_q = agent.target_q(next_obs).gather(1, next_a.unsqueeze(1)).squeeze(1)
        else:
            next_q = agent.target_q(next_obs).max(dim=1).values
        target = rewards + agent.gamma * next_q * (1.0 - dones)
    return F.smooth_l1_loss(q, target)


def _batch(B=16, all_done=False):
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(B, OBS_DIM, generator=g),
        torch.randint(0, N_ACTIONS, (B,), generator=g),
        torch.randn(B, generator=g),
        torch.randn(B, OBS_DIM, generator=g),
        torch.ones(B) if all_done else (torch.rand(B, generator=g) < 0.3).float(),
    )


def test_compute_loss():
    for double in (True, False):
        agent = DQNAgent(device="cpu", double_dqn=double)
        torch.manual_seed(0)
        batch = _batch()
        loss = agent.compute_loss(batch)
        tag = "double_dqn" if double else "vanilla"

        assert torch.is_tensor(loss), f"[{tag}] must return a tensor, got {type(loss)}"
        assert loss.dim() == 0, (
            f"[{tag}] loss must be a scalar, got shape {tuple(loss.shape)}. "
            "If it is (B, B) you hit the broadcasting bug -- q and target must both be (B,).")
        assert loss.requires_grad, (
            f"[{tag}] loss has no grad_fn -- did you wrap the ONLINE q(obs) in "
            "torch.no_grad()? Only the target may be no_grad (hint b).")

        ref = _reference_loss(agent, batch)
        assert torch.allclose(loss, ref, atol=1e-5), (
            f"[{tag}] loss {loss.item():.6f} != expected {ref.item():.6f}. "
            "Common causes: forgot the (1 - dones) mask (hint c); used .max() instead "
            "of .gather() for the taken action (hint a); used MSE instead of "
            "smooth_l1_loss (hint d); used self.q instead of self.target_q for the "
            "bootstrap value (hint b).")

    # done-masking gets its own test -- it is the single most common bug, and it
    # is exactly what teaches the agent that the -1 death penalty matters.
    agent = DQNAgent(device="cpu")
    obs, actions, rewards, next_obs, _ = _batch(all_done=True)
    batch = (obs, actions, rewards, next_obs, torch.ones(len(rewards)))
    loss = agent.compute_loss(batch)
    q = agent.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    expected = F.smooth_l1_loss(q, rewards)  # all done -> target is the reward alone
    assert torch.allclose(loss, expected, atol=1e-5), (
        f"when every transition is terminal the target must be the reward alone "
        f"(no bootstrap), so the loss should be {expected.item():.6f}, got {loss.item():.6f}. "
        "You are missing the (1 - dones) mask (hint c).")


if __name__ == "__main__":
    print("\nenvironment (already implemented)")
    check("FlappyEnv obs shape / reward shaping", test_env)
    print("\nyour TODOs")
    check("TODO 1  ReplayBuffer.push", test_push)
    check("TODO 2  ReplayBuffer.sample", test_sample)
    check("TODO 3  DQNAgent.select_action", test_select_action)
    check("TODO 4  DQNAgent.compute_loss", test_compute_loss)
    n = sum(results)
    print(f"\n{n}/{len(results)} passing"
          + ("  -- ready to train\n" if n == len(results) else "\n"))
