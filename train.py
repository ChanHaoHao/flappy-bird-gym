"""Training loop. Fully implemented -- it calls the four functions you write in dqn.py.

    uv run --project . python train.py
    uv run --project . python train.py --steps 200000 --run-name my_try

Training episodes are UNBOUNDED on purpose: a never-ending episode means the
policy works, and DQN samples from a step-indexed replay buffer so it does not
care where episodes end. Only evaluation gets a step cap (otherwise the eval
loop would never return), and that cap is reported as truncation, not death.
"""

import argparse
import os
import time

import numpy as np

from dqn import DQNAgent, ReplayBuffer
from flappy_env import FlappyEnv
from plotting import Logger


def linear_schedule(step, start, end, decay_steps):
    """Interpolate start -> end linearly over decay_steps, then hold.

    Used for both epsilon and the align penalty. decay_steps <= 0 means "jump
    straight to end", which is how you disable a term from step 1.
    """
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def evaluate(agent, episodes=5, max_steps=10_000):
    """Greedy rollouts (epsilon=0). Capped so it terminates once the agent is good.

    Always evaluated on the plain reward (no align penalty), so the number means
    the same thing regardless of what shaping training is currently using.

    Returns (mean_score, mean_return, n_truncated). Watch n_truncated: once it
    is nonzero the cap is censoring the top of the distribution and mean_score
    is an underestimate -- raise --eval-max-steps rather than believing it.
    """
    env = FlappyEnv(max_steps=max_steps)
    scores, returns, n_trunc = [], [], 0
    for _ in range(episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        n_trunc += int(truncated)
        scores.append(info["score"])
        returns.append(total)
    env.close()
    return float(np.mean(scores)), float(np.mean(returns)), n_trunc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000, help="total environment steps")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="~100-frame horizon; deaths are caused ~10-30 frames earlier, so 0.99 is plenty")
    p.add_argument("--learning-starts", type=int, default=5_000,
                   help="fill the buffer with random play before the first gradient step")
    p.add_argument("--target-update", type=int, default=1_000, help="steps between target net syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--plot-every", type=int, default=20, help="episodes between plot refreshes")
    p.add_argument("--align-penalty", type=float, default=0.0,
                   help="subtract this * |v_dist| per frame. Off by default; turn it on "
                        "(try 0.05) if training stalls at return ~9.0, which means the "
                        "agent is hovering high and dying on the first pipe forever")
    p.add_argument("--align-penalty-final", type=float, default=None,
                   help="anneal --align-penalty to this value over --align-anneal-steps. "
                        "Defaults to no annealing (holds --align-penalty throughout). "
                        "Set 0.0 to phase the shaping out once scores have taken off: it "
                        "biases toward gap-centring, which the task does not require, so "
                        "leaving it on caps final play")
    p.add_argument("--align-anneal-steps", type=int, default=100_000,
                   help="steps over which --align-penalty reaches --align-penalty-final")
    p.add_argument("--init-model", type=str, default=None,
                   help="path to a model.pt to start from, for fine-tuning. The replay "
                        "buffer still starts empty, so it refills with this policy's own "
                        "play -- keep --eps-start low or you will scribble over it")
    p.add_argument("--eval-max-steps", type=int, default=10_000,
                   help="step cap per greedy eval episode")
    p.add_argument("--run-name", type=str, default=time.strftime("dqn_%Y%m%d_%H%M%S"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    logger = Logger(run_dir)
    env = FlappyEnv(max_steps=None, align_penalty=args.align_penalty)  # unbounded during training
    agent = DQNAgent(lr=args.lr, gamma=args.gamma, device=args.device)
    buffer = ReplayBuffer(args.buffer_size)
    align_final = (args.align_penalty if args.align_penalty_final is None
                   else args.align_penalty_final)
    if args.init_model:
        agent.load(args.init_model)
        print(f"resumed from {args.init_model}")
    print(f"device={agent.device}  run={run_dir}")
    print(f"align penalty {args.align_penalty} -> {align_final} over "
          f"{args.align_anneal_steps} steps")

    obs = env.reset()
    ep_reward, ep_len, episode, last_loss = 0.0, 0, 0, float("nan")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        epsilon = linear_schedule(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        # Anneal the shaping term in place. Safe mid-episode: the penalty only
        # affects rewards emitted from here on, and past transitions keep the
        # reward they were actually collected under.
        env.align_penalty = linear_schedule(
            step, args.align_penalty, align_final, args.align_anneal_steps
        )
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # `terminated` only -- never store a truncation as done, or the agent
        # learns that surviving to the cap is as bad as crashing.
        buffer.push(obs, action, reward, next_obs, float(terminated))

        obs = next_obs
        ep_reward += reward
        ep_len += 1

        if step >= args.learning_starts:
            last_loss = agent.update(buffer.sample(args.batch_size, agent.device))
        if step % args.target_update == 0:
            agent.sync_target()

        if terminated or truncated:
            episode += 1
            logger.log_episode(episode, step, ep_reward, info["score"], ep_len, epsilon, last_loss)
            if episode % args.plot_every == 0:
                logger.plot()
                print(f"step {step:>7}  ep {episode:>5}  return {ep_reward:8.1f}  "
                      f"score {info['score']:>4}  len {ep_len:>5}  eps {epsilon:.3f}  "
                      f"loss {last_loss:.4f}  {step / (time.time() - t0):.0f} steps/s")
            obs = env.reset()
            ep_reward, ep_len = 0.0, 0

        if step % args.eval_every == 0:
            mean_score, mean_return, n_trunc = evaluate(agent, max_steps=args.eval_max_steps)
            print(f"  [eval @ {step}] mean score {mean_score:.1f}  mean return {mean_return:.1f}"
                  + (f"  ({n_trunc}/5 hit the {args.eval_max_steps}-step cap -- "
                     f"score is an underestimate)" if n_trunc else ""))
            agent.save(os.path.join(run_dir, "model.pt"))

    agent.save(os.path.join(run_dir, "model.pt"))
    print("final plot:", logger.plot())
    env.close()


if __name__ == "__main__":
    main()
