"""Watch a trained agent play, with the pygame window. Fully implemented.

    uv run --project . python play.py --model runs/<run-name>/model.pt
"""

import argparse

from dqn import DQNAgent
from flappy_env import FlappyEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    import pygame

    agent = DQNAgent(device=args.device)
    agent.load(args.model)
    env = FlappyEnv(max_steps=args.max_steps, render=True)
    clock = pygame.time.Clock()

    for ep in range(args.episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            obs, reward, terminated, truncated, info = env.step(
                agent.select_action(obs, epsilon=0.0)
            )
            total += reward
            done = terminated or truncated
            clock.tick(args.fps)
        print(f"episode {ep + 1}: score {info['score']}  return {total:.1f}  "
              f"frames {info['steps']}" + ("  (hit step cap)" if truncated else ""))
    env.close()


if __name__ == "__main__":
    main()
