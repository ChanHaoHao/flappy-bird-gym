"""DQN agent. The four functions marked TODO are yours to implement.

Everything else (network, device handling, optimiser wiring) is done. Work
through the TODOs in this order -- each one is checkable on its own:

    1. ReplayBuffer.push
    2. ReplayBuffer.sample
    3. DQNAgent.select_action
    4. DQNAgent.compute_loss     <- the one that actually matters

Run `python check.py` after each; it tests them one at a time.
"""

import numpy as np
import torch
import torch.nn as nn

from flappy_env import N_ACTIONS, OBS_DIM


class QNetwork(nn.Module):
    """Maps a state to one Q-value per action. Q(s, a) = expected discounted return."""

    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),  # no activation: Q-values are unbounded
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer of transitions, stored as flat numpy arrays.

    Why a buffer at all: consecutive frames of Flappy Bird are almost identical,
    so training on them in order gives highly correlated gradients. Sampling
    uniformly at random from the last N transitions decorrelates them.
    """

    def __init__(self, capacity: int, obs_dim: int = OBS_DIM):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # 1.0 only on real death
        self.ptr = 0   # where the next transition gets written
        self.size = 0  # how many valid transitions exist (<= capacity)

    def push(self, obs, action, reward, next_obs, done):
        """TODO 1 -- store one transition.

        Hints, in order:
          a. Write each of the five values into its array at index `self.ptr`.
             e.g. self.obs[self.ptr] = obs
          b. Advance the pointer, wrapping around when it reaches the end:
             self.ptr = (self.ptr + 1) % self.capacity
             This is what makes it circular -- old transitions get overwritten.
          c. Grow self.size up to (but never past) self.capacity:
             self.size = min(self.size + 1, self.capacity)

        Note on `done`: pass 1.0 ONLY when the bird actually died, never on a
        truncation cap. The reason is in compute_loss.
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        """TODO 2 -- return a random minibatch as torch tensors on `device`.

        Hints, in order:
          a. Draw indices uniformly from the FILLED region only, not the whole
             capacity: idx = np.random.randint(0, self.size, size=batch_size)
             (Sampling from self.capacity would return zero-filled garbage rows
             early in training -- a classic silent bug.)
          b. Index each array with `idx` to get the batch.
          c. Convert with torch.as_tensor(arr[idx]).to(device).
          d. Return them in this order, since compute_loss unpacks it:
             (obs, actions, rewards, next_obs, dones)

        Expected shapes for batch_size B:
             obs      (B, OBS_DIM)  float32
             actions  (B,)          int64
             rewards  (B,)          float32
             next_obs (B, OBS_DIM)  float32
             dones    (B,)          float32
        """

        idx = np.random.randint(0, self.size, size=batch_size)
        obs_batch = torch.as_tensor(self.obs[idx]).to(device)
        actions_batch = torch.as_tensor(self.actions[idx]).to(device)
        rewards_batch = torch.as_tensor(self.rewards[idx]).to(device)
        next_obs_batch = torch.as_tensor(self.next_obs[idx]).to(device)
        dones_batch = torch.as_tensor(self.dones[idx]).to(device)

        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, lr=5e-4, gamma=0.99, device="cuda", double_dqn=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.double_dqn = double_dqn

        self.q = QNetwork().to(self.device)
        self.target_q = QNetwork().to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())  # start identical
        for p in self.target_q.parameters():
            p.requires_grad_(False)  # target net is never trained, only copied into

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """TODO 3 -- epsilon-greedy action selection.

        Hints, in order:
          a. With probability epsilon, return a uniformly random action:
             np.random.randint(N_ACTIONS). This is exploration.
          b. Otherwise return the greedy action -- the argmax over Q-values:
               - wrap in `with torch.no_grad():` (no gradients when just acting)
               - obs is shape (OBS_DIM,); the network needs a batch dimension,
                 so make it (1, OBS_DIM) via torch.as_tensor(obs).unsqueeze(0)
               - move it to self.device
               - q_values = self.q(x)              -> shape (1, N_ACTIONS)
               - return int(q_values.argmax(dim=1).item())

        Watch out: a bird that never flaps dies in ~1 second, so with epsilon=1
        early episodes are extremely short. That is expected, not a bug.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(N_ACTIONS)
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q(obs_tensor)
                return int(q_values.argmax(dim=1).item())

    def compute_loss(self, batch) -> torch.Tensor:
        """TODO 4 -- the Bellman/TD loss. This is the heart of DQN.

        batch is exactly what ReplayBuffer.sample returned:
            obs, actions, rewards, next_obs, dones

        Hints, in order:

          a. Q-values for the actions we ACTUALLY took:
                 all_q = self.q(obs)                    # (B, N_ACTIONS)
                 q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)
             gather picks, for each row, the column named by that row's action.
             Do NOT use .max() here -- we need the taken action's value, and it
             must stay attached to the graph so gradients flow.

          b. The TD target, under torch.no_grad() (targets are constants; if you
             let gradients flow into them the training diverges):

                 vanilla DQN:
                     next_q = self.target_q(next_obs).max(dim=1).values

                 double DQN (self.double_dqn) -- pick the action with the ONLINE
                 net, evaluate it with the TARGET net. This removes the
                 overestimation bias from max-ing over noisy estimates:
                     next_a = self.q(next_obs).argmax(dim=1)
                     next_q = self.target_q(next_obs).gather(1, next_a.unsqueeze(1)).squeeze(1)

          c. Bootstrap, masked by done:
                 target = rewards + self.gamma * next_q * (1.0 - dones)

             The (1 - dones) is the crux. On death there is no future, so the
             target is just the reward. This is the ONLY place the -1 death
             penalty enters the value function -- get it wrong and the agent
             never learns that crashing is bad.

             And this is why truncation must not be marked done: hitting an
             evaluation step cap does not mean the future is worthless, so you
             still want to bootstrap there.

          d. Return the loss between q and target:
                 torch.nn.functional.smooth_l1_loss(q, target)
             Huber over MSE -- TD errors get large and MSE lets one outlier
             blow up the gradient.

        Shape check: q and target must both be (B,). If one is (B, 1) you will
        get silent broadcasting to (B, B) and a loss that looks fine but is
        meaningless. Assert it if unsure.
        """
        obs, actions, rewards, next_obs, dones = batch
        all_q = self.q(obs)
        q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_a = self.q(next_obs).argmax(dim=1)
                next_q = self.target_q(next_obs).gather(1, next_a.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_q(next_obs).max(dim=1).values

        target = rewards + self.gamma * next_q * (1.0 - dones)

        assert q.shape == target.shape

        return torch.nn.functional.smooth_l1_loss(q, target)
        

    def update(self, batch) -> float:
        """One gradient step. Implemented -- calls your compute_loss."""
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        """Hard copy online -> target. Implemented. Called every target_update steps.

        Why it exists: without it you regress Q toward a target computed from
        the same weights you are changing, which is a moving goalpost and
        oscillates. Freezing the target for N steps stabilises it.
        """
        self.target_q.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q.load_state_dict(self.q.state_dict())
