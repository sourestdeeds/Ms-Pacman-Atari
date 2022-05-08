from atari_wrapper import make_atari_env
import torch
from copy import deepcopy
import numpy as np
from torch import nn
import torch.nn.functional as F

from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy, to_torch_as
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

import os

from torch.utils.tensorboard import SummaryWriter


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning."""

    def __init__(
        self,
        c, # Channels - In this case, stacked frames.
        h, # Height 
        w, # Width
        action_shape, # Action Space
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        output_dim=None,
    ):
        super().__init__()
        self.device = device
        self.DQN = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=16, stride=4), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=12, stride=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=4, stride=1), 
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.DQN(torch.zeros(1, c, h, w)).shape[1:])

        if output_dim is not None:
            self.DQN = nn.Sequential(
                self.DQN, nn.Linear(self.output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(self, obs, state=None, info={}):
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.DQN(obs), state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning."""

    def __init__(
            self,
            c, # Channels - In this case, stacked frames.
            h, # Height 
            w, # Width
            action_shape, # Action Space
            num_atoms: int = 51,
            noisy_std: float = 0.5,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(c, h, w, action_shape, device)
        self.action_num = np.prod(action_shape)

        self.num_atoms = num_atoms

        self.Double_Q = nn.Sequential(
            NoisyLinear(self.output_dim, 512, noisy_std), nn.ReLU(inplace=True),
            NoisyLinear(512, self.action_num * self.num_atoms, noisy_std)
        )

        self.Dueling = nn.Sequential(
            NoisyLinear(self.output_dim, 512, noisy_std), nn.ReLU(inplace=True),
            NoisyLinear(512, self.num_atoms, noisy_std)
        )
        self.output_dim = self.action_num * self.num_atoms


    def forward(self, obs, state=None, info={}):
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        # Split the Q table into advantage/value
        q = self.Double_Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        # Dueling
        v = self.Dueling(obs)
        v = v.view(-1, 1, self.num_atoms)
        logits = q - q.mean(dim=1, keepdim=True) + v
        probs = logits.softmax(dim=2)
        return probs, state


class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.
        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(
            self, in_features, out_features, noisy_std=0.5
    ):
        super().__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def f(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(x.size(0), device=x.device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))  # type: ignore
        self.eps_q.copy_(self.f(self.eps_q))  # type: ignore

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * (
                self.eps_q.ger(self.eps_p)  # type: ignore
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here)."""

    def __init__(
            self,
            model,
            optim,
            discount_factor=0.99,
            estimation_step=1,
            target_update_freq=0,
            reward_normalization=False,
            is_double=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double

    def set_eps(self, eps):
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode=True):
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer, indices):
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
            self, batch, buffer, indices):
        """Compute the n-step return for Q-learning targets."""
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(self, logits, mask):
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
            self,
            batch,
            state=None,
            model="model",
            input="obs",
            **kwargs,
    ):
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch, **kwargs):
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q
        loss = (td_error.pow(2) * weight).mean()
        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(self, act, batch):
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act


class C51Policy(DQNPolicy):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887."""

    def __init__(
            self,
            model,
            optim,
            discount_factor=0.99,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            estimation_step=3,
            target_update_freq=500,
            reward_normalization=False,
            **kwargs,
    ):
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq,
            reward_normalization, **kwargs
        )
        assert num_atoms > 1, "num_atoms should be greater than 1"
        assert v_min < v_max, "v_max should be larger than v_min"
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max
        self.support = torch.nn.Parameter(
            torch.linspace(self._v_min, self._v_max, self._num_atoms),
            requires_grad=False,
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _target_q(self, buffer, indices):
        return self.support.repeat(len(indices), 1)  # shape: [bsz, num_atoms]

    def compute_q_value(self, logits, mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return super().compute_q_value((logits.to(device) * self.support.to(device)).sum(2), mask)

    def _target_dist(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._target:
            act = self(batch, input="obs_next").act
            next_dist = self(batch, model="model_old", input="obs_next").logits
        else:
            next_batch = self(batch, input="obs_next")
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        target_support = batch.returns.clamp(self._v_min, self._v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (
                              1 - (target_support.unsqueeze(1).to(device) - self.support.view(1, -1, 1).to(
                          device)).abs() /
                              self.delta_z
                      ).clamp(0, 1) * next_dist.unsqueeze(1).to(device)
        return target_dist.sum(-1)

    def learn(self, batch, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        weight = batch.pop("weight", 1.0)
        curr_dist = self(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy.to(device) * weight.to(device)).mean()
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/agent.py L94-100
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}


class RainbowPolicy(C51Policy):
    """Implementation of Rainbow DQN. arXiv:1710.02298."""

    def learn(self, batch, **kwargs):
        sample_noise(self.model)
        if self._target and sample_noise(self.model_old):
            self.model_old.train()  # so that NoisyLinear takes effect
        return super().learn(batch, **kwargs)


def sample_noise(model: nn.Module):
    """Sample the random noises of NoisyLinear modules in the model."""
    done = False
    for m in model.modules():
        if isinstance(m, NoisyLinear):
            m.sample()
            done = True
    return done


def main():
    env, train_envs, test_envs = make_atari_env(
        task='MsPacmanNoFrameskip-v4',
        seed=0,
        training_num=10,
        test_num=10,
        scale=0,
        frame_stack=4,
    )
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN(*state_shape, action_shape, device).to(device)

    def RLPipeline():
        """Rainbow Pipeline."""

        def train_fn(epoch, env_step):
            eps = eps_train
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})
                # Assumes 25M frames or 250 epochs of training
                beta_anneal_step = 25000000
                beta_final = 1.
                beta = 0.01
                if env_step <= beta_anneal_step:
                    beta = beta - env_step / beta_anneal_step * \
                           (beta - beta_final)
                else:
                    beta = beta_final
                buffer.set_beta(beta)
                if env_step % 1000 == 0:
                    logger.write("train/env_step", env_step, {"train/beta": beta})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        def stop_fn(mean_rewards):
            if env.spec.reward_threshold:
                return mean_rewards >= env.spec.reward_threshold
            else:
                return False

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join('log/rainbow', "rainbow_death_memory_250.pth"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        # Neural Network
        net = Rainbow(*state_shape, action_shape).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        # Policy
        policy = RainbowPolicy(net, optim)
        # Prioritized Experience Buffer
        buffer = PrioritizedVectorReplayBuffer(
            total_size=500000,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=4,
            alpha=0.4,
            beta=0.01,
            weight_norm=True
        )

        # Collect the resultsc
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        logger = TensorboardLogger(SummaryWriter('log/rainbow'))
        train_collector.collect(n_step=batch_size * train_num)

        result = offpolicy_trainer(
            policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
            test_num, batch_size, update_per_step=0.1,  # 1 / step_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )
        print(f'Finished training! Use {result["duration"]}')

        torch.save(policy.state_dict(), 'rainbow_death_memory.pth')
        policy.load_state_dict(torch.load('rainbow_death_memory.pth'))
        policy.eval()
        policy.set_eps(eps_test)
        test_collector.reset()
        result = test_collector.collect(n_episode=30)
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    eps_train = 0.01
    eps_test = 0.001
    eps_train_final = 0.01
    train_num = 10
    test_num = 10
    batch_size = 32
    epoch = 250
    step_per_epoch = 100000
    step_per_collect = 10
    RLPipeline()


if __name__ == '__main__':
    main()