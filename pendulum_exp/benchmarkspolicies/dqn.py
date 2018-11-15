from itertools import chain
import torch
from abstract import ParametricFunction, Arrayable, Noise, StateDict, Policy

from convert import arr_to_th, th_to_arr
from stats import penalize_mean
from config import PolicyConfig
from policies.shared import SharedAdvantagePolicy
from mylog import log
from logging import info


class DiscreteDQNPolicy(Policy):
    def __init__(self,
                 qnet_function: ParametricFunction,
                 target_function: ParametricFunction,
                 qnet_noise: Noise,
                 policy_config: PolicyConfig,
                 device) -> None:
        super().__init__(policy_config)
        self._qnet_function = qnet_function.to(device)
        self._target_function = target_function.to(device)
        self._qnet_noise = qnet_noise

        # optimization/storing
        self._device = device
        self._optimizer = torch.optim.SGD(self._qnet_function.parameters(), lr=policy_config.lr * policy_config.dt)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, policy_config.lr_decay)
        # logging
        self._cum_loss = 0
        self._log = 100

    def act(self, obs: Arrayable):
        with torch.no_grad():
            pre_action = self._adv_noise.perturb_output(
                obs, function=self._qnet_function)
            self._adv_noise.step()
            return pre_action.argmax(axis=-1)

    def learn(self):
        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            try:
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done = self._sampler.sample()
                    # for now, discrete actions
                    indices = arr_to_th(action, self._device).long()

                    q = self._qnet_function(obs).gather(1, indices.view(-1, 1)).squeeze()
                    target = torch.max(self._target_function(next_obs), dim=1)[0].squeeze()

                    exp_q = arr_to_th(reward, self._device) + self._gamma ** self._dt 
                    if self._gamma == 1.:
                        assert (1 - done).all(), "Gamma set to 1. with a potentially episodic problem..."
                        exp_q += target.detach()
                    else:
                        done = arr_to_th(done.astype('float'), self._device)
                        exp_q += self._gamma ** self._dt * target#.detach()

                    loss = ((q - exp_q) ** 2).mean()
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    self._scheduler.step()

                    self._cum_loss += loss.item()
                    self._learn_count += 1

                log("Avg_adv_loss", self._cum_loss / self._learn_count, self._learn_count)
                info(f'At iteration {self._learn_count}, avg_loss: {self._cum_loss/self._learn_count}')

            except IndexError:
                # If not enough data in the buffer, do nothing
                pass

        if self.count % self.steps_btw_catchup == self.steps_btw_catchup -1:
            pass


    def train(self):
        self._train = True
        self._qnet_function.train()

    def eval(self):
        self._train = False
        self._qnet_function.eval()

    def value(self, obs: Arrayable):
        return th_to_arr(torch.max(self._qnet_function(obs), dim=1))

    def advantage(self, obs: Arrayable):
        q = self._qnet_function(obs)
        return th_to_arr(q - torch.max(q, dim=1))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers.load_state_dict(state_dict['optimizer'])
        self._qnet_function.load_state_dict(state_dict['qnet_function'])
        self._target_function.load_state_dict(state_dict['target_function'])
        self._schedulers.last_epoch = state_dict['iteration']

    def state_dict(self):
        return {
            "optimizer": self._optimizers.state_dict(),
            "qnet_function": self._adv_function.state_dict(),
            "target_function": self._val_function.state_dict(),
            "iteration": self._scheduler.last_epoch}
