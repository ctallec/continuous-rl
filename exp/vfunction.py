# pylint: disable=too-many-arguments, too-many-locals
""" Learning V-function. """
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as f
from data import generate_trajectories, OrnsteinUlhenbeckParameters
from model import Model
import matplotlib.pyplot as plt

def to_tensor(x: np.ndarray, device) -> torch.Tensor:
    """ From numpy to tensor. """
    return torch.from_numpy(x).float().to(device)

def train(model: Model,
          optimizer: torch.optim.Optimizer,
          data: np.ndarray,
          nb_steps: int,
          begin_index: int,
          true_gamma: float,
          beta: float,
          dt: float,
          thresh: float,
          device) -> List[float]:
    """ Train. """
    losses = []
    for i in range(nb_steps):
        np_next_state = data[:, begin_index + i + 1]
        reward = to_tensor(reward_function(np_next_state, beta, dt, thresh), device)
        state = to_tensor(data[:, begin_index + i], device)
        next_state = to_tensor(np_next_state, device)
        next_next_state = to_tensor(data[:, begin_index + i + 2], device)

        V = model(state).squeeze()
        V_next = model(next_state).squeeze()
        V_next_next = model(next_next_state).squeeze()
        expected_V = reward + true_gamma * V_next
        hessian_loss = f.mse_loss((V_next_next - V_next).detach(), V_next - V) / (dt ** 4)
        loss = f.mse_loss(V, expected_V.detach()) / (dt ** 2) # + 0 * hessian_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

def evaluate(model: Model, state_limits: Tuple[float, float], pixel_width: int,
             beta: float, dt: float, train_data: np.ndarray, test_data: np.ndarray,
             thresh: float, device):
    """ Eval. """
    oned_states = np.linspace(*state_limits, num=pixel_width)
    states = np.stack(np.meshgrid(oned_states, oned_states), axis=-1)
    rs = reward_function(states, beta, dt, thresh)
    with torch.no_grad():
        Vs = model(to_tensor(states, device)).cpu().numpy().squeeze()
    plt.clf()
    plt.subplot(141)
    plt.imshow(rs) # pylint: disable=no-member

    plt.subplot(142)
    plt.imshow(Vs) # pylint: disable=no-member
    print(rs)
    print(Vs)

    plt.subplot(143)
    plt.plot(train_data[0, :, 0], train_data[0, :, 1])

    plt.subplot(144)
    plt.plot(test_data[0, :, 0], test_data[0, :, 1])
    plt.show()
    plt.pause(.01)

def run(batch_size: int,
        T: int,
        w: float,
        nb_hiddens: int,
        nb_inputs: int,
        gamma: float,
        ou_params: OrnsteinUlhenbeckParameters, beta: float,
        steps_per_epoch: int,
        thresh: float):
    """ Run. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = ou_params.dt

    data = generate_trajectories(
        batch_size=batch_size, T=T,
        w=w, ornstein_ulhenbeck_params=OrnsteinUlhenbeckParameters(dt, sigma, theta))
    print(f"Maximum and minimum values achieved in data: {data.max()}, {data.min()}")
    test_data = generate_trajectories(
        batch_size=batch_size, T=int(10 / ou_params.dt),
        w=w, ornstein_ulhenbeck_params=OrnsteinUlhenbeckParameters(dt, sigma, theta))

    model = Model(nb_hiddens, nb_inputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1 * dt)

    nb_epochs = (T - 1) // steps_per_epoch
    step = 0
    for e in range(nb_epochs):
        if e % 10 == 0:
            evaluate(model, (-5, 5), 50, beta, dt, data, test_data, thresh, device)
        losses = train(model, optimizer, data, steps_per_epoch, step,
                       gamma ** dt, beta, dt, thresh, device)
        print(f"At epoch {e}, average loss: {np.mean(losses)}, std loss: {np.std(losses)}")

def reward_function(state: np.ndarray, beta: float, dt: float, thres: float) -> np.ndarray:
    """ Reward function. """
    threshold = np.sum(state[..., :] ** 2, axis=-1) < thres
    return 1 / (1 + np.exp(-beta*state[..., 0])) * dt * threshold

if __name__ == '__main__':
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', default=1e-2, type=float)
    args = parser.parse_args()

    batch_size = 32
    T = 100000
    omega = .1
    dt = args.dt
    sigma = .3
    theta = .01
    beta = 30
    nb_hiddens = 32
    nb_inputs = 2
    gamma = .01
    thresh = 2
    steps_per_epoch = 100

    run(batch_size, T, omega, nb_hiddens, nb_inputs,
        gamma, OrnsteinUlhenbeckParameters(dt, sigma, theta),
        beta, steps_per_epoch, thresh=thresh)
