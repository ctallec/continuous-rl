# pylint: disable=too-many-arguments, too-many-locals
""" Learning V-function. """
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as f
from data import generate_trajectories, OrnsteinUlhenbeckParameters
from noise import OrnsteinUlhenbeck
from model import Model
from femvfunction import solve_exact
import matplotlib.pyplot as plt

def to_tensor(x: np.ndarray, device) -> torch.Tensor:
    """ From numpy to tensor. """
    return torch.from_numpy(x).float().to(device)

def train(model: Model,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          data: np.ndarray,
          nb_steps: int,
          begin_index: int,
          true_gamma: float,
          beta: float,
          dt: float,
          thresh: float,
          ou_process: OrnsteinUlhenbeck,
          device) -> List[float]:
    """ Train. """
    losses = []
    for i in range(nb_steps):
        # index = begin_index + i
        index = np.random.randint(0, data.shape[1] - 1)
        np_next_state = data[:, index + 1]
        reward = to_tensor(reward_function(np_next_state, beta, dt, thresh), device)
        state = to_tensor(data[:, index], device)
        next_state = to_tensor(np_next_state, device)
        # you need the copy, otherwise you modify data
        ou_process.reset(data[:, index].copy())
        fake_next_state = to_tensor(ou_process.step(), device)

        V = model(state).squeeze()
        V_next = model(next_state).squeeze()
        V_fake_next = model(fake_next_state).squeeze()
        if True:
            V_mixed_next = V_next.detach()
        else:
            V_mixed_next = V_fake_next - V_fake_next.detach() + V_next.detach()
        
        loss = f.mse_loss(V - true_gamma * V_mixed_next, reward) / dt # + 0 * hessian_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
    return losses

def zero_nan(x: torch.Tensor) -> torch.Tensor:
    """ Turn nan to zeros. """
    mask = (x != x)
    x[mask] = 0
    return x

def evaluate(model: Model, state_limits: Tuple[float, float], pixel_width: int,
             beta: float, dt: float, train_data: np.ndarray, test_data: np.ndarray,
             thresh: float, exact_solution, gamma: float, device):
    """ Eval. """
    oned_states = np.linspace(*state_limits, num=pixel_width)
    x, y = np.meshgrid(oned_states, oned_states)
    exact_V = exact_solution(x, y)
    states = np.stack([x, y], axis=-1)

    with torch.no_grad():
        exact_test_V = zero_nan(
            to_tensor(exact_solution(test_data[..., 0], test_data[..., 1]), device))
        test_V = model(to_tensor(test_data, device)).squeeze()
        delta_V = exact_test_V - test_V
        l2_test_loss = (delta_V ** 2).mean().item()
        dirichlet_test_loss = ((((1 - (1 - gamma) * dt) * delta_V[:, 1:] -
                                 delta_V[:, :-1]) / dt) ** 2).mean().item()

    print(f"Evaluation -- l2_loss: {l2_test_loss} -- dirichlet: {dirichlet_test_loss}")
    rs = reward_function(states, beta, dt, thresh)
    with torch.no_grad():
        Vs = model(to_tensor(states, device)).cpu().numpy().squeeze()
    plt.clf()
    plt.subplot(151)
    plt.imshow(rs)

    plt.subplot(152)
    plt.imshow((Vs - exact_V) ** 2) # pylint: disable=no-member

    plt.subplot(153)
    plt.imshow(Vs) # pylint: disable=no-member

    plt.subplot(154)
    plt.imshow(exact_V)

    plt.subplot(155)
    plt.plot(train_data[0, :, 0], train_data[0, :, 1])

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
        thresh: float,
        exact_solution):
    """ Run. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = ou_params.dt

    data, ou_process = generate_trajectories(
        batch_size=batch_size, T=T,
        w=w, ornstein_ulhenbeck_params=OrnsteinUlhenbeckParameters(dt, sigma, theta))
    print(f"Maximum and minimum values achieved in data: {data.max()}, {data.min()}")
    test_data, _ = generate_trajectories(
        batch_size=batch_size, T=int(10 / ou_params.dt),
        w=w, ornstein_ulhenbeck_params=OrnsteinUlhenbeckParameters(dt, sigma, theta))

    model = Model(nb_hiddens, nb_inputs).to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=2 * dt)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1 / np.power(t * dt / 2 + 1, 3/5))

    nb_epochs = (T - 1) // steps_per_epoch
    step = 0
    for e in range(nb_epochs):
        if e % 100 == 0:
            evaluate(model, (-5, 5), 100, beta, dt, data,
                     test_data, thresh, exact_solution, gamma, device)
        losses = train(model, optimizer, scheduler, data, steps_per_epoch, step,
                       1 - (1 - gamma) * dt, beta, dt, thresh, ou_process, device)
        print(f"At epoch {e}, average loss: {np.mean(losses)}, std loss: {np.std(losses)}")

def reward_function(state: np.ndarray, beta: float, dt: float, thres: float) -> np.ndarray:
    """ Reward function. """
    threshold = np.sum(state[..., :] ** 2, axis=-1) < thres
    return (1 / (1 + np.exp(-beta*state[..., 0])) - 1 / 2) * dt * threshold

if __name__ == '__main__':
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', default=1e-2, type=float)
    args = parser.parse_args()

    batch_size = 32
    T = 1000000
    omega = .1
    dt = args.dt
    sigma = .3
    theta = .01
    beta = 30
    nb_hiddens = 32
    nb_inputs = 2
    gamma = 1
    thresh = 2
    steps_per_epoch = 100

    _, _, exact_solution = solve_exact(sigma, theta, omega, beta, gamma, thresh, (-5, 5))

    run(batch_size, T, omega, nb_hiddens, nb_inputs,
        gamma, OrnsteinUlhenbeckParameters(dt, sigma, theta),
        beta, steps_per_epoch, thresh, exact_solution)
