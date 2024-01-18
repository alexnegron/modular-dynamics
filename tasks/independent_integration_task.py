import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def OU_process(alpha, mu, sigma, dt, num_timesteps, bound, x):
    for t in range(1, num_timesteps):
        dx = alpha * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[t] = np.clip(x[t-1] + dx, -bound, bound)
    return x

def OU(alpha=0, mu=0, sigma=1, dt=0.1, num_timesteps=100, bound=0.5):
    x = np.zeros(num_timesteps)
    x[0] = 0
    return OU_process(alpha, mu, sigma, dt, num_timesteps, bound, x)

def zero_inflated_OU(alpha=0, mu=0, sigma=1, dt=0.1, num_timesteps=100, bound=0.5, p_zero=0.1):
    x = np.zeros(num_timesteps)
    x[0] = 0
    for t in range(1, num_timesteps):
        if np.random.uniform(0, 1) < p_zero:
            dx = 0
        else:
            dx = alpha * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[t] = x[t-1] + dx
        x[t] = np.clip(x[t], -bound, bound)
    return x

def generate_binary_omega(num_timesteps, flip_freq=0.01, refractory_period_ratio=1, value=0.005):
    omegas = value*np.ones(num_timesteps)
    refractory_period = int(refractory_period_ratio * num_timesteps)
    last_flip = -refractory_period
    for t in range(1, num_timesteps):
        if t - last_flip >= refractory_period and np.random.uniform(0, 1) < flip_freq:
            omegas[t] = -omegas[t-1]
            last_flip = t
        else:
            omegas[t] = omegas[t-1]
    return omegas

# def generate_dataset(num_samples, num_timesteps, omega_process='zero_inflated_OU', dt=0.5, **kwargs):
#     inputs = np.zeros((num_samples, num_timesteps, 3))
#     targets = np.zeros((num_samples, num_timesteps, 2))

#     # np.random.seed(kwargs.get('seed', 42))

#     for i in range(num_samples):
#         if omega_process == 'zero_inflated_OU':
#             omegas = zero_inflated_OU(num_timesteps=num_timesteps, **kwargs)
#         elif omega_process == 'binary':
#             omegas = generate_binary_omega(num_timesteps, **kwargs)
#         else:
#             raise ValueError(f'Unknown omega_process: {omega_process}')

#         theta0 = 0
#         theta = theta0
#         x0 = np.cos(theta0)
#         y0 = np.sin(theta0)

#         for j in range(num_timesteps):
#             omega = omegas[j]
#             k1 = omega * dt
#             k2 = (omega + 0.5 * k1) * dt
#             k3 = (omega + 0.5 * k2) * dt
#             k4 = (omega + k3) * dt
#             dtheta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
#             theta = (theta + dtheta) % (2 * np.pi)

#             inputs[i, j, 0] = omega
#             inputs[i, j, 1] = x0
#             inputs[i, j, 2] = y0

#             targets[i, j, 0] = np.cos(theta)
#             targets[i, j, 1] = np.sin(theta)

#     return inputs, targets

def plot_helper(x, y, title, x_label, y_label, plot_type='plot'):
    if plot_type == 'hist':
        plt.hist(x, bins=30)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_data(inputs, targets, num_timesteps, sample_idx=0):
    fig, axs = plt.subplots(1, 2, figsize=(8,4), gridspec_kw={'width_ratios': [1, 0.05]})
    circle = plt.Circle((0,0), radius=1, fill=False)
    axs[0].add_patch(circle)
    axs[0].set_xlim([-1.1, 1.1])
    axs[0].set_ylim([-1.1, 1.1])
    axs[0].set_aspect('equal')

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_timesteps))

    sample_idx = 0

    for i in range(num_timesteps):
        axs[0].plot(targets[sample_idx, i, 0], targets[sample_idx, i, 1], marker='o', markersize=8, color=colors[i])

    sc = axs[1].scatter([],[], c=[], cmap='viridis')
    clb = plt.colorbar(sc, cax=axs[1])
    clb.ax.invert_yaxis()
    clb.set_label('Time')

    plt.tight_layout()
    plt.show()

    plot_helper(range(num_timesteps), np.arctan2(targets[sample_idx, :, 1], targets[sample_idx, :, 0]), 'Direction (rad)', 'Timestep', 'Direction', 'plot')
    plt.yticks(np.linspace(-np.pi, np.pi, 3), [r'$-\pi$', r'$0$', r'$\pi$'])

    velocities = inputs[sample_idx, :, 0].reshape(-1,1)
    plot_helper(velocities, None, 'Distribution of Angular Velocities', 'Angular Velocity', 'Count', 'hist')
    plot_helper(range(num_timesteps), velocities, 'Angular Velocities', 'Timestep', 'Angular Velocity', 'plot')



class BinaryTrajectoryGenerator:
    def __init__(self, num_timesteps, dt=0.5, flip_rate=0.01, omega_value=0.005, gain=False, **kwargs):
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.flip_rate = flip_rate
        self.gain = gain
        if gain == True:
            self.g = np.random.uniform(0.5, 1.5)
        else: 
            self.g = 1.0

        self.omega_value = omega_value * self.g if gain else omega_value
        self.kwargs = kwargs

    def generate_binary_omega(self):
        omegas =  self.omega_value * np.ones(self.num_timesteps)
        initial_direction = np.random.choice([-1, 1])

        # Generate flip times following an exponential distribution: E[num_flips] = flip_rate * num_timesteps
        flip_times = np.cumsum(np.random.exponential(scale=1/self.flip_rate, size=self.num_timesteps))
        flip_times = flip_times[flip_times < self.num_timesteps].astype(int) 
        
        omegas[:flip_times[0]] *= initial_direction # flip initial direction negative (sometimes)

        flip_index = 1
        for t in range(flip_times[0], self.num_timesteps):
            if flip_index < len(flip_times) and t == flip_times[flip_index]:
                omegas[t:] *= -1
                flip_index += 1

        return omegas

    def generate_trajectory(self):
        omegas = self.generate_binary_omega()
        theta0 = 0
        theta = theta0
        inputs = np.zeros((self.num_timesteps, 2 if self.gain else 1))
        targets = np.zeros((self.num_timesteps, 2))

        for j in range(self.num_timesteps):
            omega = omegas[j]
            k1 = omega * self.dt
            k2 = (omega + 0.5 * k1) * self.dt
            k3 = (omega + 0.5 * k2) * self.dt
            k4 = (omega + k3) * self.dt
            dtheta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            theta = (theta + dtheta) % (2 * np.pi)

            inputs[j, 0] = omega 
            if self.gain: 
                inputs[j, 1] = self.g

            targets[j, 0] = np.cos(theta)
            targets[j, 1] = np.sin(theta)
            
        return inputs, targets
    
class ConstantTrajectoryGenerator(BinaryTrajectoryGenerator):
    def generate_binary_omega(self):
        omegas = self.omega_value * np.ones(self.num_timesteps) * self.alpha 
        return omegas

def generate_dataset(num_samples, num_timesteps, num_trajectories, trajectory_type='binary', **kwargs):
    if trajectory_type == 'binary':
        generator = BinaryTrajectoryGenerator(num_timesteps, **kwargs)
    elif trajectory_type == 'constant':
        generator = ConstantTrajectoryGenerator(num_timesteps, **kwargs)
    else:
        raise ValueError("Invalid trajectory_type. Expected 'binary' or 'constant'.")

    inputs = np.zeros((num_samples, num_timesteps, (2 if generator.gain else 1) * num_trajectories))
    targets = np.zeros((num_samples, num_timesteps, 2 * num_trajectories))

    for i in range(num_samples):
        for j in range(num_trajectories):
            inputs_single, targets_single = generator.generate_trajectory()
            inputs[i, :, (2 if generator.gain else 1)*j:(2 if generator.gain else 1)*(j+1)] = inputs_single
            targets[i, :, 2*j:2*(j+1)] = targets_single

    return inputs, targets