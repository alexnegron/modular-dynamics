import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class BinaryTrajectoryGenerator:
    def __init__(self, num_timesteps, dt=0.5, num_flips=5, omega_value=0.005, include_initial_position=False, **kwargs):
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.omega_value = omega_value
        self.num_flips = num_flips
        self.include_initial_position = include_initial_position

        self.kwargs = kwargs

    def generate_binary_omega(self):
        omegas =  self.omega_value * np.ones(self.num_timesteps)
        omegas[0] = self.omega_value * np.random.choice([-1,1]) # random initial direction

        # min_distance_between_flips = self.num_timesteps // (self.num_flips + 1)
        min_distance_between_flips = 50
        flip_times = []
        for _ in range(self.num_flips):
            if flip_times:
                start = flip_times[-1] + min_distance_between_flips
            else:
                start = min_distance_between_flips
            if start >= self.num_timesteps - min_distance_between_flips:
                continue
            flip_time = np.random.choice(range(start, self.num_timesteps - min_distance_between_flips))
            flip_times.append(flip_time)

        flip_times = np.sort(flip_times)

        flip_index = 0
        for t in range(1, self.num_timesteps):
            if flip_index < len(flip_times) and t == flip_times[flip_index]:
                omegas[t:] *= -1
                flip_index += 1
        return omegas

    def generate_trajectory(self):
        omegas = self.generate_binary_omega()
        theta0 = 0
        theta = theta0
        x0 = np.cos(theta0)
        y0 = np.sin(theta0)
        inputs = np.zeros((self.num_timesteps, 3 if self.include_initial_position else 1))
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
            if self.include_initial_position:
                inputs[j, 1] = x0
                inputs[j, 2] = y0

            targets[j, 0] = np.cos(theta)
            targets[j, 1] = np.sin(theta)

        return inputs, targets
    
class ConstantTrajectoryGenerator(BinaryTrajectoryGenerator):
    def generate_binary_omega(self):
        omegas = self.omega_value * np.ones(self.num_timesteps)
        return omegas

def generate_dataset(num_samples, num_timesteps, num_trajectories, trajectory_type='binary', include_initial_position=False, integrate_max=False, **kwargs):
    if integrate_max and num_trajectories < 2:
        raise ValueError("integrate_max version requires at least two trajectories")

    inputs = np.zeros((num_samples, num_timesteps, (3 if include_initial_position else 1) * num_trajectories))
    targets = np.zeros((num_samples, num_timesteps, 2 * num_trajectories + (1 if integrate_max else 0))) 

    if trajectory_type == 'binary':
        generator = BinaryTrajectoryGenerator(num_timesteps, include_initial_position=include_initial_position, **kwargs)
    elif trajectory_type == 'constant':
        generator = ConstantTrajectoryGenerator(num_timesteps, include_initial_position=include_initial_position, **kwargs)
    else:
        raise ValueError("Invalid trajectory_type. Expected 'binary' or 'constant'.")

    for i in range(num_samples):
        angular_positions = []
        for j in range(num_trajectories):
            inputs_single, targets_single = generator.generate_trajectory()
            inputs[i, :, (3 if include_initial_position else 1)*j:(3 if include_initial_position else 1)*(j+1)] = inputs_single
            targets[i, :, 2*j:2*(j+1)] = targets_single
            angular_positions.append(np.arctan2(targets_single[:, 1], targets_single[:, 0]))

        if integrate_max:
            targets[i, :, -1] = np.max(angular_positions, axis=0)  # max of the positions at each time point

    return inputs, targets


def plot_data(inputs, targets, num_timesteps, include_initial_position=False, sample_idx=0):
    num_trajectories = inputs.shape[2] // (3 if include_initial_position else 1)
    time = range(num_timesteps)

    # Plot omegas and positions for each trajectory
    for j in range(num_trajectories):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot omegas
        omega = inputs[sample_idx, :, (3 if include_initial_position else 1)*j]
        axs[0].plot(time, omega, label=f'Trajectory {j+1}')
        axs[0].set_title(f'Angular Velocities for Trajectory {j+1}')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('Angular Velocity')

        # Plot positions
        position = np.arctan2(targets[sample_idx, :, 2*j+1], targets[sample_idx, :, 2*j])
        axs[1].plot(time, position, label=f'Trajectory {j+1}')
        axs[1].set_title(f'Positions for Trajectory {j+1}')
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Position')

        plt.tight_layout()
        plt.show()

    # Plot max(positions) values over time
    if targets.shape[2] > 2 * num_trajectories:
        max_positions = targets[sample_idx, :, -1]
        plt.figure(figsize=(8, 4))
        plt.plot(time, max_positions)
        plt.title('Max Positions Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Max Position')
        plt.show()