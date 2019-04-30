"""
MDP - Markov Decision Process class for the Structures Project
Author: John Jackson
"""

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.linalg import eigh


class FrequencyChooserMDP:

    def __init__(self, blackbox_sys):

        # Discretized control space - each action is a frequency. Cost is associated with the frequency.
        self.control_space = np.linspace(0, 100, num=1000)

        self.state_history = []
        self.reward_history = []

        # Blackbox_sys is an anonymous function.
        self.real_system = blackbox_sys
        self.system_model = SystemModel(2)

    def take_action(self, action):

        frequency = action[0]
        trajectory_simulated = self.system_model.simulate_trajectory(frequency)

        # Take action here!
        trajectory_real = self.excite_real_system(frequency)
        # Calculate the reward!

        errors = np.power(trajectory_real - trajectory_simulated, 2)
        x_new = np.mean(np.sqrt(errors))

        reward = self.calculate_reward(errors, action)
        self.state_history.append(np.copy(x_new))

        self.update_parameters(trajectory_real, action)

        self.reward_history.append(reward)

        return reward

    def excite_real_system(self, frequency):

        trajectory_real = self.real_system(frequency)
        return trajectory_real

    def calculate_reward(self, errors, action):

        sim_reward = np.log10(1./np.mean(np.abs(errors)))

        # Control action costs
        control_cost = -np.power(action[0], 1)
        total_reward = sim_reward + control_cost

        return total_reward

    def update_parameters(self, trajectory_real, action):

        def test_parameters(params, trajectory_real):

            system_model = SystemModel(2)

            m = params[0]
            Ig = params[1]
            e = 0.5                 #Assume we know e
            ku = params[3]
            kt = params[4]

            system_model.M = np.array([[m, -m * e], [-m * e, m*e**2 + Ig]])
            system_model.K = np.array([[ku, 0], [0, kt]])

            trajectory_sim = system_model.simulate_trajectory(action[0])

            mse = np.mean(np.sqrt(np.power(trajectory_real - trajectory_sim, 2)))

            return mse

        params_to_try = 10000

        best_params = self.system_model.params
        best_mse = test_parameters(best_params, trajectory_real)

        for i in range(0, params_to_try):

            m = 5*np.random.rand(1)[0]
            Ig = 20*np.random.rand(1)[0]
            e = 0.5                         #Assume we know e
            ku = 20*np.random.rand(1)[0]
            kt = 20*np.random.rand(1)[0]

            params = [m, Ig, e, ku, kt]

            mse = test_parameters(params, trajectory_real)

            if mse < best_mse:
                best_params = params

        print('Best MSE: {}'.format(best_mse))
        print('Best parameters: {}'.format(best_params))
        print('True parameters: {}'.format([1, 10, 0.5, 2, 10]))
        # Then update parameters
        self.system_model.update_model(best_params)

    def reset_mdp(self):
        self.state_history = []


class SystemModel(object):

    def __init__(self, dims):

        self.params = [1, 1, 1, 1, 1]

        self.M = np.eye(dims)
        # self.C = np.random.random_integers(0, 100, (dims, dims))
        self.K = np.eye(dims)

        self.noisydata = False #change to turn on/off noise in trajectories

    def update_model(self, params):

        # print('Old M and K:{} {}'.format(self.M, self.K))
        self.params = params
        self.M = np.array([[params[0], -params[0]*params[2]], [-params[0]*params[2],params[0]*params[2]**2 + params[1]]])
        self.K = np.array([[params[3], 0], [0, params[4]]])
        # print('New M and K:{} {}'.format(self.M, self.K))

    def simulate_trajectory(self, forced_frequency):

        #Spectral Expansion Technique to simulate forced trajectories
        t = np.linspace(0., 10., 100) #simulate for 10 seconds
        evs, ems = eigh(self.M, self.K, eigvals_only=False)

        em1 = ems[:, 0]
        em2 = ems[:, 1]
        ef1 = evs[0]
        ef2 = evs[1]
        e = 0.5
        wbar = forced_frequency

        A = lambda xs, e: xs[0] - e * xs[1]
        den = lambda ws, wbar : 1. / (ws ** 2 - wbar ** 2)
        rho = lambda A, den : A * den

        n1 = lambda t, em1, ef1, e, wbar:  A(em1, e) * den(ef1, wbar) * (np.cos(wbar * t) - np.cos(ef1* t))
        n2 = lambda t, em2, ef2, e, wbar:  A(em2, e) * den(ef2, wbar) * (np.cos(wbar * t) - np.cos(ef2* t))

        u = em1[0] * n1(t, em1, ef1, e, wbar) + em2[0] * n2(t, em2, ef2, e, wbar)
        theta = em1[1] * n1(t, em1, ef1, e, wbar) + em2[1] * n2(t, em2, ef2, e, wbar)

        trajectory = np.vstack((u, theta))

        if self.noisydata == True:
            trajectory = self.inject_noise(trajectory)

        return trajectory

    def inject_noise(self, sim_traj):

        u_traj = sim_traj[0]
        theta_traj = sim_traj[1]

        up = max(sim_traj[0])
        um = min(sim_traj[0])

        tp = max(sim_traj[1])
        tm = min(sim_traj[1])

        #create noise vector for u:
        sig_u = (up - um) / 2
        u_noise = np.random.normal(scale = sig_u, size = u_traj.shape)

        #create noise vector for theta:
        sig_theta = (tp - tm) / 2
        theta_noise = np.random.normal(scale = sig_theta, size = theta_traj.shape)

        u = u_traj + u_noise
        theta = theta_traj + theta_noise

        trajectory = np.vstack((u, theta))

        return trajectory


def real_system(frequency):

    #Blackbox real system with real parameters
    m = 1.
    Ig = 10.
    e = 0.5
    ku = 2.
    kt = 10.

    system_model = SystemModel(2)
    system_model.noisydata = True
    system_model.M = np.array([[m, -m*e], [-m*e, m*e**2 + Ig]])
    system_model.K = np.array([[ku, 0], [0, kt]])

    trajectory = system_model.simulate_trajectory(frequency)

    return trajectory


if __name__ == '__main__':

    np.random.seed(11)

    mdp = FrequencyChooserMDP(real_system)
    actions = [(0.3,), (3.,), (30.,), (5.,), (0.4,)]

    for action in actions:
        reward = mdp.take_action(action)

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(mdp.state_history, 'k-')
    plt.ylabel('Best MSE')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(mdp.reward_history), 'k-')
    plt.xlabel('Action Number')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    plt.show()

    plt.tight_layout()

    plt.savefig('update.png')
