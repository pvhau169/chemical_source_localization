import random

import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt

from simulation.Rectangle import Rectangle
from scipy import signal
from celluloid import Camera
from tqdm.notebook import tqdm
import site

np.set_printoptions(precision=2, suppress=True)


def applyKernel(matrix, kernel):
    dx = signal.convolve(matrix, kernel.T, 'same')
    dy = signal.convolve(matrix, kernel, 'same')

    return np.dstack((dx, dy))


def first_derivative(matrix):
    dx = matrix[2:, 1:-1] - matrix[0:-2, 1:-1]
    dy = matrix[1:-1, 2:] - matrix[1:-1, 0:-2]
    return np.dstack((dx, dy))


def second_derivative(matrix):
    dx = matrix[2:, 1:-1] - 2 * matrix[1:-1, 1:-1] + matrix[0:-2, 1:-1]
    dy = matrix[1:-1, 2:] - 2 * matrix[1:-1, 1:-1] + matrix[1:-1, 0:-2]
    return np.dstack((dx, dy))


class WindModel:
    def __init__(self,
                 sim_region=Rectangle([(0, 300), (-50, 50)]),
                 map_x=15,
                 map_y=5,
                 velocity_init_vector=[5, 0],
                 limit_time=200):
        self.sim_region = sim_region
        self.map_x, self.map_y = map_x, map_y

        # diffusion constant
        self.k_x = 5
        self.k_y = 5

        # compute grid node spacing
        self.max_u = 10
        self.max_v = 3
        self.velocity_init_vector = velocity_init_vector
        self.dx = self.sim_region.width / (map_x + 1)
        self.dy = self.sim_region.height / (map_y + 1)

        # velocity field values
        self.x_points = np.linspace(self.sim_region.x_min, self.sim_region.x_max, map_x)
        self.y_points = np.linspace(self.sim_region.y_min, self.sim_region.y_max, map_y)
        self.velocity = np.ones(
            (self.map_x + 2, self.map_y + 2, len(self.velocity_init_vector))) * self.velocity_init_vector

        self.corner_means = np.array(self.velocity_init_vector).repeat(4)

        self._ramp_x = np.linspace(0., 1., map_x + 2)
        self._ramp_y = np.linspace(0., 1., map_y + 2)
        # interpolators parameters
        self.interp_x, self.interp_y = None, None
        self.set_interp = False

        self.limit_time = limit_time + 30
        # self.limit_time = limit_time + 30
        self.t = -1

    def initCorner(self):
        for time_step in range(self.limit_time):
            self.noise_gen.update(dt=0.2)
            self.corner.append(self.noise_gen.output + self.corner_means)
            # self.corner.append(np.random.random(8) + self.corner_means)

    def randomWaterFlow(self):
        self.velocity_init_vector = np.array(
            [random.uniform(0, self.max_u) + 3, random.uniform(-self.max_v, self.max_v)])

    def reset(self, create_velo=True):
        self.t = -1
        self.velocity = np.ones(
            (self.map_x + 2, self.map_y + 2, len(self.velocity_init_vector))) * self.velocity_init_vector

        # intialize corner
        noise_damp = 0.1
        noise_gain = 2
        use_original_noise_updates = False
        noise_bandwidth = 0.2
        self.noise_gen = ColouredNoiseGenerator(
            np.zeros((2, 8)), noise_damp, noise_bandwidth, noise_gain,
            use_original_noise_updates)
        self.corner = []
        self.initCorner()
        if create_velo:
            self.createVelocity()

    def createVelocity(self):
        self.velocity_field_created = []
        for i in range(self.limit_time):
            # print(i, " create")
            self.update()
            # print(self.velocity_field[0][0])
            self.velocity_field_created.append(self.velocity_field.copy())

    def setInterp(self, t):
        self.interp_x = interp.RectBivariateSpline(self.x_points, self.y_points,
                                                   self.velocity_field_created[t][:, :, 0])
        self.interp_y = interp.RectBivariateSpline(self.x_points, self.y_points,
                                                   self.velocity_field_created[t][:, :, 1])

    @property
    def velocity_field(self):
        return self.velocity[1:-1, 1:-1]

    def getTurbulenceVector(self, pos, t):
        x, y = pos
        if t != self.t:
            self.t = t
            self.setInterp(t)
        x_mag = self.interp_x(x, y).item()
        y_mag = self.interp_y(x, y).item()
        return np.array([x_mag, y_mag])

    def setBoundary(self):
        # print(self.t)

        u_tl, u_tr, u_bl, u_br, v_tl, v_tr, v_bl, v_br = self.corner[self.t]

        self.velocity[:, 0, 0] = u_tl + self._ramp_x * (u_tr - u_tl)  # u top edge
        self.velocity[:, -1, 0] = u_bl + self._ramp_x * (u_br - u_bl)  # u bottom edge
        self.velocity[0, :, 0] = u_tl + self._ramp_y * (u_bl - u_tl)  # u left edge
        self.velocity[-1, :, 0] = u_tr + self._ramp_y * (u_br - u_tr)  # u right edge

        self.velocity[:, 0, 1] = v_tl + self._ramp_x * (v_tr - v_tl)  # v top edge
        self.velocity[:, -1, 1] = v_bl + self._ramp_x * (v_br - v_bl)  # v bottom edge
        self.velocity[0, :, 1] = v_tl + self._ramp_y * (v_bl - v_tl)  # v left edge
        self.velocity[-1, :, 1] = v_tr + self._ramp_y * (v_br - v_tr)  # v right edge

    def update(self, dt=1):
        self.t += 1
        # print(self.t, " update")
        self.setBoundary()
        first_kernel = np.atleast_2d([1, 0, -1])
        second_kernel = np.atleast_2d([1, -2, 1])
        du = applyKernel(self.velocity[:, :, 1], first_kernel) / (2 * self.dx)
        dv = applyKernel(self.velocity[:, :, 1], first_kernel) / (2 * self.dy)

        d2u = applyKernel(self.velocity[:, :, 0], second_kernel) / (self.dx ** 2)
        d2v = applyKernel(self.velocity[:, :, 1], second_kernel) / (self.dy ** 2)

        du_dt = -self.velocity * du + 0.3 * np.array([self.k_x, self.k_y]) * d2u
        dv_dt = -self.velocity * dv + 0.3 * np.array([self.k_x, self.k_y]) * d2v

        du_dt = np.sum(du_dt, axis=-1)
        dv_dt = np.sum(dv_dt, axis=-1)

        max_du_dt = 3
        max_dv_dt = 3

        self.velocity[:, :, 0] = np.clip(du_dt, -max_du_dt, max_du_dt)
        self.velocity[:, :, 1] = np.clip(dv_dt, -max_dv_dt, max_dv_dt)

        velocity_dt = np.dstack((du_dt, dv_dt))
        self.velocity += velocity_dt * dt

        max_du = self.max_u * 2
        max_dv = self.max_v * 2
        self.velocity[:, :, 0] = np.clip(self.velocity[:, :, 0], -max_du, max_du)
        self.velocity[:, :, 1] = np.clip(self.velocity[:, :, 1], -max_dv, max_dv)
        self.set_interp = False

    def createGif(self, time_step_max=30, file_name='gif/water_flow.gif'):
        self.reset()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title('Simulation time = ---- seconds')

        camera = Camera(fig)
        for time_step in tqdm(range(self.limit_time)):
            self.update()
            ax.quiver(self.x_points, self.y_points,
                      self.velocity_field.T[0],
                      self.velocity_field.T[1])
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', fps=4)


class ColouredNoiseGenerator(object):

    def __init__(self, init_state, damping=0.1, bandwidth=0.2, gain=1.,
                 use_original_updates=False, rng=None):

        rng = np.random
        # set up state space matrices
        self.a_mtx = np.array([
            [0., 1.], [-bandwidth ** 2, -2. * damping * bandwidth]])
        self.b_mtx = np.array([[0.], [gain * bandwidth ** 2]])
        # initialise state
        self.state = init_state
        self.rng = rng
        self.use_original_updates = use_original_updates

    @property
    def output(self):
        return self.state[0, :]

    def update(self, dt):
        # get normal random input
        n = self.rng.normal(0, 0.5, size=(1, self.state.shape[1]))
        if self.use_original_updates:
            # apply Farrell et al. (2002) update
            self.state += dt * (self.a_mtx.dot(self.state) + self.b_mtx.dot(n))
        else:
            # apply update with Euler-Maruyama integration
            self.state += (
                    dt * self.a_mtx.dot(self.state) + self.b_mtx.dot(n) * dt ** 0.5)
