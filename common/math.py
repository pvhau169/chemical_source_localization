import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

map_x = 40
# real_range = np.array([[-300, 300], [-300, 300]])
real_range = np.array([[-300, 300], [-300, 300]])
simulated_range = np.array([map_x, map_x])
interval = np.sum(np.abs(real_range), axis=0) / simulated_range

centroid_x = np.array(range(map_x)) - map_x / 2
centroids = np.stack(np.meshgrid(centroid_x, centroid_x), axis=-1)
radius_ring = (real_range[0, 1] - real_range[0, 0]) / map_x / 2


def findAngleArcCos(A, B):
    old_shape = A.shape
    A, B = torch.from_numpy(A).reshape(-1, 2).type(torch.float), torch.from_numpy(B).reshape(-1, 2).type(torch.float)
    # print(A.shape, B.shape)
    cos_between = torch.sum(A * B, axis=-1) / (torch.norm(B, dim=-1) * torch.norm(A, dim=-1))
    error_angle = torch.arccos(cos_between)
    error_angle[torch.isnan(error_angle)] = -1  # bsize, time_len
    return error_angle.reshape(old_shape[:-1]).numpy()


def getAngle(vector, deg=False):
    vector = vector[:, 0] + vector[:, 1] * 1j
    angle = np.angle(vector, deg=deg)
    return angle


def findAngleNpAngle(A, B):
    old_shape = A.shape
    A, B = A.reshape(-1, 2).astype(np.float), B.reshape(-1, 2).astype(np.float)

    angle_between = getAngle(A) - getAngle(B)

    return angle_between.reshape(old_shape[:-1])


def getRotateMatrix(vectors, base=np.array([-1, 0])):
    base_vectors = np.repeat(base[np.newaxis, :], vectors.shape[0], axis=0)

    rotate_angle = findAngleNpAngle(base_vectors, vectors)
    rotate_sin = np.sin(rotate_angle)
    rotate_cos = np.cos(rotate_angle)

    rotate_matrix = np.stack([rotate_cos, -rotate_sin, rotate_sin, rotate_cos], axis=1).reshape(vectors.shape[0], 2, 2)
    return rotate_matrix


def agentPositionRespectWaterFlowForce(agent_pos, water_flow_force):
    rotate_matrix = getRotateMatrix(water_flow_force)
    agent_pos_res_water_flow = np.einsum('aec,abc->abe', rotate_matrix, agent_pos)

    rotated_water_flow = np.einsum('aec, ac -> ae', rotate_matrix, water_flow_force)

    return agent_pos_res_water_flow, rotated_water_flow


def agentPositionRespectSourcePosition(agent_pos, source_pos):
    rotate_matrix = getRotateMatrix(source_pos, base=np.array([1, 0]))
    rotated_source_pos = np.einsum('aec, ac -> ae', rotate_matrix, source_pos)

    agent_pos_res_source_pos = np.einsum('aec,abc->abe', rotate_matrix, agent_pos)
    agent_pos_res_source_pos = agent_pos_res_source_pos - rotated_source_pos[:, np.newaxis, :]

    return agent_pos_res_source_pos, rotated_source_pos


def getShownMap(agent_pos):
    # print(agent_pos.shape)
    agent_pos = agent_pos.reshape(-1, 2)
    agent_pos = (agent_pos - real_range[:, 0]) // interval - (map_x / 2)

    distance_map = torch.cdist(torch.tensor(agent_pos, dtype=float), torch.tensor(centroids.reshape(-1, 2),
                                                                                  dtype=float)).numpy()  # bsize, time_len, n_centroids
    distance_map = distance_map.reshape(distance_map.shape[0], map_x, map_x)

    shown_map = distance_map == distance_map.min(axis=(-1, -2))[..., np.newaxis, np.newaxis]

    # print(distance_map.shape, np.product(distance_map.shape), np.sum((distance_map <= radius_ring)))
    shown_map = shown_map * (distance_map <= 0.5)
    shown_map = shown_map.sum(axis=0)

    return shown_map


def processMap(agent_pos, source_pos, water_flow_force, done):
    def getValidPos(pos):
        return pos.reshape(-1, 2)[valid.reshape(-1)]

    agent_pos_res_water_flow, rotated_water_flow = agentPositionRespectWaterFlowForce(agent_pos.copy(), water_flow_force)
    agent_pos_res_source_pos, rotated_source_pos = agentPositionRespectSourcePosition(agent_pos.copy(), source_pos)

    valid = (1 - done).astype(np.bool)
    # print(valid.shape, agent_pos.shape)
    # print(agent_pos[:2, :5])
    # print(getValidPos(agent_pos)[:5])
    # print(getValidPos(agent_pos)[40:45])
    agent_centric_map = getShownMap(getValidPos(agent_pos))
    # plt.contour(agent_centric_map, cmap='magma')
    # plt.show()
    source_centric_map = getShownMap(getValidPos(agent_pos_res_source_pos))
    water_centric_map = getShownMap(getValidPos(agent_pos_res_water_flow))

    return agent_centric_map.tolist(), source_centric_map.tolist(), water_centric_map.tolist()


def shownMap(map, source = False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.contourf(map, cmap='magma')

    if source:
        ax.scatter([20], [20], s=200)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x - map_x / 2) * interval[0] * 0.1))
    ax.xaxis.set_major_formatter(ticks_x)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x - map_x / 2) * interval[0] * 0.1))
    ax.yaxis.set_major_formatter(ticks_x)

    ax.set_xlabel("distance (m)")
    ax.set_ylabel("distance (m)")

def saveShownMap(map_name, map, shown_map_path, map_save_name):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.contourf(map, cmap='magma')

    if "source" in map_name:
        ax.scatter([20], [20], s=200)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x - map_x / 2) * interval[0] * 0.1))
    ax.xaxis.set_major_formatter(ticks_x)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x - map_x / 2) * interval[0] * 0.1))
    ax.yaxis.set_major_formatter(ticks_x)

    extra_folder = '/epsilon/' if "epsilon" in map_name else '/free/'

    fig.savefig(shown_map_path + extra_folder + map_save_name + map_name + ".png")
