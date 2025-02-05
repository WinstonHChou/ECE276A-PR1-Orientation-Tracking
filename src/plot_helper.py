from utils import *
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
from tqdm import tqdm
import transforms3d
import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def plot_euler_angles(Q, vicon_data, trained_Q, prefix):
    Q_to_V = jnp.zeros((3, Q.shape[1]))
    trained_Q_to_V = jnp.zeros((3, trained_Q[0, :].shape[0]))

    for v in range(Q.shape[1]):
        Q_to_V = Q_to_V.at[:, v].set(jnp.array(transforms3d.euler.quat2euler(Q[:, v])))
        trained_Q_to_V = trained_Q_to_V.at[:, v].set(jnp.array(transforms3d.euler.quat2euler(trained_Q[:, v])))

    if vicon_data:
        V = jnp.zeros((3, vicon_data['rots'].shape[2])) if vicon_data else None
        for v in range(vicon_data['rots'].shape[2]):
            V = V.at[:, v].set(jnp.array(transforms3d.euler.mat2euler(vicon_data['rots'][:, :, v]))) if vicon_data else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Dataset {prefix} Euler Angles')
    titles = ['Euler angle X (Roll)', 'Euler angle Y (Pitch)', 'Euler angle Z (Yaw)']
    for i in range(3):
        axes[i].set_title(titles[i])
        axes[i].plot(Q_to_V[i, :])
        axes[i].plot(trained_Q_to_V[i, :])
        if vicon_data:
            axes[i].plot(V[i, :])
            axes[i].legend(['Predicted Values from Omega',"Trained Motion model", "Values from VICON"])
        else:
            axes[i].legend(['Predicted Values from Omega',"Trained Motion model", ])
    plt.savefig(f'{prefix}_AngVel.jpg')
    plt.close()
    return trained_Q_to_V


def plot_acceleration(A, A_corrected, trained_A, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Dataset {prefix} Acceleration')
    titles = ['Acceleration X', 'Acceleration Y', 'Acceleration Z']
    for i in range(3):
        axes[i].set_title(titles[i])
        axes[i].plot(A[i, :])
        axes[i].plot(trained_A[i, :])
        axes[i].plot(A_corrected[i])
        axes[i].legend(["Predicted values from Quaternions", "Trained Observation model", 'Values from IMU', ])
    plt.savefig(f'{prefix}_Accel.jpg')
    plt.close()


def plot_cost(SUM2_loss, dataset_number):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.set_title('Loss for Dataset {}'.format(dataset_number))
    axes.plot(SUM2_loss)
    plt.savefig('{}_Cost.jpg'.format(dataset_number))
    plt.close()


def create_panorama(vicon_data, cam_data, rough=False):
    cam_time_map = {}
    HEIGHT, WIDTH = 960, 1920
    for ts in cam_data['ts'][0]:
        temp = np.abs(vicon_data['ts'][0] - ts)
        index = np.argmin(temp)
        cam_time_map[ts] = index

    phi, theta = np.linspace(90-(FOV_V/2), 90+(FOV_V/2)-1, int(PIXEL_V)) * np.pi/180, np.linspace(90-(FOV_H/2), 90+(FOV_H/2)-1, int(PIXEL_H))*np.pi/180
    cartesian_temp, cartesian = np.ones((PIXEL_V, PIXEL_H, 4)), np.ones((PIXEL_V, PIXEL_H, 3))
    for i in range(phi.shape[0]):
        cartesian_temp[i, :, 0] = np.cos(theta)
    for i in range(theta.shape[0]):
        cartesian_temp[:, i, 1] = np.cos(phi)
    for i in range(phi.shape[0]):
        cartesian_temp[i, :, 2] = np.sin(theta)
    for i in range(theta.shape[0]):
        cartesian_temp[:, i, 3] = np.sin(phi)

    cartesian[:, :, 0] = np.multiply(cartesian_temp[:, :, 3], cartesian_temp[:, :, 0])
    cartesian[:, :, 1] = np.multiply(cartesian_temp[:, :, 3], cartesian_temp[:, :, 2])
    cartesian[:, :, 2] = cartesian_temp[:, :, 1]
    print('Created cartesian coordinates map')

    world_frame_cartesian = np.zeros((PIXEL_V, PIXEL_H, 3, len(cam_time_map.keys())))
    for i in range(cam_data['cam'].shape[3]):
        world_frame_cartesian[:, :, :, i] = np.dot(cartesian, vicon_data['rots'][:, :, cam_time_map[cam_data['ts'][0][i]]])
    print('Created world frame')

    del cartesian_temp, cartesian, phi, theta, cam_time_map

    spherical_from_cartesian = np.zeros((PIXEL_V, PIXEL_H, 3, cam_data['cam'].shape[3]))

    spherical_from_cartesian_r = np.linalg.norm(world_frame_cartesian, axis=2)
    spherical_from_cartesian[:, :, 0,:] = spherical_from_cartesian_r  # rho => z
    spherical_from_cartesian[:, :, 1, :] = np.arctan2(world_frame_cartesian[:, :, 1, :], world_frame_cartesian[:, :, 0, :])  # theta => x
    spherical_from_cartesian[:, :, 2, :] = np.arccos(world_frame_cartesian[:, :, 2, :]/spherical_from_cartesian_r)  # phi => y
    del spherical_from_cartesian_r

    sx, sy = (2*np.pi/WIDTH), (np.pi/HEIGHT)
    spherical_from_cartesian[:, :, 1, :] += np.pi
    spherical_from_cartesian[:, :, 1, :] /= sx
    spherical_from_cartesian[:, :, 2, :] /= sy
    spherical_from_cartesian[:, :, 2, :] -= np.min(spherical_from_cartesian[:, :, 2, :])
    spherical_from_cartesian[:, :, 1, :] -= np.min(spherical_from_cartesian[:, :, 1, :])
    spherical_from_cartesian = spherical_from_cartesian.astype(np.int32)
    print('Created spherical projection map\nCreating image:')

    image = np.zeros((HEIGHT, WIDTH, 3)).astype(np.int32)
    for r in tqdm(range(0, cam_data['cam'].shape[3], 30 if rough else 1)):
        for i in range(cam_data['cam'].shape[0]):
            for j in range(cam_data['cam'].shape[1]):
                _, x, y = spherical_from_cartesian[i, j, :, r]
                image[y, x, :] = cam_data['cam'][i, j, :, r]
    print('Image created')
    return image