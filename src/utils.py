from quaternion_helper import *
from jax.numpy.linalg import norm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import transforms3d
from transforms3d.euler import euler2mat, mat2euler, quat2euler
import jax.numpy as jnp
from jax import grad
from jax import config
config.update("jax_enable_x64", True)

# Voltage Units: mV
V_REF = 3300                                # mV
GRAVITY = 9.81                              # m/s^2
MAXIMUM_VALUE = 1023                        # 10 bit
GYRO_SENSITIVITY = 3.33 / (jnp.pi / 180.0)  # mV/(rad/sec)
ACCEL_SENSITIVITY = 300                     # mV/g
FOV_H = 60                                  # deg
FOV_V = 45                                  # deg
PIXEL_H = 320                               # pixel
PIXEL_V = 240                               # pixel

def degToRad(value):
    return value * jnp.pi / 180.0

def radToDeg(value):
    return value / (jnp.pi / 180.0)

def calculateIMUBias(end_index, data_row):
    return jnp.average(data_row[1:, :end_index], axis=1)

def getIMUValue(bias, raw, sensitivity):
    scale_factor = V_REF / MAXIMUM_VALUE / sensitivity
    return (raw - bias[:, jnp.newaxis]) * scale_factor

def getGyroValue(bias, raw):
    return getIMUValue(bias, raw, GYRO_SENSITIVITY)

def getAccelValue(bias, raw):
    return getIMUValue(bias, raw, ACCEL_SENSITIVITY)

def motionModel(qt, taut, wt):
    inner = jnp.transpose(jnp.column_stack((0., jnp.atleast_2d(jnp.multiply(taut, wt) / 2))))
    exp = quat_exp_v(inner)
    result = quat_mult_v(qt, exp)
    # print(result)
    return exp, result

def observationModel(qt):
    gravityVector = quat_vectorize(jnp.array([0.,0.,0.,-GRAVITY], dtype='float64'))
    # print(gravityVector)
    # result = quat_mult_v(quat_mult_v(quat_inv_v(qt), gravityVector), qt)
    result = quat_mult_v(quat_inv_v(qt), quat_mult_v(gravityVector, qt))
    return result[1:,:]
