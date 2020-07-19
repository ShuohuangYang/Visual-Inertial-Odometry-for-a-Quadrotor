#%% Imports

import numpy as np
import scipy as sp
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()
    new_p = p + v * dt + 0.5 * (R.dot(a_m - a_b) + g) * dt * dt
    new_v = v + (R.dot(a_m - a_b) + g) * dt
    theta = (w_m - w_b) * dt
    theta_skew = np.array([[0, -theta[2], theta[1]],
                           [theta[2], 0, -theta[0]],
                           [-theta[1], theta[0], 0]])
    delta_q = sp.linalg.expm(theta_skew)
    new_q = R.dot(delta_q)
    new_q = Rotation.from_matrix(new_q)
    a_b = a_b
    w_b = w_b
    g = g


    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()
    I = np.identity(3)
    Fx = np.identity(18)
    Fx[0:3, 3:6] = I * dt
    a = a_m - a_b
    a_skew = np.array([[0, -a[2], a[1]],
                       [a[2], 0, -a[0]],
                       [-a[1], a[0], 0]])
    dv_coef1 = -R.dot(a_skew) * dt
    dv_coef2 = -R * dt
    dv_coef3 = I * dt
    theta = (w_m - w_b) * dt
    theta_skew = np.array([[0, -theta[2], theta[1]],
                       [theta[2], 0, -theta[0]],
                       [-theta[1], theta[0], 0]])
    theta_skew_exp = sp.linalg.expm(theta_skew)
    dq_coef1 = theta_skew_exp.transpose()
    dq_coef2 = -I * dt
    Fx[3:6, 6:9] = dv_coef1
    Fx[3:6, 9:12] = dv_coef2
    Fx[3:6, 15:18] = dv_coef3
    Fx[6:9, 6:9] = dq_coef1
    Fx[6:9, 12:15] = dq_coef2
    Fi = np.identity(18)
    Fi[0:3, 0:3] = np.zeros((3, 3))
    Fi[15:18, 15:18] = np.zeros((3, 3))
    Qi = np.zeros((18, 18))
    vi = accelerometer_noise_density**2 * dt**2 * I
    thetai = gyroscope_noise_density**2 * dt**2 * I
    ai = accelerometer_random_walk**2 * dt * I
    omegai = gyroscope_random_walk**2 * dt * I
    Qi[3:6, 3:6] = vi
    Qi[6:9, 6:9] = thetai
    Qi[9:12, 9:12] = ai
    Qi[12:15, 12:15] = omegai

    new_P = Fx.dot(error_state_covariance).dot(Fx.transpose()) + Fi.dot(Qi).dot(Fi.transpose())

    return new_P

    # return an 18x18 covariance matrix


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R = q.as_matrix()
    Pc = R.transpose().dot(Pw - p)
    u_predict = Pc[0] / Pc[2]
    v_predict = Pc[1] / Pc[2]
    uv_predict = np.array([u_predict, v_predict]).reshape(2,1)
    dzdp = 1/Pc[2] * np.array([[1, 0, -u_predict],
                     [0, 1, -v_predict]])
    dpdtheta = np.array([[0, -Pc[2], Pc[1]],
                       [Pc[2], 0, -Pc[0]],
                       [-Pc[1], Pc[0], 0]])
    dpdp = -R.transpose()

    Ht = np.zeros((2,18))
    Ht[0:2, 0:3] = dzdp.dot(dpdp)
    Ht[0:2, 6:9] = dzdp.dot(dpdtheta)
    inner = np.linalg.inv(Ht.dot(error_state_covariance).dot(Ht.transpose()) + Q)
    Kt = error_state_covariance.dot(Ht.transpose()).dot(inner)
    innovation = uv - uv_predict
    if norm(innovation) < error_threshold:
        # inliers, else outlier, ignore it and no updates occur
        dx = Kt.dot(uv - uv_predict)

        # update state
        p = p + dx[0:3]
        v = v + dx[3:6]
        dq = Rotation.from_rotvec(dx[6:9].reshape(3,))
        q = R.dot(dq.as_matrix())
        q = Rotation.from_matrix(q)
        a_b = a_b + dx[9:12]
        w_b = w_b + dx[12:15]
        g = g + dx[15:18]

        # update error state covariance
        I = np.identity(18)
        A = I - Kt.dot(Ht)
        error_state_covariance = A.dot(error_state_covariance).dot(A.transpose()) + Kt.dot(Q).dot(Kt.transpose())



    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
