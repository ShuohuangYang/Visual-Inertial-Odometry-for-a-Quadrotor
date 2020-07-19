# %% Imports

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

import proj2_2.code.vio as vio


# %%

class TestBase(unittest.TestCase):

    def error_covariance_update_test(self, fname):
        with open(fname, 'r') as file:
            print('running error_covariance_update_test : ' + fname)

            d = json.load(file)

            in_p = np.array(d['in_p'])
            in_v = np.array(d['in_v'])
            in_q = Rotation.from_quat(d['in_q'])
            in_a_b = np.array(d['in_a_b'])
            in_w_b = np.array(d['in_w_b'])
            in_g = np.array(d['in_g'])

            in_error_state_covariance = np.array(d['in_error_state_covariance'])

            w_m = np.array(d['w_m'])
            a_m = np.array(d['a_m'])
            dt = d['dt']

            accelerometer_noise_density = d['accelerometer_noise_density']
            gyroscope_noise_density = d['gyroscope_noise_density']
            accelerometer_random_walk = d['accelerometer_random_walk']
            gyroscope_random_walk = d['gyroscope_random_walk']

            out_error_state_covariance = np.array(d['out_error_state_covariance'])

            nominal_state = in_p, in_v, in_q, in_a_b, in_w_b, in_g

            # Run a test
            error_state_covariance = vio.error_covariance_update(nominal_state, in_error_state_covariance, w_m, a_m, dt,
                                                                 accelerometer_noise_density, gyroscope_noise_density,
                                                                 accelerometer_random_walk, gyroscope_random_walk)

            delta = error_state_covariance - out_error_state_covariance

            self.assertTrue(norm(delta.ravel()) < 1e-5, 'failed ' + fname)

    def nominal_state_update_test(self, fname):
        with open(fname, 'r') as file:
            print('running nominal_state_update_test : ' + fname)

            d = json.load(file)

            in_p = np.array(d['in_p'])
            in_v = np.array(d['in_v'])
            in_q = Rotation.from_quat(d['in_q'])
            in_a_b = np.array(d['in_a_b'])
            in_w_b = np.array(d['in_w_b'])
            in_g = np.array(d['in_g'])

            w_m = np.array(d['w_m'])
            a_m = np.array(d['a_m'])
            dt = d['dt']

            out_p = np.array(d['out_p'])
            out_v = np.array(d['out_v'])
            out_q = Rotation.from_quat(d['out_q'])
            out_a_b = np.array(d['out_a_b'])
            out_w_b = np.array(d['out_w_b'])
            out_g = np.array(d['out_g'])

            nominal_state = in_p, in_v, in_q, in_a_b, in_w_b, in_g

            # Run a test
            p, v, q, a_b, w_b, g = vio.nominal_state_update(nominal_state, w_m, a_m, dt)

            self.assertTrue(norm(out_p - p) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_v - v) < 1e-5, 'failed ' + fname)
            delta = out_q.inv() * q
            self.assertTrue(delta.magnitude() < 1e-4, 'failed ' + fname)
            self.assertTrue(norm(out_a_b - a_b) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_w_b - w_b) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_g - g) < 1e-5, 'failed ' + fname)

    def measurement_update_step_test(self, fname):
        with open(fname, 'r') as file:
            print('running measurement_update_step_test: ' + fname)

            d = json.load(file)

            in_p = np.array(d['in_p'])
            in_v = np.array(d['in_v'])
            in_q = Rotation.from_quat(d['in_q'])
            in_a_b = np.array(d['in_a_b'])
            in_w_b = np.array(d['in_w_b'])
            in_g = np.array(d['in_g'])

            in_error_state_covariance = np.array(d['in_error_state_covariance'])

            uv = np.array(d['uv'])
            Pw = np.array(d['Pw'])
            error_threshold = d['error_threshold']
            image_measurement_covariance = np.array(d['image_measurement_covariance'])

            out_p = np.array(d['out_p'])
            out_v = np.array(d['out_v'])
            out_q = Rotation.from_quat(d['out_q'])
            out_a_b = np.array(d['out_a_b'])
            out_w_b = np.array(d['out_w_b'])
            out_g = np.array(d['out_g'])

            out_error_state_covariance = np.array(d['out_error_state_covariance'])

            out_inno = np.array(d['out_inno'])

            nominal_state = in_p, in_v, in_q, in_a_b, in_w_b, in_g

            # Run a test
            nominal_state, error_state_covariance, inno = vio.measurement_update_step(nominal_state,
                                                                                      in_error_state_covariance,
                                                                                      uv, Pw, error_threshold,
                                                                                      image_measurement_covariance)

            p, v, q, a_b, w_b, g = nominal_state

            self.assertTrue(norm(out_p - p) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_v - v) < 1e-5, 'failed ' + fname)
            delta = out_q.inv() * q
            self.assertTrue(delta.magnitude() < 1e-4, 'failed ' + fname)
            self.assertTrue(norm(out_a_b - a_b) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_w_b - w_b) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_g - g) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(out_inno - inno) < 1e-5, 'failed ' + fname)

            delta = error_state_covariance - out_error_state_covariance
            self.assertTrue(norm(delta.ravel()) < 1e-5, 'failed ' + fname)

    # test error_covariance_update
    def test_error_covariance_update_1020(self):
        self.error_covariance_update_test('test_imu_update_1020.json')

    def test_error_covariance_update_1150(self):
        self.error_covariance_update_test('test_imu_update_1150.json')

    def test_error_covariance_update_2000(self):
        self.error_covariance_update_test('test_imu_update_2000.json')

    def test_error_covariance_update_2450(self):
        self.error_covariance_update_test('test_imu_update_2450.json')

    def test_error_covariance_update_2900(self):
        self.error_covariance_update_test('test_imu_update_2900.json')

    # test nominal_state_update
    def test_nominal_state_update_1020(self):
        self.nominal_state_update_test('test_imu_update_1020.json')

    def test_nominal_state_update_1150(self):
        self.nominal_state_update_test('test_imu_update_1150.json')

    def test_nominal_state_update_2000(self):
        self.nominal_state_update_test('test_imu_update_2000.json')

    def test_nominal_state_update_2450(self):
        self.nominal_state_update_test('test_imu_update_2450.json')

    def test_nominal_state_update_2900(self):
        self.nominal_state_update_test('test_imu_update_2900.json')

    # test measurement_update_step
    def test_measurement_update_step_10(self):
        self.measurement_update_step_test('test_stereo_update_10.json')

    def test_measurement_update_step_50(self):
        self.measurement_update_step_test('test_stereo_update_50.json')

    def test_measurement_update_step_100(self):
        self.measurement_update_step_test('test_stereo_update_100.json')

    def test_measurement_update_step_125(self):
        self.measurement_update_step_test('test_stereo_update_125.json')

    def test_measurement_update_step_175(self):
        self.measurement_update_step_test('test_stereo_update_175.json')


if __name__ == '__main__':
    unittest.main()
