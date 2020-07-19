vio.py

Kalman Filter
Implemented Error State Kalman Filter specifically for the drone with stereo odometer measurement.



def nominal_state_update(nominal_state, w_m, a_m, dt):

This function updates the nominal state of the UAV with the old nominal state value and the acceleration and angular velocity measured by accelerometer and gyroscope.
The nominal state include position, velocity, rotation, accelerometer bias, pyrometer bias and gravity vector. The values are updated in discrete space with dt. 




def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):

This function update the error state covariance for the UAV with the nominal state values, old error state covariance, measurements and noise and bias from the measurement. This is the core change to the ESKF since the covariance here is of the error state(dx), not the state value itself. This promise an optimal covariance due to the characteristic of Kalman Filter by taking the value at the minimal covariance. 
accelerometer noise density == standard deviation of accelerometer noise == (std dev)a_n
gyroscope noise density == standard deviation of gyro noise == (std dev)omega_n
accelerometer random walk == accelerometer random walk rate == (std dev)a_w
gyroscope random walk == gyro random walk rate == (std dev)omega_w




def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):

This function updates the nominal state and the error state covariance with the measurements value. The Measured value is uv, the observed location of the feature point in the camera frame. The predicted location of the feature point in the camera frame(Pc) could be calculated by the nominal states estimate(orientation--rotation, and translation--position) and the known point location in the world frame Pw. Normalized Pc with the third element give uv_predicted. The innovation is the difference between the measured and predicted value. 
The Jacobian Ht (2x18) is computed by the taking partial derivative of zt wrt error state.
The Kalman gain matrix(18x2) is computed by error state covariance, Ht(Jacobian) and the Qt(covariance matrix associated with the image measurement(u,v). 
The error state is computed by Kt*innovation.
After the error state is updated, the nominal state is updated immediately and the error state is set to 0. 
The error state covariance is updated with Kt, Ht, error_state_covariance and Qt and identity matrix with size(18) 
