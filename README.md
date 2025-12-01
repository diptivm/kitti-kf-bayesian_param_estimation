# kitti-kf-bayesian_param_estimation
Linear KF and Bayesian Parameter Estimation for GPS + IMU fusion on KITTI dataset trajectory
We explore the application of the linear Kalman filter to localization problems in robotics, using a sample
trajectory of a car navigating an outdoor environment from the KITTI dataset. Specifically, the process and measurement noise parameters may not be known in many situations, and while specifications such as sensor accuracy may offer good starting priors, they are often hand-tuned for best accuracy of the estimated state against a known groundtruth. We demonstrate the application of the Kalman filter for accurate localization of the car fusing GPS measurement and acceleration input, first hand-tuning the process and measurement noise, and then experimenting with Bayesian inference for adaptive tuning of  these noise parameters using an Expectation Maximisation approach.

Please refer to "Project Report" for details on the setup, methodology and results.

LinearKF2D.py: Linear Kalman Filter class with prediction and correction steps
EM_KalmanFilter.py: EM algorithm for estimation of process and measurement noise parameters in the Kalman Filter
KF0.ipynb: Notebook applying Kalman Filter on KITTI dataset trajectory
KF0_EMKF.ipynb: Notebook applying Kalman Filter with EM for parameter estimation on KITTI dataset trajectory
