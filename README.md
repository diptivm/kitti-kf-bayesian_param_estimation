# kitti-kf-bayesian_param_estimation
Linear KF and Bayesian Parameter Estimation for GPS + IMU fusion on KITTI dataset trajectory
We explore the application of the linear Kalman filter to localization problems in robotics, using a sample
trajectory of a car navigating an outdoor environment from the KITTI dataset. Specifically, the process and measurement noise parameters may not be known in many situations, and while specifications such as sensor accuracy may offer good starting priors, they are often hand-tuned for best accuracy of the estimated state against a known groundtruth. We demonstrate the application of the Kalman filter for accurate localization of the car fusing GPS measurement and acceleration input, first hand-tuning the process and measurement noise, and then experimenting with Bayesian inference for adaptive tuning of  these noise parameters using an Expectation Maximisation approach.

Please refer to "Project Report" for details on the setup, methodology and results.

## Installation

To install the required Python dependencies, run:

```
pip install -r requirements.txt
```

## Usage

You can run the provided notebooks to try out the Kalman Filter (KF) and Expectation Maximization (EM) methods:

### 1. Linear Kalman Filter

```
KF0.ipynb
```
Applies the linear Kalman Filter on the KITTI dataset trajectory.

### 2. Kalman Filter with EM for Parameter Estimation

```
KF0_EMKF.ipynb
```
Applies the Linear Kalman Filter with estimation of the process and measurement noise parameters using an Expectation Maximisation approach
---

**Alternatively, to use the Python scripts directly:**

- Use `LinearKF2D.py` for the linear Kalman Filter.
- Use `EM_KalmanFilter.py` for the EM-based parameter estimation on the Linear Kalman Filter.

Example minimal usage:

```python
from LinearKF2D import LinearKF
from EM_KalmanFilter import EMKalmanFilter

# Instantiate and use the filters according to the examples in the notebooks.
```





