import numpy as np

class EM_KalmanFilter:
    def __init__(self, kf_class, max_iter=10):
        self.kf = kf_class
        self.max_iter = max_iter

        # Prior Strength
        target_Q = 1.0
        target_R = 1.0
        prior_strength_Q = 200
        prior_strength_R = 200  
        
        # Priors
        self.alpha_Q = prior_strength_Q  
        self.beta_Q  = target_Q * (self.alpha_Q + 1)        

        self.alpha_R = prior_strength_R
        self.beta_R = target_R * (self.alpha_R + 1)
        
        # Final filtered output
        self.filtered_mus = None
        self.filtered_covs = None

        self.use_accel_bias = False 

    def fit(self, measurements, dt_list, accel_inputs, p0=None, v0=None):
        N = len(measurements)
        
        if p0 is None: p0 = measurements[0]
        else: p0 = np.asarray(p0)
            
        if v0 is None: v0 = np.array([0.0, 0.0])
        else: v0 = np.asarray(v0)
        
        Q_history = []
        R_history = []
        
        smooth_mus = None
        smooth_covs = None

        self.accel_bias = np.array([0.0, 0.0])
        
        for iteration in range(self.max_iter):
            # E-step: Forward Filter and Backward Smoother

            if self.use_accel_bias:
                corrected_accels = accel_inputs - self.accel_bias
            else:
                corrected_accels = accel_inputs
                self.accel_bias = np.array([0.0, 0.0])  

            mus, covs = self._run_batch_filter(measurements, dt_list, corrected_accels, p0, v0)
            smooth_mus, smooth_covs = self._run_rts_smoother(mus, covs, dt_list, corrected_accels)
            
            if iteration == self.max_iter - 1: # store forward pass of last iteration
                self.filtered_mus = mus
                self.filtered_covs = covs

                sigma_xs = []
                sigma_ys = []
                for cov in covs:
                    sigma_x = np.sqrt(cov[0, 0])  
                    sigma_y = np.sqrt(cov[1, 1])  
                    sigma_xs.append(sigma_x)
                    sigma_ys.append(sigma_y)
                self.sigma_xs = np.array(sigma_xs)
                self.sigma_ys = np.array(sigma_ys)

                print("final self.filtered_mus.shape: ", self.filtered_mus.shape)
                print("final self.filtered_covs.shape: ", self.filtered_covs.shape)
                print("final sigma_xs.shape: ", self.sigma_xs.shape)
                print("final sigma_ys.shape: ", self.sigma_ys.shape)
            
            # M-step: Update Q and R
            
            # Update R (Measurement Variance)
            res_meas = measurements - smooth_mus[:, 0:2]
            sse_meas = np.sum(res_meas**2)
            N_samples_R = N * 2 
            new_R = (self.beta_R + 0.5 * sse_meas) / (self.alpha_R + 0.5 * N_samples_R + 1)
            print("new_R: ", new_R)
            print("sse_meas: ", sse_meas)

            # Update Q (Process Variance)
            vx = smooth_mus[:, 2]
            vy = smooth_mus[:, 3]
            accel_res_list = []
            raw_errors_x = []
            raw_errors_y = []

            for k in range(N - 1):
                dt = dt_list[k]
                if self.use_accel_bias:
                    u_accel = accel_inputs[k] - self.accel_bias
                else:
                    u_accel = accel_inputs[k]
                dv_actual = np.array([vx[k+1] - vx[k], vy[k+1] - vy[k]])
                dv_expected = u_accel * dt
                dv_error = dv_actual - dv_expected
                raw_errors_x.append(dv_error[0]/dt)
                raw_errors_y.append(dv_error[1]/dt)

            raw_errors_x = np.array(raw_errors_x)
            raw_errors_y = np.array(raw_errors_y)

            if self.use_accel_bias:
                bias_x = np.mean(raw_errors_x)
                bias_y = np.mean(raw_errors_y)
                self.accel_bias = np.array([bias_x, bias_y])
            else:
                bias_x = 0.0
                bias_y = 0.0
                self.accel_bias = np.array([0.0, 0.0])

            print("accel_bias: ", self.accel_bias)

            centered_err_x = raw_errors_x - bias_x
            centered_err_y = raw_errors_y - bias_y

            all_centered_errors = np.concatenate([centered_err_x, centered_err_y])
            sse_process = np.sum(all_centered_errors**2)

            N_samples_Q = (N - 1) * 2
            new_Q = (self.beta_Q + 0.5 * sse_process) / (self.alpha_Q + 0.5 * N_samples_Q + 1)
            print("new_Q: ", new_Q)
            print("sse_process: ", sse_process)

            # Update parameters
            self.kf.process_var = new_Q
            self.kf.meas_var = new_R
            
            Q_history.append(new_Q)
            R_history.append(new_R)
            
            print(f"Iter {iteration+1}: Q_est={new_Q:.4f}, R_est={new_R:.4f}")

        return smooth_mus, smooth_covs, Q_history, R_history

    def _run_batch_filter(self, zs, dts, accels, p0, v0):
        mus = []
        covs = []
        
        self.kf.initialize(p0_xy=p0, v0_xy=v0, P0_scale=1.0)
        mus.append(self.kf.x.flatten())  
        covs.append(self.kf.P.copy())
        
        for k in range(len(zs)-1):
            self.kf.predict(a_world_xy=accels[k], dt=dts[k]) 
            self.kf.update(z_xy=zs[k+1])
            mus.append(self.kf.x.flatten())  
            covs.append(self.kf.P.copy())
            
        print(np.array(mus).shape)
        print(np.array(covs).shape)
        return np.array(mus), np.array(covs)

    def _run_rts_smoother(self, mus, covs, dts, accels):
        N = len(mus)
        smooth_mus = np.zeros_like(mus)
        smooth_covs = np.zeros_like(covs)
        
        smooth_mus[-1] = mus[-1]
        smooth_covs[-1] = covs[-1]
        
        for k in range(N - 2, -1, -1):
            dt = dts[k]
            u_accel = accels[k]
            
            A, B, _ = self.kf._compute_matrices(dt)
            u_vec = np.array([[u_accel[0]], [u_accel[1]]])
            x_k = mus[k].reshape(4, 1)
            x_pred_next = A @ x_k + B @ u_vec
            
            _, _, Q = self.kf._compute_matrices(dt)
            P_pred_next = A @ covs[k] @ A.T + Q
            
            J = covs[k] @ A.T @ np.linalg.inv(P_pred_next)
            
            smooth_mus_kp1 = smooth_mus[k+1].reshape(4, 1)
            smooth_mus_k = x_k + J @ (smooth_mus_kp1 - x_pred_next)
            smooth_mus[k] = smooth_mus_k.flatten()  
            smooth_covs[k] = covs[k] + J @ (smooth_covs[k+1] - P_pred_next) @ J.T
            
        return smooth_mus, smooth_covs