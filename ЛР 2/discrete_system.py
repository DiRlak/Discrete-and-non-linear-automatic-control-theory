from control import tf, feedback, step_response, series, sample_system, pole_zero_map
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

T1 = 0.02
T2 = 0.45
T3 = 0.5

num = [1]
den = [1, 0]
den1 = [T1, 1]
den2 = [T2, 1]
den3 = [T3, 1]

W = tf(num, den)
W1 = tf(num, den1)
W2 = tf(num, den2)
W3 = tf(num, den3)

W_p = series(W, W1, W2, W3)
print('W_p(s) =', W_p)

T0 = 0.01
W_z = sample_system(W_p, T0, method='zoh')
print('W_z(z) =', W_z)

Kp = 2
Ki = 1
Kd = 0.5

num_pid = [Kd, Kp, Ki]
den_pid = [1, 0]
W_c_z = tf(num_pid, den_pid, dt=True)
print('W_c_z(z) =', W_c_z)

W_cl_z = feedback(W_c_z * W_z, 1, sign=-1)
print('W_cl_z(s) = ', W_cl_z)

t, y = step_response(W_cl_z)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control for discrete system')
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
pole_zero_map(W_cl_z).plot()
plt.axvline(0, color='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Pole-Zero Plot with PID Control')
plt.grid()
plt.show()

Kp = 0.01
Ki = 0
Kd = 0
tau = 0.01

num_pid = [Kd, Kp, Ki]
den_pid = [tau, 1]
W_c_z = tf(num_pid, den_pid, dt=True)
print('W_c_z(z) =', W_c_z)

W_cl_z = feedback(W_c_z * W_z, 1, sign=-1)
print('W_cl_z(s) = ', W_cl_z)

t, y = step_response(W_cl_z)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control for discrete system')
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
pole_zero_map(W_cl_z).plot()
plt.axvline(0, color='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Pole-Zero Plot with PID Control')
plt.grid()
plt.show()

Ku = Kp
peaks, _ = find_peaks(y, prominence=0.5)
peak_times = t[peaks]
periods = np.diff(peak_times)
pu = np.mean(periods)
print('Mean period of fluctuations:', pu)

Kp = 0.6 * Ku
Ki = 1.2 * Ku / pu
Kd = 3 * Ku * pu / 40

num_pid = [Kd, Kp, Ki]
den_pid = [1, 0]
W_c_z = tf(num_pid, den_pid, dt=True)
print('W_c_z(z) =', W_c_z)

W_cl_z = feedback(W_c_z * W_z, 1, sign=-1)
print('W_cl_z(s) = ', W_cl_z)

t, y = step_response(W_cl_z)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control for discrete system')
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
pole_zero_map(W_cl_z).plot()
plt.axvline(0, color='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Pole-Zero Plot with PID Control')
plt.grid()
plt.show()
