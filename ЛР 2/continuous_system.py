from control import tf, feedback, step_response
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

T = 0.03
xi = 0.1
num = [1]
den = [T ** 2, 2 * T * xi, 1, 0]
T_p = tf(num, den)
print('T_p(s) =', T_p)

Kp = 2
Ki = 1
Kd = 0.5

num_pid = [Kd, Kp, Ki]
den_pid = [1, 0]
T_c = tf(num_pid, den_pid)
print('T_c(s) =', T_c)

T_cl = feedback(T_c * T_p, 1, sign=-1)
print('T_cl(s) =', T_cl)

t, y = step_response(T_cl)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control')
plt.grid()
plt.show()

Kp = 6.67
Ki = 0
Kd = 0

num_pid = [Kd, Kp, Ki]
den_pid = [1, 0]
T_c = tf(num_pid, den_pid)

T_cl = feedback(T_c * T_p, 1, sign=-1)

t, y = step_response(T_cl)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control')
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
T_c = tf(num_pid, den_pid)

T_cl = feedback(T_c * T_p, 1, sign=-1)

t, y = step_response(T_cl)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response with PID Control')
plt.grid()
plt.show()