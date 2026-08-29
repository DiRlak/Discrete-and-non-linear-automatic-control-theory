import control as ctrl
import numpy as np
from matplotlib import pyplot as plt

# Задаем параметры параметры переходного процесса и находим параметры генератора колебаний
t_end = 3.0  # Время моделирования, с
N = 15  # Количество колебаний за время моделирования
A = 10.0  # Амплитуда в начальный момент времени

lamb = 2 * np.pi * N / t_end  # Угловая частота затухающих колебаний λ
xi = 1.5 / np.sqrt((np.pi * N) ** 2 + 2.25)  # Декремент затухания ξ
omega = lamb / np.sqrt(1 - xi ** 2)  # Угловая частота гармонических колебаний Ω
gamma = xi * omega  # γ = ξΩ — коэффициент затухания


def oscillator(t):
    # Определяем функцию генератора колебаний
    return A * np.cos(lamb * t) * np.exp(-gamma * t)


# Записываем сгенерированные колебания
sim_time = 15.0
steps_per_sec = 1000
time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
oscillations = oscillator(time_grid)
plt.plot(time_grid, oscillations)
plt.grid()
plt.show()

# Задаем параметры нелинейностей
a = 0.25
b = 2.25
c = 2.0
assert a > 0 and b > 0 and c > 0 and b >= a

# Подаем сгенерированный сигнал на вход нелинейностей

# а) Идеальное двухпозиционное реле
relay = ctrl.relay_hysteresis_nonlinearity(c, 0.0)
relay_output = np.zeros(time_grid.shape)
for i in range(time_grid.shape[0]):
    relay_output[i] = relay(oscillations[i])
plt.plot(time_grid, relay_output)
plt.grid()
plt.show()


# б) Усилитель с ограничением и зоной нечувствительности
def linear_saturation_with_deadzone(x, a, b, c):
    if x < -b:
        return -c
    elif x < -a:
        m = c / (b - a)  # Slope for linear regions
        return -c + m * (x + b)
    elif x <= a:
        return 0
    elif x <= b:
        m = c / (b - a)  # Same slope
        return m * (x - a)
    else:
        return c


saturation_output = np.zeros(time_grid.shape)
for i in range(time_grid.shape[0]):
    saturation_output[i] = linear_saturation_with_deadzone(oscillations[i], a, b, c)

plt.plot(time_grid, saturation_output)
plt.grid()
plt.show()

# в) Трехпозиционное реле
relay = ctrl.relay_hysteresis_nonlinearity(c, 0.0)
relay_output = np.zeros(time_grid.shape)
for i in range(time_grid.shape[0]):
    relay_output[i] = relay(oscillations[i]) if (oscillations[i] < -a or oscillations[i] > a) else 0.0
plt.plot(time_grid, relay_output)
plt.grid()
plt.show()

# г) Двухпозиционное реле с гистерезисом
relay = ctrl.relay_hysteresis_nonlinearity(c, a)
relay_output = np.zeros(time_grid.shape)
for i in range(time_grid.shape[0]):
    relay_output[i] = relay(oscillations[i])
plt.plot(time_grid, relay_output)
plt.grid()
plt.show()
