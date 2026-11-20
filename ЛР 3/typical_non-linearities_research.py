"""Исследование типичных нелинейностей"""

import numpy as np

import control as ctrl
from matplotlib import pyplot as plt

a = 0.25
b = 2.25
c = 2
T_END = 3.0  # Время затухания колебательного процесса, с
N = 12 # Количество колебаний за время затухания колебательного процесса
A = 5.0 # Амплитуда в начальный момент времени

def oscillator(t, lamb, gamma, amplitude = A):
    """Функция, описывающая колебания на выходе колебательного звена"""
    return amplitude * np.cos(lamb * t) * np.exp(-gamma * t)


def linear_saturation_with_deadzone(x, a, b, c):
    """Функция, описывающая усилитель с ограничением и зоной нечувствительности"""
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


def plot_element_output_on_time(
    time_grid, oscillations, relay_output, title: str, amplitude: float = A
) -> None:
    """Отображает выходы звеньев во времени и уровни амплитуды +A и -A."""
    figure, axes = plt.subplots()

    axes.plot(time_grid, oscillations, label="Выход колебательного звена")
    axes.plot(time_grid, relay_output, linestyle="--", label="Выход нелинейного элемента")

    amplitude_levels = (
        (amplitude, f"A = {amplitude:g}", "tab:red", 6, "bottom"),
        (-amplitude, f"-A = {-amplitude:g}", "tab:red", -6, "top"),
    )
    for value, label, color, offset, alignment in amplitude_levels:
        axes.axhline(value, color=color, linestyle=":", alpha=0.7)
        axes.annotate(
            label,
            xy=(0, value),
            xycoords=axes.get_yaxis_transform(),
            xytext=(6, offset),
            textcoords="offset points",
            color=color,
            verticalalignment=alignment,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
        )

    axes.margins(y=0.1)
    axes.set_xlabel("Время, с")
    axes.set_ylabel("Значение сигнала")
    axes.set_title(title)
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    plt.show()


def plot_characteristic_of_nonlinearity(input_values, output_values, title: str) -> None:
    """Отображает характеристику нелинейности с отметками ±a, ±b и ±c на осях."""
    figure, axes = plt.subplots()

    axes.plot(input_values, output_values)

    axes.spines["left"].set_position("zero")
    axes.spines["bottom"].set_position("zero")
    axes.spines[["top", "right"]].set_visible(False)

    axes.set_xticks(
        [-b, -a, a, b],
        [rf"$-b = {-b:g}$", rf"$-a = {-a:g}$", rf"$a = {a:g}$", rf"$b = {b:g}$"],
    )
    axes.set_yticks([-c, c], [rf"$-c = {-c:g}$", rf"$c = {c:g}$"])
    axes.tick_params(axis="both", pad=6)
    for tick, alignment in zip(axes.get_xticklabels(), ("right", "right", "left", "left")):
        tick.set_horizontalalignment(alignment)
    for tick in axes.xaxis.get_major_ticks()[:2]:
        tick.set_pad(-10)
        tick.label1.set_verticalalignment("bottom")
    axes.get_yticklabels()[0].set_verticalalignment("top")
    axes.get_yticklabels()[1].set_verticalalignment("bottom")

    axes.update_datalim([[-b, -c], [b, c]])
    axes.margins(x=0.15, y=0.2)

    axes.set_xlabel("Вход системы - e")
    axes.set_ylabel("Выход системы - g")
    axes.xaxis.set_label_coords(0.5, -0.06)
    axes.yaxis.set_label_coords(-0.06, 0.5)
    axes.set_title(title)
    axes.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Исследование типичных нелинейностей и эксперименты с амплитудой"""

    # Нахождение параметров генератора колебаний
    lamb = 2 * np.pi * N / T_END  # Угловая частота затухающих колебаний λ
    xi = 1.5 / np.sqrt((np.pi * N) ** 2 + 2.25)  # Декремент затухания ξ
    omega = lamb / np.sqrt(1 - xi ** 2)  # Угловая частота гармонических колебаний Ω
    gamma = xi * omega  # γ = ξΩ — коэффициент затухания
    print(f"λ = {lamb:.6f}\nξ = {xi:.6}\nΩ = {omega:.6f}\nγ = {gamma:.6f}")

    # Запись сгенерированных колебаний
    sim_time = 5.0 # Время моделирования
    steps_per_sec = 1000 # Шаг моделирования
    time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
    oscillations = oscillator(time_grid, lamb, gamma)

    # Проверка на корректность параметров неленейных элементов
    assert a > 0 and b > 0 and c > 0 and b >= a

    # Подача сгенерированного сигнала на вход каждой из нелинейностей

    # а) Идеальное двухпозиционное реле
    relay = ctrl.relay_hysteresis_nonlinearity(c, 0.0)
    relay_output = np.zeros(time_grid.shape)
    for i in range(time_grid.shape[0]):
        relay_output[i] = relay(oscillations[i])
    title = "Выход идеального двухпозиционного реле"
    plot_element_output_on_time(time_grid, oscillations, relay_output, title, amplitude=A)
    plot_characteristic_of_nonlinearity(oscillations, relay_output, title)

    # б) Усилитель с ограничением и зоной нечувствительности
    saturation_output = np.zeros(time_grid.shape)
    for i in range(time_grid.shape[0]):
        saturation_output[i] = linear_saturation_with_deadzone(oscillations[i], a, b, c)
    title = "Выход усилителя с ограничением и зоной нечувствительности"
    plot_element_output_on_time(time_grid, oscillations, saturation_output, title, amplitude=A)
    plot_characteristic_of_nonlinearity(oscillations, saturation_output, title)

    # в) Трехпозиционное реле
    relay = ctrl.relay_hysteresis_nonlinearity(c, 0.0)
    relay_output = np.zeros(time_grid.shape)
    for i in range(time_grid.shape[0]):
        relay_output[i] = relay(oscillations[i]) if (oscillations[i] < -a or oscillations[i] > a) else 0.0
    title = "Выход трехпозиционного реле"
    plot_element_output_on_time(time_grid, oscillations, relay_output, title, amplitude=A)
    plot_characteristic_of_nonlinearity(oscillations, relay_output, title)

    # г) Двухпозиционное реле с гистерезисом
    relay = ctrl.relay_hysteresis_nonlinearity(c, a)
    relay_output = np.zeros(time_grid.shape)
    for i in range(time_grid.shape[0]):
        relay_output[i] = relay(oscillations[i])
    title = "Выход двухпозиционного реле с гистерезисом"
    plot_element_output_on_time(time_grid, oscillations, relay_output, title, amplitude=A)
    plot_characteristic_of_nonlinearity(oscillations, relay_output, title)

    # д) Люфт
    backlash_output = np.zeros(time_grid.shape)
    previous_output = 0.0
    for i in range(time_grid.shape[0]):
        x = oscillations[i]
        previous_output = np.clip(previous_output, x - a, x + a)
        backlash_output[i] = previous_output
    title = "Выход люфта"
    plot_element_output_on_time(time_grid, oscillations, backlash_output, title, amplitude=A)
    plot_characteristic_of_nonlinearity(oscillations, backlash_output, title)
    

    # Эксперименты для различных элементов при разной амплитуде гармонического сигнала A
    # и нулевом значении коэффициента демпфироввания ξ

    # Нахождение параметров генератора колебаний (ξ = 0)
    lamb = 2 * np.pi * N / T_END  # Угловая частота затухающих колебаний λ
    xi = 0  # Декремент затухания ξ
    omega = lamb / np.sqrt(1 - xi ** 2)  # Угловая частота гармонических колебаний Ω
    gamma = xi * omega  # γ = ξΩ — коэффициент затухания

    # Усилитель с ограничением и зоной нечувствительности (ξ = 0) и разные A
    for amplitude in (0.2, 2, 5):
        sim_time = 0.625 # Время моделирования
        steps_per_sec = 1000 # Шаг моделирования
        time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
        oscillations = oscillator(time_grid, lamb, gamma, amplitude)

        saturation_output = np.zeros(time_grid.shape)
        for i in range(time_grid.shape[0]):
            saturation_output[i] = linear_saturation_with_deadzone(oscillations[i], a, b, c)
        title = f"Выход усилителя с ограничением и зоной нечувствительности A={amplitude}, ξ={xi}"
        plot_element_output_on_time(time_grid, oscillations, saturation_output, title, amplitude=amplitude)

    # Трехпозиционное реле (ξ = 0) и разные A
    for amplitude in (0.2, 1):
        sim_time = 0.625 # Время моделирования
        steps_per_sec = 1000 # Шаг моделирования
        time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
        oscillations = oscillator(time_grid, lamb, gamma, amplitude)

        relay = ctrl.relay_hysteresis_nonlinearity(c, 0.0)
        relay_output = np.zeros(time_grid.shape)
        for i in range(time_grid.shape[0]):
            relay_output[i] = relay(oscillations[i]) if (oscillations[i] < -a or oscillations[i] > a) else 0.0
        title = f"Выход трехпозиционного реле A={amplitude}, ξ={xi}"
        plot_element_output_on_time(time_grid, oscillations, relay_output, title, amplitude=amplitude)

    # Двухпозиционное реле с гистерезисом (ξ = 0) и разные A
    for amplitude in (0.2, 1):
        sim_time = 0.625 # Время моделирования
        steps_per_sec = 1000 # Шаг моделирования
        time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
        oscillations = oscillator(time_grid, lamb, gamma, amplitude)

        relay = ctrl.relay_hysteresis_nonlinearity(c, a)
        relay_output = np.zeros(time_grid.shape)
        for i in range(time_grid.shape[0]):
            relay_output[i] = relay(oscillations[i])
        title = f"Выход двухпозиционного реле с гистерезисом A={amplitude}, ξ={xi}"
        plot_element_output_on_time(time_grid, oscillations, relay_output, title, amplitude=amplitude)

    # Люфт (ξ = 0) и разные A
    for amplitude in (0.2, 1):
        sim_time = 0.625 # Время моделирования
        steps_per_sec = 1000 # Шаг моделирования
        time_grid = np.linspace(0.0, sim_time, int(sim_time * steps_per_sec) + 1)
        oscillations = oscillator(time_grid, lamb, gamma, amplitude)

        backlash_output = np.zeros(time_grid.shape)
        previous_output = 0.0
        for i in range(time_grid.shape[0]):
            x = oscillations[i]
            previous_output = np.clip(previous_output, x - a, x + a)
            backlash_output[i] = previous_output
        title = f"Выход люфта A={amplitude}, ξ={xi}"
        plot_element_output_on_time(time_grid, oscillations, backlash_output, title, amplitude=amplitude)

if __name__ == "__main__":
    main()
