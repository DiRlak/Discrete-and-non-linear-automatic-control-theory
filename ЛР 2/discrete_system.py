"""Настройка ПИД-регулятора (ПСР-регулятора) методом Зиглера-Николса для дискретной системы"""

import numpy as np

from control import TransferFunction, feedback, poles, step_response, tf, sample_system, series, pole_zero_map
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

T1 = 0.02
T2 = 0.45
T3 = 0.5
T0 = 0.01


def create_filtered_pid(Kp: float, Ki: float, Kd: float, tau: float) -> TransferFunction:
    """Создать ПИД-регулятор с фильтром низких частот: C(s) = Kp + Ki / s + (Kd * s) / (tau * s + 1)."""
    return tf([Kp * tau + Kd, Kp + Ki * tau, Ki], [tau, 1, 0])


def plot_step_response(system: TransferFunction, title: str) -> None:
    """Показать отклик системы на единичное ступенчатое воздействие."""
    time, response = step_response(system)

    figure, axes = plt.subplots()
    axes.plot(time, response)
    axes.set_xlabel("Время, с")
    axes.set_ylabel("Выход системы")
    axes.set_title(title)
    axes.grid(True)
    figure.tight_layout()
    plt.show()


def plot_discrete_pz(system: TransferFunction, title: str) -> None:
    """Отобразить нули и полюса дискретной системы"""
    figure, axes = plt.subplots()
    pole_zero_map(system).plot(ax=axes)

    axes.add_patch(
        Circle(
            (0, 0),
            radius=1,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
    )

    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Re(z)")
    axes.set_ylabel("Im(z)")
    axes.set_title(title)
    axes.grid(True)
    figure.tight_layout()
    plt.show()


def plot_closed_loop(
    system: TransferFunction,
    controller: TransferFunction,
    Kp: float,
    Ki: float = 0,
    Kd: float = 0,
    tau: float = 0,
) -> TransferFunction:
    """Замкнуть систему отрицательной обратной связью и показать оба графика.

    system и controller — дискретные передаточные функции объекта и регулятора.
    Kp, Ki, Kd и tau используются только для подписей графиков;
    передаточная функция регулятора задаётся аргументом controller.
    Возвращает передаточную функцию замкнутой системы.
    """
    closed_loop = feedback(controller * system, 1, sign=-1)
    parameters = f"Kp={Kp:g}, Ki={Ki:g}, Kd={Kd:g}, tau={tau:g}"

    plot_step_response(
        closed_loop,
        f"ПХ дискретной системы: {parameters}",
    )
    plot_discrete_pz(
        closed_loop,
        f"Нули и полюса дискретной системы: {parameters}",
    )

    return closed_loop


def main() -> None:
    """Рассчитать коэффициенты дискретного ПИД(ПСР)-регулятора методом Зиглера-Николса"""

    # Передаточная функция объекта управления: W(s) = 1 / s(T1 * s + 1)(T2 * s + 1)(T3 * s + 1)
    W = tf([1], [1, 0])
    W1 = tf([1], [T1, 1])
    W2 = tf([1], [T2, 1])
    W3 = tf([1], [T3, 1])
    W_continuous_system = series(W, W1, W2, W3)
    print("W(s) =", W_continuous_system)

    # Дискретизация непрерывной передаточной функции ОУ экстраполятором 0-го порядка 
    W_discrete_system = sample_system(W_continuous_system, T0, method='zoh')
    print("W_ds(z) =", W_discrete_system)

    # Исходные коэффициенты ПИД-регулятора
    Kp = 2
    Ki = 1
    Kd = 0.5
    tau = T0/10
    W_continuous_pid = create_filtered_pid(Kp, Ki, Kd, tau)
    print("W_cp(s) =", W_continuous_pid)

    # Дискретизация ПИД экстраполятором 0-го порядка
    W_discrete_pid = sample_system(W_continuous_pid, T0, method='zoh')
    print("W_dp(z) =", W_discrete_pid)

    # Построение ПХ замкнутой дискретной системы с исходным ПИД
    plot_closed_loop(W_discrete_pid * W_discrete_system, 1, Kp, Ki, Kd, tau)

    #--------- Настройка ПИД-регулятора методом Зиглера-Николса ---------

    # Иллюстрация ПХ замкнутой дискретной системы с П-регулятором и нулей и полюсов на комплексной плоскости (Kp = 1)
    Kp = 1
    Ki = 0
    Kd = 0
    W_discrete_pid = tf([Kp], [1], T0)
    plot_closed_loop(W_discrete_pid * W_discrete_system, 1, Kp, Ki, Kd, tau)

    # Определение границ, где находится предельное значение Kp
    Kp_step = 0.1
    Kp_left = None

    for Kp in np.arange(1, 10 + Kp_step/2, Kp_step):
        W_discrete_pid = tf([Kp], [1], T0)
        W_discrete_closed_loop = feedback(W_discrete_pid * W_discrete_system, 1, sign=-1)

        poles_cl = poles(W_discrete_closed_loop)
        max_abs_pole = np.max(np.abs(poles_cl))

        print("Kp =", round(Kp, 2),
            "Самое большое абсолютное значение полюса:", max_abs_pole)

        if max_abs_pole >= 1:
            if Kp_left is None:
                raise ValueError("Нужно уменьшить начальное значение Kp")

            Kp_right = Kp
            break

        Kp_left = Kp

    else:
        raise ValueError("В заданном диапазоне граница не найдена")

    print("Последнее устойчивое Kp:", Kp_left)
    print("Первое неустойчивое Kp:", Kp_right)
    print()

    # Иллюстрация ПХ замкнутой дискретной системы с П-регулятором и нулей и полюсов на комплексной плоскости между (последнее устойчивое Kp)
    Kp = Kp_left
    W_discrete_pid = tf([Kp], [1], T0)
    plot_closed_loop(W_discrete_pid * W_discrete_system, 1, Kp, Ki, Kd, tau)

    # Иллюстрация ПХ замкнутой дискретной системы с П-регулятором и нулей и полюсов на комплексной плоскости между (первое неустойчивое Kp)
    Kp = Kp_right
    W_discrete_pid = tf([Kp], [1], T0)
    plot_closed_loop(W_discrete_pid * W_discrete_system, 1, Kp, Ki, Kd, tau)

    # Уточние предельного значения коэффециента П-регулятора
    eps = 1e-6

    while Kp_right - Kp_left > eps:
        Kp = (Kp_left + Kp_right) / 2

        W_discrete_pid = tf([Kp], [1], T0)
        W_discrete_closed_loop = feedback(W_discrete_pid * W_discrete_system, 1, sign=-1)

        poles_cl = poles(W_discrete_closed_loop)
        max_abs_pole = np.max(np.abs(poles_cl))

        if max_abs_pole < 1:
            Kp_left = Kp
        else:
            Kp_right = Kp

    Ku = (Kp_left + Kp_right) / 2

    print("Уточненный интервал:", Kp_left, Kp_right)
    print("Ku =", Ku)
    print()

    # Нахождение периода автоколебаний по аргументу критического полюса при Kp = Ku
    W_discrete_pid = tf([Ku], [1], T0)
    W_discrete_closed_loop = feedback(W_discrete_pid * W_discrete_system, 1, sign=-1)

    poles_cl = poles(W_discrete_closed_loop)

    # Выбор комплексных полюсов в верхней полуплоскости
    complex_poles = poles_cl[np.imag(poles_cl) > 1e-8]

    # Критический полюс находится ближе всего к единичной окружности
    index = np.argmin(np.abs(np.abs(complex_poles) - 1))
    z_u = complex_poles[index]

    w_u = np.angle(z_u) / T0
    pu = 2 * np.pi / w_u

    print('Критический полюс:', z_u)
    print('Абсолютное значение критического полюса:', np.abs(z_u))
    print('w_u =', w_u)
    print('pu =', pu)
    print()

    # Расчёт коэффициентов ПИД-регулятора настроенного методом Зиглера-Николса
    Kp = 0.6 * Ku
    Ki = 1.2 * Ku / pu
    Kd = 3 * Ku * pu / 40

    print('Kp =', Kp)
    print('Ki =', Ki)
    print('Kd =', Kd)
    print()

    W_continuous_pid = create_filtered_pid(Kp, Ki, Kd, tau)
    W_discrete_pid = sample_system(W_continuous_pid, T0, method='zoh')
    W_discrete_closed_loop = plot_closed_loop(W_discrete_pid * W_discrete_system, 1, Kp, Ki, Kd, tau)
    
    poles_cl = poles(W_discrete_closed_loop)
    max_abs_pole = np.max(np.abs(poles_cl))

    print('Полюсы:', poles_cl)
    print('Самое большое абсолютное значение полюса:', max_abs_pole)
    print('Система устойчива:', max_abs_pole < 1)


if __name__ == "__main__":
    main()