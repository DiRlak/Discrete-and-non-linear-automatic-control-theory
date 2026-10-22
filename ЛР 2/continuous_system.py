"""Настройка ПИД-регулятора методом Зиглера-Николса для непрерывной системы"""

from math import pi

from control import TransferFunction, feedback, poles, step_response, tf
from matplotlib import pyplot as plt

T = 0.03
XI = 0.1


def create_pid(Kp: float, Ki: float, Kd: float) -> TransferFunction:
    """Создать ПИД-регулятор: C(s) = Kp + Ki / s + Kd * s."""
    return tf([Kd, Kp, Ki], [1, 0])


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


def main() -> None:
    """Рассчитать и показать настройки одной и той же системы с разными коэффициентами ПИД"""
    
    # Передаточная функция объекта управления: W(s) = 1 / (T² * s³ + 2 * T * ξ * s² + s).
    plant_denominator = [T ** 2, 2 * T * XI, 1, 0]
    W_continuous_system = tf([1], plant_denominator)
    print("W(s) =", W_continuous_system)

    # 1. Исходные коэффициенты ПИД-регулятора.
    Kp = 2
    Ki = 1
    Kd = 0.5
    W_continuous_pid = create_pid(Kp, Ki, Kd)
    print("W_pid(s) =", W_continuous_pid)

    # Построение ПХ замкнутой непрерывной системы с исходным ПИД
    W_continuous_closed_loop = feedback(W_continuous_pid * W_continuous_system, 1, sign=-1)
    print("W_cl(s) =", W_continuous_closed_loop)
    title = f" ПХ непрерывной системы с исходным ПИД: Kp={Kp:g}, Ki={Ki:g}, Kd={Kd:g}"
    plot_step_response(W_continuous_closed_loop, title)
    print("Полюса исходной системы:", poles(W_continuous_closed_loop))
    print()

    # 2. П-регулятор на границе устойчивости (Ki = Kd = 0).
    Kp = plant_denominator[1] / plant_denominator[0]
    W_continuous_pid = tf([Kp], [1])
    W_continuous_closed_loop = feedback(W_continuous_pid * W_continuous_system, 1, sign=-1)
    title = "ПХ непрерывной системы с П-регулятор при предельных колебаниях"
    plot_step_response(W_continuous_closed_loop, title)

    Ku = Kp
    omega_u = 1 / T
    pu = 2 * pi / omega_u
    print(f"Период предельных колебаний pu = {pu:.6g} с")
    print()

    # 3. Настройка ПИД-регулятора по предельным колебаниям.
    Kp = 0.6 * Ku
    Ki = 1.2 * Ku / pu
    Kd = 3 * Ku * pu / 40
    print(f"Настроенный ПИД: Kp={Kp:.6g}, Ki={Ki:.6g}, Kd={Kd:.6g}")
    print()

    # Построение ПХ замкнутой непрерывной системы с ПИД, настроенным методом Зиглера-Николса
    W_continuous_pid = create_pid(Kp, Ki, Kd)
    W_continuous_closed_loop = feedback(W_continuous_pid * W_continuous_system, 1, sign=-1)
    title = "ПХ непрерывной системы с ПИД, настроеным по предельным колебаниям"
    plot_step_response(W_continuous_closed_loop, title)
    print("Полюса настроенной системы:", poles(W_continuous_closed_loop))
    print()


if __name__ == "__main__":
    main()
