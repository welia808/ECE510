import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_ode(ode_func, initial_conditions, t_span, method='RK45', dense_output=True, **kwargs):
    """
    Solves a system of ordinary differential equations (ODEs).

    Args:
        ode_func (callable): A function that computes the derivatives of the state variables.
            It should have the signature: `dydt = ode_func(t, y)`, where `t` is the time
            and `y` is a NumPy array of the state variables.
        initial_conditions (list or array-like): The initial values of the state variables.
        t_span (tuple): The time span of the integration, (t_start, t_end).
        method (str, optional): The integration method to use. Defaults to 'RK45'.
            Other options include 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.
        dense_output (bool, optional): Whether to compute a dense output and allow continuous evaluation of the solution. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `scipy.integrate.solve_ivp`.

    Returns:
        scipy.integrate.OdeSolution: An object representing the solution of the ODE.
    """
    solution = solve_ivp(ode_func, t_span, initial_conditions, method=method, dense_output=dense_output, **kwargs)
    return solution

def plot_solution(solution, labels=None, title="ODE Solution"):
    """
    Plots the solution of an ODE.

    Args:
        solution (scipy.integrate.OdeSolution): The solution object returned by `solve_ode`.
        labels (list of str, optional): Labels for the state variables. Defaults to None.
        title (str, optional): The title of the plot.
    """
    plt.figure()
    if solution.y.ndim == 1:
        plt.plot(solution.t, solution.y)
        if labels:
            plt.ylabel(labels[0])
    else:
        for i in range(solution.y.shape[0]):
            if labels:
                plt.plot(solution.t, solution.y[i], label=labels[i])
            else:
                plt.plot(solution.t, solution.y[i])
        if labels:
            plt.legend()
    plt.xlabel('time')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Example usage: Solving the Lorenz system
def lorenz(t, y, sigma=10, beta=8/3, rho=28):
    """
    Lorenz system of ODEs.

    Args:
        t (float): Time.
        y (array-like): State variables [x, y, z].
        sigma (float, optional): Lorenz system parameter. Defaults to 10.
        beta (float, optional): Lorenz system parameter. Defaults to 8/3.
        rho (float, optional): Lorenz system parameter. Defaults to 28.

    Returns:
        array-like: Derivatives of the state variables.
    """
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

if __name__ == "__main__":
    initial_conditions = [1, 1, 1]
    t_span = (0, 40)

    solution = solve_ode(lorenz, initial_conditions, t_span)
    plot_solution(solution, labels=['x', 'y', 'z'], title="Lorenz System")

    # Example with different method and dense output usage.
    t_eval = np.linspace(0, 40, 400) #dense output usage.
    solution2 = solve_ode(lorenz, initial_conditions, t_span, method='LSODA', t_eval=t_eval)
    plot_solution(solution2, labels = ['x', 'y', 'z'], title = "Lorenz System with LSODA and dense output")