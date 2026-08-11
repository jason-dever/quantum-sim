import numpy as np
from sim_essentials import Grid, get_empty_sim
import dirac
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

def gaussian(mu, sigma, momentum, x):
    return np.exp(-(x-mu)**2/(2*sigma**2)) * np.exp(1j*momentum*x)

def transport_solution_1(f, g, x, t): # First spinor component
    return 1/2*(f(x-t) + g(x-t) + f(x+t) - g(x+t))

def transport_solution_2(f, g, x, t): # Second component
    return 1/2*(f(x-t) + g(x-t) - f(x+t) + g(x+t))

# The justification for the analytical solutions and method for testing here
# is given in dirac.tex. detailed=0 gives pass/fail + avg error, detailed=1
# gives individual error for each iter, detailed=2 plots probabilities against
# ideal on the same figure, detailed=3 gives each iteration its own figure
def unit_test(grid: Grid, times, f, g, term, m=1, detailed=0, tolerance=1e-4):
    errors = []

    initial_psi = np.concatenate((f(grid.x), g(grid.x)))
    initial_psi /= np.linalg.norm(initial_psi)
    for t in times:
        # These two variables are used for graphing the ideal curve if detailed >= 2.
        num_pts = 1024
        x_fine = np.linspace(-grid.d, grid.d, num_pts, endpoint=False)

        match term:
            case "transport":
                dynamics = dirac.transport(grid, t)
                ideal_psi = np.concatenate((transport_solution_1(f, g, grid.x, t), 
                                            transport_solution_2(f, g, grid.x, t)))
                ideal_curve = abs(transport_solution_1(f, g, x_fine, t))**2 + abs(transport_solution_2(f, g, x_fine, t))**2
            case "mass":
                dynamics = dirac.mass(grid, t, m)
                ideal_psi = np.concatenate((np.exp(-1j*m*t)*initial_psi[:grid.N], np.exp(1j*m*t)*initial_psi[grid.N:]))
                ideal_curve = abs(f(x_fine))**2 + abs(g(x_fine))**2 # Phase accumulated is not relevant to measurement probabilities

        qc = get_empty_sim(grid, num_spinor=1)
        qc.initialize(Statevector(initial_psi))
        qc.compose(dynamics, inplace=True)

        ideal_psi /= np.linalg.norm(ideal_psi)
        final_statevector = Statevector.from_circuit(qc)

        error =  abs(1-abs(final_statevector.inner(Statevector(ideal_psi))))
        errors.append(error)

        if detailed:
            print(f"{term}, t={t}, error={error}")

            if detailed >= 2:
                ideal_curve /= ideal_curve.sum()

                if detailed >= 3:
                    plt.figure()

                plt.plot(x_fine, ideal_curve*num_pts/grid.N, "r-")
                plt.bar(grid.x, final_statevector.probabilities(qargs=list(range(grid.num_qubits))), width=0.7*grid.dx)

    print(f"{term} average error {sum(errors)/len(errors)}", end=", ")
    if max(errors) < tolerance:
        print("pass")
    else:
        print("fail")

def test_all():
    grid = Grid(num_qubits = 8, d=6*np.pi)

    unit_test(grid, [6*t for t in range(3)], lambda x: gaussian(0, 1.2, 2, x), lambda x: gaussian(1, 0.8, -2, x), "transport")
    unit_test(grid, [t/10 for t in range(10)], lambda x: gaussian(5, 3, 0, x), lambda x: 0*x, "transport")
    unit_test(grid, [t for t in range(12)], lambda x: 0*x, lambda x: gaussian(0, 1, 0, x), "transport")
    unit_test(grid, [t for t in range(10)], lambda x: gaussian(0, 1, 0, x), lambda x: gaussian(0, 1, 0, x), "transport")

    unit_test(grid, [4*t for t in range(4)], lambda x: gaussian(-5, 1/np.sqrt(2), 1, x), lambda x: gaussian(5, 1, -2, x), "mass", m=2)
    unit_test(grid, [t/10 for t in range(10)], lambda x: gaussian(10, 3, 0, x), lambda x: 0*x, "mass")
    unit_test(grid, [t for t in range(12)], lambda x: 0*x, lambda x: gaussian(0, 1, 0, x), "mass", m=0)
    unit_test(grid, [t for t in range(12)], lambda x: 0*x, lambda x: gaussian(0, 1, 0, x), "mass")
    unit_test(grid, [t for t in range(10)], lambda x: gaussian(0, 1, 0, x), lambda x: gaussian(0, 1, 0, x), "mass", m=-2)

test_all()
plt.show()
