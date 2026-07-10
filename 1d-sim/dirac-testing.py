from sim_essentials import *
from itertools import chain
import dirac
import matplotlib.pyplot as plt

def gaussian(mu, sigma, momentum, x):
    return np.exp(-(x-mu)**2/(2*sigma**2)) * np.exp(1j*momentum*x)

# detailed=0, total pass/fail + avg error, detailed=1 gives individual error for each iter, detailed=2 plots probabilities against ideal
def test_linear_transport(grid: Grid, times, f, g, detailed=0, tolerance=1e-1):
    # Here we test our implementation of the solution to the 1d linear transport equation
    # partial_t phi_1 = -partial_x phi_1
    # partial_t phi_2 = partial_x phi_2.
    # The rest of Dirac for a free particle is just a few gates, and this in particular 
    # is very easy to test against an analytical solution, simply given by translation.

    errors = []

    initial_psi = np.concatenate((f(grid.x)*grid.fftshift_correction, g(grid.x)*grid.fftshift_correction))
    initial_psi /= np.linalg.norm(initial_psi)
    for t in times:
        qc = get_empty_sim(grid, num_spinor=1)
        qc.initialize(Statevector(initial_psi))

        # dirac.transport() implements a sigma_x term by diagonalizing sigma_x = H sigma_z H.
        # In this test, we are concerned with the already diagonal linear transport equation,
        # so we remove the Hadamard gates at the start and end of the circuit, noting that H^-1 = H.
        indicator_idx = qc.num_qubits-1
        qc.h(indicator_idx)
        qc.compose(dirac.transport(grid, t), inplace=True)
        qc.h(indicator_idx)

        ideal_psi = np.concatenate((f(grid.x - t)*grid.fftshift_correction, g(grid.x + t)*grid.fftshift_correction))
        ideal_psi /= np.linalg.norm(ideal_psi)

        final_statevector = Statevector.from_circuit(qc)
        error = np.linalg.norm(ideal_psi - final_statevector.data)
        errors.append(error)

        if detailed:
            print(f"linear transport, t={t}, error={error}")
        if detailed == 2:
            num_pts = 512
            x_fine = np.linspace(-grid.d, grid.d, num_pts, endpoint=False)
            ideal_curve = abs(f(x_fine-t)**2) + abs(g(x_fine+t))**2
            ideal_curve /= ideal_curve.sum()
            plt.plot(x_fine, ideal_curve*num_pts/grid.N, "r-")
            plt.bar(grid.x, final_statevector.probabilities(qargs=list(range(grid.num_qubits))), width=0.7*grid.dx)

    print(f"linear transport average error {sum(errors)/len(errors)}", end=", ")
    if max(errors) < tolerance:
        print("pass")
    else:
        print("fail")

    
grid = Grid(num_qubits = 7, d=6*np.pi)
test_linear_transport(grid, [4*t for t in range(4)], lambda x: gaussian(-1, 1, 1, x), lambda x: gaussian(-1, 1/np.sqrt(2), -1, x), detailed=2)

plt.show()