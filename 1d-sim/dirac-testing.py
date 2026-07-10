from sim_essentials import *
from itertools import chain
import dirac

def gaussian(mu, sigma, momentum, x):
    return np.exp(-(x-mu)**2/(2*sigma**2)) * np.exp(1j*momentum*x)

def test_linear_transport(grid: Grid, f, g, detailed=False, tolerance=1e-6):
    # Here we test our implementation of the solution to the 1d linear transport equation
    # partial_t phi_1 = -partial_x phi_1
    # partial_t phi_2 = partial_x phi_2.
    # The rest of Dirac for a free particle is just a few gates, and this in particular 
    # is very easy to test against an analytical solution, simply given by translation.

    num_steps = 24
    errors = []

    initial_psi = np.concatenate((f(grid.x)*grid.fftshift_correction, g(grid.x)*grid.fftshift_correction))
    initial_psi /= np.linalg.norm(initial_psi)
    for s in range(0, num_steps//2+1):
        t = s/4
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

        error = np.linalg.norm(ideal_psi - Statevector.from_circuit(qc).data)
        errors.append(error)

        if detailed:
            print(f"linear transport, t: {t}, error: {error}")

    print(f"linear transport average error {sum(errors)/len(errors)}", end=", ")
    if max(errors) < tolerance:
        print("pass")
    else:
        print("fail")

    
grid = Grid(num_qubits = 6, d=4*np.pi)
test_linear_transport(grid, lambda x: gaussian(1, 1, 0, x), lambda x: gaussian(-2, 1/np.sqrt(2), 0, x), detailed=False)