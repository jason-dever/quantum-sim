from sim_essentials import *
import dirac
import matplotlib.pyplot as plt

def gaussian(mu, sigma, momentum, x):
    return np.exp(-(x-mu)**2/(2*sigma**2)) * np.exp(1j*momentum*x)

def transport_solution_1(f, g, x, t): # First spinor component
    return 1/2*(f(x-t) + g(x-t) + f(x+t) - g(x+t))

def transport_solution_2(f, g, x, t): # Second component
    return 1/2*(f(x-t) + g(x-t) - f(x+t) + g(x+t))

# Here we test our implementation of the solution to the transport term
# partial_t psi = -partial_x sigma_x psi
# The justification for the analytical solution is given in dirac.tex.
# detailed=0 gives pass/fail + avg error, detailed=1 gives individual 
# error for each iter, detailed=2 plots probabilities against ideal
def test_transport(grid: Grid, times, f, g, detailed=0, tolerance=1e-4):
    errors = []

    initial_psi = np.concatenate((f(grid.x)*grid.fftshift_correction, g(grid.x)*grid.fftshift_correction))
    initial_psi /= np.linalg.norm(initial_psi)
    for t in times:
        qc = get_empty_sim(grid, num_spinor=1)
        qc.initialize(Statevector(initial_psi))
        qc.compose(dirac.transport(grid, t), inplace=True)

        ideal_psi = np.concatenate((transport_solution_1(f, g, grid.x, t)*grid.fftshift_correction, 
                                    transport_solution_2(f, g, grid.x, t)*grid.fftshift_correction))
        ideal_psi /= np.linalg.norm(ideal_psi)
        final_statevector = Statevector.from_circuit(qc)

        error =  abs(1-abs(final_statevector.inner(Statevector(ideal_psi))))
        errors.append(error)

        if detailed:
            print(f"transport, t={t}, error={error}")
        if detailed == 2:
            num_pts = 1024
            x_fine = np.linspace(-grid.d, grid.d, num_pts)    
            ideal_curve = abs(transport_solution_1(f, g, x_fine, t))**2 + abs(transport_solution_2(f, g, x_fine, t))**2
            ideal_curve /= ideal_curve.sum()

            plt.plot(x_fine, ideal_curve*num_pts/grid.N, "r-")
            plt.bar(grid.x, final_statevector.probabilities(qargs=list(range(grid.num_qubits))), width=0.7*grid.dx)

    print(f"transport average error {sum(errors)/len(errors)}", end=", ")
    if max(errors) < tolerance:
        print("pass")
    else:
        print("fail")

    
grid = Grid(num_qubits = 7, d=6*np.pi)
test_transport(grid, [4*t for t in range(4)], lambda x: gaussian(0, 1, 0, x), lambda x: 0, detailed=2)

plt.show()