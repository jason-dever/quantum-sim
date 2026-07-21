from sim_essentials import *
import numpy as np
from math import floor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate

# Uncomment these two and comment the last one to run on an IBM QPU
# service = QiskitRuntimeService()
# backend = service.least_busy(simulator=False, operational=True)
backend = FakeCasablancaV2()

def kinetic(grid: Grid, dt) -> QuantumCircuit:
    qc = QuantumCircuit(grid.num_qubits)
    qc.compose(QFTGate(grid.num_qubits).inverse(), inplace=True)

    phase = (np.pi/grid.length)**2
    for j in range(grid.num_qubits):
        qc.rz(2**(j+2)*phase*dt, j)
        for l in range(j+1, grid.num_qubits):
            qc.rzz(2**(j+l+2)*phase*dt, j, l)

    qc.compose(QFTGate(grid.num_qubits), inplace=True)
    return qc

def harmonic_potential(grid: Grid, dt) -> QuantumCircuit:
    qc = QuantumCircuit(grid.num_qubits)
    for j in range(grid.num_qubits):
        qc.rz(2**j * grid.dx * dt * (2*grid.d + grid.dx*(1-2**grid.num_qubits)), j)
        for l in range(j+1, grid.num_qubits):
            qc.rzz(2**(j+l) * grid.dx**2 * dt, j, l)
    return qc

# The kinetic energy circuit is the same in all cases, so we don't need to let
# it vary by passing it into this function.
def get_one_iter(grid: Grid, potential_qc: QuantumCircuit, dt) -> QuantumCircuit:
    qc = QuantumCircuit(grid.num_qubits)
    qc.compose(kinetic(grid, dt), inplace=True)
    qc.compose(potential_qc, inplace=True)

    return qc

def get_sim_circuit(grid: Grid, get_one_iter: Callable[[Grid, QuantumCircuit, float], QuantumCircuit], potential: QuantumCircuit, dt, final_t) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=0)
    num_iter = 0 if final_t == 0 else floor(final_t/dt)

    # Using operator splitting to approximately solve the equation, we take 
    # timesteps of length dt from t=0 to the last step before/at final_t.
    for _ in range(num_iter):
        qc.compose(get_one_iter(grid, potential, dt), inplace=True)

    return qc

# The analytical solutions don't need normalization here
# because we have to normalize again before we plot anyway.

def analytical_solution_free(momentum, x, t):
    # For intial sigma = 1/np.sqrt(2)
    return abs(np.sqrt(1j/(-4*t+1j))*np.exp((-1j*x**2 - momentum*x + momentum**2 * t)/(-4*t+1j)))**2


def analytical_solution_qho(initial_mu, initial_sigma, x, t):
    # For zero initial momentum
    omega = 2
    Sigma2_t = (initial_sigma * np.cos(omega*t))**2 + (1/initial_sigma * np.sin(omega*t))**2
    return np.exp(-(x-initial_mu*np.cos(omega*t))**2/Sigma2_t)

grid = Grid(num_qubits=6, d=2*np.pi)

# The curve of measurement probabilities, ie abs(psi)**2, will be a Gaussian with
# mean mu and standard deviation sigma/sqrt(2).

mu = 2
sigma = 1/np.sqrt(2)
momentum = 0

psi = np.exp(-(grid.x - mu)**2 / (2 * sigma**2)) * np.exp(1j * momentum * grid.x)
psi *= grid.fftshift_correction
psi /= np.linalg.norm(psi)

fig, axes = plt.subplots(3, 3, squeeze=False, figsize=(15, 8))
for ax, t in zip(axes.flat, [t*0.1 for t in range(9)]):
    ax.set_ylim(top=max(abs(psi)**2)*1.05)
    # These two variables are used to plot the ideal curve from the analytical solution (if desired).
    num_pts = 500
    x_fine = np.linspace(-grid.d, grid.d, num_pts, endpoint=False)

    potential = "no"
    num_steps = 128
    dt = t/num_steps

    match potential:
        case "qho":
            potential_qc = harmonic_potential(grid, dt)
            ideal_curve = analytical_solution_qho(mu, sigma, x_fine, t)
        case "no": # Free particle
            potential_qc = QuantumCircuit(grid.num_qubits)
            ideal_curve = analytical_solution_free(momentum, x_fine-mu, t)
            dt = t # No need for splitting in this case: we can compute for exact t
        case _:
            print("what?")

    dynamics = get_sim_circuit(grid, get_one_iter, potential_qc, dt, t)
    probs = exact_sim(grid, psi, dynamics)

    ax.bar(grid.x, probs, width=grid.dx*0.7)
    ax.set_xlabel("position")
    ax.set_ylabel("probability")
    ax.set_title(f"t={t} (initial p={momentum}), {potential} potential")

    # Note that the analytical solutions are implemented for only 
    # a set of special cases (see each function for which ones),
    # so this plot is only accurate in those situations.
    do_plot_ideal = True
    if do_plot_ideal: 
        ideal_curve /= ideal_curve.sum()
        ax.plot(x_fine, ideal_curve*num_pts/grid.N, "r-")

plt.tight_layout()
plt.show()