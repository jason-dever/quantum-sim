import numpy as np
import time
from math import ceil, floor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from qiskit.visualization import plot_state_city

# Consider the wavefunction over the interval [-d, d] at grid distance dx.
num_qubits = 5
d = 2*np.pi
length = 2*d
N = 2**num_qubits

service = QiskitRuntimeService()
# backend = service.least_busy(simulator=False, operational=True)
backend = FakeCasablancaV2()

def kinetic(n, dt):
    qc = QuantumCircuit(n)
    qc.compose(QFTGate(n).inverse(), inplace=True)

    phase = (np.pi/length)**2
    for j in range(n):
        for l in range(j+1, n):
            qc.rzz(2**(j+l+2)*phase*dt, j, l)
    for j in range(n):
        qc.rz(2**(j+2)*phase*dt, j)

    qc.compose(QFTGate(n), inplace=True)
    return qc

# The kinetic energy circuit is the same in all cases, so we don't need to let
# it vary by passing it into this function.
def get_one_iter(potential_qc, dt):
    qc = QuantumCircuit(num_qubits)
    qc.compose(kinetic(num_qubits, dt), inplace=True)
    qc.compose(potential_qc, inplace=True)

    return qc

def get_sim_circuit(potential_qc, dt, final_t):
    qc = QuantumCircuit(num_qubits)
    num_iter = 0 if final_t == 0 else floor(final_t/dt)

    # Since the operators for potential and kinetic energy do not commute
    # we apply the Trotter formula, taking timesteps of length dt from t=0 to t=final_t.
    for k in range(num_iter):
        qc.compose(get_one_iter(potential_qc, dt), inplace=True)

    residual = final_t - dt*num_iter
    if residual > 1e-6:
        qc.compose(get_one_iter(potential_qc, residual), inplace=True)

    return qc


def approx_sim(initial_statevector, potential_qc, dt, final_t, backend, num_shots=256):
    sim = QuantumCircuit(num_qubits)
    sim.initialize(initial_statevector)
    sim.compose(get_sim_circuit(potential_qc, dt, final_t), inplace=True)
    sim.measure_all()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(sim)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = num_shots  

    job = sampler.run([isa_circuit])
    # print(f"Job ID: {job.job_id()}")
    counts = job.result()[0].data.meas.get_counts()

    probs = []
    for k in range(N):
        k_str = format(k, f"0{num_qubits}b")

        count = counts.get(k_str, 0)
        probs.append(count/num_shots)

    return probs

def exact_sim(initial_statevector, potential_qc, dt, final_t):
    sim = QuantumCircuit(num_qubits)
    sim.initialize(initial_statevector)
    sim.compose(get_sim_circuit(potential_qc, dt, final_t), inplace=True)
    return Statevector.from_circuit(sim).probabilities()

def analytic_solution_no_potential(mu, momentum, x, t):
    # For sigma = 1/np.sqrt(2)
    return np.sqrt(1j/(-4*t+1j))*np.exp((-1j*x**2 - momentum*x + momentum**2 * t)/(-4*t+1j))

mu = 0
sigma = 1/np.sqrt(2)
momentum = -2*np.pi

dx = 2*d/N
x = np.linspace(-d, d, num=N, endpoint=False)

psi = np.exp(-(x - mu)**2 / (2 * sigma**2)) * np.exp(1j * momentum * x)
j_idx = np.arange(N)
psi *= (-1)**j_idx
psi /= np.linalg.norm(psi)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, t in zip(axes.flat, [x/10 for x in range(6)]):
    potential = QuantumCircuit(num_qubits)
    probs = exact_sim(psi, potential, t, t)

    ax.bar(x, probs, width=dx*0.75)
    ax.set_xlabel("position")
    ax.set_ylabel("probability")
    ax.set_title(f"t={t} (p={momentum})")

    num_pts = 500
    x_fine = np.linspace(-d, d, num_pts, endpoint=False)
    curve = abs(analytic_solution_no_potential(mu, momentum, x_fine, t))**2
    curve /= curve.sum()
    ax.plot(x_fine, curve*num_pts/N, "r-")

plt.tight_layout()
plt.show()