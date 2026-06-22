import numpy as np
import time
from math import ceil, floor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from qiskit.visualization import plot_state_city

# Consider the wavefunction over the interval [-d, d] at grid distance dx.
dx = np.pi/8
d = np.pi
length = 2*d
num_qubits = ceil(np.log2(length/dx + 1))

service = QiskitRuntimeService()
# backend = service.least_busy(simulator=False, operational=True)
backend = FakeBelemV2()

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
    num_iter = floor(final_t/dt)

    # Since the operators for potential and kinetic energy do not commute
    # we apply the Trotter formula, taking timesteps of length dt from t=0 to t=final_t.
    for k in range(num_iter):
        qc.compose(get_one_iter(potential_qc, dt), inplace=True)

    # If final_t is not a multiple of dt, we iterate to the largest
    # time step before final_t in the loop above and step to final_t here.
    qc.compose(get_one_iter(potential_qc, final_t-dt*num_iter), inplace=True)

    return qc


def sim(initial_statevector, potential_qc, dt, final_t, backend):
    sim = QuantumCircuit(num_qubits)
    sim.initialize(initial_statevector)
    sim.compose(get_sim_circuit(potential_qc, dt, final_t), inplace=True)
    sim.measure_all()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(sim)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 512  

    job = sampler.run([isa_circuit])
    # print(f"Job ID: {job.job_id()}")
    counts = job.result()[0].data.meas.get_counts()

    # Fill counts with all possible measurement outcomes; by default 
    # outcomes that are not observed are not included in the dict by Qiskit.
    for k in range(2**num_qubits):
        k_str = format(k, f"0{num_qubits}b")
        if k_str not in counts:
            counts[k_str] = 0

    return counts

mu = 0
sigma = 0.3
momentum = 0
dt = 0.8

x = np.linspace(-d, d, num=2**num_qubits)

psi_0 = np.exp(-(x - mu)**2 / (2 * sigma**2)) * np.exp(1j * momentum * x)
psi_0 /= np.linalg.norm(psi_0)
initial_statevector = Statevector(psi_0)

# qc = QuantumCircuit(num_qubits)
# qc.initialize(initial_statevector)
# qc.compose(get_sim_circuit(QuantumCircuit(num_qubits), 0.05, 0.05), inplace=True)
# final = Statevector.from_circuit(qc)
# plot_state_city(final)

for t in [x/20 for x in range(1, 6)]:
    counts = sim(initial_statevector, QuantumCircuit(num_qubits), dt, t, backend)
    plot_histogram(counts, title=f"after time {t} (p=0)")
plt.show()