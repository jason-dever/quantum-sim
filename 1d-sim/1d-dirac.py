from sim_essentials import *
from qiskit.circuit.library import QFTGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def transport(grid: Grid, dt, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)
    pos_reg = next(r for r in qc.qregs if r.name == "pos")

    qc.compose(QFTGate(grid.num_qubits).inverse(), pos_reg, inplace=True)
    indicator_idx = qc.num_qubits-1 # Spinor indicator is most significant qubit

    phase = np.pi/grid.length
    for j in range(grid.num_qubits):
        qc.crz(phase * 2**(j+1) * dt, indicator_idx, j)

    qc.x(indicator_idx)
    for j in range(grid.num_qubits):
        qc.crz(-phase * 2**(j+1) * dt, indicator_idx, j)
    qc.x(indicator_idx)

    qc.compose(QFTGate(grid.num_qubits), pos_reg, inplace=True)

    return qc

grid = Grid(num_qubits=6, d=np.pi)

mu = 0
sigma = 0.5
momentum = 0

psi_1 = np.exp(-(grid.x - mu)**2 / (2 * sigma**2)) * np.exp(1j * momentum * grid.x)
psi_1 *= grid.fftshift_correction
psi = np.concatenate((psi_1, psi_1))
psi /= np.linalg.norm(psi)

initial_statevector = Statevector(psi)
qc = get_empty_sim(grid, num_spinor=1)
qc.initialize(initial_statevector)
qc.compose(transport(grid, 0.75), inplace=True)
plot_histogram(Statevector.from_circuit(qc).probabilities_dict(qargs=list(range(grid.num_qubits))))

# qc.draw("latex", filename="transport.png")
plt.show()