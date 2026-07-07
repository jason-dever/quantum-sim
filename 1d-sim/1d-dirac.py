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
        qc.cp(phase * 2**(j+1) * dt, indicator_idx, j)

    qc.x(indicator_idx)
    for j in range(grid.num_qubits):
        qc.cp(-phase * 2**(j+1) * dt, indicator_idx, j)
    qc.x(indicator_idx)

    qc.p(-2 * np.pi * grid.N/grid.length * dt, indicator_idx)

    qc.compose(QFTGate(grid.num_qubits), pos_reg, inplace=True)

    return qc

grid = Grid(num_qubits=6, d=2*np.pi)

mu = 0
sigma_1 = 1/np.sqrt(2)
sigma_2 = 0.3   
momentum_1 = 3
momentum_2 = -2

psi_1 = np.exp(-(grid.x - mu)**2 / (2 * sigma_1**2)) * np.exp(1j * momentum_1 * grid.x)
psi_1 *= grid.fftshift_correction
psi_2 = np.exp(-(grid.x - mu)**2 / (2 * sigma_2**2)) * np.exp(1j * momentum_2 * grid.x)
psi_2 *= grid.fftshift_correction
psi = np.concatenate((psi_1, psi_2))
psi /= np.linalg.norm(psi)

initial_statevector = Statevector(psi)
fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(15, 8))

for ax, t in zip(axes.flat, [t for t in range(6)]):
    # num_pts = 500
    # x_fine = np.linspace(-grid.d, grid.d, num_pts, endpoint=False)
    # ideal_curve = abs(np.exp(-(x_fine - mu - t)**2 / (2 * sigma_1**2)))**2 + abs(np.exp(-(x_fine - mu + t)**2 / (2 * sigma_2**2)))**2
    # ideal_curve /= ideal_curve.sum()
    # ax.plot(x_fine, ideal_curve*num_pts/grid.N, "r-")

    dynamics = get_empty_sim(grid, num_spinor=1)
    dynamics.initialize(initial_statevector)
    dynamics.compose(transport(grid, t), inplace=True)

    # probs = exact_sim(grid, psi, dynamics, num_spinor=1)

    # ax.bar(grid.x, probs, width=grid.dx*0.7)
    # ax.set_xlabel("position")
    # ax.set_ylabel("probability")
    # ax.set_title(f"t={t}")

    psi_1_ideal = np.exp(-(grid.x - mu - t)**2 / (2 * sigma_1**2)) * np.exp(1j * momentum_1 * (grid.x-t))
    psi_1_ideal *= grid.fftshift_correction
    psi_2_ideal = np.exp(-(grid.x - mu + t)**2 / (2 * sigma_2**2)) * np.exp(1j * momentum_2 * (grid.x+t))
    psi_2_ideal *= grid.fftshift_correction
    psi_ideal = np.concatenate((psi_1_ideal, psi_2_ideal))
    psi_ideal /= np.linalg.norm(psi_ideal)
    psi_ideal = Statevector(psi_ideal)
    print(abs(Statevector.from_circuit(dynamics).inner(Statevector(psi_ideal)))**2)

# plt.tight_layout()
# plt.show()