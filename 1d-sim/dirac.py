from sim_essentials import *
from qiskit.circuit.library import QFTGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def transport(grid: Grid, dt, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)
    indicator_idx = qc.num_qubits-1 # Spinor indicator is most significant qubit

    qc.h(indicator_idx)

    pos_reg = next(r for r in qc.qregs if r.name == "pos")
    qc.compose(QFTGate(grid.num_qubits).inverse(), pos_reg, inplace=True)

    phase = dt*np.pi/grid.length
    for j in range(grid.num_qubits):
        qc.cp(phase * 2**(j+1), indicator_idx, j)

    qc.x(indicator_idx)
    for j in range(grid.num_qubits):
        qc.cp(-phase * 2**(j+1), indicator_idx, j)
    qc.x(indicator_idx)

    qc.rz(-2*phase*grid.N, indicator_idx)

    qc.compose(QFTGate(grid.num_qubits), pos_reg, inplace=True)

    qc.h(indicator_idx)

    return qc

def mass(grid: Grid, dt, m, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)
    indicator_idx = qc.num_qubits-1

    qc.rz(2*m*dt, indicator_idx)
    return qc

def linear_potential(grid: Grid, dt, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)

    for j in range(grid.num_qubits):
        qc.p(-2**j * grid.dx * dt, j)

    return qc

def qho_potential(grid: Grid, dt, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)

    for j in range(grid.num_qubits):
        qc.p(2**j * grid.dx * dt *(2*grid.d - 2**j * grid.dx), j)
        for l in range(j+1, grid.num_qubits):
            qc.cp(-2**(j+l+1) * grid.dx**2 * dt, j, l)

    return qc

def get_one_iter(grid: Grid, dt, m, potential, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)

    qc.compose(transport(grid, dt/2, num_spinor), inplace=True)
    qc.compose(mass(grid, dt, m, num_spinor), inplace=True)
    qc.compose(potential(grid, dt, num_spinor), inplace=True)
    qc.compose(transport(grid, dt/2, num_spinor), inplace=True)

    return qc

def get_sim_circuit(grid: Grid, get_one_iter, dt, final_t, m, potential=lambda grid, dt, num_spinor: QuantumCircuit(grid.num_qubits + num_spinor)) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)
    num_iter = 0 if final_t == 0 else floor(final_t/dt)

    # Using operator splitting to approximately solve the equation, we take 
    # timesteps of length dt from t=0 to the last step before/at final_t.
    for _ in range(num_iter):
        qc.compose(get_one_iter(grid, dt, m, potential), inplace=True)

    residual = final_t - dt*num_iter
    if residual > 1e-6:
        qc.compose(get_one_iter(grid, residual, m, potential), inplace=True)

    return qc

if __name__ == "__main__":
    grid = Grid(num_qubits=7, d=4*np.pi)

    mu = 5
    sigma_1 = 1
    sigma_2 = 1
    momentum_1 = 0
    momentum_2 = 0

    psi_1 = np.exp(-(grid.x - mu)**2 / (2 * sigma_1**2)) * np.exp(1j * momentum_1 * grid.x)
    psi_1 *= grid.fftshift_correction
    # psi_2 = np.exp(-(grid.x - mu)**2 / (2 * sigma_2**2)) * np.exp(1j * momentum_2 * grid.x)
    psi_2 = 0*grid.x
    psi_2 *= grid.fftshift_correction
    psi = np.concatenate((psi_1, psi_2))
    psi /= np.linalg.norm(psi)

    # max_y = max(abs(psi_1/np.linalg.norm(psi_1))**2 + abs(psi_2/np.linalg.norm(psi_2))**2)*1.05/2
    initial_statevector = Statevector(psi)
    fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(15, 8))
    for ax, t in zip(axes.flat, [t for t in range(6)]):
        # ax.set_ylim(top=max_y)

        dt = 1/8
        num_steps = t/dt
        m=1

        dynamics = get_empty_sim(grid, num_spinor=1)
        dynamics.initialize(initial_statevector)
        dynamics.compose(get_sim_circuit(grid, get_one_iter, dt, t, m, potential=qho_potential), inplace=True)

        probs = exact_sim(grid, psi, dynamics, num_spinor=1)

        ax.bar(grid.x, probs, width=0.7*grid.dx)
        ax.set_xlabel("position")
        ax.set_ylabel("probability")
        ax.set_title(f"t={t}")

    plt.tight_layout()
    plt.show()