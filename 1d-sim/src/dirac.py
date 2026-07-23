from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from sim_essentials import Grid, get_empty_sim, exact_sim, approx_sim

def transport(grid: Grid, dt, num_spinor=1) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor)
    indicator_idx = qc.num_qubits-1 # Spinor indicator is most significant qubit

    qc.h(indicator_idx)

    pos_reg = next(r for r in qc.qregs if r.name == "pos")
    qc.compose(QFTGate(grid.num_qubits).inverse(), pos_reg, inplace=True)

    phases = [dt*np.pi*2**(j+1)/grid.length for j in range(grid.num_qubits)]
    phases[-1] *= -1
    for j, phase in enumerate(phases):
        qc.cp(phase, indicator_idx, j)

    qc.x(indicator_idx)
    for j, phase in enumerate(phases):
        qc.cp(-phase, indicator_idx, j)
    qc.x(indicator_idx)

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
        qc.p(2**j * grid.dx * dt * (2*grid.d - 2**j * grid.dx), j)
        for l in range(j+1, grid.num_qubits):
            qc.cp(-2**(j+l+1) * grid.dx**2 * dt, j, l)

    return qc

def get_sim_circuit(grid: Grid, dt, final_t, m, potential=lambda grid, dt: QuantumCircuit(grid.num_qubits)) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)
    if final_t == 0:
        return qc

    num_iter = ceil(final_t/dt)
    dt = final_t/num_iter

    # Note that the splitting contains consecutive transport(dt/2) terms
    # in the loop, which we combine into a single transport(dt) to lessen
    # the depth of the final circuit.
    qc.compose(transport(grid, dt/2), inplace=True)
    for _ in range(num_iter-1):
        qc.compose(mass(grid, dt, m), inplace=True)
        qc.compose(potential(grid, dt), inplace=True)
        qc.compose(transport(grid, dt), inplace=True)

    qc.compose(mass(grid, dt, m), inplace=True)
    qc.compose(potential(grid, dt), inplace=True)
    qc.compose(transport(grid, dt/2), inplace=True)

    return qc

def plot_zitterbewegung():
    grid = Grid(num_qubits=8, d=10*np.pi)

    psi_1 = np.exp(-(grid.x)**2 / 2)
    psi = np.concatenate((psi_1, psi_1))
    psi /= np.linalg.norm(psi)
    psi = Statevector(psi)

    dt = 1/16
    m=1

    expected_positions = []
    time_resolution = 1/8
    final_t = 30
    times = [t*time_resolution for t in range(ceil(final_t/time_resolution)+1)]
    for t in times:
        dynamics = get_empty_sim(grid, num_spinor=1)
        dynamics.initialize(psi)
        dynamics.compose(get_sim_circuit(grid, dt, time_resolution, m), inplace=True)
        psi = Statevector.from_circuit(dynamics)
        probs = psi.probabilities(qargs=list(range(grid.num_qubits)))

        expected_x = 0
        for k, prob in enumerate(probs):
            x = -grid.d + k*grid.dx
            expected_x += x*prob

        # print(expected_x, t)
        expected_positions.append(expected_x)

    plt.xlabel("time")
    plt.ylabel("expected position")
    plt.title("initial psi_1 = psi_2 = exp(-x^2/2), expected position over time")
    plt.plot(times, expected_positions)

def plot_snapshots():
    grid = Grid(num_qubits=7, d=4*np.pi)

    mu = 0
    sigma_1 = 1
    sigma_2 = 1
    momentum_1 = 2
    momentum_2 = 2

    psi_1 = np.exp(-(grid.x - mu)**2 / (2 * sigma_1**2)) * np.exp(1j * momentum_1 * grid.x)
    psi_2 = np.exp(-(grid.x - mu)**2 / (2 * sigma_2**2)) * np.exp(1j * momentum_2 * grid.x)
    # psi_2 = 0*grid.x
    psi = np.concatenate((psi_1, psi_2))
    psi /= np.linalg.norm(psi)

    # max_y = max(abs(psi_1/np.linalg.norm(psi_1))**2 + abs(psi_2/np.linalg.norm(psi_2))**2)*1.05/2
    fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(15, 8))
    for ax, t in zip(axes.flat, [t for t in range(9)]):
        # ax.set_ylim(top=max_y)

        dt = 1/8
        m=1

        dynamics = get_empty_sim(grid, num_spinor=1)
        dynamics.compose(get_sim_circuit(grid, dt, t, m), inplace=True)

        probs = exact_sim(grid, psi, dynamics, num_spinor=1)

        ax.bar(grid.x, probs, width=0.7*grid.dx)
        ax.set_xlabel("position")
        ax.set_ylabel("probability")
        ax.set_title(f"t={t}")


if __name__ == "__main__":
    plot_snapshots()
    # plot_zitterbewegung()
    plt.tight_layout()
    plt.show()
