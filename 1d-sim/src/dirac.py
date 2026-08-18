from math import ceil
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sim_essentials import Grid, get_empty_sim, exact_sim, approx_sim, gaussian
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2

def transport(grid: Grid, dt) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)
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

def mass(grid: Grid, dt, m) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)
    indicator_idx = qc.num_qubits-1

    qc.rz(2*m*dt, indicator_idx)
    return qc

def linear_potential(grid: Grid, dt) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)

    for j in range(grid.num_qubits):
        qc.p(-2**j * grid.dx * dt, j)

    return qc

def qho_potential(grid: Grid, dt) -> QuantumCircuit:
    qc = get_empty_sim(grid, num_spinor=1)

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

def one_step_runtime(num_grid_qubits, potential = lambda grid, dt: QuantumCircuit(grid.num_qubits)):
    grid = Grid(num_grid_qubits, 2*np.pi)
    qc = get_empty_sim(grid, num_spinor=1, measuring=True)
    qc.compose(get_sim_circuit(grid, 1, 1, 1, potential), inplace=True)

    pos_reg = next(r for r in qc.qregs if r.name == "pos")
    measurement_reg = next(r for r in qc.cregs if r.name == "meas")
    qc.measure(pos_reg, measurement_reg)

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)

    nshots = 4096
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = nshots

    isa_circuit = pm.run(qc)
    job = sampler.run([isa_circuit])
    span = job.result().metadata["execution"]["execution_spans"]

    runtime = span.stop-span.start
    print(f"{num_grid_qubits} grid qubits, runtime {runtime/timedelta(microseconds=1)}us")
    return runtime/timedelta(microseconds=1)


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
    x_hat = position_operator(grid)
    for t in times:
        dynamics = get_empty_sim(grid, num_spinor=1)
        dynamics.initialize(psi)
        dynamics.compose(get_sim_circuit(grid, dt, time_resolution, m), inplace=True)

        psi = Statevector.from_circuit(dynamics)
        print(psi.expectation_value(x_hat))
        expected_positions.append(psi.expectation_value(x_hat))

    plt.xlabel("time")
    plt.ylabel("expected position")
    plt.title("initial psi_1 = psi_2 = exp(-x^2/2), expected position over time")
    plt.plot(times, expected_positions)

def position_operator(grid: Grid):
    terms = []
    terms.append(("I"*(grid.num_qubits+1), grid.dx*(grid.N-1)/2 - grid.d))

    for k in range(grid.num_qubits):
        z_k = "I"*(grid.num_qubits-k) + "Z" + "I"*k
        terms.append((z_k, -grid.dx * 2**(k-1)))

    return SparsePauliOp.from_list(terms)

def plot_snapshots():
    grid = Grid(num_qubits=6, d=2.5*np.pi)

    mu = 0
    sigma_1 = 1
    sigma_2 = 1
    momentum_1 = 5
    momentum_2 = 0

    psi_1 = np.exp(-(grid.x - mu)**2 / (2 * sigma_1**2)) * np.exp(1j * momentum_1 * grid.x)
    # psi_2 = np.exp(-(grid.x - mu)**2 / (2 * sigma_2**2)) * np.exp(1j * momentum_2 * grid.x)
    psi_2 = 0*grid.x
    psi = np.concatenate((psi_1, psi_2))
    psi /= np.linalg.norm(psi)

    # max_y = max(abs(psi_1/np.linalg.norm(psi_1))**2 + abs(psi_2/np.linalg.norm(psi_2))**2)*1.05/2
    fig, axes = plt.subplots(3, 2, squeeze=False, figsize=(15, 8))
    for ax, t in zip(axes.flat, [t for t in range(9)]):
        # ax.set_ylim(top=max_y)

        dt = 1
        m=0
        print(f"num_steps: {ceil(t/dt)}")

        dynamics = get_empty_sim(grid, num_spinor=1)
        dynamics.compose(get_sim_circuit(grid, dt, t, m), inplace=True)

        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)
        # backend = FakeCasablancaV2()

        probs = approx_sim(grid, psi, dynamics, num_spinor=1, backend=backend, num_shots=2048)
        # probs = exact_sim(grid, psi, dynamics, num_spinor=1)

        ax.bar(grid.x, probs, width=0.7*grid.dx)
        ax.set_xlabel("position")
        ax.set_ylabel("probability")
        ax.set_title(f"t={t}")


if __name__ == "__main__":
    # times = []
    # ngrid_positions = []
    # for nqubits in range(1, 24):
    #     times.append(one_step_runtime(nqubits, qho_potential))
    #     ngrid_positions.append(2**nqubits)

    # plt.plot(ngrid_positions, times)

    # grid = Grid(num_qubits=6, d=2*np.pi)
    # one_step_circuit = get_sim_circuit(grid, 1, 1, 1, qho_potential)
    # one_step_circuit.draw(output="mpl", filename="one_step_circuit_1d.png")

    # plot_snapshots()
    plot_zitterbewegung()
    plt.tight_layout()
    plt.show()
