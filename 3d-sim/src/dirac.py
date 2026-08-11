from math import ceil
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QFTGate, PauliGate
from qiskit.quantum_info import Pauli, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from matplotlib import pyplot as plt

class Dimension(IntEnum):
    X=0
    Y=1
    Z=2

alpha = [Pauli(f"X{dim.name}") for dim in Dimension]
beta = Pauli("ZI")

@dataclass(frozen=True)
class Grid:
    num_qubits: tuple[int, int, int]
    # We consider the wavefunction on the box [-x_max, x_max] x [-y_max, y_max] x [-z_max, z_max].
    axis_maxes: tuple[float, float, float]

    @property
    def length(self) -> tuple[float, ...]:
        return tuple(2*d_j for d_j in self.axis_maxes)

    @property
    def N(self) -> tuple[int, ...]:
        return tuple(2**n_j for n_j in self.num_qubits)

    @property
    def dspace(self) -> tuple[float, ...]:
        return tuple(length_j/N_j for length_j, N_j in zip(self.length, self.N))

    @property
    def grid(self) -> tuple[np.ndarray, ...]:
        return tuple(np.linspace(-j_max, j_max, num=N_j, endpoint=False) for j_max, N_j in zip(self.axis_maxes, self.N))

    @property
    def num_grid_positions(self):
        num_positions = 1
        for dim in Dimension:
            num_positions *= self.N[dim]
        return num_positions

def get_empty_qc(grid: Grid, measuring: list[Dimension]) -> QuantumCircuit:
    regs = []
    for dim in Dimension:
        regs.append(QuantumRegister(grid.num_qubits[dim], f"pos {dim.name}"))
        if dim in measuring:
            regs.append(ClassicalRegister(grid.num_qubits[dim], f"meas {dim.name}"))

    regs.append(QuantumRegister(2, "spin"))

    return QuantumCircuit(*regs)

def transport(grid: Grid, dim: Dimension, dt) -> QuantumCircuit:
    qc = get_empty_qc(grid, measuring=[])
    pos_reg = next(r for r in qc.qregs if r.name == f"pos {dim.name}")

    s_1 = qc.num_qubits-1
    s_0 = qc.num_qubits-2

    csigma = PauliGate(dim.name).control()

    qc.append(csigma, [s_1, s_0])
    qc.h(s_1)

    qc.compose(QFTGate(grid.num_qubits[dim]).inverse(), pos_reg, inplace=True)

    phases = [dt*np.pi*2**(j+1)/grid.length[dim] for j in range(grid.num_qubits[dim])]
    phases[-1] *= -1
    for j, phase in enumerate(phases):
        qc.cp(phase, s_1, pos_reg[j])

    qc.x(s_1)
    for j, phase in enumerate(phases):
        qc.cp(-phase, s_1, pos_reg[j])
    qc.x(s_1)

    qc.compose(QFTGate(grid.num_qubits[dim]), pos_reg, inplace=True)

    qc.h(s_1)
    qc.append(csigma, [s_1, s_0])

    return qc

def mass(grid: Grid, m, dt) -> QuantumCircuit:
    qc = get_empty_qc(grid, measuring=[])
    s_1 = qc.num_qubits-1

    qc.rz(2*m*dt, s_1)
    return qc

def complete_transport(grid: Grid, dt) -> QuantumCircuit:
    qc = get_empty_qc(grid, measuring=[])
    for dim in Dimension:
        qc.compose(transport(grid, dim, dt), inplace=True)
    return qc

def get_sim_dynamics(grid: Grid, dt, final_t, m) -> QuantumCircuit:
    qc = get_empty_qc(grid, measuring=[])
    if final_t == 0:
        return qc

    num_iter = ceil(final_t/dt)
    dt = final_t/num_iter

    qc.compose(mass(grid, m, dt/2), inplace=True)
    # qc.compose(potential(grid, dt/2), inplace=True)
    for _ in range(num_iter-1):
        qc.compose(complete_transport(grid, dt), inplace=True)
        qc.compose(mass(grid, m, dt), inplace=True)
        # qc.compose(potential(grid, dt), inplace=True)

    qc.compose(complete_transport(grid, dt), inplace=True)
    qc.compose(mass(grid, m, dt/2), inplace=True)
    # qc.compose(potential(grid, dt/2), inplace=True)

    return qc

def estimate_analytics(grid: Grid, num_steps, initialize=False):
    qc = get_empty_qc(grid, measuring=[])
    if initialize:
        psi_1 = get_gaussian_component(grid)
        zero = np.zeros((grid.num_grid_positions,), dtype=complex)
        psi = np.concatenate((psi_1, zero, zero, zero))
        psi /= np.linalg.norm(psi)
        qc.initialize(psi)

    qc.compose(get_sim_dynamics(grid, dt=1, final_t=num_steps, m=1).decompose(), inplace=True)

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    isa_circuit = pm.run(qc)

    fidelity = isa_circuit.estimate_fidelity(target=backend.target)
    print(isa_circuit.count_ops())
    print(f'''backend: {backend.name}
num steps: {num_steps}
estimated fidelity: {fidelity}''')

def get_gaussian_component(grid: Grid):
    gaussian = lambda x, y, z: np.exp(-(x**2+y**2+z**2)/2)

    psi = np.empty((grid.num_grid_positions,), dtype=complex)
    for l in range(grid.N[Dimension.Z]):
        for k in range(grid.N[Dimension.Y]):
            for j in range(grid.N[Dimension.X]):
                x = -grid.axis_maxes[Dimension.X]+j*grid.dspace[Dimension.X]
                y = -grid.axis_maxes[Dimension.Y]+k*grid.dspace[Dimension.Y]
                z = -grid.axis_maxes[Dimension.Z]+l*grid.dspace[Dimension.Z]

                psi[j + 2**(grid.num_qubits[Dimension.X])*k +
                              2**(grid.num_qubits[Dimension.X]+grid.num_qubits[Dimension.Y])*l] = gaussian(x, y, z)
    return psi


grid = Grid((3, 3, 3), (2*np.pi, 2*np.pi, 2*np.pi))
for num_steps in range(12):
    estimate_analytics(grid, num_steps, initialize=True)
    print("")

# psi = np.concatenate((psi_component, psi_component, psi_component, psi_component))
# psi /= np.linalg.norm(psi)
# # print(len(psi))

# qc = get_empty_qc(grid, measuring=[])
# qc.initialize(psi)
# # qc.compose(transport(grid, Dimension.X, 0.5), inplace=True)
# # qc.compose(transport(grid, Dimension.Y, 1), inplace=True)
# qc.compose(transport(grid, Dimension.Z, 2), inplace=True)
# vec = Statevector.from_circuit(qc)
# # qc_decomposed = qc.decompose()
# # print(qc.depth())
# # print(qc_decomposed.depth())
# # qc.decompose().draw(output="mpl", filename="decomposed-transport.png")
# # qc.draw(output="mpl", filename="transport.png")
# # # for qubit in qc.qubits:
# # #     print(qubit)
# plt.bar(np.arange(32), vec.probabilities(qargs=list(range(6, 11))))
# plt.show()
