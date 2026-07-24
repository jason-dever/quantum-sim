from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

class Dims(IntEnum):
    x=0
    y=1
    z=2

@dataclass(frozen=True)
class Grid:
    n: (int, int, int)
    # We consider the wavefunction on the box [-x_max, x_max] x [-y_max, y_max] x [-z_max, z_max].
    axis_maxes: (float, float, float)

    @property
    def length(self) -> tuple[float, ...]:
        return tuple(2*d_j for d_j in self.axis_maxes)

    @property
    def N(self) -> tuple[int, ...]:
        return tuple(2**n_j for n_j in self.n)

    @property
    def dspace(self) -> tuple[float, ...]:
        return tuple(length_j/N_j for length_j, N_j in zip(self.length, self.N))

    @property
    def grid(self) -> tuple[np.ndarray, ...]:
        return tuple(np.linspace(-j_max, j_max, num=N_j, endpoint=False) for j_max, N_j in zip(self.axis_maxes, self.N))

def get_empty_sim(grid: Grid, measuring: list[str]) -> QuantumCircuit:
    regs = []
    for dim in Dims:
        regs.append(QuantumRegister(grid.n[dim], f"pos {dim.name}"))
        if dim.name in measuring:
            regs.append(ClassicalRegister(grid.n[dim], f"meas {dim.name}"))

    regs.append(QuantumRegister(2, "spin"))

    return QuantumCircuit(*regs)

grid = Grid((3, 4, 5), (np.pi, 2*np.pi, 4*np.pi))
qc = get_empty_sim(grid, ["x", "y", "z"])
qc.draw(output="latex", filename="circ.png")
for qubit in qc.qubits:
    print(qubit)
