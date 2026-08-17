import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2 as Sampler
from math import floor
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager

@dataclass(frozen=True)
class Grid: # This is a centralized config object for discretizing a 1D grid.
    num_qubits: int
    d: float # Considering the wavefunction on the interval [-d, d] at steps dx.

    @property
    def length(self) -> float:
        return 2*self.d

    @property
    def N(self) -> int: # Number of grid positions
        return 2**self.num_qubits

    @property
    def dx(self) -> float:
        return self.length/self.N

    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.d, self.d, num=self.N, endpoint=False)

    @property
    def fftshift_correction(self) -> np.ndarray:
        return (-1)**np.arange(self.N)

def get_empty_sim(grid: Grid, num_spinor: int, measuring=False):
    pos_reg = QuantumRegister(grid.num_qubits, "pos")
    regs = [pos_reg]

    if num_spinor:
        regs.append(QuantumRegister(num_spinor, "spin"))
    if measuring:
        regs.append(ClassicalRegister(grid.num_qubits, "meas"))

    return QuantumCircuit(*regs)

def approx_sim(grid: Grid, initial_statevector, dynamics: QuantumCircuit, backend, num_spinor=0, num_shots=256):
    sim = get_empty_sim(grid, num_spinor, measuring=True)
    sim.initialize(initial_statevector)
    sim.compose(dynamics, inplace=True)

    pos_reg = next(r for r in sim.qregs if r.name == "pos")
    measurement_reg = next(r for r in sim.cregs if r.name == "meas")
    sim.measure(pos_reg, measurement_reg)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(sim)
    print(f"isa circuit ops: {dict(isa_circuit.count_ops())}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = num_shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True

    job = sampler.run([isa_circuit])
    # print(f"Job ID: {job.job_id()}")
    counts = job.result()[0].data.meas.get_counts()

    probs = []
    for k in range(grid.N):
        k_str = format(k, f"0{grid.num_qubits}b")

        count = counts.get(k_str, 0)
        probs.append(count/num_shots)

    return probs

def exact_sim(grid: Grid, initial_statevector, dynamics: QuantumCircuit, num_spinor=0):
    sim = get_empty_sim(grid, num_spinor)
    sim.initialize(initial_statevector)
    sim.compose(dynamics, inplace=True)

    return Statevector.from_circuit(sim).probabilities(qargs=list(range(grid.num_qubits)))
