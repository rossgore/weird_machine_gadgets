"""
Weird Machine Gadget Library (Simulation Version)
==================================================

This lightweight version simulates the Modbus-based weird machine logic
entirely in memory without requiring network connections or hardware.

The architecture and variable names match the original Modbus library,
but underlying operations are replaced with in-memory dictionaries that
mimic register and coil reads/writes.

Gadget layers:
 - ProtocolGadget: Simulated Modbus read/write and atomic updates
 - ControlGadget:  Loops and conditional logic
 - StateMachine:   Register-based state transitions
 - ConditionalExecutor: Threshold/event triggers
"""

import time
from typing import Optional, Callable


# =============================================================================
# LAYER 1: Protocol Processing Gadgets (PPG)
# =============================================================================

class ProtocolGadget:
    """Simulated Modbus layer implementing memory-based read/write."""

    def __init__(self, host: str, port: int = 502, unit: int = 1):
        self.host = host
        self.port = port
        self.unit = unit
        self._connected = False
        # Simulated data maps
        self.registers = {}
        self.coils = {}

    # -------------------------------------------------------------------------
    # Connection simulate
    # -------------------------------------------------------------------------
    def connect(self) -> bool:
        print(f"[Simulated Connect] host={self.host}, port={self.port}")
        self._connected = True
        return self._connected

    def disconnect(self):
        if self._connected:
            print(f"[Simulated Disconnect] host={self.host}:{self.port}")
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # -------------------------------------------------------------------------
    # Register Operations (Simulated FC03 / FC06)
    # -------------------------------------------------------------------------
    def read_register(self, address: int) -> Optional[int]:
        val = self.registers.get(address, 0)
        print(f"[Sim FC03] Read register {address} = {val}")
        return val

    def write_register(self, address: int, value: int) -> bool:
        self.registers[address] = int(value)
        print(f"[Sim FC06] Write register {address} = {value}")
        return True

    # -------------------------------------------------------------------------
    # Coil Operations (Simulated FC01 / FC05)
    # -------------------------------------------------------------------------
    def read_coil(self, address: int) -> Optional[bool]:
        val = self.coils.get(address, False)
        print(f"[Sim FC01] Read coil {address} = {val}")
        return val

    def write_coil(self, address: int, value: bool) -> bool:
        self.coils[address] = bool(value)
        print(f"[Sim FC05] Write coil {address} = {value}")
        return True

    # -------------------------------------------------------------------------
    # Atomic Operations
    # -------------------------------------------------------------------------
    def increment_register(self, address: int) -> Optional[int]:
        val = self.registers.get(address, 0) + 1
        self.registers[address] = val
        print(f"[Sim Atomic] Increment register {address} → {val}")
        return val

    def decrement_register(self, address: int) -> Optional[int]:
        val = max(0, self.registers.get(address, 0) - 1)
        self.registers[address] = val
        print(f"[Sim Atomic] Decrement register {address} → {val}")
        return val


# =============================================================================
# LAYER 2: Control Logic Gadgets (CLG)
# =============================================================================

class ControlGadget:
    """Simulated loop and condition control abstraction."""

    def __init__(self, protocol: ProtocolGadget):
        self.protocol = protocol

    def wait_for_condition(self, condition: Callable[[], bool],
                           timeout: float = None,
                           poll_interval: float = 0.1) -> bool:
        """Polls until a condition returns True."""
        start = time.time()
        while True:
            if condition():
                return True
            if timeout and (time.time() - start >= timeout):
                return False
            time.sleep(poll_interval)

    def wait_for_register_value(self, address: int, value: int,
                                timeout: float = None) -> bool:
        return self.wait_for_condition(
            lambda: self.protocol.read_register(address) == value,
            timeout=timeout
        )

    def wait_for_coil_state(self, address: int, state: bool,
                            timeout: float = None) -> bool:
        return self.wait_for_condition(
            lambda: self.protocol.read_coil(address) == state,
            timeout=timeout
        )

    def conditional_write(self, register: int,
                          condition_value: int,
                          true_value: int,
                          false_value: int) -> bool:
        """Simulate if-then-else for register operation."""
        val = self.protocol.read_register(register)
        set_val = true_value if val == condition_value else false_value
        print(f"[Sim Conditional] register {register}: {val} → {set_val}")
        return self.protocol.write_register(register, set_val)

    def repeat_until(self, condition: Callable[[], bool],
                     body: Callable[[], None],
                     delay: float = 0.1,
                     max_iterations: int = None) -> int:
        count = 0
        while True:
            if condition():
                break
            if max_iterations and count >= max_iterations:
                break
            body()
            count += 1
            if delay > 0:
                time.sleep(delay)
        return count


# =============================================================================
# LAYER 3: Composite Gadgets (StateMachine / ConditionalExecutor)
# =============================================================================

class StateMachine:
    """Simulates Modbus-like state machine transitions."""

    def __init__(self, protocol: ProtocolGadget,
                 register: int = 40002, transition_coil: int = 10002):
        self.protocol = protocol
        self.register = register
        self.transition_coil = transition_coil
        self.states = {}

    def define_state(self, name: str, value: int):
        self.states[name] = value
        print(f"[Sim State] Defined {name} = {value}")

    def set_state(self, name: str) -> bool:
        if name not in self.states:
            print(f"[Sim State] Unknown state {name}")
            return False
        code = self.states[name]
        self.protocol.write_register(self.register, code)
        print(f"[Sim State] Transition → {name} (code={code})")
        return True

    def get_state(self) -> Optional[str]:
        code = self.protocol.read_register(self.register)
        for name, val in self.states.items():
            if val == code:
                return name
        return f"UNKNOWN({code})"


class ConditionalExecutor:
    """Simulates logic-based triggers that execute actions."""

    def __init__(self, protocol: ProtocolGadget):
        self.protocol = protocol
        self.conditions = []

    def add_condition(self, register: int, threshold: int,
                      comparison: str, action: Callable):
        self.conditions.append({
            "register": register,
            "threshold": threshold,
            "comparison": comparison,
            "action": action,
        })
        print(f"[Sim Trigger] Registered condition on {register} {comparison} {threshold}")

    def check_conditions(self) -> list:
        triggered = []
        for i, cond in enumerate(self.conditions):
            value = self.protocol.read_register(cond["register"])
            c = cond["comparison"]
            satisfied = (
                (c == ">" and value > cond["threshold"])
                or (c == "<" and value < cond["threshold"])
                or (c == "==" and value == cond["threshold"])
                or (c == ">=" and value >= cond["threshold"])
                or (c == "<=" and value <= cond["threshold"])
            )
            if satisfied:
                print(f"[Sim Trigger] Condition {i} satisfied: {value} {c} {cond['threshold']}")
                cond["action"]()
                triggered.append(i)
        return triggered

    def monitor(self, interval: float = 0.1, duration: float = None):
        start = time.time()
        while True:
            self.check_conditions()
            if duration and (time.time() - start >= duration):
                break
            time.sleep(interval)


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "0.3.0-sim"
__all__ = ["ProtocolGadget", "ControlGadget", "StateMachine", "ConditionalExecutor"]

#Also make SimpleWeirdMachine Available
from weird_machine_gadgets.simple import SimpleWeirdMachine