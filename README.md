
# Weird Machine Gadgets

A Python library for constructing **weird machines** — emergent computational systems built from the composition of **architectural gadgets** in cyber-physical and control systems. The library provides a framework to explore how low-level protocol primitives, such as Modbus function codes, can be combined to form computational logic without deploying any software on embedded targets.

This project was developed as part of the **Old Dominion University Wind Energy Testbed Initiative** to teach **emergent computation** and **industrial control system security** using programmable control protocols.

---

## What Are Weird Machines?

A *weird machine* is an unintended computational system that emerges within existing architectures when legitimate operations can be composed into meaningful computation.  

This library demonstrates how **industrial control instructions**—such as Modbus reads, writes, and coils—can act as **Turing-complete computation primitives** when systematically combined.

**Key Idea:**  
Modbus function codes are not just commands — they are *computational instructions*. Weird Machine Gadgets allows you to build buffers, counters, timers, and even full state machines purely from Modbus operations.

---

## Architecture

The library follows a modular architecture consisting of three conceptual layers:

### Layer 1: Protocol Processing Gadgets (PPG)
Implements Modbus operations as computational primitives.

Functions:
- `read_register()` / `write_register()` — Read and write numeric values  
- `read_coil()` / `write_coil()` — Boolean operations (FC01, FC05 analogues)  
- `increment_register()` / `decrement_register()` — Atomic arithmetic steps  

Serves as the foundation for all other gadget layers.

---

### Layer 2: Control Logic Gadgets (CLG)
Implements logic and control behaviors through compositional functions:
- `wait_for_condition()` — Conditional polling  
- `wait_for_register_value()` — Synchronization primitive  
- `conditional_write()` — Simple IF-THEN-ELSE  
- `repeat_until()` — Loop construction without Python iteration  

These functions enable the creation of algorithmic flow using structured protocol sequences.

---

### Layer 3: Composite Gadgets
Complex behaviors composed from Protocol and Control Gadgets.

Classes:
- `StateMachine` — Finite-state system built from register transitions  
- `ConditionalExecutor` — Threshold-triggered actions and logic predicates  

These enable full weird-machine composition using architectural semantics alone.

---

## Installation

```

git clone https://github.com/your-org/weird-machine-gadgets.git
cd weird-machine-gadgets
pip install -r requirements.txt

```

---

## Running the Examples

To run examples, execute from the **top-level directory** (outside the `weird_machine_gadgets` package).  
Each example must be run as a module:

```

python3 -m weird_machine_gadgets.examples.counter_example
python3 -m weird_machine_gadgets.examples.timer_example
python3 -m weird_machine_gadgets.examples.simple_average_example

```

---

## Example Programs

### 1. Counter Example (`counter_example.py`)

Demonstrates a **tally counter** built only from read/write primitives.  
Simulates hardware event counting—such as button presses or sensor triggers.

Concepts:
- Uses `repeat_until()` for architectural looping  
- Increments a Modbus register each event  
- Cleanly stops upon reaching maximum threshold  

**Execution:**
```

python3 -m weird_machine_gadgets.examples.counter_example

```

Expected outcome:
- Register `40010` holds the running count  
- Coil `10010` simulates event pulses  
- Stops when count reaches configured maximum  

---

### 2. Timer Example (`timer_example.py`)

Implements a procedural **countdown timer** using Modbus operations instead of language loops.  
All control is implemented through architectural primitives such as read, write, and coil modulation.

Concepts:
- Countdown performed via `repeat_until()` loop gadget  
- Registers used as counters  
- Coil flags indicate timer active/inactive states  
- Includes a payload phase at completion (e.g. trigger an operation)

**Execution:**
```

python3 -m weird_machine_gadgets.examples.timer_example

```

---

### 3. Average Example (`simple_average_example.py`)

Implements an **integer average** calculator composed entirely of Modbus register operations.  
Each number is injected into a register space; the sum, count, and average are updated step-by-step.

Concepts:
- Demonstrates mathematical operations built from register I/O  
- Teaches arithmetic control in deterministic sequences  
- Uses `40060–40063` range for input, sum, count, and average  

**Execution:**
```

python3 -m weird_machine_gadgets.examples.simple_average_example

```

---

## Educational Use

Weird Machine Gadgets is an instructional and research framework designed for:

- Cyber-physical system security education  
- Industrial protocol behavior analysis  
- Emergent computation theory demonstrations  
- Safe experimentation with control system computation  

### Learning Objectives
Students and researchers learn to:
1. Compose protocol operations into computation.
2. Recognize architecture-level computation.
3. Understand how unintended programmability emerges.
4. Identify and defend against control-plane weird machines.

---

## API Reference

### ProtocolGadget
Implements Modbus-style register and coil operations.
```

class ProtocolGadget(host: str, port: int = 502, unit: int = 1)

```
Methods:
- `read_register(address: int)`  
- `write_register(address: int, value: int)`  
- `read_coil(address: int)`  
- `write_coil(address: int, value: bool)`  
- `increment_register(address: int)`  
- `decrement_register(address: int)`  

### ControlGadget
Implements control primitives using Modbus semantics.
```

class ControlGadget(protocol: ProtocolGadget)

```
- `wait_for_condition(condition, timeout=None)`  
- `conditional_write(register, condition_value, true_value, false_value)`  
- `repeat_until(condition, body, delay)`  

### StateMachine
```

class StateMachine(protocol: ProtocolGadget, register: int = 40002, transition_coil: int = 10002)

```
- `define_state(name: str, value: int)`  
- `set_state(name: str)`  
- `get_state()`  

### ConditionalExecutor
```

class ConditionalExecutor(protocol: ProtocolGadget)

```
- `add_condition(register, threshold, comparison, action)`  
- `check_conditions()`  
- `monitor(interval, duration)`  

---

