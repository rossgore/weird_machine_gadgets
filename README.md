
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

# Essential Methods to Add to SimpleWeirdMachine

## Analysis: Mapping Gadget Categories to Core Methods

After analyzing the comprehensive dataset of weird machine gadgets extracted from industrial control system manuals, I've identified **4 essential methods** that would provide the foundation for supporting all major gadget categories while maintaining the beginner-friendly nature of `SimpleWeirdMachine`.

The existing class already handles:

- **Variables (registers)**: `set_variable()`, `get_variable()`, `increment_variable()`, `decrement_variable()`
- **Flags (coils)**: `set_flag()`, `get_flag()`
- **Basic control flow**: `wait_until()`, `repeat_until_done()`

**What's missing**: The ability to perform arithmetic operations on remote data, make decisions based on remote state comparisons, and compose multiple operations into sequences.

These 4 methods address those gaps and unlock the full spectrum of weird machine gadgets found in real industrial control systems.

---

## The 4 Essential Methods

### 1. `compute(expression: str) -> int`

**Category**: Arithmetic/Computation Gadgets

**What it does**: Evaluates a mathematical expression using variables stored on the remote PLC without requiring multiple round-trip communications. Think of it as a calculator that operates on the PLC's memory rather than Python's memory.

**Why it's essential**: In real PLCs, the CPT (Compute) instruction allows you to write expressions like `Output = (Input - 32) * 5 / 9` all in one operation. Currently, `SimpleWeirdMachine` would require you to fetch the input value to Python, perform the calculation locally, and write it back. This method mirrors how real PLCs work by expressing the computation as a single operation.

**Method signature**:
```python
def compute(self, expression: str) -> int
```

**Parameters**:
- `expression` (str): A mathematical expression using variable names, numbers, and operators (+, -, *, /, %, parentheses)

**Returns**:
- `int`: The computed result

**Detailed behavior**:

1. Parse the expression string to identify all variable names
2. For each variable name found, fetch its current value from the PLC using the existing `get_variable()` method
3. Substitute the variable names in the expression with their actual numeric values
4. Evaluate the mathematical expression safely (no arbitrary code execution)
5. Return the integer result

The result is **not automatically written back** to the PLC. This gives you flexibility to use the result in Python logic or explicitly store it with `set_variable()`.

**Example 1: Temperature conversion (Fahrenheit to Celsius)**

Setup:
```python
machine.set_variable('temp_f', 98)
```

Call:
```python
result = machine.compute('(temp_f - 32) * 5 / 9')
```

What happens internally:
- The method sees `temp_f` in the expression
- Reads `temp_f` from PLC → gets value 98
- Substitutes: `(98 - 32) * 5 / 9`
- Evaluates: `66 * 5 / 9 = 36` (integer division)
- Returns: `36`

**Example 2: Sensor scaling (ADC to engineering units)**

Setup:
```python
machine.set_variable('raw_adc', 2048)    # 12-bit ADC reading
machine.set_variable('adc_min', 0)
machine.set_variable('adc_max', 4095)
machine.set_variable('eng_min', 0)
machine.set_variable('eng_max', 100)
```

Call:
```python
scaled = machine.compute('(raw_adc - adc_min) * (eng_max - eng_min) / (adc_max - adc_min) + eng_min')
machine.set_variable('pressure_psi', scaled)
```

What happens:
- Reads all 5 variables from PLC
- Substitutes: `(2048 - 0) * (100 - 0) / (4095 - 0) + 0`
- Evaluates: `2048 * 100 / 4095 + 0 = 50`
- Returns: `50`
- You then store it back as `pressure_psi`

**Example 3: Polynomial approximation (thermistor linearization)**

Setup:
```python
# Polynomial coefficients for temperature curve
machine.set_variable('a0', 25)
machine.set_variable('a1', 15)
machine.set_variable('a2', -2)
machine.set_variable('voltage', 3)
```

Call:
```python
temp = machine.compute('a0 + a1 * voltage + a2 * voltage * voltage')
```

What happens:
- Reads coefficients and voltage
- Substitutes: `25 + 15 * 3 + (-2) * 3 * 3`
- Evaluates: `25 + 45 + (-18) = 52`
- Returns: `52` (degrees Celsius)

**Value**: 

This demonstrates that PLC instructions aren't fundamentally different from Python—they're just operating on data in a different memory space. Users learn that expressions like `CPT` in Logix 5000 or `CALCULATE` in SIEMENS are simply arithmetic operations composed together, not magic. The key insight is that weird machines can exploit these arithmetic gadgets by chaining them in unintended ways to perform computation the PLC designer never anticipated.

**Gadget categories supported**:
- Arithmetic composition (ADD/SUB/MUL/DIV chains)
- Linear scaling networks
- Polynomial approximation
- Power calculations (voltage × current)

**Implementation considerations**:

1. **Expression parsing and variable identification**: Use regular expressions to identify variable names in the expression. A pattern like `r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'` will match valid Python identifiers. Be careful to exclude Python keywords and built-in function names. You'll need to iterate through all matches and check if each identifier exists in `self._variable_map` to distinguish variables from potential function names or keywords.

2. **Safe evaluation without security vulnerabilities**: Never use `eval()` directly on user input, as it can execute arbitrary Python code. Instead, after substituting variable values, use `eval()` with restricted globals and locals: `eval(expression, {"__builtins__": {}}, {})`. This prevents access to dangerous functions like `open()` or `import`. Alternatively, consider using the `ast` module to parse the expression into an Abstract Syntax Tree and validate that it only contains safe operations (arithmetic operators, numbers, parentheses) before evaluating.

3. **Handling operator precedence and integer arithmetic**: Python's `eval()` naturally respects mathematical operator precedence (multiplication before addition, etc.), which matches PLC behavior. However, be aware that Python 3 uses true division (`/`) by default, which returns floats. Since PLCs typically use integer arithmetic, you may want to convert the final result to `int()`. Consider whether you want to use floor division (`//`) in expressions or convert after evaluation. Document this behavior clearly so users understand how division is handled.

---

### 2. `compare(left_var: str, operator: str, right_value: Union[int, str]) -> bool`

**Category**: Control-Flow Gadgets (Branching)

**What it does**: Compares a variable stored on the PLC against either a constant value or another PLC variable, returning a boolean result. This enables decision-making based on remote state without fetching the data to Python first.

**Why it's essential**: Real PLCs make control decisions using comparison instructions like LES (Less Than), GRT (Greater Than), and LIM (Limit Test). For example, "only enable the pump if tank level is below 80%". Currently, you'd need to fetch the tank level to Python and use a Python `if` statement. This method mirrors how PLCs actually implement branching logic—by comparing values in their own memory space.

**Method signature**:
```python
def compare(self, left_var: str, operator: str, right_value: Union[int, str]) -> bool
```

**Parameters**:
- `left_var` (str): Name of the variable on the PLC (left side of comparison)
- `operator` (str): Comparison operator: `'=='`, `'!='`, `'<'`, `'>'`, `'<='`, `'>='`
- `right_value` (int or str): Either a constant integer, or a string naming another PLC variable

**Returns**:
- `bool`: `True` if the comparison holds, `False` otherwise

**Detailed behavior**:

1. Read the value of `left_var` from the PLC
2. Determine if `right_value` is a constant or variable name:
   - If it's an integer, use it directly
   - If it's a string, read that variable's value from the PLC
3. Apply the comparison operator
4. Return the boolean result

**Example 1: Threshold detection (constant comparison)**

Setup:
```python
machine.set_variable('temperature', 85)
```

Call:
```python
is_overtemp = machine.compare('temperature', '>', 80)
```

What happens:
- Reads `temperature` from PLC → gets 85
- `right_value` is integer 80 (constant)
- Performs comparison: `85 > 80`
- Returns: `True`

Usage in control logic:
```python
if machine.compare('temperature', '>', 80):
    machine.set_flag('cooling_enable', True)
    print("Temperature exceeds threshold, enabling cooling")
```

**Example 2: Variable-to-variable comparison**

Setup:
```python
machine.set_variable('current_position', 1500)
machine.set_variable('target_position', 2000)
```

Call:
```python
at_target = machine.compare('current_position', '==', 'target_position')
```

What happens:
- Reads `current_position` → gets 1500
- Sees `right_value` is string `'target_position'`, so reads it → gets 2000
- Performs comparison: `1500 == 2000`
- Returns: `False`

**Example 3: Interlock logic (safety system)**

Setup:
```python
machine.set_variable('pressure', 45)
machine.set_variable('pressure_limit', 50)
machine.set_flag('emergency_stop_active', False)
```

Call:
```python
pressure_ok = machine.compare('pressure', '<', 'pressure_limit')
estop_clear = not machine.get_flag('emergency_stop_active')

if pressure_ok and estop_clear:
    machine.set_flag('pump_enable', True)
else:
    machine.set_flag('pump_enable', False)
    print("Cannot enable pump: pressure or E-stop condition")
```

What happens:
- First comparison: pressure (45) < pressure_limit (50) → `True`
- Checks emergency stop flag → `False` (not active), so `estop_clear` is `True`
- Both conditions met → pump enabled

**Example 4: Range checking (LIM instruction equivalent)**

For range checking (value within bounds), you can combine two comparisons:

Setup:
```python
machine.set_variable('sensor_reading', 75)
```

Call:
```python
in_range = (machine.compare('sensor_reading', '>=', 60) and 
            machine.compare('sensor_reading', '<=', 80))
```

Returns: `True` (75 is between 60 and 80)

This mimics the LIM (Limit Test) instruction in Logix 5000, which checks if a value falls within specified bounds.

**Value**:

Users learn that control flow in PLCs is fundamentally about comparing values stored in memory and making decisions based on those comparisons. Unlike a Python program where you write `if x > 10:`, a PLC compares register values and activates outputs based on those comparisons. This method helps users understand that weird machines exploit this by manipulating the values being compared to redirect control flow in unintended ways.

In ladder logic, a comparison is literally a "rung condition"—if the comparison is true, the rung "conducts" and executes its output instructions. This method brings that conceptual model into Python.

**Gadget categories supported**:
- Threshold gates and triggers
- State selection logic (if state == 3, do action)
- Safety interlocks (multiple conditions must be true)
- Range validation

**Implementation considerations**:

1. **Type detection for right_value parameter**: Use Python's `isinstance(right_value, int)` to determine if the right operand is a constant or variable name. If it's an integer, use it directly. If it's a string, call `get_variable(right_value)` to fetch its value from the PLC. Consider adding type hints (`Union[int, str]`) to make the parameter expectations clear. You might also want to handle edge cases where a string looks like a number (e.g., `"123"`) by checking if it exists as a variable name first before attempting numeric conversion.

2. **Operator dispatch using a dictionary**: Create a dictionary mapping operator strings to Python comparison functions. For example: `{'<': operator.lt, '>': operator.gt, '==': operator.eq}`. This is cleaner than a long if-elif chain. You can use Python's built-in `operator` module which provides functions like `operator.lt`, `operator.gt`, `operator.eq`, etc. This approach also makes it easy to add new operators later if needed (like 'in_range' for LIM instruction equivalents).

3. **Error handling for invalid operators or missing variables**: Validate that the operator string is one of the supported operators before attempting the comparison. If an invalid operator is provided, raise a `ValueError` with a helpful message listing the valid operators. Similarly, if `left_var` doesn't exist in `self._variable_map`, raise a descriptive error. For variable-to-variable comparisons, check that the right-side variable also exists before attempting to read it. Good error messages help users debug their code and understand what went wrong.

---

### 3. `conditional_action(condition_func: Callable[[], bool], action_func: Callable[[], None], else_action_func: Callable[[], None] = None) -> bool`

**Category**: Composition/Chaining & Control-Flow Gadgets

**What it does**: Executes one function if a condition is true, and optionally executes a different function if the condition is false. The key difference from a regular Python `if` statement is that the condition is evaluated based on PLC state, making the branching decision dependent on remote data.

**Why it's essential**: This method encapsulates the pattern of "check PLC state, then take action" which is fundamental to ladder logic programming. In ladder logic, each rung has conditions (comparisons, flags) and actions (outputs, moves, calculations). This method makes that pattern explicit and reusable, and it's essential for building state machines and conditional sequences.

**Method signature**:
```python
def conditional_action(self, 
                      condition_func: Callable[[], bool],
                      action_func: Callable[[], None], 
                      else_action_func: Callable[[], None] = None) -> bool
```

**Parameters**:
- `condition_func` (callable): A function (often a lambda) that returns `True` or `False`. This typically reads PLC state using `get_variable()`, `get_flag()`, or `compare()`
- `action_func` (callable): A function to execute if condition is `True`. Usually calls `set_variable()`, `set_flag()`, or other machine methods
- `else_action_func` (callable, optional): A function to execute if condition is `False`

**Returns**:
- `bool`: `True` if `action_func` was executed, `False` if `else_action_func` was executed (or if condition was false with no else action)

**Detailed behavior**:

1. Call `condition_func()` to evaluate the condition
2. If the result is `True`:
   - Execute `action_func()`
   - Return `True`
3. If the result is `False`:
   - If `else_action_func` is provided, execute it
   - Return `False`

The power of this method is that it makes the dependency on PLC state explicit. You're not just writing `if x > 10:` with a local Python variable—you're making a control decision based on what's happening in the PLC's memory.

**Example 1: Mode-based execution (state machine)**

Setup:
```python
machine.set_variable('mode', 1)
machine.set_variable('output', 0)
```

Call:
```python
machine.conditional_action(
    condition_func=lambda: machine.get_variable('mode') == 1,
    action_func=lambda: machine.set_variable('output', 100),
    else_action_func=lambda: machine.set_variable('output', 0)
)
```

What happens:
- Condition evaluates: reads `mode` from PLC → 1, checks if 1 == 1 → `True`
- Executes `action_func`: sets `output` to 100 on the PLC
- Returns `True`

If mode were 2:
- Condition evaluates: 2 == 1 → `False`
- Executes `else_action_func`: sets `output` to 0
- Returns `False`

**Example 2: Safety interlock (multiple conditions)**

Setup:
```python
machine.set_flag('system_ready', True)
machine.set_flag('fault_active', False)
machine.set_variable('pressure', 45)
```

Call:
```python
machine.conditional_action(
    condition_func=lambda: (
        machine.get_flag('system_ready') and 
        not machine.get_flag('fault_active') and
        machine.compare('pressure', '<', 50)
    ),
    action_func=lambda: (
        machine.set_flag('pump_enable', True),
        print("All conditions met, enabling pump")
    )
)
```

What happens:
- Condition evaluates all three checks:
  - `system_ready` flag → `True`
  - `fault_active` flag → `False`, so `not False` → `True`
  - pressure (45) < 50 → `True`
  - All three `True`, so overall → `True`
- Executes `action_func`: enables pump, prints message
- Returns `True`

Note: To execute multiple statements in a lambda, use a tuple: `lambda: (statement1, statement2)`

**Example 3: Temperature control with hysteresis**

Setup:
```python
machine.set_variable('temperature', 78)
machine.set_variable('setpoint', 75)
machine.set_variable('deadband', 2)
machine.set_flag('heater_on', False)
```

Heating logic (turn on if below setpoint - deadband):
```python
machine.conditional_action(
    condition_func=lambda: machine.compare('temperature', '<', 
                                          machine.get_variable('setpoint') - machine.get_variable('deadband')),
    action_func=lambda: machine.set_flag('heater_on', True)
)
```

What happens:
- Reads: setpoint (75) - deadband (2) = 73
- Condition: temperature (78) < 73 → `False`
- No action taken, heater stays off

If temperature were 72:
- Condition: 72 < 73 → `True`
- Action: turns heater on

**Example 4: State machine transition**

Setup:
```python
machine.set_variable('state', 0)
machine.set_variable('count', 0)
```

State machine logic:
```python
# State 0: Initialization
machine.conditional_action(
    condition_func=lambda: machine.get_variable('state') == 0,
    action_func=lambda: (
        machine.set_variable('count', 0),
        machine.set_variable('state', 1)  # Transition to state 1
    )
)

# State 1: Running
machine.conditional_action(
    condition_func=lambda: machine.get_variable('state') == 1,
    action_func=lambda: machine.increment_variable('count')
)

# State 1 → State 2 transition (when count reaches 10)
machine.conditional_action(
    condition_func=lambda: (
        machine.get_variable('state') == 1 and 
        machine.compare('count', '>=', 10)
    ),
    action_func=lambda: machine.set_variable('state', 2)
)
```

This creates a simple state machine where:
- State 0: Initialize and move to state 1
- State 1: Increment counter
- State 1 → State 2: When counter reaches 10

**Value**:

This method teaches users that PLCs execute "conditional logic" rather than "sequential programs." In ladder logic, multiple rungs can all evaluate their conditions on every scan cycle, and outputs activate based on which conditions are currently true. This is fundamentally different from a Python script that executes top-to-bottom once.

By wrapping condition-action pairs in a method, users see the pattern clearly: weird machines work by manipulating the conditions (the data being tested) to cause unintended actions to execute. If an attacker can control the `mode` variable in Example 1, they control which output value gets written.

**Gadget categories supported**:
- Conditional branching
- Interlocks (AND/OR of multiple conditions)
- State machines
- Mode selection
- Event-driven actions

**Implementation considerations**:

1. **Exception handling in condition and action functions**: Wrap the calls to `condition_func()` and `action_func()` in try-except blocks to catch and handle exceptions gracefully. If the condition function raises an exception (e.g., trying to read a non-existent variable), you should decide whether to treat this as "condition failed" (return `False`) or re-raise the exception with additional context. For action functions, consider whether to suppress exceptions or let them propagate. Clear error handling helps users debug their state machine logic when variables are missing or operations fail.

2. **Supporting multiple actions in a single call**: Lambda functions can execute multiple statements using tuple syntax: `lambda: (action1(), action2(), action3())`. However, this can be confusing for beginners. Consider documenting this pattern clearly with examples, or alternatively, allow `action_func` to be a list of callables that you execute in sequence. For example: `for func in action_func: func()` if `isinstance(action_func, list)`. This makes multi-step actions more explicit and readable.

3. **Return value semantics for tracking execution paths**: The return value (`True` if action executed, `False` otherwise) allows users to track which branch was taken. This is useful for logging, debugging, and testing state machines. Consider enhancing this by storing execution history in an optional internal list (e.g., `self._execution_log`) that records which conditions evaluated to true/false and which actions were executed. This would be invaluable for users debugging complex state machines where multiple conditional_action calls interact.

---

### 4. `compose_operations(stages: List[Dict]) -> Dict`

**Category**: Composition/Chaining Gadgets

**What it does**: Executes a sequence of operations in order, where each operation can read results from previous operations. This allows you to build multi-step processes like cascade control loops, data aggregation pipelines, or batch sequences without manually managing intermediate variables.

**Why it's essential**: Real industrial control systems chain operations together—the output of one block becomes the input to the next. For example, in cascade control, a primary PID controller's output becomes the setpoint for a secondary PID controller. This method provides a structured way to express such pipelines, mirroring how Sequential Function Charts (SFC) and function block diagrams work in real PLCs.

**Method signature**:
```python
def compose_operations(self, stages: List[Dict]) -> Dict
```

**Parameters**:
- `stages` (list of dict): Each dictionary describes one operation with keys like:
  - `'action'`: What to do (e.g., `'compute'`, `'compare'`, `'pide'`, `'read_multiple'`, `'aggregate'`)
  - `'input'` or specific input keys: Where to get data from
  - `'output'`: Where to store the result
  - Additional parameters specific to the action

**Returns**:
- `dict`: A dictionary mapping output names to their computed values

**Detailed behavior**:

1. Initialize an empty results dictionary
2. For each stage in order:
   - Identify the action type
   - Fetch inputs (from PLC variables or previous stage results)
   - Execute the action
   - Store the output in the results dictionary
   - Write outputs back to PLC if specified
3. Return the complete results dictionary

The key concept is **data flow**: outputs from one stage can be referenced by name in subsequent stages.

**Example 1: Cascade control (two-loop control system)**

Scenario: Control liquid level in a tank by adjusting flow rate. The primary controller compares actual level to desired level and outputs a flow rate setpoint. The secondary controller compares actual flow to that setpoint and outputs a valve position.

Setup:
```python
machine.set_variable('tank_level', 45)          # Current level (%)
machine.set_variable('level_setpoint', 50)      # Desired level (%)
machine.set_variable('flow_rate', 120)          # Current flow (L/min)
```

Call:
```python
results = machine.compose_operations([
    {
        'action': 'pide',
        'pv': 'tank_level',              # Process Variable (what we measure)
        'sp': 'level_setpoint',          # Setpoint (what we want)
        'cv_out': 'flow_setpoint',       # Control Variable output
        'gains': {'kp': 2.0, 'ki': 0.5, 'kd': 0.1}
    },
    {
        'action': 'pide',
        'pv': 'flow_rate',               # Measure actual flow
        'sp': 'flow_setpoint',           # Use output from stage 1 as setpoint
        'cv_out': 'valve_position',      # Final output to valve
        'gains': {'kp': 1.5, 'ki': 0.3, 'kd': 0.05}
    }
])

print(f"Flow setpoint: {results['flow_setpoint']}")
print(f"Valve position: {results['valve_position']}")
```

What happens:
1. **Stage 1 (Level control)**:
   - Reads `tank_level` (45) and `level_setpoint` (50)
   - Error = 50 - 45 = 5
   - PID calculation produces output (e.g., 130 L/min)
   - Writes 130 to `flow_setpoint` on PLC
   - Stores in results: `{'flow_setpoint': 130}`

2. **Stage 2 (Flow control)**:
   - Reads `flow_rate` (120) and `flow_setpoint` (130, from stage 1)
   - Error = 130 - 120 = 10
   - PID calculation produces output (e.g., 65% valve opening)
   - Writes 65 to `valve_position` on PLC
   - Stores in results: `{'flow_setpoint': 130, 'valve_position': 65}`

3. **Returns**: `{'flow_setpoint': 130, 'valve_position': 65}`

This is called "cascade control" because the loops cascade—the primary (outer) loop manipulates the setpoint of the secondary (inner) loop.

**Example 2: Data aggregation pipeline**

Scenario: Monitor temperature from multiple sensors, compute the average, and set an alarm if average exceeds threshold.

Setup:
```python
machine.set_variable('zone1_temp', 72)
machine.set_variable('zone2_temp', 75)
machine.set_variable('zone3_temp', 78)
machine.set_variable('zone4_temp', 70)
```

Call:
```python
results = machine.compose_operations([
    {
        'action': 'read_multiple',
        'sources': ['zone1_temp', 'zone2_temp', 'zone3_temp', 'zone4_temp'],
        'output': 'all_temps'
    },
    {
        'action': 'aggregate',
        'input': 'all_temps',
        'operation': 'average',
        'output': 'avg_temp'
    },
    {
        'action': 'compare',
        'var': 'avg_temp',
        'op': '>',
        'threshold': 74,
        'output': 'overtemp_alarm'
    }
])

print(f"Average temperature: {results['avg_temp']}")
print(f"Alarm active: {results['overtemp_alarm']}")
```

What happens:
1. **Stage 1 (Read multiple)**:
   - Reads all 4 temperature variables from PLC
   - Stores: `{'all_temps': [72, 75, 78, 70]}`

2. **Stage 2 (Aggregate)**:
   - Takes `all_temps` from stage 1: [72, 75, 78, 70]
   - Computes average: (72 + 75 + 78 + 70) / 4 = 73.75 ≈ 74 (integer)
   - Writes 74 to `avg_temp` on PLC
   - Stores: `{'all_temps': [...], 'avg_temp': 74}`

3. **Stage 3 (Compare)**:
   - Reads `avg_temp` (74 from stage 2)
   - Compares: 74 > 74 → `False`
   - Sets `overtemp_alarm` flag to `False` on PLC
   - Stores: `{'all_temps': [...], 'avg_temp': 74, 'overtemp_alarm': False}`

4. **Returns**: Full results dictionary

**Example 3: Batch production sequence**

Scenario: Execute a multi-phase production batch—fill tank, heat to temperature, hold for time, then drain.

Setup:
```python
machine.set_variable('batch_phase', 0)
machine.set_variable('tank_level', 0)
machine.set_variable('temperature', 25)
```

Call (simplified single scan):
```python
results = machine.compose_operations([
    {
        'action': 'read_multiple',
        'sources': ['batch_phase', 'tank_level', 'temperature'],
        'output': 'current_state'
    },
    {
        'action': 'conditional_action',
        'condition': lambda: results['current_state'][0] == 0,  # Phase 0: Fill
        'true_action': lambda: (
            machine.set_flag('fill_valve', True),
            machine.set_variable('batch_phase', 1) if machine.compare('tank_level', '>=', 80) else None
        )
    },
    {
        'action': 'conditional_action',
        'condition': lambda: results['current_state'][0] == 1,  # Phase 1: Heat
        'true_action': lambda: (
            machine.set_flag('fill_valve', False),
            machine.set_flag('heater', True),
            machine.set_variable('batch_phase', 2) if machine.compare('temperature', '>=', 60) else None
        )
    }
])
```

What happens (if starting in phase 0 with level at 50):
1. **Stage 1**: Reads current state [0, 50, 25]
2. **Stage 2**: Phase is 0, so opens fill valve, checks if level ≥ 80 (no), stays in phase 0
3. **Stage 3**: Phase is not 1, so skips

On subsequent scans as level rises:
- When level reaches 80, transitions to phase 1
- Phase 1 closes fill valve, turns on heater
- When temperature reaches 60, transitions to phase 2 (not shown)

**Example 4: Feedforward + feedback control**

Scenario: Control a process where external disturbances can be measured. Use feedforward to preemptively compensate for the disturbance, and feedback to correct remaining error.

Setup:
```python
machine.set_variable('disturbance_flow', 50)    # Measured inlet disturbance
machine.set_variable('process_temp', 72)
machine.set_variable('temp_setpoint', 75)
```

Call:
```python
results = machine.compose_operations([
    {
        'action': 'compute',
        'expression': 'disturbance_flow * 0.8',  # Feedforward gain
        'output': 'ff_compensation'
    },
    {
        'action': 'pide',
        'pv': 'process_temp',
        'sp': 'temp_setpoint',
        'cv_out': 'fb_output',
        'gains': {'kp': 1.0, 'ki': 0.2, 'kd': 0.0}
    },
    {
        'action': 'compute',
        'expression': 'ff_compensation + fb_output',
        'output': 'total_output'
    }
])

print(f"Feedforward: {results['ff_compensation']}")
print(f"Feedback: {results['fb_output']}")
print(f"Total control output: {results['total_output']}")
```

What happens:
1. **Stage 1**: Computes feedforward = 50 × 0.8 = 40
2. **Stage 2**: PID computes error-based correction (e.g., output = 15)
3. **Stage 3**: Combines both: total = 40 + 15 = 55

The feedforward anticipates the effect of the disturbance, while feedback corrects any remaining error.

**Value**:

This method teaches users that complex control behaviors emerge from composing simple operations. A cascade control system isn't a magical "cascade controller"—it's two PID blocks where one's output feeds into the other's setpoint. 

Users learn to think in terms of **data flow graphs**: data enters at one point, flows through transformations (compute, PID, compare), and produces outputs. This is exactly how function block diagrams work in IEC 61131-3 (the international standard for PLC programming).

For weird machines, this demonstrates how attackers chain gadgets: the output of one gadget becomes the input to the next, creating computation paths the system designer never intended. By understanding legitimate composition, users can reason about malicious composition.

**Gadget categories supported**:
- Cascade control
- Feedforward compensation
- Data aggregation and statistics
- Multi-phase sequences
- Pipeline processing
- Event chains

**Implementation considerations**:

1. **Action dispatch mechanism with extensibility**: Use a dictionary to map action strings to handler methods: `action_handlers = {'compute': self._handle_compute, 'compare': self._handle_compare, 'pide': self._handle_pide}`. Then dispatch with `action_handlers[stage['action']](stage, results)`. This makes the code clean and easy to extend—adding a new action type just means adding a new handler method and dictionary entry. Each handler method should follow a consistent signature, taking the stage configuration and current results dictionary as parameters.

2. **Results dictionary for inter-stage data flow**: Maintain a `results` dictionary that accumulates outputs from each stage. When a stage references an input like `'input': 'all_temps'`, first check if `'all_temps'` exists in the results dictionary (from a previous stage). If not found in results, try reading it as a PLC variable name. This dual lookup strategy allows stages to reference both intermediate pipeline results and actual PLC variables seamlessly. Clear documentation of this lookup order helps users understand how data flows through the pipeline.

3. **Validation of stage configurations**: Before executing stages, validate that required keys exist in each stage dictionary. For example, every stage should have an `'action'` key, and most will need an `'output'` key. Some actions require specific parameters (e.g., `'pide'` needs `'pv'`, `'sp'`, `'cv_out'`, `'gains'`). Raise descriptive `ValueError` or `KeyError` exceptions if required keys are missing, with messages like "Stage 2 (pide) missing required key 'gains'". This validation catches configuration errors early and helps users debug their pipeline definitions.

---

## Summary: How These Methods Work Together

These 4 methods form a coherent toolkit:

1. **`compute()`** performs arithmetic on remote data
2. **`compare()`** makes decisions based on remote data
3. **`conditional_action()`** executes actions based on those decisions
4. **`compose_operations()`** chains everything into complex sequences

### Example: Complete control system

Here's how they work together in a realistic temperature control system:

```python
# Initialize
machine.set_variable('current_temp', 68)
machine.set_variable('setpoint', 72)
machine.set_variable('mode', 1)  # 1 = heating, 2 = cooling

# Complex control logic
results = machine.compose_operations([
    # Stage 1: Compute temperature error
    {
        'action': 'compute',
        'expression': 'setpoint - current_temp',
        'output': 'error'
    },
    
    # Stage 2: Determine if heating or cooling needed based on mode
    {
        'action': 'conditional_action',
        'condition': lambda: machine.get_variable('mode') == 1,
        'true_action': lambda: (
            # Heating mode: if error positive, turn on heater
            machine.set_flag('heater', machine.compare('error', '>', 0))
        ),
        'false_action': lambda: (
            # Cooling mode: if error negative, turn on cooler
            machine.set_flag('cooler', machine.compare('error', '<', 0))
        )
    }
])
```

This single composition:
- Uses `compute()` to calculate error
- Uses `compare()` to check if action needed
- Uses `conditional_action()` to route logic by mode
- Chains them in `compose_operations()` for complete behavior

---

## Implementation Priority

**Start with these two**:
1. `compute()` — unlocks arithmetic gadgets
2. `compare()` — unlocks branching gadgets

These provide immediate value and are relatively simple to implement.

**Then add**:
3. `conditional_action()` — unlocks control flow patterns

**Finally**:
4. `compose_operations()` — unlocks complex multi-stage systems

This staged approach lets users learn progressively while building toward full industrial control system capabilities.

