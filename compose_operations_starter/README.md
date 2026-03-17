# Understanding `compose_operations`

## What Problem Does It Solve?

When a weird machine does something interesting, it rarely does it in a single step.
Real PLC programs are built from **chains of simple operations**: read a sensor, scale
the raw value, compare it to a threshold, and take action based on the result. Each
step feeds data into the next.

Without a composition mechanism, you would express that chain like this in Python:

```python
raw = machine.get_variable("raw_adc")
scaled = int((raw / 4095) * 100)
machine.set_variable("pressure_pct", scaled)

over_limit = machine.compare("pressure_pct", ">", 80)
machine.set_flag("pressure_alarm", over_limit)

if over_limit:
    machine.set_flag("pump_enable", False)
```

This works, but it mixes Python control flow with PLC operations in a way that quickly
becomes hard to read and hard to reason about when chains get longer. More importantly,
it does not mirror how the underlying PLC actually thinks about the problem.

A PLC does not run a sequential script. It runs **a scan cycle**: on every cycle, a
series of blocks each evaluate their inputs, produce an output, and hand it off to the
next block. The output of one block is the input of the next. This is called a
**data-flow pipeline**, and it is the mental model that `compose_operations` brings into
Python.

The same chain written as a pipeline:

```python
results = machine.compose_operations([
    {
        "action": "scale",
        "input": "raw_adc", "in_min": 0, "in_max": 4095,
        "out_min": 0, "out_max": 100,
        "output": "pressure_pct", "write_to": "pressure_pct",
    },
    {
        "action": "compare",
        "left": "pressure_pct", "op": ">", "right": 80,
        "output": "over_limit", "write_to": "pressure_alarm",
    },
    {
        "action": "conditional_action",
        "condition": lambda: results.get("over_limit"),
        "true_action": lambda: machine.set_flag("pump_enable", False),
    },
])
```

The pipeline version is declarative: you say *what* should happen at each step, not
*how* Python should sequence its local variables. That matches how PLC programs are
written and, critically, how weird machine gadgets are composed.

---

## Why Does This Matter for Weird Machines?

A **weird machine gadget** is a unit of computation that already exists in a system —
it just was not placed there intentionally by an attacker. The power of weird machines
comes from **chaining gadgets**: the output of one gadget becomes the input of the
next, building up useful computation from pieces the designer never intended to combine
that way.

`compose_operations` is a direct model of this. Each stage in the pipeline is a gadget.
The `results` dictionary is the data bus connecting them. When you write a pipeline,
you are doing exactly what a weird machine exploit does: picking gadgets, ordering them,
and wiring their outputs to the next gadget's inputs.

Understanding how legitimate composition works — and what the mechanism looks like from
the inside — is essential groundwork before studying how composition can be exploited.

---

## The Core Mechanism

Every call to `compose_operations` follows the same four steps for each stage:

```
┌─────────────────────────────────────────────────────────────┐
│  For each stage in the pipeline list, in order:             │
│                                                             │
│  1. VALIDATE  -- does this stage have an "action" key?      │
│                                                             │
│  2. DISPATCH  -- look up the handler for this action type   │
│                  in the handlers dictionary                 │
│                                                             │
│  3. EXECUTE   -- call the handler; it reads from the        │
│                  machine or from `results`, does its work,  │
│                  and returns a single value                 │
│                                                             │
│  4. STORE     -- if the stage has an "output" key, put the  │
│                  return value into `results` under that key │
│                  (optionally also write it to the machine   │
│                  if "write_to" is set)                      │
└─────────────────────────────────────────────────────────────┘
```

The `results` dictionary is the key. It starts empty and grows as stages run. A stage
that runs early can place a value into `results["error"]`, and a later stage can read
`results["error"]` as its input simply by naming the string `"error"`. This lookup
is handled automatically by `_resolve_value()`, described in the next section.

---

## A Close Look at `_resolve_value`

Every handler needs to read its inputs. Those inputs might be:

- A literal number the caller typed directly into the stage dict (`"right": 74`)
- A machine variable that should be read live (`"left": "avg_temp"`)
- The output of an earlier stage (`"input": "all_temps"`)

Rather than making every handler figure this out individually, `_resolve_value` handles
the lookup in a consistent priority order:

| What you put in the stage dict | What gets returned |
|---|---|
| `74` (an int or float) | `74` |
| `True` / `False` (a bool) | `True` / `False` |
| `[1, 2, 3]` (a list) | `[1, 2, 3]` |
| `{"from_result": "error"}` | `results["error"]` |
| `{"from_variable": "x"}` | `machine.get_variable("x")` |
| `{"from_flag": "ready"}` | `machine.get_flag("ready")` |
| `"avg_temp"` (matches a results key) | `results["avg_temp"]` |
| `"avg_temp"` (matches a variable name) | `machine.get_variable("avg_temp")` |
| `"ready"` (matches a flag name) | `machine.get_flag("ready")` |

The string-based lookup (rows 7–9) is why you can write `"left": "avg_temp"` in a
compare stage and have it automatically resolve to the current value — whether that
value came from a previous stage result or directly from the machine.

---

## What Is Already Implemented

### The machine itself

`SimpleWeirdMachine` in the starter file is a self-contained in-memory implementation.
It uses plain Python dicts instead of Modbus registers, so you can run and test every
pipeline without a PLC connected. The public API is the same as the real class:

| Method | What it does |
|---|---|
| `set_variable(name, value)` | Store an integer under a name |
| `get_variable(name)` | Read it back |
| `increment_variable(name)` | Add 1 |
| `decrement_variable(name)` | Subtract 1 |
| `set_flag(name, value)` | Store a boolean under a name |
| `get_flag(name)` | Read it back |
| `compute(expression)` | Safely evaluate an arithmetic expression |
| `compare(left, op, right)` | Compare a variable to a constant or another variable |
| `conditional_action(...)` | Execute one of two callables based on a condition |

### Stage handlers already working

Five action types are fully implemented and registered in the dispatch table:

**`read_multiple`** — reads a list of named sources (variables, flags, or prior results)
and returns them as a Python list. This is typically the first stage in a pipeline that
needs to pull in several sensor readings before processing them.

```python
{"action": "read_multiple", "sources": ["zone1", "zone2", "zone3"], "output": "readings"}
```

**`aggregate`** — applies a statistical reduction to a list: `"average"`, `"sum"`,
`"min"`, or `"max"`. The input is almost always the output of a preceding
`read_multiple` stage.

```python
{"action": "aggregate", "input": "readings", "operation": "average", "output": "avg"}
```

**`compute`** — evaluates a safe arithmetic expression. Both machine variables and
numeric values from earlier stages are available by name in the expression string.

```python
{"action": "compute", "expression": "setpoint - avg", "output": "error"}
```

**`compare`** — compares two values (variable, result, or literal) and returns a
boolean. Supports `==`, `!=`, `<`, `>`, `<=`, `>=`.

```python
{"action": "compare", "left": "error", "op": ">", "right": 0, "output": "heating_needed"}
```

**`conditional_action`** — runs a `true_action` callable when a condition is met,
optionally a `false_action` otherwise. Bridges the gap between a boolean result
computed earlier in the pipeline and an actual action taken on the machine.

```python
{
    "action": "conditional_action",
    "condition": lambda: machine.get_flag("heating_needed"),
    "true_action":  lambda: machine.set_flag("heater_on", True),
    "false_action": lambda: machine.set_flag("heater_on", False),
}
```

### The `_execution_log`

Every call to `compose_operations` appends a summary of each stage to
`machine._execution_log`. After a pipeline runs you can print it to see exactly which
stages executed, in what order, and what value each one produced. This is useful for
tracing data flow while you are building and debugging pipelines.

```python
for entry in machine._execution_log:
    print(entry)
```

---

## What Still Needs to Be Implemented

Two handlers are stubbed out and will raise `NotImplementedError` until you fill them
in. Both follow the exact same pattern as the five handlers above.

### `_handle_write`

**What it should do:** Take a resolved value and write it back to the machine — to a
variable if it is an integer, to a flag if it is a boolean. This is useful as the final
stage in a pipeline when you want to persist a computed result under a new name.

**Stage format:**
```python
{
    "action": "write",
    "destination": "alarm_active",   # name to write to on the machine
    "value": "overtemp_alarm",       # result key, variable name, or literal
    "output": "write_result",        # optional: also store in results
}
```

**Steps to implement:**
1. Call `self._resolve_value(stage.get("value"), results)` to get the concrete value.
2. Check `destination = stage.get("destination")`; raise `ValueError` if it is `None`.
3. Call `self.set_flag(destination, value)` if `value` is a `bool`, otherwise call
   `self.set_variable(destination, int(value))`.
4. Return `value`.
5. Add `"write": self._handle_write` to the `handlers` dict in `compose_operations`.

---

### `_handle_scale`

**What it should do:** Linearly map a value from one numeric range to another. This is
the standard formula for converting a raw ADC reading into an engineering unit such as
PSI, degrees Celsius, or percent:

```
output = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
```

**Stage format:**
```python
{
    "action": "scale",
    "input":   "raw_adc",   # variable, result key, or literal to scale
    "in_min":  0,
    "in_max":  4095,
    "out_min": 0,
    "out_max": 100,
    "output":  "pressure_pct",
    "write_to": "pressure_pct",
}
```

**Steps to implement:**
1. Resolve all five inputs (`input`, `in_min`, `in_max`, `out_min`, `out_max`) using
   `_resolve_value`. All five may reference earlier results or variable names.
2. Apply the formula. Return `int(result)`.
3. Guard against division by zero when `in_max == in_min`.
4. Add `"scale": self._handle_scale` to the `handlers` dict in `compose_operations`.

---

### Extend `_handle_aggregate`

The existing implementation supports `"average"`, `"sum"`, `"min"`, and `"max"`.
Adding `"median"` and `"std_dev"` requires only new branches in the `if/elif` chain.
Both are available in Python's `statistics` standard library module.

---

## Running the Starter

Two demo pipelines are provided at the bottom of the file and run automatically:

```
python compose_operations_starter.py
```

**`demo_temperature_pipeline`** shows a three-stage chain: compute a temperature error,
compare it to zero, then enable or disable a heater flag based on the mode setting.

**`demo_aggregation_pipeline`** shows four stages: read four zone temperatures into a
list, average them, compare the average to a threshold, and set an alarm flag.

Run these before writing any code so you can see what correct pipeline output looks
like. After implementing `_handle_write` and `_handle_scale`, try adding them as
additional stages to one of the existing demos to verify your implementations.
