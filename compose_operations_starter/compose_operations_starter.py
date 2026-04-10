"""
compose_operations_starter.py

Overview
--------
compose_operations() lets you describe a multi-step data-processing pipeline
as a list of stage dictionaries. Each stage reads inputs (from the machine or
from a prior stage), does one unit of work, and writes its result forward into
a shared results dictionary. The next stage can then use that result as its
input.

This is called a data-flow pipeline. Think of it like a Unix pipe, but the
data flowing through it are PLC variables and computed values:

    [read sensor values]
        -> [compute average]
            -> [compare to threshold]
                -> [set alarm flag]

Each arrow above is one stage in the list you pass to compose_operations().

What you need to implement
--------------------------
Two stage handlers are stubbed out and marked TODO:

    _handle_write   -- writes a resolved value back to a variable or flag
    _handle_scale   -- linearly maps a value from one range to another

An additional TODO inside _handle_aggregate asks you to extend the supported
operations beyond average/sum/min/max.

No external PLC libraries are required. Run this file directly to see the
two working demo pipelines before you start.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Optional, Union


Number = Union[int, float]
StageDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Safe expression evaluator (infrastructure -- no changes needed here)
# ---------------------------------------------------------------------------

class SafeExpressionEvaluator(ast.NodeVisitor):
    """Walk an AST and evaluate only arithmetic operations."""

    ALLOWED_BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    ALLOWED_UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, variables: Dict[str, Number]):
        self.variables = variables

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.variables:
            raise ValueError(f"Unknown variable in expression: {node.id}")
        return self.variables[node.id]

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        return node.value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in self.ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return self.ALLOWED_BINOPS[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in self.ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return self.ALLOWED_UNARYOPS[op_type](operand)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _safe_eval(expression: str, variables: Dict[str, Number]) -> Number:
    tree = ast.parse(expression, mode="eval")
    return SafeExpressionEvaluator(variables).visit(tree)


# ---------------------------------------------------------------------------
# SimpleWeirdMachine
# ---------------------------------------------------------------------------

class SimpleWeirdMachine:
    """
    Lightweight in-memory machine for running and testing pipelines.

    Variables hold integer values. Flags hold boolean values.
    Both are stored in plain Python dicts here; in the real system they
    map to Modbus registers and coils over the network.
    """

    def __init__(self, host: str = "demo", port: int = 502):
        self.host = host
        self.port = port
        self._variables: Dict[str, int] = {}
        self._flags: Dict[str, bool] = {}
        self._execution_log: List[str] = []

    # --- core variable and flag API ----------------------------------------

    def set_variable(self, name: str, value: Number) -> None:
        self._variables[name] = int(value)

    def get_variable(self, name: str) -> int:
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'")
        return self._variables[name]

    def increment_variable(self, name: str) -> None:
        self.set_variable(name, self.get_variable(name) + 1)

    def decrement_variable(self, name: str) -> None:
        self.set_variable(name, self.get_variable(name) - 1)

    def set_flag(self, name: str, value: bool) -> None:
        self._flags[name] = bool(value)

    def get_flag(self, name: str) -> bool:
        if name not in self._flags:
            raise KeyError(f"Unknown flag: '{name}'")
        return self._flags[name]

    # --- building-block methods --------------------------------------------

    def compute(self, expression: str) -> int:
        """Evaluate an arithmetic expression over current variable values."""
        return int(_safe_eval(expression, dict(self._variables)))

    def compare(self, left_var: str, op: str, right: Union[int, str]) -> bool:
        """Compare a variable to a constant or to another variable."""
        left_val = self.get_variable(left_var)
        right_val = self.get_variable(right) if isinstance(right, str) else right
        ops = {
            "==": operator.eq, "!=": operator.ne,
            "<": operator.lt,  ">": operator.gt,
            "<=": operator.le, ">=": operator.ge,
        }
        if op not in ops:
            raise ValueError(f"Unsupported operator: {op}")
        return bool(ops[op](left_val, right_val))

    def conditional_action(
        self,
        condition_func: Callable[[], bool],
        action_func: Callable[[], Any],
        else_action_func: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """Run action_func if condition_func() is true, else_action_func otherwise."""
        if condition_func():
            action_func()
            return True
        if else_action_func is not None:
            else_action_func()
        return False

    # -----------------------------------------------------------------------
    # compose_operations -- pipeline infrastructure
    #
    # The methods below are the internal helpers that compose_operations()
    # calls. Each one handles exactly one stage action type. They all share
    # the same signature:
    #
    #     _handle_<action>(self, stage: StageDict, results: Dict) -> Any
    #
    #   stage   -- the stage dict from the caller's pipeline list
    #   results -- the shared dict that accumulates outputs from every stage
    #              that has run so far in this pipeline execution
    #
    # A handler reads whatever it needs from `stage` and `results`, does its
    # work, and returns a single value. compose_operations() takes care of
    # storing that return value under the stage's 'output' key.
    # -----------------------------------------------------------------------

    def _validate_stage(self, stage: StageDict, index: int) -> None:
        """Raise early with a clear message if a stage is malformed."""
        if not isinstance(stage, dict):
            raise TypeError(
                f"Stage {index} must be a dict, got {type(stage).__name__}"
            )
        if "action" not in stage:
            raise KeyError(f"Stage {index} is missing required key 'action'")

    def _resolve_value(self, value: Any, results: Dict[str, Any]) -> Any:
        """
        Turn a stage input specification into a concrete value.

        Stage dictionaries often refer to data by name rather than by value
        directly. For example, a compare stage might say:

            {"left": "avg_temp", "op": ">", "right": 74}

        Here "avg_temp" needs to be resolved -- is it a key in the results
        dict from a previous stage, or a live variable on the machine?

        This method handles that lookup so individual handlers don't have to.

        Resolution order
        ----------------
        1. Literal number, bool, list, or tuple    -> returned as-is
        2. Dict with 'from_result'   key            -> results[key]
           Dict with 'from_variable' key            -> get_variable(key)
           Dict with 'from_flag'     key            -> get_flag(key)
        3. Callable                                -> returned as-is
        4. String matching a results key           -> results[string]
        5. String matching a variable name         -> get_variable(string)
        6. String matching a flag name             -> get_flag(string)
        7. Anything else                           -> returned as-is
        """
        if isinstance(value, (int, float, bool, list, tuple)):
            return value

        if isinstance(value, dict):
            if "from_result" in value:
                key = value["from_result"]
                if key not in results:
                    raise KeyError(f"Result key not found: {key}")
                return results[key]
            if "from_variable" in value:
                return self.get_variable(value["from_variable"])
            if "from_flag" in value:
                return self.get_flag(value["from_flag"])
            return value

        if callable(value):
            return value

        if isinstance(value, str):
            if value in results:
                return results[value]
            if value in self._variables:
                return self.get_variable(value)
            if value in self._flags:
                return self.get_flag(value)

        return value

    def _store_output(
        self, output_name: Optional[str], value: Any, results: Dict[str, Any]
    ) -> None:
        """Write a handler's return value into the shared results dict."""
        if output_name is not None:
            results[output_name] = value

    # --- stage handlers (implemented) --------------------------------------

    def _handle_read_multiple(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Resolve a list of named sources and return them as a Python list.

        Required stage keys:
            sources  -- list of variable names, flag names, or result keys

        Example stage:
            {
                "action": "read_multiple",
                "sources": ["zone1_temp", "zone2_temp", "zone3_temp"],
                "output": "all_temps"
            }
        """
        sources = stage.get("sources")
        if not isinstance(sources, list):
            raise ValueError("read_multiple stage requires 'sources' to be a list")
        return [self._resolve_value(src, results) for src in sources]

    def _handle_aggregate(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Apply a statistical operation to a list of values.

        Required stage keys:
            input      -- name of a results key that holds a list
            operation  -- one of: "average", "sum", "min", "max"

        Example stage:
            {
                "action": "aggregate",
                "input": "all_temps",
                "operation": "average",
                "output": "avg_temp"
            }

        TODO: Add support for additional operations such as "median" and
        "std_dev". Both are straightforward with the statistics module from
        the Python standard library. Add them to the if/elif chain below
        following the same pattern as the existing operations.
        """
        data = self._resolve_value(stage.get("input"), results)
        operation_name = stage.get("operation")

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("aggregate requires a non-empty list as input")

        if operation_name == "average":
            return int(sum(data) / len(data))
        if operation_name == "sum":
            return int(sum(data))
        if operation_name == "min":
            return min(data)
        if operation_name == "max":
            return max(data)

        raise ValueError(f"Unsupported aggregate operation: {operation_name}")

    def _handle_compute(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Evaluate an arithmetic expression, with access to both machine
        variables and previous stage results.

        Required stage keys:
            expression -- arithmetic string, e.g. "setpoint - current_temp"

        Example stage:
            {
                "action": "compute",
                "expression": "setpoint - current_temp",
                "output": "error"
            }
        """
        expression = stage.get("expression")
        if not isinstance(expression, str):
            raise ValueError("compute stage requires an 'expression' string")

        # Build a context that includes both live variables and any numeric
        # values already computed earlier in this pipeline.
        context = dict(self._variables)
        for key, val in results.items():
            if isinstance(val, (int, float)):
                context[key] = val

        return int(_safe_eval(expression, context))

    def _handle_compare(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Compare two values and return a boolean.

        Required stage keys:
            left  -- variable name, result key, or literal value
            op    -- comparison operator string: ==, !=, <, >, <=, >=
            right -- variable name, result key, or literal value

        Example stage:
            {
                "action": "compare",
                "left": "avg_temp",
                "op": ">=",
                "right": 74,
                "output": "overtemp_alarm"
            }
        """
        left_val = self._resolve_value(stage.get("left", stage.get("var")), results)
        right_val = self._resolve_value(
            stage.get("right", stage.get("threshold")), results
        )
        op = stage.get("op")
        ops = {
            "==": operator.eq, "!=": operator.ne,
            "<": operator.lt,  ">": operator.gt,
            "<=": operator.le, ">=": operator.ge,
        }
        if op not in ops:
            raise ValueError(f"Unsupported comparison operator: {op}")
        return bool(ops[op](left_val, right_val))

    def _handle_conditional_action(
        self, stage: StageDict, results: Dict[str, Any]
    ) -> Any:
        """
        Run a true_action callable if a condition is met, optionally a
        false_action otherwise. This brings conditional_action() into a
        pipeline stage so branching can be part of a composed sequence.

        Required stage keys:
            condition   -- callable or resolvable value (truthy/falsy)
            true_action -- callable executed when condition is true

        Optional stage keys:
            false_action -- callable executed when condition is false

        Example stage:
            {
                "action": "conditional_action",
                "condition": lambda: machine.get_flag("needs_heat"),
                "true_action": lambda: machine.set_flag("heater_on", True),
                "false_action": lambda: machine.set_flag("heater_on", False),
                "output": "branch_taken"
            }
        """
        condition_spec = stage.get("condition")
        true_action = stage.get("true_action")
        false_action = stage.get("false_action")

        if condition_spec is None or true_action is None:
            raise ValueError(
                "conditional_action stage requires 'condition' and 'true_action'"
            )
        if not callable(true_action):
            raise ValueError("true_action must be callable")
        if false_action is not None and not callable(false_action):
            raise ValueError("false_action must be callable when provided")

        if callable(condition_spec):
            condition_func = condition_spec
        else:
            resolved = self._resolve_value(condition_spec, results)
            condition_func = lambda: bool(resolved)

        return self.conditional_action(condition_func, true_action, false_action)

    # --- stage handlers (to implement) ------------------------------------

    def _handle_write(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Write a resolved value back to a named variable or flag on the machine.

        This is useful at the end of a pipeline to persist a computed result
        so that subsequent code (or a future pipeline run) can read it by name.

        Required stage keys:
            destination -- the variable or flag name to write to
            value       -- the value to write; resolved via _resolve_value(),
                           so it can be a literal, a variable name, or a
                           results key from a previous stage

        The handler should write a bool to a flag and an int to a variable.
        Return the written value so compose_operations() can store it in results.

        Example stage:
            {
                "action": "write",
                "destination": "alarm_active",
                "value": "overtemp_alarm",
                "output": "write_result"
            }

        TODO: Implement this handler following the pattern of the handlers
        above. Use _resolve_value() to obtain the concrete value, then call
        set_flag() for booleans and set_variable() for everything else.
        Raise a ValueError if destination is None.
        """

        destination = stage.get("destination")
        if destination is None:
            raise ValueError("write stage requires 'destination' key")

        value = stage.get("value")
        if value is None:
            raise ValueError("Write stage requires 'value' key")
        resolved_value = self._resolve_value(value, results)
        if isinstance(resolved_value, bool):
            self.set_flag(destination, resolved_value)
        else: self.set_variable(destination, resolved_value)
        return resolved_value
    
    def _handle_scale(self, stage: StageDict, results: Dict[str, Any]) -> Any:
        """
        Linearly map a value from an input range to an output range.

        This is the standard sensor-scaling formula:

            output = (value - in_min) * (out_max - out_min)
                     / (in_max - in_min) + out_min

        It is equivalent to the CPT scaling instructions used in real PLCs
        to convert a raw ADC reading into an engineering unit like PSI or °C.

        Required stage keys:
            input    -- variable name, result key, or literal to scale
            in_min   -- lower bound of the input range
            in_max   -- upper bound of the input range
            out_min  -- lower bound of the output range
            out_max  -- upper bound of the output range

        Return the scaled value as an int.

        Example stage:
            {
                "action": "scale",
                "input": "raw_adc",
                "in_min": 0,
                "in_max": 4095,
                "out_min": 0,
                "out_max": 100,
                "output": "pressure_pct"
            }

        TODO: Implement this handler. Use _resolve_value() for the input and
        all four range bounds (they may reference earlier results). Apply the
        formula above and return int(result). Consider what should happen when
        in_max == in_min to avoid a division-by-zero error.
        """
        value = self._resolve_value(stage.get("input"), results)
        in_min = self._resolve_value(stage.get("in_min"), results)
        in_max = self._resolve_value(stage.get("in_max"), results)
        out_min = self._resolve_value(stage.get("out_min"), results)
        out_max = self._resolve_value(stage.get("out_max"), results)

        if in_max == in_min:
            raise ValueError("Cannot divide by zero in scale stage: in_max and in_min cannot be equal")
        else:
            output = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

        return int(output)

    # --- compose_operations -----------------------------------------------

    def compose_operations(self, stages: List[StageDict]) -> Dict[str, Any]:
        """
        Execute a data-flow pipeline described as a list of stage dicts.

        How it works
        ------------
        compose_operations() runs stages one by one in the order you supply
        them. Before the first stage starts, a shared dictionary called
        `results` is created and starts empty.

        For each stage:
            1. The stage's "action" key is used to look up the right handler
               method in the `handlers` dispatch dictionary below.
            2. The handler is called. It reads whatever inputs it needs --
               either from the machine's variables/flags, or from `results`
               using _resolve_value() -- and returns a single computed value.
            3. If the stage has an "output" key, that return value is stored
               in `results` under that key. This makes it available to every
               stage that comes after.
            4. If the stage also has a "write_to" key, the return value is
               written back to the machine (as a variable or flag) so it
               persists beyond the pipeline.

        After all stages finish, the full `results` dictionary is returned.

        Why this matters for weird machines
        ------------------------------------
        Each stage is a gadget. compose_operations() is the mechanism that
        chains gadgets together so the output of one becomes the input of the
        next. The `results` dict is the data bus connecting them. This is
        exactly the same composition pattern found in real PLC function-block
        diagrams and Sequential Function Charts.

        Supported actions
        -----------------
            read_multiple      -- read a list of named sources
            aggregate          -- compute stats over a list (avg, sum, min, max)
            compute            -- evaluate an arithmetic expression
            compare            -- compare two values, returns bool
            conditional_action -- branch based on a condition
            write              -- TODO: write a value to a variable or flag
            scale              -- TODO: linearly map a value between two ranges

        Stage dictionary pattern
        ------------------------
            {
                "action":   "compute",          # required
                "output":   "error",            # optional: store in results
                "write_to": "error",            # optional: also write to machine
                ...                             # action-specific keys
            }

        TODO: Add a "compose" action type that accepts a nested list of stages
        under a "stages" key and calls compose_operations() recursively. This
        would allow sub-pipelines to be packaged as reusable units.

        TODO: Add per-stage tracing so each stage's action, resolved inputs,
        and output value are printed or logged when a debug flag is set.
        """
        if not isinstance(stages, list):
            raise TypeError("stages must be a list of stage dictionaries")

        results: Dict[str, Any] = {}

        # Dispatch table: action name -> handler method.
        #
        # TODO: Once _handle_write and _handle_scale are implemented above,
        # add them here:
        #     "write": self._handle_write,
        #     "scale": self._handle_scale,
        handlers = {
            "read_multiple":      self._handle_read_multiple,
            "aggregate":          self._handle_aggregate,
            "compute":            self._handle_compute,
            "compare":            self._handle_compare,
            "conditional_action": self._handle_conditional_action,
            "write":              self._handle_write,
            "scale":              self._handle_scale,   
        }

        for index, stage in enumerate(stages):
            self._validate_stage(stage, index)
            action_name = stage["action"]

            if action_name not in handlers:
                raise ValueError(
                    f"Stage {index}: unsupported action '{action_name}'. "
                    f"Available actions: {sorted(handlers)}"
                )

            # Run the handler and capture the result.
            value = handlers[action_name](stage, results)

            # Store the result in the shared results dict if the stage named an output.
            output_name = stage.get("output") or stage.get("cv_out")
            self._store_output(output_name, value, results)

            # Optionally persist the result back to the machine.
            write_to = stage.get("write_to")
            if write_to is not None:
                if isinstance(value, bool):
                    self.set_flag(write_to, value)
                else:
                    self.set_variable(write_to, int(value))

            self._execution_log.append(
                f"stage {index} ({action_name}): output={output_name!r} value={value!r}"
            )

        return results


# ---------------------------------------------------------------------------
# Demo pipelines
# ---------------------------------------------------------------------------

def demo_temperature_pipeline() -> None:
    """
    Pipeline: compute error -> check if heating needed -> act on the result.

    This shows how three stages chain together:
      Stage 0 computes error and stores it in results["error"].
      Stage 1 reads results["error"] (via _resolve_value) and returns a bool.
      Stage 2 reads that bool from results and decides which branch to run.
    """
    print("\n--- demo_temperature_pipeline ---")
    machine = SimpleWeirdMachine()
    machine.set_variable("current_temp", 68)
    machine.set_variable("setpoint", 72)
    machine.set_variable("mode", 1)

    results = machine.compose_operations([
        {
            "action": "compute",
            "expression": "setpoint - current_temp",
            "output": "error",
            "write_to": "error",
        },
        {
            "action": "compare",
            "left": "error",
            "op": ">",
            "right": 0,
            "output": "needs_heat",
            "write_to": "needs_heat",
        },
        {
            "action": "conditional_action",
            "condition": lambda: (
                machine.get_variable("mode") == 1
                and machine.get_flag("needs_heat")
            ),
            "true_action":  lambda: machine.set_flag("heater_on", True),
            "false_action": lambda: machine.set_flag("heater_on", False),
            "output": "branch_taken",
        },
    ])

    print("Pipeline results:", results)
    print("Machine variables:", machine._variables)
    print("Machine flags:", machine._flags)
    print("Execution log:")
    for entry in machine._execution_log:
        print(" ", entry)


def demo_aggregation_pipeline() -> None:
    """
    Pipeline: read sensors -> average them -> trigger alarm if over threshold.

    This shows how a list produced in stage 0 flows into stage 1,
    and how the scalar produced in stage 1 flows into stage 2.
    """
    print("\n--- demo_aggregation_pipeline ---")
    machine = SimpleWeirdMachine()
    machine.set_variable("zone1_temp", 72)
    machine.set_variable("zone2_temp", 75)
    machine.set_variable("zone3_temp", 78)
    machine.set_variable("zone4_temp", 70)

    results = machine.compose_operations([
        {
            "action": "read_multiple",
            "sources": ["zone1_temp", "zone2_temp", "zone3_temp", "zone4_temp"],
            "output": "all_temps",
        },
        {
            "action": "aggregate",
            "input": "all_temps",
            "operation": "average",
            "output": "avg_temp",
            "write_to": "avg_temp",
        },
        {
            "action": "compare",
            "left": "avg_temp",
            "op": ">=",
            "right": 74,
            "output": "overtemp_alarm",
            "write_to": "overtemp_alarm",
        },
    ])

    print("Pipeline results:", results)
    print("Machine variables:", machine._variables)
    print("Machine flags:", machine._flags)
    print("Execution log:")
    for entry in machine._execution_log:
        print(" ", entry)


if __name__ == "__main__":
    demo_temperature_pipeline()
    demo_aggregation_pipeline()
