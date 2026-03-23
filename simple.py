"""
Simplified Weird Machine Library for Beginners

This wrapper makes the library feel more like regular Python programming.
Perfect for students with Python/Java background but unfamiliar with Modbus.

Instead of thinking about "registers" and "coils", just think about
"variables" and "flags" that happen to be stored remotely.
"""

from weird_machine_gadgets import ProtocolGadget, ControlGadget
import operator
import re
import ast

class SimpleWeirdMachine:
    """
    Beginner-friendly wrapper for the weird machine library.
    
    Hides Modbus terminology and makes it feel like regular Python.
    
    Example:
        machine = SimpleWeirdMachine('192.168.1.100')
        machine.set_variable('counter', 10)
        value = machine.get_variable('counter')
        machine.set_flag('ready', True)
    """
    
    def __init__(self, ip_address: str, port: int = 502):
        """
        Connect to the weird machine system.
        
        Args:
            ip_address: IP address of the PLC (like '192.168.1.100')
            port: Modbus port (default 502, you rarely need to change this)
        """
        self.modbus = ProtocolGadget(ip_address, port=port)
        self.control = ControlGadget(self.modbus)
        self.modbus.connect()
        
        # Keep track of which register/coil numbers we've used
        # So students don't have to remember addresses
        self._variable_map = {}
        self._flag_map = {}
        self._next_register = 40001
        self._next_coil = 10001
    
    def __enter__(self):
        """Allow using 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically disconnect when done."""
        self.disconnect()
    
    def disconnect(self):
        """Close connection to the PLC."""
        self.modbus.disconnect()
    
    # ==================== Variables (Integers) ====================
    
    def set_variable(self, name: str, value: int):
        """
        Set a variable (like x = 5 in Python).
        
        Example:
            machine.set_variable('counter', 10)
            machine.set_variable('temperature', 72)
        """
        # Assign a register number if this is a new variable
        if name not in self._variable_map:
            self._variable_map[name] = self._next_register
            self._next_register += 1
        
        register = self._variable_map[name]
        self.modbus.write_register(register, value)
    
    def get_variable(self, name: str) -> int:
        """
        Get a variable value (like reading x in Python).
        
        Example:
            value = machine.get_variable('counter')
        """
        if name not in self._variable_map:
            raise ValueError(f"Variable '{name}' not found. Use set_variable() first.")
        
        register = self._variable_map[name]
        return self.modbus.read_register(register)
    
    def increment_variable(self, name: str) -> int:
        """
        Add 1 to a variable (like x += 1 in Python).
        
        Returns the new value.
        """
        register = self._variable_map[name]
        return self.modbus.increment_register(register)
    
    def decrement_variable(self, name: str) -> int:
        """
        Subtract 1 from a variable (like x -= 1 in Python).
        
        Returns the new value.
        """
        register = self._variable_map[name]
        return self.modbus.decrement_register(register)
    
    # ==================== Flags (Booleans) ====================
    
    def set_flag(self, name: str, value: bool):
        """
        Set a flag (like flag = True in Python).
        
        Example:
            machine.set_flag('ready', True)
            machine.set_flag('done', False)
        """
        # Assign a coil number if this is a new flag
        if name not in self._flag_map:
            self._flag_map[name] = self._next_coil
            self._next_coil += 1
        
        coil = self._flag_map[name]
        self.modbus.write_coil(coil, value)
    
    def get_flag(self, name: str) -> bool:
        """
        Get a flag value (like reading a boolean in Python).
        
        Example:
            if machine.get_flag('ready'):
                print("System ready!")
        """
        if name not in self._flag_map:
            raise ValueError(f"Flag '{name}' not found. Use set_flag() first.")
        
        coil = self._flag_map[name]
        return self.modbus.read_coil(coil)
    
    # ==================== Control Flow ====================
    
    def wait_until(self, condition, timeout: float = None):
        """
        Wait until a condition is true (like a while loop).
        
        Args:
            condition: A function that returns True when done
            timeout: Maximum seconds to wait (None = wait forever)
        
        Example:
            # Wait until counter reaches 0
            machine.wait_until(lambda: machine.get_variable('counter') == 0)
        """
        return self.control.wait_for_condition(condition, timeout)
    
    def repeat_until_done(self, condition, action, delay: float = 0.1):
        """
        Repeat an action until a condition is met.
        
        This is like a while loop, but using weird machine operations.
        
        Args:
            condition: Function that returns True when done
            action: Function to run each loop iteration
            delay: Seconds to wait between loops
        
        Example:
            # Countdown from 10 to 0
            machine.set_variable('counter', 10)
            machine.repeat_until_done(
                condition=lambda: machine.get_variable('counter') == 0,
                action=lambda: machine.decrement_variable('counter'),
                delay=0.5
            )
        """
        return self.control.repeat_until(condition, action, delay)

    def compare(self, left_var: str, operation: str, right_value: int | str) -> bool:
        """
        Compare a variable with a value or second variable.

        This handles comparisons and other operations present in Python's built-in operator class.
        Args:
            left_var: First variable in comparison
            operation: Operator to use (i.e., '<', '>', '==')
            right_value: Integer value or string to compare against

        Example:
            # Evaluate counter > 10
            machine.set_variable('counter', 9)
            machine.compare('counter', '>', 10)
        """

        left_reg_value = self.get_variable(left_var)

        operator_map = {'<': operator.lt, '>': operator.gt, '==': operator.eq, '<=': operator.le, '>=': operator.ge, '!=': operator.ne}
        if operation not in operator_map:
            raise ValueError(f"Invalid operator. Use one of the following: {list(operator_map.keys())}.")

        if isinstance(right_value, str):
            right_reg_value = self.get_variable(right_value)
        else:
            right_reg_value = int(right_value)

        return operator_map[operation](left_reg_value, right_reg_value)

    def conditional_action(self, condition_func: Callable[[], bool], action_func: Callable[[], None], else_action_func: Callable[[], None] = None) -> bool:
        """
        Execute a function if a condition is true.

        This encapsulates conditional function execution based on remote PLC state.
        Args:
            condition_func: Condition to check for validity
            action_func: Function to execute if true
            else_action_func: Function to execute if false

        Example:
            # If mode == 1, output = 100. Else, output = 0.
            machine.conditional_action(
                condition_func=lambda: machine.get_variable('mode') == 1,
                action_func=lambda: machine.set_variable('output', 100),
                else_action_func=lambda: machine.set_variable('output', 0)
            )
        """
        condition_result = condition_func()

        if condition_result:
            action_func()
        elif else_action_func is not None:
            else_action_func()

        return condition_result

    # ==================== Arithmetic ====================

    def eval_node(self, node):
        _OPERATORS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.floordiv,  # Treat / as integer division (//) to mirror modbus functionality
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        if isinstance(node, ast.Expression):
            return self.eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](self.eval_node(node.left), self.eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](self.eval_node(node.operand))
        raise ValueError(f"Disallowed expression: {ast.dump(node)}")

    def compute(self, expression: str) -> int:
        """
        Calculate the result of an arithmetic expression.

        This computes arithmetic expressions using PLC variables and attached values.
        Args:
            expression: String expression to evaluate

        Example:
            # Calculate temp_f - temp_f2
            machine.set_variable('temp_f', 98)
            machine.set_variable('temp_f2', 53)
            result = machine.compute('temp_f - temp_f2')
        """

        var_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)

        for var_name in var_names:
            # Find value of var_name
            var_value = self.get_variable(var_name)

            # Replace var_name with the associated value
            expression = expression.replace(var_name, str(var_value), 1)

        if re.search(r'[a-zA-Z_]', expression):
            raise ValueError("Unresolved variable or illegal token in expression")

        return self.eval_node(ast.parse(expression, mode="eval"))


# ==================== Simple Examples ====================

def example_variables():
    """Example: Using variables (like normal Python)."""
    print("\n=== Variable Example ===\n")
    
    with SimpleWeirdMachine('192.168.1.100') as machine:
        # Set variables (like x = 10, y = 20)
        machine.set_variable('x', 10)
        machine.set_variable('y', 20)
        
        # Read them back
        x = machine.get_variable('x')
        y = machine.get_variable('y')
        print(f"x = {x}, y = {y}")
        
        # Calculate sum
        total = x + y
        machine.set_variable('total', total)
        print(f"total = {machine.get_variable('total')}")


def example_flags():
    """Example: Using boolean flags."""
    print("\n=== Flag Example ===\n")
    
    with SimpleWeirdMachine('192.168.1.100') as machine:
        # Set flags
        machine.set_flag('ready', True)
        machine.set_flag('error', False)
        
        # Check flags (like if statements)
        if machine.get_flag('ready'):
            print("System is ready!")
        
        if not machine.get_flag('error'):
            print("No errors detected")


def example_countdown():
    """Example: Countdown timer (the classic weird machine)."""
    print("\n=== Countdown Timer Example ===\n")
    
    with SimpleWeirdMachine('192.168.1.100') as machine:
        # Initialize counter
        machine.set_variable('counter', 5)
        print("Starting countdown from 5...\n")
        
        # Define when to stop
        def is_zero():
            return machine.get_variable('counter') == 0
        
        # Define what to do each loop
        def countdown_step():
            value = machine.get_variable('counter')
            print(f"Counter: {value}")
            machine.decrement_variable('counter')
        
        # Run the countdown loop
        machine.repeat_until_done(
            condition=is_zero,
            action=countdown_step,
            delay=0.5
        )
        
        print("\nCountdown complete!")


def example_accumulator():
    """Example: Add up a list of numbers."""
    print("\n=== Accumulator Example ===\n")
    
    with SimpleWeirdMachine('192.168.1.100') as machine:
        numbers = [5, 10, 15, 20, 25]
        print(f"Adding numbers: {numbers}\n")
        
        # Initialize sum
        machine.set_variable('sum', 0)
        
        # Add each number
        for num in numbers:
            current_sum = machine.get_variable('sum')
            new_sum = current_sum + num
            machine.set_variable('sum', new_sum)
            print(f"Added {num}: sum = {new_sum}")
        
        print(f"\nFinal sum: {machine.get_variable('sum')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Weird Machine Examples")
    print("=" * 60)
    
    # Run examples (uncomment to try)
    example_variables()
    # example_flags()
    # example_countdown()
    # example_accumulator()
