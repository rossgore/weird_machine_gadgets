"""
Simplified Weird Machine Library for Beginners

This wrapper makes the library feel more like regular Python programming.
Perfect for students with Python/Java background but unfamiliar with Modbus.

Instead of thinking about "registers" and "coils", just think about
"variables" and "flags" that happen to be stored remotely.
"""

from weird_machine_gadgets import ProtocolGadget, ControlGadget


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
