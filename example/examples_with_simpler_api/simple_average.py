"""
Example: Simple Average Calculator Using SimpleWeirdMachine

This is the average example, but using the friendly SimpleWeirdMachine
wrapper. Use descriptive names like 'running_sum', 'count', and 'average'!

This example demonstrates:
- Multiple named variables for computation
- Accumulation pattern (running sum)
- Mathematical operations with remote storage
"""

from weird_machine_gadgets.simple import SimpleWeirdMachine


def compute_average_simple(machine: SimpleWeirdMachine, 
                          numbers: list):
    """
    Calculate the average of a list of numbers using SimpleWeirdMachine.
    
    Args:
        machine: SimpleWeirdMachine instance
        numbers: List of integers to average
    """
    
    print("\n=== Simple Average Calculation (SimpleWeirdMachine Version) ===")
    print(f"Numbers to average: {numbers}\n")
    
    # Initialize all variables with descriptive names
    print("Initializing calculator...")
    machine.set_variable('current_input', 0)
    machine.set_variable('running_sum', 0)
    machine.set_variable('count', 0)
    machine.set_variable('average', 0)
    print("Calculator ready.\n")
    
    # Process each number
    for idx, number in enumerate(numbers, 1):
        print(f"Processing number {idx}: {number}")
        
        # Store the current input
        machine.set_variable('current_input', number)
        
        # Add to running sum
        current_sum = machine.get_variable('running_sum')
        new_sum = current_sum + number
        machine.set_variable('running_sum', new_sum)
        
        # Increment count (one simple operation!)
        machine.increment_variable('count')
        
        # Calculate new average
        sum_value = machine.get_variable('running_sum')
        count_value = machine.get_variable('count')
        avg = sum_value // count_value
        machine.set_variable('average', avg)
        
        print(f"  Sum: {sum_value}")
        print(f"  Count: {count_value}")
        print(f"  Running average: {avg}\n")
    
    # Get final result
    final_avg = machine.get_variable('average')
    print(f"Final computed average: {final_avg}\n")
    return final_avg


def main():
    """Main function - calculates average of preset numbers."""
    TESTBED_IP = "192.168.1.100"
    numbers = [8, 15, 28]  # The three numbers to average
    
    print("=" * 70)
    print("Simple Average Calculator (SimpleWeirdMachine Version)")
    print("=" * 70)
    print("\nThis calculates the average of three numbers using")
    print("descriptive variable names instead of register numbers!\n")
    print(f"Connecting to {TESTBED_IP}...")
    
    # Connect
    machine = SimpleWeirdMachine(TESTBED_IP)
    print("Connected.\n")
    
    # Calculate average
    average = compute_average_simple(machine, numbers)
    
    print("=" * 70)
    print("Average Calculation Complete")
    print("=" * 70)
    print(f"\nNumbers: {numbers}")
    print(f"Average: {average}")
    
    # Disconnect
    machine.disconnect()


if __name__ == "__main__":
    main()