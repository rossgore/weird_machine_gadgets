"""
Example: Simple Running Average Calculation (Non-Interactive)

Computes the average of three preset numbers ("8, 15, 28") using
architectural gadgets and Modbus register operations. There is
no user input—the numbers are injected automatically. This is
like a simple grade or sensor average calculator.

Demonstrates:
- Modbus storage of values (input, sum, count, average)
- Accumulation and division through register manipulation
- Reusable pattern for embedded computation with gadgets
"""

from weird_machine_gadgets import ProtocolGadget

def compute_average_of_three(
    modbus: ProtocolGadget,
    numbers,
    input_register=40060,
    sum_register=40061,
    count_register=40062,
    average_register=40063
):
    """
    Feeds three preset numbers into architectural registers to calculate the average.
    Args:
        modbus: ProtocolGadget for Modbus operations
        numbers: List of exactly 3 integers
        input_register: Register used for current input
        sum_register: Register for total running sum
        count_register: Register for number of inputs
        average_register: Register for running average
    """
    print("\n=== Simple Average Calculation ===")
    print(f"Numbers: {numbers}")
    print(f"Registers: input({input_register}), sum({sum_register}), count({count_register}), avg({average_register})\n")
    
    # Initialize all registers
    modbus.write_register(sum_register, 0)
    modbus.write_register(count_register, 0)
    modbus.write_register(average_register, 0)
    
    for idx, number in enumerate(numbers):
        print(f"Processing number {idx + 1}: {number}")
        
        # Store input
        modbus.write_register(input_register, number)
        
        # Read and update sum
        running_sum = modbus.read_register(sum_register)
        running_sum = running_sum + number
        modbus.write_register(sum_register, running_sum)
        
        # Update count
        count = modbus.read_register(count_register) + 1
        modbus.write_register(count_register, count)
        
        # Compute average (integer division for demonstration)
        average = running_sum // count
        modbus.write_register(average_register, average)
        
        print(f"  New sum: {running_sum}")
        print(f"  Count: {count}")
        print(f"  New average: {average}\n")
    
    final_average = modbus.read_register(average_register)
    print(f"Final computed average: {final_average}\n")
    return final_average

def main():
    TESTBED_IP = "192.168.1.100"
    MODBUS_PORT = 502
    numbers = [8, 15, 28]  # The three numbers to average

    print("=" * 60)
    print("Simple Average — Architectural Weird Machine")
    print("=" * 60)

    with ProtocolGadget(TESTBED_IP, port=MODBUS_PORT) as modbus:
        print("Connected.\n")
        compute_average_of_three(modbus, numbers)

        print("=" * 60)
        print("Simple Average Example Complete")
        print("=" * 60)

if __name__ == "__main__":
    main()
