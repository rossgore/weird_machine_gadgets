"""
Example: Simple Event Counter Using Architectural Gadgets

This example shows how to build a basic counter/tally system using 
Modbus operations. Think of it like a visitor counter at a museum 
or a click counter - simple, useful, and easy to understand.

The counter:
- Increments each time an event is detected (coil goes HIGH)
- Can be reset to zero
- Can read the current count

All operations use Modbus primitives (no Python loops or state).
"""

from weird_machine_gadgets import ProtocolGadget, ControlGadget
import time


def build_counter(modbus: ProtocolGadget,
                 count_register: int = 40010,
                 event_coil: int = 10010,
                 max_count: int = 10):
    """
    Build a simple event counter.
    
    Args:
        modbus: ProtocolGadget for communication
        count_register: Register storing the count
        event_coil: Coil that triggers count increment
        max_count: Maximum count before stopping
    """
    
    print("\n=== Event Counter ===")
    print(f"Count register: {count_register}")
    print(f"Event trigger coil: {event_coil}")
    print(f"Max count: {max_count}\n")
    
    # Initialize counter to zero
    print("Initializing counter to 0...")
    modbus.write_register(count_register, 0)
    print("Counter ready.\n")
    
    control = ControlGadget(modbus)
    
    # Track how many events we've simulated
    events_simulated = 0
    
    # Stop condition: counter reached max
    def reached_max():
        current = modbus.read_register(count_register)
        if current >= max_count:
            print(f"\nCounter reached maximum ({max_count}).")
            return True
        return False
    
    # Event detection and increment
    def check_and_increment():
        nonlocal events_simulated
        
        # Simulate an event occurring (in real use, this would come from
        # external hardware, sensors, or another program)
        if events_simulated < max_count:
            modbus.write_coil(event_coil, True)
            events_simulated += 1
            time.sleep(0.05)
        
        # Check if event coil is triggered
        event_triggered = modbus.read_coil(event_coil)
        
        if event_triggered:
            # Read current count
            current = modbus.read_register(count_register)
            new_count = current + 1
            
            # Write new count
            modbus.write_register(count_register, new_count)
            print(f"Event detected! Count: {current} -> {new_count}")
            
            # Reset the event coil
            modbus.write_coil(event_coil, False)
    
    # Run the counter loop
    print("Counter running...\n")
    
    iterations = control.repeat_until(
        condition=reached_max,
        body=check_and_increment,
        delay=0.2
    )
    
    final_count = modbus.read_register(count_register)
    print(f"\nCounter stopped. Final count: {final_count}")
    print(f"Processed {iterations} check cycles.")
    
    return final_count


def reset_counter(modbus: ProtocolGadget, count_register: int = 40010):
    """Reset the counter to zero."""
    print("\nResetting counter to 0...")
    modbus.write_register(count_register, 0)
    print("Counter reset.\n")


def read_counter(modbus: ProtocolGadget, count_register: int = 40010):
    """Read the current counter value."""
    count = modbus.read_register(count_register)
    print(f"\nCurrent count: {count}\n")
    return count


def main():
    """Main example showing simple counter functionality."""
    TESTBED_IP = "192.168.1.100"
    MODBUS_PORT = 502
    
    print("=" * 70)
    print("Simple Event Counter (Constructive Weird Machine)")
    print("=" * 70)
    print("\nThis demonstrates a basic tally/counter system built from")
    print("architectural gadgets. Like counting visitors, button presses,")
    print("or any discrete events.\n")
    print(f"Connecting to {TESTBED_IP}:{MODBUS_PORT}...")
    
    with ProtocolGadget(TESTBED_IP, port=MODBUS_PORT) as modbus:
        print("Connected.\n")
        
        # Example: Count up to 10 events
        build_counter(
            modbus=modbus,
            count_register=40010,
            event_coil=10010,
            max_count=10
        )
        
        # Read final value
        final = read_counter(modbus, count_register=40010)
        
        # Could reset if needed
        # reset_counter(modbus, count_register=40010)
        
        print("=" * 70)
        print("Counter Example Complete")
        print("=" * 70)
        print("\nThis shows how architectural gadgets can implement")
        print("simple, useful tools - not just exploits.")
        print("The counter uses only FC03 (read) and FC06 (write) operations.\n")


if __name__ == "__main__":
    main()
