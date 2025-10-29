"""
Example: Event Counter Using SimpleWeirdMachine

This is the counter example, but using the friendly SimpleWeirdMachine
wrapper. Use 'count' and 'event_signal' instead of register/coil numbers!

This example demonstrates:
- Named counter variable
- Named event flag
- Simple increment operation
- Event-driven counting
"""

from weird_machine_gadgets.simple import SimpleWeirdMachine
import time


def build_counter_simple(machine: SimpleWeirdMachine,
                        max_count: int = 10):
    """
    Build a simple event counter using SimpleWeirdMachine.
    
    Args:
        machine: SimpleWeirdMachine instance
        max_count: Maximum count before stopping
    """
    
    print("\n=== Event Counter (SimpleWeirdMachine Version) ===")
    print(f"Will count up to {max_count} events\n")
    
    # Initialize counter and event flag
    print("Initializing counter...")
    machine.set_variable('count', 0)
    machine.set_flag('event_signal', False)
    print("Counter ready.\n")
    
    # Track how many events we've simulated
    events_simulated = 0
    
    # Stop condition: counter reached max
    def counted_enough():
        current = machine.get_variable('count')
        if current >= max_count:
            print(f"\nCounter reached maximum ({max_count}).")
            return True
        return False
    
    # Event detection and increment
    def simulate_and_count():
        nonlocal events_simulated
        
        # Simulate an event occurring (in real use, this would come from
        # external hardware, sensors, or another program)
        if events_simulated < max_count:
            machine.set_flag('event_signal', True)
            events_simulated += 1
            time.sleep(0.05)
        
        # Check if event flag is triggered
        event_triggered = machine.get_flag('event_signal')
        
        if event_triggered:
            # Increment the counter (one simple operation!)
            machine.increment_variable('count')
            
            # Read and display current count
            current = machine.get_variable('count')
            print(f"Event detected! Count: {current}")
            
            # Reset the event signal
            machine.set_flag('event_signal', False)
    
    # Run the counter loop
    print("Counter running...\n")
    
    machine.repeat_until_done(
        condition=counted_enough,
        action=simulate_and_count,
        delay=0.2
    )
    
    # Display final count
    final_count = machine.get_variable('count')
    print(f"\nCounter stopped. Final count: {final_count}")
    
    return final_count


def reset_counter(machine: SimpleWeirdMachine):
    """Reset the counter to zero."""
    print("\nResetting counter to 0...")
    machine.set_variable('count', 0)
    print("Counter reset.\n")


def read_counter(machine: SimpleWeirdMachine):
    """Read the current counter value."""
    count = machine.get_variable('count')
    print(f"\nCurrent count: {count}\n")
    return count


def main():
    """Main example showing simple counter functionality."""
    TESTBED_IP = "192.168.1.100"
    
    print("=" * 70)
    print("Simple Event Counter (SimpleWeirdMachine Version)")
    print("=" * 70)
    print("\nThis demonstrates a basic tally/counter system built from")
    print("architectural gadgets, using descriptive variable names.\n")
    print(f"Connecting to {TESTBED_IP}...")
    
    # Connect
    machine = SimpleWeirdMachine(TESTBED_IP)
    print("Connected.\n")
    
    # Example: Count up to 10 events
    build_counter_simple(
        machine=machine,
        max_count=10
    )
    
    # Read final value
    final = read_counter(machine)
    
    # Could reset if needed
    # reset_counter(machine)
    
    print("=" * 70)
    print("Counter Example Complete")
    print("=" * 70)
    
    # Disconnect
    machine.disconnect()


if __name__ == "__main__":
    main()