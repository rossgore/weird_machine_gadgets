"""
Example: Countdown Timer Using SimpleWeirdMachine

This is the timer example, but using the friendly SimpleWeirdMachine
wrapper. No register numbers to remember - just use descriptive variable names!

This example demonstrates:
- Named variables instead of register addresses
- Simple increment/decrement operations
- Clean loop syntax
- Boolean flags with meaningful names
"""

from weird_machine_gadgets.simple import SimpleWeirdMachine


def build_timer_simple(machine: SimpleWeirdMachine,
                       initial_count: int = 10,
                       cycle_time: float = 0.1):
    """
    Build a countdown timer using SimpleWeirdMachine.
    
    Args:
        machine: SimpleWeirdMachine instance
        initial_count: Number to count down from
        cycle_time: Seconds between each count
    """
    
    print("\n=== Building Timer (SimpleWeirdMachine Version) ===")
    print(f"Starting countdown from {initial_count}\n")
    
    # ------------------------------------------------------------
    # STEP 1: Initialize
    # ------------------------------------------------------------
    print("Step 1: Initialization")
    
    # Set countdown variable (like: countdown = 10)
    machine.set_variable('countdown', initial_count)
    print(f"  Set 'countdown' = {initial_count}")
    
    # Set timer active flag (like: timer_active = True)
    machine.set_flag('timer_active', True)
    print("  Set 'timer_active' = True")
    print("  Timer initialized\n")
    
    # ------------------------------------------------------------
    # STEP 2: The Countdown Loop
    # ------------------------------------------------------------
    print("Step 2: Countdown Loop\n")
    
    # Stop condition - check if countdown reached 0
    def is_zero():
        value = machine.get_variable('countdown')
        if value == 0:
            print("  Counter reached 0!\n")
        return value == 0
    
    # Loop body - print current value and decrement
    def countdown_step():
        current = machine.get_variable('countdown')
        print(f"  Countdown: {current}")
        machine.decrement_variable('countdown')
    
    # Execute the countdown loop
    machine.repeat_until_done(
        condition=is_zero,
        action=countdown_step,
        delay=cycle_time
    )
    
    # ------------------------------------------------------------
    # STEP 3: Cleanup
    # ------------------------------------------------------------
    print("Step 3: Cleanup")
    machine.set_flag('timer_active', False)
    print("  Set 'timer_active' = False")
    print("  Timer deactivated\n")


def main():
    """Main function - runs the countdown timer."""
    TESTBED_IP = "192.168.1.100"
    
    print("=" * 70)
    print("Countdown Timer (SimpleWeirdMachine Version)")
    print("=" * 70)
    print("\nThis is the same timer example, but using descriptive")
    print("variable names instead of register numbers!\n")
    print(f"Connecting to {TESTBED_IP}...")
    
    # Connect using SimpleWeirdMachine
    machine = SimpleWeirdMachine(TESTBED_IP)
    print("Connected.\n")
    
    # Build and run a 10-second countdown
    build_timer_simple(
        machine=machine,
        initial_count=10,
        cycle_time=0.1
    )
    
    # Optional: Execute a payload when done
    print("=" * 70)
    print("EXECUTING PAYLOAD")
    print("=" * 70)
    print("\nTimer complete. Setting wind speed to maximum...\n")
    
    # Store the wind speed value
    machine.set_variable('wind_speed', 1023)
    print(f"Wind speed set to: {machine.get_variable('wind_speed')}")
    
    print("\n" + "=" * 70)
    print("Timer Complete")
    print("=" * 70)
    print("\nAll computation done using named variables!")
    print("No register numbers needed!\n")
    
    # Disconnect when done
    machine.disconnect()


if __name__ == "__main__":
    main()