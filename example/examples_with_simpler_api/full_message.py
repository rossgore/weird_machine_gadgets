"""
Example: Full Message Post Using SimpleWeirdMachine

This is an example of a message posted all at once on a
system using the SimpleWeirdMachine wrapper.

This example demonstrates:
- Named character stored as integer variable
- Named unread flag
- Event-driven message retrieval
"""

from weird_machine_gadgets.simple import SimpleWeirdMachine

def build_full_message_simple(machine: SimpleWeirdMachine,
                              message: str = "Hello World!"):
    """
    Build a simple message poster using SimpleWeirdMachine.

    Args:
        machine: SimpleWeirdMachine instance
        message: Message to be posted
    """

    print("\n=== Full Message Posts (SimpleWeirdMachine Version) ===")
    print(f"Will control one message\n")

    # Initialize message and unread flag
    print("Initializing message variables...")
    machine.set_variable('message', 0)
    machine.set_flag('unread', False)
    print("Message ready.\n")

    final_message = ""

    # Sending/receiving character
    def send_rcv_message():
        nonlocal message
        nonlocal final_message

        # Simulate posting a message (in real use, this would come from
        # external hardware, sensors, or another program)
        previous = machine.get_variable('message')
        new_message = int.from_bytes(message.encode('utf-8'), 'big')
        if previous != new_message:
            machine.set_variable('message', int.from_bytes(message.encode('utf-8'), 'big'))
            # Trigger unread flag
            machine.set_flag('unread', True)

        # Check if unread flag is triggered
        unread_message = machine.get_flag('unread')

        # Display if there is a new message
        if unread_message:
            final_message = string_decode_int(machine)
            print(f"\nMessage retrieved from server: {final_message}")
            machine.set_flag('unread', False)

    print("Retrieving message...\n")
    send_rcv_message()

    return final_message

def int_encode_string(machine: SimpleWeirdMachine):
    """Encode the message string"""
    print("\nEncoding message string to integer...")
    retrieved_message_value = machine.get_variable('message')
    message_string = retrieved_message_value.to_bytes((retrieved_message_value.bit_length() + 7) // 8, "big").decode()
    print(f"\nMessage encoded: {message_string}")

def string_decode_int(machine: SimpleWeirdMachine):
    """Decode the message integer"""
    print("\nDecoding message integer to string...")
    retrieved_message_value = machine.get_variable('message')
    message_string = retrieved_message_value.to_bytes((retrieved_message_value.bit_length() + 7) // 8, "big").decode()
    print(f"\nMessage decoded: {message_string}")
    return message_string

def read_message(machine: SimpleWeirdMachine):
    """Read the current message value."""
    print(f"\nCurrent message: {string_decode_int(machine)}\n")

def main():
    """Main example showing simple full message transmission functionality."""
    TESTBED_IP = "192.168.1.100"

    print("=" * 70)
    print("Simple Full Message System (SimpleWeirdMachine Version)")
    print("=" * 70)
    print("\nThis demonstrates a basic message posting system built from")
    print("architectural gadgets, using descriptive variable names.\n")
    print(f"Connecting to {TESTBED_IP}...")

    # Connect
    machine = SimpleWeirdMachine(TESTBED_IP)
    print("Connected.\n")

    # Example: Send and a retrieve a simple message all at once
    build_full_message_simple(
        machine=machine,
        message="This is a really cool message! "
                "How long can I make this message? Infinite (maybe)!"
    )

    # Read final value
    read_message(machine)

    print("=" * 70)
    print("Full Message Complete")
    print("=" * 70)

    # Disconnect
    machine.disconnect()


if __name__ == "__main__":
    main()
