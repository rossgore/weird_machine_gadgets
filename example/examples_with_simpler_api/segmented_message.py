"""
Example: Segmented Message Post Using SimpleWeirdMachine

This is an example of a message posted character by character on a system
using the SimpleWeirdMachine wrapper.

This example demonstrates:
- Named character stored as integer variable
- Named unread flag
- Event-driven message retrieval
"""

from weird_machine_gadgets.simple import SimpleWeirdMachine

def build_segmented_message_simple(machine: SimpleWeirdMachine,
                                   message: str = "Hello World!"):
    """
    Build a simple message poster using SimpleWeirdMachine.

    Args:
        machine: SimpleWeirdMachine instance
        message: Message to be posted
    """

    print("\n=== Segmented Message Posts (SimpleWeirdMachine Version) ===")
    print(f"Will control one message\n")

    # Initialize message and unread flag
    print("Initializing message variables...")
    machine.set_variable('current_char', 0)
    machine.set_flag('unread', False)
    print("Message ready.\n")

    # Stop condition: empty message
    def message_empty():
        if message == "":
            print(f"\nFull message retrieved.")
            return True
        return False

    final_message = ""

    # Sending/receiving character
    def send_rcv_char():
        nonlocal message
        nonlocal final_message

        # Retrieve and remove first char in message
        next_char = message[0]
        message = message[1:]

        # Simulate posting the current char of the message
        machine.set_variable('current_char', ord(next_char))
        machine.set_flag('unread', True)

        # Check if new char is available on server
        char_available = machine.get_flag('unread')

        if char_available:
            # Retrieve and decode the character
            read_char = chr(machine.get_variable('current_char'))

            # Display current character and append to retrieved message
            print(f"Character detected: {read_char}")
            final_message += read_char

            # Reset the event signal
            machine.set_flag('unread', False)

    # Run the message retrieval loop, one char per iteration
    print("Retrieving message...\n")

    machine.repeat_until_done(
        condition=message_empty,
        action=send_rcv_char,
        delay=0.2
    )

    # Display the new message
    print(f"\nMessage retrieved from server: {final_message}")

    return final_message

def main():
    """Main example showing simple character-by-character message transmission functionality."""
    TESTBED_IP = "192.168.1.100"

    print("=" * 70)
    print("Simple Segmented Message System (SimpleWeirdMachine Version)")
    print("=" * 70)
    print("\nThis demonstrates a basic message posting system built from")
    print("architectural gadgets, using descriptive variable names.\n")
    print(f"Connecting to {TESTBED_IP}...")

    # Connect
    machine = SimpleWeirdMachine(TESTBED_IP)
    print("Connected.\n")

    # Example: Send and retrieve a simple message character by character
    build_segmented_message_simple(
        machine=machine,
        message="This is a really cool message! How long can I make this message? Infinite (maybe)!")

    print("=" * 70)
    print("Segmented Message Example Complete")
    print("=" * 70)

    # Disconnect
    machine.disconnect()


if __name__ == "__main__":
    main()
