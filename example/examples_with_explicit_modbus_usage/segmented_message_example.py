"""
Example: Segmented Message Post Using Architectural Gadgets

This is an example of a message posted character by character on a system
using the Architectural Gadgets ProtocolGadget and ControlGadget.

This example demonstrates:
- Named character stored as integer variable
- Named unread flag
- Event-driven message retrieval
"""

from weird_machine_gadgets import ProtocolGadget, ControlGadget

def build_segmented_message(machine: ProtocolGadget,
                                   char_register: int = 40010,
                                   unread_coil: int = 10010,
                                   message: str = "Hello World!"):
    """
    Build a message poster using Architectural Gadgets.

    Args:
        machine: ProtocolGadget instance
        message: Message to be posted
    """

    print("\n=== Segmented Message Posts (Architectural Gadget Version) ===")
    print(f"Will control one message\n")

    # Initialize message and unread flag
    print("Initializing message variables...")
    machine.write_register(char_register, 0)
    machine.write_coil(unread_coil, False)
    print("Message ready.\n")

    control = ControlGadget(machine)

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
        machine.write_register(char_register, ord(next_char))
        machine.write_coil(unread_coil, True)

        # Check if new char is available on server
        char_available = machine.read_coil(unread_coil)

        if char_available:
            # Retrieve and decode the character
            read_char = chr(machine.read_register(char_register))

            # Display current character and append to retrieved message
            print(f"Character detected: {read_char}")
            final_message += read_char

            # Reset the event signal
            machine.write_coil(unread_coil, False)

    # Run the message retrieval loop, one char per iteration
    print("Retrieving message...\n")

    control.repeat_until(
        condition=message_empty,
        body=send_rcv_char,
        delay=0.2
    )

    # Display the new message
    print(f"\nMessage retrieved from server: {final_message}")

    return final_message

def main():
    """Main example showing character-by-character message transmission functionality."""
    TESTBED_IP = "192.168.1.100"
    MODBUS_PORT = 500

    print("=" * 70)
    print("Segmented Message System (Constructive Weird Machine)")
    print("=" * 70)
    print("\nThis demonstrates a basic message posting system built from")
    print("architectural gadgets, using descriptive variable names.\n")
    print(f"Connecting to {TESTBED_IP}:{MODBUS_PORT}...")

    # Connect
    machine = ProtocolGadget(TESTBED_IP, port=MODBUS_PORT)
    print("Connected.\n")

    # Example: Send and retrieve a simple message character by character
    build_segmented_message(
        machine=machine,
        char_register=40010,
        unread_coil=10010,
        message="This is a really cool message! How long can I make this message? Infinite (maybe)!")

    print("=" * 70)
    print("Segmented Message Example Complete")
    print("=" * 70)

    # Disconnect
    machine.disconnect()


if __name__ == "__main__":
    main()
