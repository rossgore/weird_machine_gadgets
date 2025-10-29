# Quick Reference: Building the Three Examples with SimpleWeirdMachine

This guide shows you how to build the same programs from the examples, but using the beginner-friendly `SimpleWeirdMachine` wrapper instead of raw Modbus operations.

---

## Example 1: Countdown Timer

### Original Version (Using ProtocolGadget)
```python
from weird_machine_gadgets import ProtocolGadget, ControlGadget

modbus = ProtocolGadget('192.168.1.100', port=502)
modbus.connect()
control = ControlGadget(modbus)

# Initialize
modbus.write_register(40001, 10)
modbus.write_coil(10001, True)

# Countdown loop
def is_zero():
    return modbus.read_register(40001) == 0

def decrement():
    current = modbus.read_register(40001)
    modbus.write_register(40001, current - 1)

control.repeat_until(condition=is_zero, body=decrement, delay=0.1)

# Cleanup
modbus.write_coil(10001, False)
modbus.disconnect()
```

### SimpleWeirdMachine Version
```python
from weird_machine_gadgets.simple import SimpleWeirdMachine

# Connect
machine = SimpleWeirdMachine('192.168.1.100')

# Initialize timer
machine.set_variable('countdown', 10)
machine.set_flag('timer_active', True)

print("Starting countdown from 10...\n")

# Countdown loop - much simpler!
def is_zero():
    value = machine.get_variable('countdown')
    if value == 0:
        print("Timer reached 0!\n")
    return value == 0

def countdown_step():
    current = machine.get_variable('countdown')
    print(f"Countdown: {current}")
    machine.decrement_variable('countdown')

machine.repeat_until_done(
    condition=is_zero,
    action=countdown_step,
    delay=0.1
)

# Cleanup
machine.set_flag('timer_active', False)
print("Timer complete!")

machine.disconnect()
```

**Simpler:**
- No register numbers to remember (40001, 10001) - just use names!
- `set_variable('countdown', 10)` instead of `write_register(40001, 10)`
- `decrement_variable('countdown')` instead of read→subtract→write

---

## Example 2: Event Counter

### Original Version (Using ProtocolGadget)
```python
from weird_machine_gadgets import ProtocolGadget, ControlGadget

modbus = ProtocolGadget('192.168.1.100', port=502)
modbus.connect()
control = ControlGadget(modbus)

# Initialize
modbus.write_register(40010, 0)

events_simulated = 0
max_count = 10

def reached_max():
    return modbus.read_register(40010) >= max_count

def check_and_increment():
    global events_simulated
    if events_simulated < max_count:
        modbus.write_coil(10010, True)
        events_simulated += 1
    
    if modbus.read_coil(10010):
        current = modbus.read_register(40010)
        modbus.write_register(40010, current + 1)
        print(f"Event {current + 1}")
        modbus.write_coil(10010, False)

control.repeat_until(reached_max, check_and_increment, delay=0.2)

modbus.disconnect()
```

### SimpleWeirdMachine Version (Easier!)
```python
from weird_machine_gadgets.simple import SimpleWeirdMachine
import time

# Connect
machine = SimpleWeirdMachine('192.168.1.100')

# Initialize counter
machine.set_variable('count', 0)
machine.set_flag('event_signal', False)

print("Starting event counter...\n")

events_to_simulate = 10
events_simulated = 0

# Stop when we've counted 10 events
def counted_enough():
    return machine.get_variable('count') >= events_to_simulate

# Simulate event and count it
def simulate_and_count():
    global events_simulated
    
    # Simulate an event happening
    if events_simulated < events_to_simulate:
        machine.set_flag('event_signal', True)
        events_simulated += 1
        time.sleep(0.05)
    
    # Check if event flag is set
    if machine.get_flag('event_signal'):
        # Increment the counter
        machine.increment_variable('count')
        current = machine.get_variable('count')
        print(f"Event detected! Count: {current}")
        
        # Reset the event signal
        machine.set_flag('event_signal', False)

# Run the counter
machine.repeat_until_done(
    condition=counted_enough,
    action=simulate_and_count,
    delay=0.2
)

final_count = machine.get_variable('count')
print(f"\nFinal count: {final_count}")

machine.disconnect()
```

**What's Simpler:**
- Named variables: `'count'` instead of remembering register 40010
- Named flags: `'event_signal'` instead of coil 10010
- `increment_variable('count')` - one line instead of read→add→write
- No need to track which registers you've used!

---

## Example 3: Simple Average Calculator

### Original Version (Using ProtocolGadget)
```python
from weird_machine_gadgets import ProtocolGadget

modbus = ProtocolGadget('192.168.1.100', port=502)
modbus.connect()

numbers = [8, 15, 28]

# Initialize registers
modbus.write_register(40060, 0)  # input
modbus.write_register(40061, 0)  # sum
modbus.write_register(40062, 0)  # count
modbus.write_register(40063, 0)  # average

# Process each number
for number in numbers:
    # Store input
    modbus.write_register(40060, number)
    
    # Update sum
    running_sum = modbus.read_register(40061)
    running_sum = running_sum + number
    modbus.write_register(40061, running_sum)
    
    # Update count
    count = modbus.read_register(40062) + 1
    modbus.write_register(40062, count)
    
    # Calculate average
    average = running_sum // count
    modbus.write_register(40063, average)
    
    print(f"Added {number}: sum={running_sum}, count={count}, avg={average}")

final_average = modbus.read_register(40063)
print(f"\nFinal average: {final_average}")

modbus.disconnect()
```

### SimpleWeirdMachine Version (Easier!)
```python
from weird_machine_gadgets.simple import SimpleWeirdMachine

# Connect
machine = SimpleWeirdMachine('192.168.1.100')

numbers = [8, 15, 28]

print(f"Calculating average of: {numbers}\n")

# Initialize variables
machine.set_variable('current_input', 0)
machine.set_variable('running_sum', 0)
machine.set_variable('count', 0)
machine.set_variable('average', 0)

# Process each number
for number in numbers:
    print(f"Processing number: {number}")
    
    # Store the current input
    machine.set_variable('current_input', number)
    
    # Add to running sum
    current_sum = machine.get_variable('running_sum')
    new_sum = current_sum + number
    machine.set_variable('running_sum', new_sum)
    
    # Increment count
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
print(f"Final average: {final_avg}")

machine.disconnect()
```

**What's Simpler:**
- Descriptive variable names: `'running_sum'`, `'count'`, `'average'`
- No register numbers to manage (40060, 40061, 40062, 40063)
- Easier to read and understand what each variable represents
- `increment_variable('count')` instead of read→add 1→write

---

## Side-by-Side Comparison

| Task | ProtocolGadget | SimpleWeirdMachine |
|------|----------------|-------------------|
| Set variable | `modbus.write_register(40001, 10)` | `machine.set_variable('counter', 10)` |
| Read variable | `x = modbus.read_register(40001)` | `x = machine.get_variable('counter')` |
| Increment | `x = modbus.read_register(40001)`<br>`modbus.write_register(40001, x+1)` | `machine.increment_variable('counter')` |
| Decrement | `x = modbus.read_register(40001)`<br>`modbus.write_register(40001, x-1)` | `machine.decrement_variable('counter')` |
| Set flag | `modbus.write_coil(10001, True)` | `machine.set_flag('ready', True)` |
| Check flag | `if modbus.read_coil(10001):` | `if machine.get_flag('ready'):` |
| Loop | `control.repeat_until(cond, body)` | `machine.repeat_until_done(cond, action)` |

---


## Need Help?

If you're stuck:
1. Write it in normal Python first (with regular variables)
2. Identify which variables you need
3. Replace `x = 5` with `machine.set_variable('x', 5)`
4. Replace `y = x` with `y = machine.get_variable('x')`
5. Test each step as you go!

Remember: You're not learning new programming, just using familiar concepts with remote storage!
