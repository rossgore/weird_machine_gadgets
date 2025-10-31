from weird_machine_gadgets.simple import SimpleWeirdMachine
import time

#This example of a weird machine takes control of Modbus registers
#to create a machine spits out a codeword every 10 seconds, for a minute

#Program starts once Machine has started
def build_machine(machine):
    machine.set_flag('is_vulnerable', False)
    print("SYSTEM STARTING.. INITIALIZING..\n")
    time.sleep(1)
    print("Machine online.\n")
    time.sleep(1)

#The list of codewords to be distributed one by one in order, every 10 seconds
def set_codeword():
    return ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "FOXTROT","SIERRA"]

#Creates timer to count from when computer turns on
def timer(machine, total_seconds: int = 60, interval:int = 10):

    #Before beginning count, checks to see if machine is ready for payload
    #If not, sets up timer at 0 to incrementally increase
    event_triggered= machine.get_flag('is_vulnerable')
    if event_triggered == False:
        print("TIMER STARTING..\n")
        print("PLEASE MONITOR AND STANDBY..\n")
        elapsed_time = 0

        #Sets flag, letting attacker know the timer is active

        machine.set_flag("timer_active", True)

        #Readies timer to fetch codewords
        codewords = set_codeword()

        for word in codewords:

            # Waits for 10 seconds and increases modbus register by 10

            time.sleep(interval)
            elapsed_time += interval
            machine.set_variable("time", elapsed_time)

            #Assigns codeword based on value of 10
            print(f"[{elapsed_time}s] CODEWORD: {word}")
            if elapsed_time == total_seconds:
                print(".....")
                time.sleep(1)
                print(".....")
                time.sleep(1)
                print("SIMULATION COMPLETE...")

                #Lets attacker know the cycle is done, and machine is now vulnerable

                machine.set_flag("is_vulnerable", True)
                machine.set_flag("timer_active", False)


    print("SYSTEM TIMER COMPLETE.")


def main():
    TESTBED_IP = "192.168.1.100"
    machine = SimpleWeirdMachine(TESTBED_IP)
    print(f"Connecting to {TESTBED_IP}...")

    build_machine(machine)
    timer(machine)
    machine.disconnect()

if __name__ == "__main__":
    main()
