import os
import time
import threading

from weird_machine_gadgets import ProtocolGadget, ControlGadget

HOST = os.getenv("WM_HOST", "127.0.0.1")
PORT = int(os.getenv("WM_PORT", "502"))
UNIT = int(os.getenv("WM_UNIT", "1"))


# Addresses
SEMA_COIL   = 21        # 00021: 0=Unlocked, 1=Locked
OWNER_REG   = 40031     # holds current owner id (0 = none)
SHARED_COUNT= 40100     # shared counter incremented in the critical section

# Small helper to sleep a bit between protocol ops (keeping logs readable)
SLEEP = float(os.getenv("WM_SLEEP", "0.05"))

class BinarySemaphore:
    """
    Binary semaphore built from architectural gadgets only:
        - Uses read_coil/write_coil and conditional_write (ControlGadget)
        - Approximates test-and-set via an owner register + immediate verification
    """
    def __init__(self, proto: ProtocolGadget, control: ControlGadget):
        self.p = proto
        self.c = control

    def take(self, task_id: int, timeout_s: float = 5.0) -> bool:
        """
        Try to acquire the semaphore within timeout:
            1) Wait until SEMA == 0 (unlocked)
            2) Claim ownership if OWNER_REG == 0 (unowned)
            3) Lock the semaphore by writing SEMA_COIL = 1
            4) verify SEMA COIL is locked (1) and OWNER_REG == task_id (verify we own it)
            5) On failure (race), release claim and retry with slight backoff
        """
        start = time.time()
        backoff = 0.5
        while time.time() - start < timeout_s:
            # wait until the flag is seen as 0
            if not self.c.wait_for_condition(lambda: self.p.read_coil(SEMA_COIL) is False, timeout=1.0):
                # Didn't open this second; try again
                continue
            time.sleep(SLEEP)

            # Try to claim ownership if currently unowned (OWNER_REG==0)
            self.c.conditional_write(OWNER_REG, 0, task_id, self.p.read_register(OWNER_REG))
            time.sleep(SLEEP)
        
            # Did we become owner?
            if self.p.read_register(OWNER_REG) != task_id:
                time.sleep(0.1)
                backoff = min(backoff * 1.5, 0.3)
                continue

            # Set the semaphore *coil* to locked
            self.p.write_coil(SEMA_COIL, True)
            time.sleep(SLEEP)
            
            
            # Verify lock state + ownership
            if self.p.read_coil(SEMA_COIL) == 1 and self.p.read_register(OWNER_REG) == task_id:
                return True
            
            # Failed (race); back off briefly and retry
            self.c.conditional_write(OWNER_REG, task_id, 0, self.p.read_register(OWNER_REG))
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 0.3)

        return False
    
    
    def give(self, task_id: int):
        """
        Release onyl if we are the owner:
            - set OWNER_REG=0 if OWNER_REG==task_id
            - set SEMA_COIL=0
        """
        if self.p.read_register(OWNER_REG) == task_id:
            self.p.write_register(OWNER_REG, 0)
            time.sleep(SLEEP)
            self.p.write_coil(SEMA_COIL, False)
        else:
            # Note the owner, leave state unchanged
            pass

def critical_section(proto: ProtocolGadget):
    """
    Critical section payload:
     read count -> increment -> write back
     we deliberately split read/increment/write to make interleaving visible
    """
    val = proto.read_register(SHARED_COUNT)
    time.sleep(SLEEP)
    val += 1
    time.sleep(SLEEP)
    proto.write_register(SHARED_COUNT, val)

def worker (name: str, task_id: int, iterations: int, proto: ProtocolGadget, control: ControlGadget):
    sem = BinarySemaphore(proto, control)
    for i in range(iterations):
        acquired = sem.take(task_id, timeout_s=10.0)
        if not acquired:
            print(f"[{name}] failed to acquire semaphore on iter {i}")
            continue
        try:
            print(f"[{name}] acquired > entering critical section (iter {i})")
            critical_section(proto)
            new_val = proto.read_register(SHARED_COUNT)
            print(f"[{name}] leaving critical section, SHARED_COUNT={new_val}")
        finally:
            sem.give(task_id)
            print(f"[{name}] released")
        time.sleep(0.2)

def main():
    proto = ProtocolGadget(HOST, PORT, UNIT)
    ctrl = ControlGadget(proto)

    # Initialize shared state
    proto.write_coil(SEMA_COIL, False)
    proto.write_register(OWNER_REG, 0)
    proto.write_register(SHARED_COUNT, 0)

    # Two contending workers
    t1 = threading.Thread(target=worker, args=("TASK-A", 1,5, proto, ctrl), daemon=True)
    t2 = threading.Thread(target=worker, args=("TASK-B", 2,5, proto, ctrl), daemon=True)

    t1.start(); t2.start()
    t1.join(); t2.join()

    print(f"[MAIN] Done. Final SHARED_COUNT={proto.read_register(SHARED_COUNT)}")

if __name__ == "__main__":
    main()