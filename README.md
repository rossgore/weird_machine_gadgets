# Weird Machine Gadgets

Research toolkit and teaching materials for discovering, composing, and studying **weird machines** — emergent computational behaviors that arise from unintended interactions between system components, protocols, and physical constraints, rather than from deliberately written code.

This repository supports an ongoing research effort into **architectural weird machines**: gadgets built not from code-level primitives (like ROP gadgets), but from the interactions of communication protocols, hardware timing, and cross-boundary control paths in cyber-physical systems. Our primary testbed is a wind energy control system built on Modbus, but the same gadget-composition ideas generalize to networking, robotics, and other protocol-driven systems.

## What Is a Weird Machine?

A weird machine is a computational artifact where execution occurs outside a system's original specification, triggered by unexpected input or unintended composition of legitimate operations. Classic examples include ROP-chain exploitation, but our work extends the concept to **architectural weird machines**, where the "gadgets" are protocol operations, timing side effects, or cross-system control paths instead of instructions.

Documented examples we've studied that motivated this project:
- **Arista EOS logging**: local device logging becomes contingent on TCP reachability of a remote syslog server, an emergent control path never specified by the intended design.
- **Robolink Zumi cars**: sensor input drives the car into unintended looping states not present in the original program logic.
- **Tekken 8 moveset swapping**: conditional/timing interactions unlock gender-locked movesets, exposing hidden state as an exploitable variable.

## Project Structure

| Directory / File | Purpose |
|---|---|
| `simple.py` | Core Modbus gadget API (Protocol Processing Gadgets, Conditional Logic Gadgets) used across examples. |
| `example/examples_with_explicit_modbus_usage/` | Worked examples showing gadgets built directly on raw Modbus function codes (FC01–FC16). |
| `codeword_messages.py` | Composite gadget implementing the Remote Mailbox / codeword-signaling weird machine (segmented and unsegmented message passing via registers). |
| `weird-gadget-llm-finetuning/` | Fine-tuning pipeline (FLAN-T5-small + DistilGPT2 ensemble) for automatically classifying weird machine gadget types from text; first step toward the LLM consortium approach. |
| `compose_operations` module (`compose_operations_starter.py`) | Data-flow pipeline mechanism for chaining gadgets, modeling how PLC scan-cycle composition mirrors weird machine gadget chaining. |
| `blog/` | Student write-ups of weird machines discovered in coursework and in the wind energy testbed. |
| `reading_lists/` | Curated academic reading list on weird machines, exploitability, and the theory of computation behind this work. |
| `tls-weird-machine/` | Demo exploring weird-machine-style behavior in TLS handshake/session state. |

## Gadget Stack (Wind Energy Testbed)

Our testbed uses Modbus function codes as an "instruction set" and builds three layers of gadgets on top:

1. **PPG (Protocol Processing Gadgets)** — register/coil read+write and increment/decrement primitives, directly on top of FC01/FC03/FC05/FC06.
2. **CLG (Conditional Logic Gadgets)** — emergent control-flow behavior (if-statements, while-loops, do-while loops) built purely from protocol interactions.
3. **Composite Gadgets** — higher-level constructs such as state machines and conditional executors, composed from PPGs and CLGs.

Weird machines composed from this stack so far include a **Binary Semaphore** (mutual exclusion from protocol gadgets alone), a **Code Signaling** machine (vulnerability state signaled via coil/register flags), and a **Remote Mailbox** (message passing through server registers, segmented and unsegmented).

We argue, by construction and analogy, that this gadget stack is Turing complete (sequence, selection, and iteration are all achievable), though a formal proof is out of scope for this project.

## Suggested Learning Path

For anyone new to the project (including interns), the intended order is:

1. **Reading list** (`reading_lists/`) — foundational papers on weird machines and exploitability theory.
2. **`simple.py` API** — understand the base gadget primitives before looking at examples.
3. **`example/examples_with_explicit_modbus_usage/`** — see the primitives applied directly to raw Modbus operations.
4. **`compose_operations` starter** — learn the data-flow composition mechanism that chains gadgets together.
5. **`codeword_messages.py`** — a full composite gadget (Remote Mailbox) as a worked example of composition in practice.
6. **`blog/`** — student write-ups showing how others approached discovery and documentation of new weird machines.
7. **`weird-gadget-llm-finetuning/`** — the current frontier: using fine-tuned LLMs to help identify gadgets in system documentation at scale.
8. **`tls-weird-machine/`** — a second protocol domain (TLS) to test whether the same gadget-composition mindset generalizes beyond Modbus.

## Research Directions (Active / Upcoming)

- **White paper and research artifacts**: formal documentation of methodology and findings, using Overleaf/Git/LaTeX workflows; publication pending sponsor review.
- **LLM consortium for gadget discovery**: multiple fine-tuned LLMs independently propose candidate gadgets from system documentation, with a dedicated reasoning model synthesizing and validating results to reduce hallucination and bias.
- **Cross-domain generalization**: testing whether the PPG/CLG/Composite gadget stack transfers cleanly from Modbus/wind energy to other protocol domains (TLS, SELinux/kernel audit paths, etc.).

## Notes on Repository Organization

This repo has grown organically as the project has taken on new directions rather than following a single planned structure from the start. Expect some directories to reflect earlier phases of the project and others to be actively maintained; when in doubt, check the README or summary file inside each subdirectory for the most current status.
