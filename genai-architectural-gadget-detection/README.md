# White Paper: Open-Source LLM Consortium for Architectural Gadget Detection in Cyber-Physical Systems

**Authors:** [Ross Gore, Jada Cumberland, Samuel Jackson, Brianne Dunn, Sachin Shetty, Old Dominion University]  


---

## Executive Summary

Weird machines—unintended computational capabilities emergent within cyber-physical system (CPS) architectures represent a critical security challenge  and an oppurtunity for industrial control systems (ICS), building automation, and critical infrastructure. Traditional detection methods rely on manual expert analysis that is slow, expensive, and non-scalable.

This white paper presents a novel **open-source LLM consortium architecture** that helps draft architectural gadget detection through:

- **Three specialized open-source LLMs**: Llama 3.1 (70B), Qwen2.5 (72B), and Pixtral-12B for multimodal document analysis
- **GPT-o1 reasoner**: Open-source aggregation and consensus framework for explainable, validated results
- **Local deployment**: Fully air-gapped operation on NIST 800-171 and CMMC Level 2 compliant infrastructure
- **High-performance computing**: 12 Nvidia A100 GPUs and 240 CPU cores enable efficient analysis at scale


---

## 1. Introduction

### 1.1 Background: The Weird Machine Problem

Weird machines arise when the architectural components of cyber-physical systems—Modbus registers, PLC timers, protocol bridges, memory buffers—can be composed to create unintended computational capabilities. These latent computers enable:

- **Timer-based exploits**: Countdown mechanisms trigger delayed actions
- **State machines**: Protocol operations encode complex logic flows
- **Memory manipulation**: Register spaces serve as general-purpose storage
- **Distributed computation**: Multi-device coordination implements algorithms

Detecting these capabilities requires understanding protocol semantics, memory organization, control logic, and timing—expertise typically restricted to a small community of specialists.

### 1.2 The Open-Source LLM Opportunity

Recent advances in open-source large language models enable distillation of expert knowledge into deployable, scalable analysis systems. However, individual models suffer from:
- **Hallucination**: Generating plausible but incorrect analyses
- **Blind spots**: Missing gadget types based on training biases
- **Inconsistency**: Variable performance across documentation styles

A consortium approach addresses these limitations through diverse model perspectives, cross-validation, and reasoned consensus.

### 1.3 Problem Statement

**Challenge:** Enable accurate, and explainable identification of architectural gadgets in CPS documentation without dependence on proprietary cloud services.

**Requirements:**
- Air-gapped operation for classified/sensitive systems
- NIST 800-171 and CMMC Level 2 compliance
- Transparency and auditability of analysis
- Scalability to hundreds of vendor systems
- Integration with existing security workflows

---

## 2. Consortium Architecture

### 2.1 Component LLMs

The consortium employs three specialized open-source models, each providing complementary strengths:

#### **Llama 3.1 (70B) - Text Analysis Specialist**
- **Provider:** Meta AI (Apache 2.0 License)
- **Architecture:** Transformer decoder, 70 billion parameters
- **Specialization:** Deep semantic understanding of technical documentation, protocol specifications, and control logic
- **Fine-tuning:** Adapted on CPS documentation corpus with architectural gadget annotations
- **Deployment:** 4x A100 GPUs (40GB VRAM each) using tensor parallelism
- **Inference:** vLLM with quantization (FP8) for 2.5x throughput improvement

**Strengths:**
- Excellent at parsing dense technical specifications
- Strong reasoning about control flow and state machines
- Robust handling of acronyms and domain terminology

#### **Qwen2.5 (72B) - Protocol and Memory Expert**
- **Provider:** Alibaba Cloud (Apache 2.0 License)
- **Architecture:** Transformer with extended context window (32K tokens)
- **Specialization:** Protocol analysis, register mapping, memory organization
- **Fine-tuning:** Enhanced with Modbus, BACnet, OPC UA specifications and memory layout examples
- **Deployment:** 4x A100 GPUs with FlashAttention-2 for extended context
- **Inference:** Optimized for structured outputs (JSON schema enforcement)

**Strengths:**
- Superior performance on register/coil address interpretation
- Excellent pattern matching for protocol function codes
- Strong numerical reasoning for memory calculations

#### **Pixtral-12B - Multimodal Document Analyst**
- **Provider:** Mistral AI (Apache 2.0 License)
- **Architecture:** Vision-language model (12B parameters)
- **Specialization:** Extracting information from diagrams, schematics, register maps, timing diagrams
- **Fine-tuning:** Trained on annotated ladder logic, network diagrams, and memory maps
- **Deployment:** 2x A100 GPUs with vision encoder optimization
- **Inference:** Native multimodal input processing

**Strengths:**
- Analyzes visual documentation (PDF diagrams, schematics)
- Interprets ladder logic and function block diagrams
- Extracts structured data from tables and register maps

### 2.2 GPT-o1 Reasoner Framework

The **GPT-o1 (Open Source)** reasoner serves as the consortium's aggregation and consensus layer:

- **Function:** Collect outputs from all three LLMs, identify agreements/disagreements, apply logical reasoning to resolve conflicts, generate unified analysis
- **Architecture:** Lightweight orchestration layer (Python-based) with structured reasoning chains
- **Implementation:** 
  - Chain-of-thought prompting for explicit reasoning steps
  - Majority voting for gadget identification
  - Weighted confidence scoring
  - Explanation generation for decisions
- **Deployment:** 2x A100 GPUs (shared with coordination tasks)

**Reasoning Process:**
1. **Parallel Inference**: Submit documentation to all three models simultaneously
2. **Output Collection**: Gather gadget identifications, risk assessments, and exploitation patterns
3. **Consensus Building**:
   - If 3/3 agree → High confidence result
   - If 2/3 agree → Majority consensus with dissenting view noted
   - If 1/1/1 split → Reasoner applies domain rules or flags for human review
4. **Explanation Generation**: Document decision rationale, cite model sources, highlight uncertainty
5. **Report Assembly**: Unified analysis with model-specific insights preserved

---

## 3. Infrastructure and Deployment

### 3.1 Computational Environment

Our ODU virtual environment provides performance and security for this project. It contains:

**Hardware Specifications:**
- **GPUs:** 12x Nvidia A100 (80GB HBM2e each) = 960GB total GPU memory
- **CPUs:** 240 computation cores (AMD EPYC or Intel Xeon Scalable)
- **Memory:** 2TB+ system RAM
- **Storage:** High-speed NVMe SSD arrays for model weights and datasets
- **Network:** 100Gbps internal fabric, segmented VLANs

**Model Allocation:**
- Llama 3.1 (70B): 4x A100 GPUs
- Qwen2.5 (72B): 4x A100 GPUs  
- Pixtral-12B: 2x A100 GPUs
- GPT-o1 Reasoner: 2x A100 GPUs (shared with coordination)
- Reserve capacity: Available for fine-tuning, batch processing, failover

**Performance:**
- **Concurrent analysis**: 10-15 system documents simultaneously
- **Throughput**: 100-150 documents per day (comprehensive analysis)
- **Latency**: 30-90 seconds per document (depending on complexity)
- **Batch processing**: 500+ documents overnight for procurement reviews

### 3.2 Security and Compliance

**NIST 800-171 Compliance:**
- Access controls with multi-factor authentication
- Audit logging of all model inferences and user interactions
- Encryption at rest (AES-256) for datasets and model weights
- Encryption in transit (TLS 1.3) for internal API communications
- Regular vulnerability scanning and patch management

**CMMC Level 2 Compliance:**
- Asset management and configuration control
- Incident response procedures
- Personnel security and training requirements
- Physical security controls for compute infrastructure
- Supply chain risk management for open-source dependencies

**Air-Gapped Deployment:**
- All models run locally—no external API calls
- Documentation never leaves secure environment
- Updates via controlled, audited process
- Offline operation for classified facilities

**Network Segmentation:**
- Isolated VLAN for LLM inference cluster
- Separate VLAN for training and fine-tuning operations
- DMZ for user-facing interfaces (if external access required)
- Zero-trust architecture with least-privilege access

---

## 4. Data Collection and Documentation Sources

### 4.1 Open-Source Documentation

**Industrial Control Systems:**
- OpenPLC Project: Open-source PLC runtime and documentation
- CODESYS: IEC 61131-3 programming system (community edition)
- UniPi: Raspberry Pi-based industrial automation platform
- Industrial Shields: Arduino-based PLC documentation
- Beckhoff TwinCAT (trial/educational documentation)

**Protocol Specifications:**
- Modbus.org: Complete Modbus RTU/TCP specification (free)
- BACnet International: BACnet protocol standard (available for purchase, redistribution allowed)
- OPC Foundation: OPC UA specification (parts freely available)
- MQTT.org: MQTT protocol specification (open standard)
- DNP3.org: DNP3 specification (available with membership)
- IEC 61850: Smart grid communication (summary documents available)

**Academic and Government Sources:**
- NIST Cybersecurity Framework documentation
- DOE energy systems testbed descriptions
- DHS CISA ICS advisories and technical details
- Published academic papers on ICS testbeds
- University research lab documentation (e.g., SCADA testbeds)

**Vendor Resources:**
- Freely available product datasheets
- Application notes and white papers
- Sample configuration files and templates
- Training materials and webinar recordings
- GitHub repositories with device configurations

### 4.2 Document Types Required

**Essential for Training (50+ examples minimum):**

1. **Protocol Specifications** (15-20 examples)
   - Register/coil address maps
   - Function code definitions
   - Message format specifications
   - Timing and sequencing requirements

2. **Controller/PLC Manuals** (15-20 examples)
   - Memory organization diagrams
   - Instruction set descriptions
   - Timer and counter specifications
   - I/O mapping tables
   - Scan cycle documentation

3. **System Integration Documents** (10-15 examples)
   - Network architecture diagrams
   - Protocol bridge configurations
   - Data flow specifications
   - Device interconnection maps
   - Timing and synchronization requirements

4. **Configuration Files** (5-10 examples)
   - Ladder logic programs
   - Function block diagrams
   - SCADA screen configurations
   - Gateway mapping files (XML/YAML)
   - Register allocation tables

### 4.3 Annotation Requirements

Each training example includes:

**Input Component:**
- System description (200-1000 words)
- Technical specifications
- Architecture diagrams (if available)
- Sample configurations

**Expert Annotation (Output):**
- **Protocol Processing Gadgets (PPG):** Identified registers, coils, function codes
- **Control Logic Gadgets (CLG):** Timers, counters, conditionals, state machines
- **Communication Bridge Gadgets (CBG):** Protocol translations, data transformations
- **Security Processing Gadgets (SPG):** Authentication, encryption capabilities
- **Motor/Actuator Control Gadgets (MACG):** Physical control outputs
- **Risk Assessment:** Turing completeness evaluation, exploitation patterns
- **Code Examples:** Concrete proof-of-concept implementations

**Quality Control:**
- Minimum 2 expert reviewers per annotation
- Validation against known exploitation patterns
- Cross-reference with published vulnerabilities
- Regular dataset audits for consistency

---

## 5. Fine-Tuning Process

### 5.1 Training Pipeline

**Phase 1: Base Model Selection and Setup (Week 1)**
- Download and validate Llama 3.1 (70B), Qwen2.5 (72B), Pixtral-12B
- Configure tensor parallelism and distributed training
- Set up monitoring and logging infrastructure

**Phase 2: Dataset Preparation (Weeks 2-3)**
- Collect 50-100 annotated examples
- Split: 70% training, 15% validation, 15% test
- Convert to instruction-following format (Alpaca, ShareGPT)
- Preprocess multimodal components for Pixtral

**Phase 3: Fine-Tuning (Week 4)**
- Use Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- Training configuration:
  - Learning rate: 1e-5 to 5e-5
  - Batch size: 4-8 (per GPU with gradient accumulation)
  - Epochs: 3-5
  - Optimizer: AdamW with cosine annealing
- Monitor validation loss and gadget identification accuracy

**Phase 4: Evaluation (Week 5)**
- Test on held-out examples
- Measure per-gadget-type precision/recall
- Evaluate risk assessment agreement with experts
- Benchmark against baseline (non-fine-tuned) models

**Phase 5: Consortium Integration (Week 6)**
- Deploy all three models with unified API
- Implement GPT-o1 reasoner orchestration
- Conduct end-to-end system testing
- Document performance and failure modes

### 5.2 Computational Requirements

**Training Resources:**
- 8x A100 GPUs for fine-tuning (4 reserved for active inference)
- 100-150 GPU-hours per model (LoRA fine-tuning)
- Total training time: 3-5 days wall-clock
- Storage: 500GB for checkpoints and logs

**Inference Optimization:**
- FP8 quantization: 2-3x faster inference
- FlashAttention-2: 50% memory reduction
- vLLM continuous batching: 4x throughput improvement
- Speculative decoding: 20-30% latency reduction

---

## 6. Consortium Workflow and Decision Logic

### 6.1 Document Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Submits Documentation                    │
│                  (PDF, Markdown, XML, Diagrams)                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Document Preprocessing                       │
│  • Extract text from PDFs                                        │
│  • Parse structured data (XML, YAML)                             │
│  • Extract diagrams and tables (for Pixtral)                     │
│  • Normalize formatting                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│   Llama 3.1 (70B)        │     │   Qwen2.5 (72B)          │
│   Text Analysis          │     │   Protocol Analysis       │
│   • Control logic        │     │   • Register mapping      │
│   • State machines       │     │   • Memory organization   │
│   • Timing analysis      │     │   • Function codes        │
└─────────────┬─────────────┘     └─────────────┬─────────────┘
              │                                 │
              │                                 │
              │          ┌──────────────────────┘
              │          │
              │          ▼
              │   ┌───────────────────────────┐
              │   │   Pixtral-12B            │
              │   │   Multimodal Analysis    │
              │   │   • Diagram extraction   │
              │   │   • Ladder logic parsing │
              │   │   • Visual register maps │
              └───┤   └───────────┬───────────┘
                  │               │
                  └───────┬───────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPT-o1 Reasoner Aggregation                    │
│  1. Collect all three model outputs                             │
│  2. Identify agreements (high confidence)                        │
│  3. Flag disagreements for reasoning                             │
│  4. Apply domain logic and voting                                │
│  5. Generate unified explanation                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Unified Report                         │
│  • Identified gadgets (PPG, CLG, CBG, SPG, MACG)                │
│  • Risk assessment and Turing completeness                       │
│  • Exploitation patterns with code examples                      │
│  • Model-specific insights and confidence levels                 │
│  • Dissenting opinions (if any)                                  │
│  • Defensive recommendations                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Consensus and Conflict Resolution

**Scenario 1: Full Agreement (3/3)**
```
Llama:   "PPG: Modbus registers 40001-40010, CLG: Timer T1"
Qwen:    "PPG: Modbus registers 40001-40010, CLG: Timer T1"
Pixtral: "PPG: Modbus registers 40001-40010, CLG: Timer T1"
→ Reasoner: HIGH CONFIDENCE - All models agree
→ Output: Direct inclusion in final report
```

**Scenario 2: Majority Consensus (2/3)**
```
Llama:   "Risk: HIGH - Sufficient memory + timers = Turing complete"
Qwen:    "Risk: HIGH - 100 registers enable complex programs"
Pixtral: "Risk: MODERATE - Limited control flow"
→ Reasoner: MODERATE-HIGH CONFIDENCE - Majority consensus
→ Output: "Risk assessed as HIGH (2/3 models), with one dissent noting limited branching"
```

**Scenario 3: Split Decision (1/1/1)**
```
Llama:   "CBG: Protocol bridge with data transformation"
Qwen:    "PPG: Simple register pass-through"
Pixtral: "CLG: Validation logic with conditionals"
→ Reasoner: LOW CONFIDENCE - No consensus
→ Output: "Conflicting interpretations detected:
           - Model 1: Bridge gadget (transformation capability)
           - Model 2: Simple forwarding (no computation)
           - Model 3: Conditional logic (validation checks)
           Recommendation: Manual expert review required"
```

**Scenario 4: Missing Detection**
```
Llama:   [No mention of counter C1]
Qwen:    "CLG: Counter C1 (accumulator)"
Pixtral: "CLG: Counter C1 (max value 100)"
→ Reasoner: MODERATE CONFIDENCE - 2/3 detected
→ Output: "Counter C1 identified by 2/3 models - included with note about detection inconsistency"
```

### 6.3 Explainability Features

**Model Attribution:**
- Each finding tagged with contributing model(s)
- Confidence scores per model for each gadget
- Direct quotes from model reasoning

**Evidence Linking:**
- Specific document sections cited
- Line numbers and page references
- Visual elements (diagrams) referenced
- Protocol specification cross-references

**Reasoning Chains:**
- Step-by-step logic for risk assessments
- "If register X + timer Y + conditional Z, then Turing-complete because..."
- Analogies to known exploited systems


---

## 7. References and Resources

**Open-Source Models:**
- Llama 3.1: https://ai.meta.com/llama/
- Qwen2.5: https://github.com/QwenLM/Qwen2.5
- Pixtral: https://mistral.ai/news/pixtral-12b/
- GPT-o1: https://github.com/[reasoner-framework]

**Protocol Specifications:**
- Modbus: https://modbus.org/specs.php
- BACnet: https://bacnet.org/
- OPC UA: https://opcfoundation.org/
- MQTT: https://mqtt.org/

**Compliance Frameworks:**
- NIST 800-171: https://csrc.nist.gov/publications/detail/sp/800-171/rev-2/final
- CMMC: https://www.acq.osd.mil/cmmc/

**Training Resources:**
- Fine-tuning guide: https://huggingface.co/docs/transformers/training
- vLLM optimization: https://docs.vllm.ai/
- PEFT/LoRA: https://github.com/huggingface/peft

---

