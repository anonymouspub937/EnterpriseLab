---
layout: default
title: "EnterpriseLab: Full-Stack Platform for Enterprise AI Agents"
---

# EnterpriseLab: A Full-Stack Platform for Developing and Deploying Agents in Enterprises

**Unified infrastructure for training privacy-preserving, cost-effective enterprise AI agents**

🏢 Enterprise AI | 🤖 Agent Training | 🔧 Tool Integration | 💰 Cost Efficiency

**Authors:** Anonymous Authors¹  
**Affiliation:** ¹Anonymous Institution

[📄 Paper (Coming Soon)](#) | [💻 GitHub (Coming Soon)](#) | [🎥 Demo Videos](#) | [📊 Dataset](#)

---

## Abstract

Deploying AI agents in enterprise environments requires balancing capability with data sovereignty and cost constraints. While frontier models like GPT-4o demonstrate strong reasoning abilities, their high inference costs ($3-$15 per million tokens) and data privacy concerns hinder enterprise adoption.

We introduce **EnterpriseLab**, a full-stack platform that unifies tool integration, data generation, and model training into a closed-loop framework. The platform enables enterprises to train small 8B-parameter models that match GPT-4o's performance while reducing inference costs by 8-10×.

### 🎯 Key Contributions

- **Unified Platform:** Closed-loop integration of tool connectivity, trajectory synthesis, model training, and evaluation
- **EnterpriseArena Benchmark:** 15 containerized applications with 140+ tools across IT, HR, sales, and engineering domains
- **Automated Data Synthesis:** Constraint-aware tool graph traversal for executable training data generation
- **Cost-Effective Models:** 8B models matching GPT-4o performance with 8-10× lower inference costs
- **Cross-Benchmark Performance:** +10% improvement on EnterpriseBench and CRMArena

### Key Results at a Glance

| Metric | Value |
|--------|-------|
| Cost Reduction vs. GPT-4o | **8-10×** |
| Enterprise Tools | **140+** |
| Expert-Curated Tasks | **500** |
| Improvement on Benchmarks | **+10%** |

---

## Introduction

### The Enterprise AI Challenge

Enterprise environments require intelligent automation across complex, cross-departmental workflows spanning HR, IT, sales, and engineering. While frontier language models demonstrate strong capabilities, their deployment faces critical constraints:

- **Data Sovereignty:** Regulations require on-premises deployment
- **High Costs:** $3-$15 per million tokens for proprietary APIs
- **API Latency:** Network delays impact user experience
- **Privacy Concerns:** Sensitive data cannot be sent to external services

### The Infrastructure Gap

Small Language Models (SLMs) in the 8B-32B parameter range offer a promising alternative through on-premises deployment and 10× cost reduction. However, effective specialization is hindered by fragmented development pipelines:

**Current Challenges:**
- Tool integration, data collection, and model training are disconnected
- Existing benchmarks measure performance but don't build agents
- Data synthesis operates independently of execution environments
- No unified infrastructure for iterative development

### The EnterpriseLab Solution

EnterpriseLab addresses these challenges by providing a unified platform that integrates:

1. **Modular Tool Environment:** MCP-based architecture for plug-and-play tool integration
2. **Automated Trajectory Synthesis:** Programmatic training data generation from environment schemas
3. **Integrated Training Pipeline:** SFT, DPO, and online RL with continuous evaluation

---

## The EnterpriseLab Platform

![EnterpriseLab Architecture](assets/images/platform-architecture.png)
*Figure 1: EnterpriseLab's three-module architecture for developing enterprise agents*

### 1. Modular Tool Environment Architecture

The environment layer implements a client-server system built on Model Context Protocol (MCP), featuring:

#### Dynamic Tool Registry
- Runtime discovery of available tools from active servers
- Unified action schemas with normalized parameter formats
- Semantic conflict resolution (e.g., mapping `repository` and `project` to standard `workspace_id`)

#### Stateful Execution Containers
- Dedicated Docker instances for each training episode
- Persistent storage across multi-turn trajectories
- Maintained authentication and database states

#### Observation Normalizer
- Captures heterogeneous tool outputs (APIs, CLI, logs)
- Transforms to token-budget JSON format
- Importance-based truncation prioritizing errors and return values

### 2. Task Synthesis Pipeline

Automated generation of high-quality, executable training data through four phases:

#### Phase 1: Tool Graph Construction
Build dependency graph where edges represent data-flow compatibility between tools. Graph ensures any path corresponds to executable sequences.

#### Phase 2: Constraint-Aware Trajectory Sampling
- Depth-first traversal from valid entry nodes (CREATE, LIST/SEARCH tools)
- Local and global memory buffers for argument satisfaction
- Collects K valid trajectories per starting node

#### Phase 3: Hierarchical Task Synthesis
- Generate low-level thoughts for consecutive tool pairs
- Compose into high-level user intents
- Example: *'create repo → add file'* becomes *"Set up a new project"*

#### Phase 4: Validation and Filtering
- De-duplication via exact and fuzzy matching (≥0.9 threshold)
- Diversity-based filtering using Maximal Marginal Relevance
- Grounding validation through environment execution

### 3. Integrated Training Infrastructure

#### Agent Scaffolding
Support for multiple execution strategies:
- **ReAct:** Interleaved reasoning and tool execution for open-weight and proprietary models
- **Function Calling:** Native API-based structured tool schemas for proprietary models
- All executions logged and cached for training

#### Offline Training Methods
- **Supervised Fine-Tuning (SFT):** Cross-entropy loss on expert trajectories with LoRA support
- **Direct Preference Optimization (DPO):** Preference-based alignment from trajectory pairs

#### Agentic GRPO: Online Reinforcement Learning
Group Relative Policy Optimization adapted for agentic settings:
- Trajectories generated via ReAct-style rollouts
- Trajectory-level rewards from environment execution
- Group-relative advantages for stable credit assignment
- Tool output tokens masked during loss computation

**Trajectory Reward Design:**  
Composite reward combining four execution-grounded signals:
- **r₁:** Tool selection accuracy
- **r₂:** Execution success (no runtime errors)
- **r₃:** Final answer correctness
- **r₄:** Format compliance (ReAct structure)

Overall reward: r(τ) = Σ wₖrₖ(τ), normalized to [0,1]

---

## EnterpriseArena: Benchmark Instantiation

EnterpriseArena demonstrates EnterpriseLab's capabilities through a comprehensive benchmark environment with 15 specialized MCP servers and 500 expert-curated tasks.

### MCP Server Ecosystem

| Domain | Applications | Tools |
|--------|-------------|-------|
| 💬 **Communication** | RocketChat, Mail System | 20 tools for messaging and email |
| 💻 **Development** | GitLab MCP | 22 tools for version control and CI/CD |
| 🎫 **Operations & IT** | Zammad, Plane (Jira) | 24 tools for ticketing and project management |
| 👥 **Human Resources** | Frappe HR, Calendar | 20 tools for employee management |
| 💾 **Data & Storage** | Mongoose MCP, OwnCloud | 15 tools for database and file operations |
| 📊 **Business (CRM)** | Dolibarr, Salesforce | 19 tools for customer relationship management |
| 💰 **Finance** | Invoice System | 7 tools for invoicing and payments |
| 🔧 **Utilities** | File System, Bash, Browser | 18 tools for system operations |

### Task Complexity and Categories

| Task Category | Description | % of Tasks |
|---------------|-------------|------------|
| **CRUD Operations** | Create, Read, Update, Delete tasks across systems | 35% |
| **Search & Orchestration** | Multi-system information retrieval and coordination | 28% |
| **Multi-entity Workflow** | Complex tasks involving multiple data entities | 18% |
| **Version Control** | Code management and development operations | 12% |
| **Cross-functional Integration** | Tasks spanning multiple departments | 7% |

### Example Complex Task

**Cross-Functional Recruitment Workflow**

**Task:** "Read the 2026 Software Engineer job description, fetch relevant resumes, identify the top three candidates based on required skills, and coordinate interview scheduling with engineering managers via email."

**Required Orchestration:**
- OwnCloud: Document retrieval
- Frappe HR: Resume database access
- Custom Logic: Skills-based candidate ranking
- Mail System: Coordinated email communication

**Complexity:** 6-8 tool invocations across 3 systems with stateful reasoning

### Stateful Environment Dependencies

Unlike static benchmarks, EnterpriseArena maintains a unified backend where data changes propagate automatically:
- Creating HR employee records updates central registry
- Updates enable subsequent CRM assignments
- Notification dispatches occur without external intervention
- API-level validation enforces enterprise constraints

### Expert Validation

Tasks developed through structured reviews with 9 domain experts across Software Engineering, Business Development, Sales, IT Security, HR, and Finance. All tasks rated "Realistic" or above on five-point Likert scale.

---

## Results and Analysis

### Performance Across Benchmarks

We evaluate Qwen3-8B models trained with EnterpriseLab across four environments: EnterpriseArena (ours), EnterpriseBench, CRMArena, and τ-Bench.

| Model | EA | EB | CRM | τ-B |
|-------|----|----|-----|-----|
| **Closed-Source Models** | | | | |
| GPT-4o (2-shot) | 0.45 | 0.47 | 0.32 | 0.54 |
| Claude-3.5-Sonnet (2-shot) | 0.60 | 0.55 | 0.34 | 0.56 |
| Gemini-2.5-Pro (2-shot) | **0.71** | 0.55 | **0.49** | **0.59** |
| **Open-Source Models** | | | | |
| Qwen3-8B Base (2-shot) | 0.31 | 0.35 | 0.25 | 0.33 |
| ToolACE (26K-trained) | 0.39 | 0.41 | 0.10 | 0.15 |
| xLAM-2-70B (60K-trained) | 0.15 | 0.40 | 0.12 | 0.17 |
| **Our Platform-Trained Models (<1K examples)** | | | | |
| Qwen3-8B SFT | 0.35 | 0.38 | 0.30 | 0.36 |
| **Qwen3-8B Agentic GRPO** | **0.43** | **0.51** | **0.35** | **0.42** |

### Key Performance Insights

| Metric | Value |
|--------|-------|
| Improvement over Base Model | **30%** |
| Performance Parity | **≈GPT-4o** |
| Advantage Over GPT-4o on EnterpriseBench | **+10%** |
| Training Data Reduction vs. Baselines | **26-60×** |

### Tool Selection Accuracy

| Model | EA | EB |
|-------|----|----|
| GPT-4o (2-shot) | 0.31 | 0.21 |
| Qwen3-8B Base (2-shot) | 0.14 | 0.14 |
| **Qwen3-8B Agentic GRPO** | **0.28** | **0.21** |

### Cost Efficiency Analysis

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| GPT-4o | $5.00 | $15.00 |
| Claude-3.5-Sonnet | $3.00 | $15.00 |
| Gemini-2.5-Pro | $1.25 | $10.00 |
| **Qwen3-8B Agentic GRPO (Self-hosted)** | **$0.50–$1.00** | (combined) |

**Result:** 8-10× cost reduction while achieving competitive performance makes EnterpriseLab-trained models ideal for cost-sensitive, large-scale deployments.

### Impact of Trajectory-Level Optimization

Comparing optimization strategies on EnterpriseBench:
- **Agentic GRPO:** ~10% improvement over token-level GRPO
- **Agentic GRPO vs. DPO:** ~15% improvement in execution accuracy
- **Tool Selection:** ~10% improvement over both baselines

*Trajectory-level optimization is critical for multi-turn agentic tasks, validating EnterpriseLab's design for complete trajectory collection and training.*

### Synthetic Data Quality Analysis

Analysis of 1,500 synthetic trajectories for EnterpriseBench:

**Diversity**
- Self-BLEU score: 0.4 (moderate diversity)
- 70 unique APIs across 5 enterprise domains
- Balanced distribution: Software Engineering (34.4%), CRM (25.3%), HR (20.8%), Operations (16.0%), IT (3.5%)

**Complexity**
- Average 3.2 turns per dialog (σ = 1.29)
- 68.1% require multi-turn reasoning
- 54.7% involve multi-tool composition with dependency chains

**Correctness**
- Rule-based validation: 100% pass rate (schema compliance)
- GPT-4 semantic evaluation: 81.9% pass rate (200-sample stratified)

### Adaptation to Environment Changes

Testing robustness with 30% tool modifications on EnterpriseBench:

| Scenario | LLM Eval | Tool Eval |
|----------|----------|-----------|
| Original environment | 0.50 | 0.20 |
| Modified environment (30% changes) | 0.43 (-15%) | 0.15 |
| **+ 200 incremental training samples** | **0.48 (95% recovery)** | **0.18** |

**Insight:** EnterpriseLab supports rapid model adaptation to evolving environments with minimal additional data, without full retraining.

### Training Efficiency

**Time to Production**
- **SFT/DPO:** 30 minutes to 2 hours on 2×A100 GPUs
- **Agentic GRPO:** 24-30 hours on 4×H200 GPUs
- **Total:** Production-ready models from raw tool schemas in under 2 days

### Error Analysis

Analysis of 50 failure cases reveals systematic patterns:

| Failure Mode | Frequency | Description |
|--------------|-----------|-------------|
| Tool Parameter Errors | 42% | Incorrect arguments causing API failures; limited error recovery |
| Domain Misselection | 28% | Ambiguous cues lead to wrong tool selection and recursion loops |
| Task Decomposition | 18% | Completing initial sub-task but failing to plan subsequent steps |
| Context Loss | 12% | Loss of coherence in longer interactions |

---

## Comparison with Existing Benchmarks

| Benchmark | Domain Focus | Multi-App Flow | Dynamic Data | Training Platform |
|-----------|--------------|----------------|--------------|-------------------|
| AgentBench | General Reasoning | ✗ | ✗ | ✗ |
| WebArena | Web UI | ✗ | ✓ | ✗ |
| SWE-bench | Software Eng. | ✗ | ✗ | ✗ |
| CRMArena | CRM | ✗ | ✗ | ✗ |
| EnterpriseBench | General Enterprise | ✓ | ✗ | ✗ |
| τ-Bench | Customer Service | ✗ | ✓ | ✗ |
| **EnterpriseLab + Arena** | **Cross-Functional Enterprise** | **✓** | **✓** | **✓** |

---

## Citation

```bibtex
@article{enterpriselab2026,
  title={EnterpriseLab: A Full-Stack Platform for Developing and Deploying Agents in Enterprises},
  author={Anonymous},
  year={2026}
}
```

---

*© 2026 Anonymous Institution. All rights reserved.*
