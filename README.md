# AI and Machine Learning Engineer — Complete Program

> From fundamentals to production: Your structured path to becoming a professional AI/ML Engineer

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/nanlabs/nan-ai-engineering-labs/actions/workflows/validate-all-modules.yml/badge.svg)](https://github.com/nanlabs/nan-ai-engineering-labs/actions/workflows/validate-all-modules.yml)
[![Modules](https://img.shields.io/badge/Modules-12%20Core%20%2B%206%20Advanced-blue)](<>)

---

## What will you learn?

This program takes you from zero AI knowledge to implementing production systems, covering the entire stack required to work as a professional AI/ML Engineer:

### Foundations (Modules 1-4)

Build solid roots in programming, mathematics, and classical ML:

- Scientific Python (NumPy, Pandas, Matplotlib)
- Linear Algebra, Calculus, and Statistics applied to ML
- ML Algorithms: Regression, Classification, Clustering, Ensemble methods
- Deep Learning: Neural Networks, Back-propagation, CNNs, RNNs

**Upon completion:** You will be able to train classical ML models and neural networks, understand technical papers, and diagnose over-fitting/under-fitting issues.

---

### Specialization (Modules 5-8)

Master the 4 critical areas of modern AI:

- **NLP and LLMs**: Fine-tuning, prompt engineering, RAG, embeddings
- **Computer Vision**: CNNs, transfer learning, object detection, segmentation
- **Time Series**: Forecasting with ARIMA/Prophet/LSTMs, anomaly detection
- **Recommender Systems**: Collaborative filtering, matrix factorization, hybrid systems

**Upon completion:** You will be able to build intelligent chatbots, classify medical images, predict future sales, and recommend personalized products.

---

### Generative AI (Module 9)

Create cutting-edge generative systems:

- GANs for generating realistic images
- VAEs and Diffusion Models (Stable Diffusion)
- Advanced Prompt Engineering for LLMs
- RAG (Retrieval-Augmented Generation) for enterprise applications

**Upon completion:** You will be able to build applications such as image generators, chatbots with memory, and Q&A systems based on proprietary documents.

---

### Ethics and Governance (Modules 10-11)

Develop professional judgment regarding ethics and security:

- Detection and mitigation of bias in models
- Explainability with SHAP, LIME, attention visualization
- Privacy: Anonymization, differential privacy, federated learning
- Security: Encryption, hashing, sensitive data tokenization

**Upon completion:** You will be able to audit models for bias, explain decisions to non-technical stakeholders, and comply with regulations like GDPR/HIPAA.

---

### Production and MLOps (Module 12)

Deploy models to production robustly:

- Deployment with FastAPI, Docker, Kubernetes
- CI/CD for ML with GitHub Actions, MLflow
- Monitoring: drift detection, latency tracking, cost optimization
- A/B testing and controlled experimentation

**Upon completion:** You will be able to deploy scalable models on AWS/GCP, configure automated pipelines, and monitor systems in production.

---

### Cutting-Edge Trends (6 Advanced Modules)

Additionally, you will master the most recent industry trends:

| Topic                | What you will learn                                    | What it is for                                                     |
|----------------------|--------------------------------------------------------|--------------------------------------------------------------------|
| **AI Agents**        | ReAct, tool use, multi-agent systems                   | Build autonomous assistants that execute complex tasks              |
| **LLM Guardrails**   | Input validation, output filtering, toxicity detection | Protect LLM applications from prompt injection and harmful content  |
| **Multimodal AI**    | CLIP, BLIP, Stable Diffusion                           | Combine text and images (visual search, VQA, generation)           |
| **LLM Evaluation**   | BLEU, ROUGE, BERTScore, benchmarks                     | Measure response quality and optimize prompts systematically       |
| **AI Observability** | Logging, tracing, cost tracking, dashboards            | Monitor LLMs in production (latency, costs, errors)                |
| **Synthetic Data**   | CTGAN, text augmentation, differential privacy         | Generate synthetic data for training while maintaining privacy      |

---

## Learning Roadmap

```text
PHASE 1: Foundations (Modules 1-4)
|-- Module 01: Python + Math for ML
|-- Module 02: Data Wrangling & Visualization
|-- Module 03: ML Fundamentals
+-- Module 04: Deep Learning Basics
    +-- Checkpoint: Implement image classifier with CNN

PHASE 2: Specialization (Modules 5-8)
|-- Module 05: NLP & Large Language Models
|-- Module 06: Computer Vision
|-- Module 07: Time Series & Anomaly Detection
+-- Module 08: Recommender Systems
    +-- Checkpoint: Fine-tune LLM and deploy chatbot

PHASE 3: Production (Modules 9-12)
|-- Module 09: Generative AI & Prompt Engineering
|-- Module 10: Ethics, Bias & Explainability
|-- Module 11: Data Privacy & Security
+-- Module 12: MLOps & AI in Production
    +-- Checkpoint: Deploy complete system with CI/CD

PHASE 4: Advanced (Trends)
|-- Trends: Agents, Guardrails, Multimodal
|-- Trends: Evaluation, Observability, Synthetic Data
+-- Final Project: Production system with complete monitoring
```

**Total Estimated Time:** 8-12 months (at your own pace)

---

## Program Structure

Each module follows a proven pedagogical structure:

```text
modules/0X-module-name/
|-- README.md          -> Overview, objectives, use cases
|-- STATUS.md          -> Your personal progress
|-- theory/            -> Theoretical material (concepts, papers, explanations)
|-- examples/          -> Functional code demonstrating concepts
|-- practices/         -> Guided exercises with solutions
|-- mini-project/      -> Module integration project
+-- evaluation/        -> Self-assessment and checklist
```

**Total Materials:**

- 12 Core Modules (foundations to production)
- 6 Advanced Modules (current trends)
- 60+ Executable Python code examples
- 40+ Guided practices with solutions
- 12 Integration mini-projects
- ~28,000 lines of code and documentation

---

## Quick Start

### 1. Review the complete program

- [**MODULES-SUMMARY.md**](docs/MODULES-SUMMARY.md) - What you will learn in each module
- [**LEARNING-PATH.md**](docs/LEARNING-PATH.md) - Complete study route by phases
- [**IMPLEMENTATION-STATUS.md**](docs/IMPLEMENTATION-STATUS.md) - Project status
- [**QUICK-START.md**](docs/QUICK-START.md) - Start in 10 minutes (setup + first example)

### 2. Setup your environment

```bash
git clone https://github.com/nanlabs/nan-ai-engineering-labs.git
cd nan-ai-engineering-labs

# create virtual env
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install deps
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Start Module 1

```bash
cd modules/01-programming-math-for-ml
cat README.md
```

**Study flow within each module:**

```text
theory/ -> examples/ -> practices/ -> mini-project/ -> evaluation/
```

### 4. Track your progress

- Mark your progress in STATUS.md for each module.
- Complete the self-assessment checklists.
- Document blockers and learnings in notes/.

---

## Complete Documentation

### Main Guides

| Document                                          | Description                        | When to read it    |
|---------------------------------------------------|------------------------------------|--------------------|
| [**README.md**](README.md)                        | This file - central hub            | Entry point        |
| [**LEARNING-PATH.md**](docs/LEARNING-PATH.md)     | Complete phase-by-phase study path | Before starting    |
| [**MODULE-SUMMARY.md**](docs/MODULE-SUMMARY.md)   | Module-by-module learning goals    | Initial planning   |
| [**STUDY-RHYTHM.md**](docs/STUDY-RHYTHM.md)       | Suggested adaptable pace           | If unsure about commitment |
| [**RESOURCES.md**](docs/RESOURCES.md)             | Recommended external resources     | During study       |

### Technical Guides

| Document                                                      | Description              | When to use it             |
|---------------------------------------------------------------|--------------------------|----------------------------|
| [**IMPLEMENTATION-STATUS.md**](docs/IMPLEMENTATION-STATUS.md) | Full project status      | Verify available content   |
| [**PHASES-CREATION.md**](docs/PHASES-CREATION.md)             | Repo construction phases | For maintainers only       |
| [**templates/**](templates/)                                  | Reusable templates       | When creating your own projects |

### Key Folders

```text
nan-ai-engineering-labs/
|-- modules/          -> 12 Core Modules (your main focus)
|-- trends-extras/    -> 6 Advanced Modules (post-core)
|-- docs/             -> Centralized documentation
|-- shared/           -> Datasets, utils, shared notebooks
+-- templates/        -> Templates for practices/projects
```

---

## Who is this program for?

### Perfect if you are:

- **Developer** looking to pivot into AI/ML
- **Data Analyst** seeking to deepen predictive modeling skills
- **Student** (CS/Math/Engineering) interested in AI
- **Disciplined Self-learner** with time for intensive practice
- **Professional** looking to update skills in LLMs and Generative AI

### Prerequisites:

- Basic Python (variables, loops, functions)
- High school level math (algebra, graphs)
- Technical English (for reading papers/docs)
- Dedication: 10-15 hours/week recommended

### Upon completion, you will be at the level of:

- **Junior AI/ML Engineer** (complete Phases 1-2)
- **Mid-level ML Engineer** (complete Phases 1-3)
- **Senior AI Engineer** (complete Phases 1-4 + personal project)

---

## Study Methodology

### Learning Cycle

Each module follows this proven cycle:

```text
1. THEORY (30%)
   +-- Read concepts, papers, and explanations

2. EXAMPLES (20%)
   +-- Run functional code, understand the details

3. PRACTICES (30%)
   +-- Solve guided exercises, compare with solutions

4. MINI-PROJECT (15%)
   +-- Build an integration project applying everything

5. EVALUATION (5%)
   +-- Self-assess with a checklist, identify gaps
```

### Progress Tracking

Use STATUS.md in each module to record:

```markdown
## Current Status: In progress

- [x] Theory completed (2024-03-01)
- [x] Examples executed (2024-03-05)
- [ ] Practices 1/2 completed
- [ ] Mini-project pending

**Current Blocker**: Overfitting in CNN model
**Next Step:** Apply data augmentation
```

### Success Tips

- Don't skip modules - Each builds upon the previous one.
- Type the code - Don't just copy/paste; write it manually.
- Modify examples - Change hyperparameters, observe results.
- Document learnings - Use notes/ for your discoveries.
- Share projects - Upload mini-projects to GitHub for your portfolio.
- Join the community - ML Discord/Reddit for questions.

---

## Projects You Will Build

Throughout the program, you will complete real projects for your portfolio:

### Mini-Project per Module

1. **Exploratory Data Analysis** of the Titanic dataset with insights.
2. **Movie Recommender System** using matrix factorization.
3. **Sentiment Classifier** for reviews using fine-tuned BERT.
4. **Anomaly Detector** in financial time series.
5. **RAG Chatbot** answering questions about PDF documents.
6. **Bias Audit** in a credit scoring model.
7. **Pipeline ML complete** with CI/CD, monitoring and A/B testing.

### Suggested Final Project

Build a **complete production system** that integrates:

- LLM with RAG about your chosen domain.
- Guardrails for security (input validation + output filtering).
- Deployment with FastAPI + Docker.
- Monitoring with Prometheus + Grafana.
- Automated evaluation with pytest.
- CI/CD with GitHub Actions.

**Domain Examples:**

- Medical assistant answering symptom queries.
- HR bot answering employee policy questions.
- Personalized educational tutor for students.
- Product recommender with explanations.

---

## Contributing

Found an error? Have suggestions? Contributions are welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on branching, commits, and quality expectations.

---

## Sibling Labs

This repository is part of the NaNLABS learning lab ecosystem:

- [nan-python-engineering-labs](https://github.com/nanlabs/nan-python-engineering-labs) — Python engineering fundamentals
- [nan-data-engineering-labs](https://github.com/nanlabs/nan-data-engineering-labs) — Data engineering track
- [nan-ai-native-engineering-labs](https://github.com/nanlabs/nan-ai-native-engineering-labs) — AI-native operator skills

---

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This program synthesizes knowledge from:

- Academic papers (BERT, GPT, Stable Diffusion, etc.).
- Courses from Stanford, MIT, and Fast.ai.
- Practical industry experience.
- Open-source ML/AI community.
