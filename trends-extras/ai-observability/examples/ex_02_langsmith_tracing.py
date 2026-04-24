"""
Distributed Tracing with LangSmith
===================================
Trace LLM chains y agents con LangSmith.
Visualiza flujo completo: retrieval → generation → post-processing.

Requirements:
    pip install langsmith langchain
"""

import time
import random
from typing import List, Dict
from datetime import datetime

# ============================================================================
# MOCK TRACING FRAMEWORK (conceptual)
# ============================================================================

class Span:
    """
    Representa un span en distributed tracing.
    Un span es una unidad de trabajo (e.g., una llamada a retrieval).
    """

    def __init__(self, name: str, parent_id: str = None):
        self.span_id = f"span_{random.randint(1000, 9999)}"
        self.parent_id = parent_id
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.metadata = {}
        self.children = []

    def end(self):
        """Finaliza el span."""
        self.end_time = time.time()

    def duration_ms(self) -> float:
        """Duración en ms."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def add_metadata(self, key: str, value: any):
        """Añade metadata al span."""
        self.metadata[key] = value

    def add_child(self, child_span: 'Span'):
        """Añade child span."""
        self.children.append(child_span)

    def print_tree(self, indent: int = 0):
        """Imprime árbol de spans."""
        prefix = "  " * indent
        duration = self.duration_ms()

        print(f"{prefix}├─ {self.name} ({duration:.0f}ms)")

        # Print metadata
        for key, value in self.metadata.items():
            if key == "cost_usd":
                print(f"{prefix}│  💰 ${value:.4f}")
            elif key == "tokens":
                print(f"{prefix}│  🎫 {value} tokens")
            elif key == "model":
                print(f"{prefix}│  🤖 {value}")

        # Print children
        for child in self.children:
            child.print_tree(indent + 1)


class SimpleTracer:
    """
    Tracer simple para demostración.
    En producción: usar LangSmith, OpenTelemetry, etc.
    """

    def __init__(self):
        self.traces = []
        self.current_span = None

    def start_span(self, name: str) -> Span:
        """Inicia un nuevo span."""
        parent_id = self.current_span.span_id if self.current_span else None
        span = Span(name, parent_id)

        if self.current_span:
            self.current_span.add_child(span)
        else:
            self.traces.append(span)

        self.current_span = span
        return span

    def end_span(self, span: Span):
        """Finaliza span."""
        span.end()
        self.current_span = span.parent_id


# ============================================================================
# TRACED RAG PIPELINE
# ============================================================================

def traced_rag_pipeline(query: str, tracer: SimpleTracer) -> str:
    """
    RAG pipeline con tracing.
    """
    # Root span
    root_span = tracer.start_span("rag_pipeline")
    root_span.add_metadata("query", query)

    # 1. Retrieval
    retrieval_span = tracer.start_span("retrieval")
    time.sleep(0.05)  # Simulate vector search
    docs = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris."
    ]
    retrieval_span.add_metadata("docs_retrieved", len(docs))
    retrieval_span.end()

    # 2. Reranking
    rerank_span = tracer.start_span("reranking")
    time.sleep(0.02)  # Simulate reranking
    top_docs = docs[:2]
    rerank_span.add_metadata("top_docs", len(top_docs))
    rerank_span.end()

    # 3. Generation
    gen_span = tracer.start_span("generation")
    gen_span.add_metadata("model", "gpt-3.5-turbo")
    time.sleep(0.15)  # Simulate LLM call
    response = "Paris is the capital of France."
    gen_span.add_metadata("tokens", 25)
    gen_span.add_metadata("cost_usd", 0.0005)
    gen_span.end()

    root_span.end()

    return response


# ============================================================================
# REAL LANGSMITH USAGE
# ============================================================================

REAL_LANGSMITH_CODE = """
from langsmith import traceable
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# ============================================================================
# Setup LangSmith
# ============================================================================

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# ============================================================================
# Traced Functions
# ============================================================================

@traceable(name="retrieval")
def retrieve_documents(query: str) -> List[str]:
    '''
    Retrieval con tracing automático.
    '''
    vector_store = FAISS.load_local("index")
    docs = vector_store.similarity_search(query, k=5)
    return docs

@traceable(name="generation")
def generate_response(query: str, context: str) -> str:
    '''
    Generation con tracing.
    '''
    llm = OpenAI(temperature=0)
    prompt = f"Context: {context}\\n\\nQuestion: {query}"
    response = llm(prompt)
    return response

@traceable(name="rag_pipeline")
def rag_pipeline(query: str) -> str:
    '''
    RAG pipeline completo con tracing.
    LangSmith automáticamente crea árbol de traces.
    '''
    # Retrieval (traced)
    docs = retrieve_documents(query)
    context = "\\n".join([d.page_content for d in docs])

    # Generation (traced)
    response = generate_response(query, context)

    return response

# ============================================================================
# Usage
# ============================================================================

result = rag_pipeline("What is the capital of France?")

# En LangSmith dashboard verás:
# rag_pipeline (230ms, $0.0012)
#   ├─ retrieval (80ms)
#   │    ├─ vector_search (75ms)
#   │    └─ rerank (5ms)
#   └─ generation (150ms, $0.0012)
#        ├─ llm_call (145ms, $0.0012)
#        └─ post_process (5ms)
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_simple_tracing():
    """Tracing simple."""
    print("="*70)
    print("DEMO 1: Simple Tracing")
    print("="*70 + "\n")

    tracer = SimpleTracer()

    print("📊 Executing RAG pipeline with tracing...\n")

    result = traced_rag_pipeline("What is the capital of France?", tracer)

    print(f"✅ Result: {result}\n")
    print("🔍 Trace Tree:\n")

    for trace in tracer.traces:
        trace.print_tree()


def demo_agent_tracing():
    """Tracing para agents."""
    print("\n" + "="*70)
    print("DEMO 2: Agent Tracing")
    print("="*70 + "\n")

    tracer = SimpleTracer()

    # Agent root span
    agent_span = tracer.start_span("agent_execution")
    agent_span.add_metadata("task", "Research task")

    # Thought 1
    thought1 = tracer.start_span("thought_1")
    thought1.add_metadata("thought", "I need to search for info")
    time.sleep(0.01)
    thought1.end()

    # Action 1: Search
    action1 = tracer.start_span("action_search")
    action1.add_metadata("query", "capital of France")
    time.sleep(0.05)
    action1.add_metadata("result", "Paris is the capital")
    action1.end()

    # Thought 2
    thought2 = tracer.start_span("thought_2")
    thought2.add_metadata("thought", "I have the answer")
    time.sleep(0.01)
    thought2.end()

    # Final answer
    answer = tracer.start_span("final_answer")
    answer.add_metadata("answer", "Paris")
    time.sleep(0.01)
    answer.end()

    agent_span.end()

    print("🤖 Agent Execution Trace:\n")
    for trace in tracer.traces:
        trace.print_tree()


def demo_trace_analysis():
    """Analizar traces."""
    print("\n" + "="*70)
    print("DEMO 3: Trace Analysis")
    print("="*70 + "\n")

    print("📊 TRACE ANALYSIS:\n")

    print("1️⃣ Identificar bottlenecks:")
    print("   rag_pipeline: 230ms")
    print("     ├─ retrieval: 80ms      (35%)")
    print("     └─ generation: 150ms    (65%) ⚠️  BOTTLENECK\n")

    print("2️⃣ Cost breakdown:")
    print("   Total: $0.0012")
    print("     ├─ retrieval: $0")
    print("     └─ generation: $0.0012    (100% of cost)\n")

    print("3️⃣ Error tracking:")
    print("   rag_pipeline: 100 calls")
    print("     ├─ success: 95 (95%)")
    print("     ├─ timeout: 3 (3%)")
    print("     └─ error: 2 (2%)\n")

    print("💡 Optimizations:")
    print("   • Cache embeddings (reduce retrieval time)")
    print("   • Use cheaper model for simple queries")
    print("   • Parallelize retrieval + reranking")


def demo_distributed_tracing():
    """Distributed tracing across services."""
    print("\n" + "="*70)
    print("DEMO 4: Distributed Tracing")
    print("="*70 + "\n")

    print("🌐 DISTRIBUTED SYSTEM:\n")
    print("""
    User Request
        │
        ▼
    ┌─────────────────┐
    │  API Gateway    │  (trace_id: abc123)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  RAG Service    │  (span: rag_pipeline)
    └────────┬────────┘
             │
             ├──► ┌─────────────────┐
             │    │ Vector DB       │  (span: retrieval)
             │    └─────────────────┘
             │
             └──► ┌─────────────────┐
                  │ LLM API         │  (span: generation)
                  └─────────────────┘

    💡 Cada span tiene mismo trace_id
    💡 Parent spans esperan a child spans
    💡 Puedes ver flujo completo end-to-end
    """)


def demo_langsmith_features():
    """Características de LangSmith."""
    print("\n" + "="*70)
    print("DEMO 5: LangSmith Features")
    print("="*70 + "\n")

    print("🎯 LANGSMITH FEATURES:\n")

    print("✅ Automatic Tracing:")
    print("   • Auto-instrument LangChain")
    print("   • No code changes needed")
    print("   • Captures all LLM calls\n")

    print("📊 Visualizations:")
    print("   • Waterfall charts (timeline)")
    print("   • Dependency graphs")
    print("   • Cost breakdown\n")

    print("🔍 Debugging:")
    print("   • Inspect inputs/outputs")
    print("   • Replay traces")
    print("   • Compare versions\n")

    print("📈 Analytics:")
    print("   • Latency percentiles")
    print("   • Error rates")
    print("   • Cost tracking\n")

    print("🧪 Testing:")
    print("   • Create test datasets from traces")
    print("   • Regression testing")
    print("   • A/B testing\n")


if __name__ == "__main__":
    print("\n🎯 DISTRIBUTED TRACING WITH LANGSMITH")
    print("🔍 Visualize LLM application flow\n")

    demo_simple_tracing()
    demo_agent_tracing()
    demo_trace_analysis()
    demo_distributed_tracing()
    demo_langsmith_features()

    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("  ✅ Trace every LLM call")
    print("  ✅ Name spans descriptively")
    print("  ✅ Add metadata (model, tokens, cost)")
    print("  ✅ Use consistent trace_id across services")
    print("  ✅ Set up alerts for slow traces")
    print("  ✅ Analyze traces weekly")
    print("  ✅ Use traces to create test cases")

    print("\n" + "="*70)
    print("🚀 SETUP LANGSMITH:")
    print("="*70)
    print("  1. Sign up: https://smith.langchain.com/")
    print("  2. Get API key")
    print("  3. Set environment variables")
    print("  4. Add @traceable decorator")
    print("  5. View traces in dashboard")

    print("\n" + "="*70)
    print("CÓDIGO REAL:")
    print("="*70)
    print(REAL_LANGSMITH_CODE)
