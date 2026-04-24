"""
Cost Tracking for LLM Applications
===================================
Track and optimize LLM call costs.
Budget alerts, cost analytics, optimization recommendations.

Requirements:
    pip install tiktoken
"""

import tiktoken
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# COST CALCULATOR
# ============================================================================

class CostCalculator:
    """
    Calculate LLM call costs based on tokens.
    """

    # Pricing (as of 2024, prices may change)
    PRICING = {
        "gpt-4": {
            "input": 0.03,   # $ per 1K tokens
            "output": 0.06,
        },
        "gpt-4-32k": {
            "input": 0.06,
            "output": 0.12,
        },
        "gpt-3.5-turbo": {
            "input": 0.0015,
            "output": 0.002,
        },
        "claude-2": {
            "input": 0.008,
            "output": 0.024,
        },
        "claude-3-opus": {
            "input": 0.015,
            "output": 0.075,
        },
    }

    def __init__(self):
        self.encodings = {}

    def _get_encoding(self, model: str):
        """Get tokenizer encoding for model."""
        if model not in self.encodings:
            try:
                self.encodings[model] = tiktoken.encoding_for_model(model)
            except:
                # Fallback to cl100k_base for unknown models
                self.encodings[model] = tiktoken.get_encoding("cl100k_base")

        return self.encodings[model]

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text.
        """
        # Simplified: count words * 1.3 (approximation)
        # In production: use real tiktoken
        return int(len(text.split()) * 1.3)

    def calculate_cost(
        self,
        prompt: str,
        completion: str,
        model: str = "gpt-3.5-turbo"
    ) -> Dict:
        """
        Calculate cost of a call.
        """
        # Count tokens
        input_tokens = self.count_tokens(prompt, model)
        output_tokens = self.count_tokens(completion, model)

        # Get pricing
        if model not in self.PRICING:
            model = "gpt-3.5-turbo"  # Fallback

        pricing = self.PRICING[model]

        # Calculate cost
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model
        }


# ============================================================================
# COST TRACKER
# ============================================================================

class CostTracker:
    """
    Track costs over time with budget alerts.
    """

    def __init__(self, daily_budget: float = 100.0, monthly_budget: float = 2000.0):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget

        self.calls = []
        self.cost_by_user = defaultdict(float)
        self.cost_by_model = defaultdict(float)
        self.cost_by_day = defaultdict(float)

    def log_call(
        self,
        user_id: str,
        model: str,
        cost: float,
        timestamp: datetime = None
    ):
        """
        Log a call.
        """
        if timestamp is None:
            timestamp = datetime.now()

        day_key = timestamp.strftime("%Y-%m-%d")

        # Store call
        self.calls.append({
            "timestamp": timestamp,
            "user_id": user_id,
            "model": model,
            "cost": cost
        })

        # Update aggregations
        self.cost_by_user[user_id] += cost
        self.cost_by_model[model] += cost
        self.cost_by_day[day_key] += cost

    def get_daily_cost(self, date: datetime = None) -> float:
        """Get daily cost."""
        if date is None:
            date = datetime.now()

        day_key = date.strftime("%Y-%m-%d")
        return self.cost_by_day.get(day_key, 0.0)

    def check_budget_alert(self) -> List[str]:
        """Check if budget was exceeded."""
        alerts = []

        daily_cost = self.get_daily_cost()
        if daily_cost > self.daily_budget:
            alerts.append(f"⚠️ Daily budget exceeded: ${daily_cost:.2f} > ${self.daily_budget:.2f}")
        elif daily_cost > self.daily_budget * 0.8:
            alerts.append(f"⚠️ Daily budget 80%: ${daily_cost:.2f} / ${self.daily_budget:.2f}")

        return alerts

    def get_top_users(self, n: int = 5) -> List[tuple]:
        """Get top N users by cost."""
        return sorted(
            self.cost_by_user.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

    def get_cost_by_model(self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        return dict(self.cost_by_model)

    def generate_report(self) -> str:
        """Generate cost report."""
        report = []
        report.append("="*70)
        report.append("COST REPORT")
        report.append("="*70)

        # Daily cost
        daily = self.get_daily_cost()
        report.append(f"\n📊 Today: ${daily:.2f} / ${self.daily_budget:.2f}")

        progress = min(daily / self.daily_budget, 1.0)
        bar = "█" * int(progress * 30)
        report.append(f"[{bar:<30}] {progress:.0%}")

        # Top users
        report.append("\n👥 Top Users:")
        for user, cost in self.get_top_users(3):
            report.append(f"   {user:20s} ${cost:.2f}")

        # Models
        report.append("\n🤖 By Model:")
        for model, cost in self.get_cost_by_model().items():
            report.append(f"   {model:20s} ${cost:.2f}")

        # Alerts
        alerts = self.check_budget_alert()
        if alerts:
            report.append("\n🚨 ALERTS:")
            for alert in alerts:
                report.append(f"   {alert}")

        return "\n".join(report)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demo_cost_calculation():
    """Calculate costs."""
    print("="*70)
    print("DEMO 1: Cost Calculation")
    print("="*70 + "\n")

    calculator = CostCalculator()

    # Different models
    prompt = "Explain quantum computing in simple terms"
    completion = "Quantum computing uses quantum mechanics principles to process information..."

    models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]

    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}\n")

    for model in models:
        cost_info = calculator.calculate_cost(prompt, completion, model)

        print(f"🤖 {model}:")
        print(f"   Tokens: {cost_info['total_tokens']:,}")
        print(f"   Cost:   ${cost_info['total_cost']:.4f}\n")


def demo_cost_tracking():
    """Track costs."""
    print("="*70)
    print("DEMO 2: Cost Tracking")
    print("="*70 + "\n")

    calculator = CostCalculator()
    tracker = CostTracker(daily_budget=10.0)

    # Simulate calls
    calls = [
        ("user_alice", "gpt-3.5-turbo", "Short query", "Short answer"),
        ("user_bob", "gpt-4", "Long analysis request...", "Detailed response..."),
        ("user_alice", "gpt-3.5-turbo", "Another question", "Another answer"),
        ("user_charlie", "claude-3-opus", "Complex task", "Comprehensive result"),
    ]

    print("📞 Simulating LLM calls...\n")

    for user, model, prompt, completion in calls:
        cost_info = calculator.calculate_cost(prompt, completion, model)
        tracker.log_call(user, model, cost_info["total_cost"])

        print(f"   {user} → {model}: ${cost_info['total_cost']:.4f}")

    print("\n" + tracker.generate_report())


def demo_budget_alerts():
    """Budget alerts."""
    print("\n" + "="*70)
    print("DEMO 3: Budget Alerts")
    print("="*70 + "\n")

    tracker = CostTracker(daily_budget=5.0)

    # Simulate exceeding budget
    costs = [1.0, 1.5, 1.2, 2.0, 1.5]  # Total: $7.2

    for i, cost in enumerate(costs, 1):
        tracker.log_call(f"user_{i}", "gpt-4", cost)

        daily_cost = tracker.get_daily_cost()
        print(f"Call {i}: +${cost:.2f} → Total: ${daily_cost:.2f}")

        # Check alerts
        alerts = tracker.check_budget_alert()
        for alert in alerts:
            print(f"   {alert}")

        print()


def demo_cost_optimization():
    """Cost optimization strategies."""
    print("="*70)
    print("DEMO 4: Cost Optimization Strategies")
    print("="*70 + "\n")

    calculator = CostCalculator()

    prompt = "What is the capital of France?" * 10  # Repeated
    completion = "Paris" * 10

    print("💡 OPTIMIZATION STRATEGIES:\n")

    # 1. Caching
    print("1️⃣ Caching:")
    print("   Without cache: 10 calls × $0.001 = $0.010")
    print("   With cache:    1 call × $0.001 = $0.001")
    print("   Savings:       90%\n")

    # 2. Model selection
    print("2️⃣ Model Selection:")
    gpt35_cost = calculator.calculate_cost(prompt, completion, "gpt-3.5-turbo")
    gpt4_cost = calculator.calculate_cost(prompt, completion, "gpt-4")

    print(f"   GPT-3.5: ${gpt35_cost['total_cost']:.4f}")
    print(f"   GPT-4:   ${gpt4_cost['total_cost']:.4f}")
    print(f"   Savings: {(1 - gpt35_cost['total_cost']/gpt4_cost['total_cost'])*100:.0f}% by using GPT-3.5\n")

    # 3. Prompt engineering
    print("3️⃣ Prompt Engineering:")
    long_prompt = "Please explain in great detail with examples..." * 20
    short_prompt = "Explain briefly:"

    long_cost = calculator.calculate_cost(long_prompt, completion, "gpt-4")
    short_cost = calculator.calculate_cost(short_prompt, completion, "gpt-4")

    print(f"   Long prompt:  ${long_cost['total_cost']:.4f}")
    print(f"   Short prompt: ${short_cost['total_cost']:.4f}")
    print(f"   Savings: {(1 - short_cost['total_cost']/long_cost['total_cost'])*100:.0f}% with concise prompts\n")

    # 4. Rate limiting
    print("4️⃣ Rate Limiting:")
    print("   Without limits: Users spam → $500/day")
    print("   With limits:    10 req/user/day → $50/day")
    print("   Savings:        90%\n")


def demo_cost_analytics_dashboard():
    """Dashboard analytics."""
    print("="*70)
    print("DEMO 5: Cost Analytics Dashboard")
    print("="*70 + "\n")

    print("📊 GRAFANA/DATADOG DASHBOARD:\n")
    print("""
┌────────────────────────────────────────────────────────────────┐
│  LLM Cost Dashboard                           Last updated: now │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  💰 Today's Spend: $85.50 / $100 (85%)          [████████▓░]  │
│                                                                 │
│  📈 Cost Trend (7 days)                                        │
│      $100 ┤                                                ╭─  │
│       $75 ┤                                         ╭──────╯   │
│       $50 ┤                              ╭──────────╯          │
│       $25 ┤                    ╭─────────╯                     │
│        $0 └─────────────────────┘                              │
│            Mon  Tue  Wed  Thu  Fri  Sat  Sun                   │
│                                                                 │
│  🔝 Top 5 Users (Today)                                        │
│     1. alice@company.com    $25.50  [████████████▓░]          │
│     2. bob@company.com      $18.20  [█████████░░░░]           │
│     3. charlie@company.com  $15.00  [███████▓░░░░░]           │
│                                                                 │
│  🤖 Cost by Model                                              │
│     GPT-4:        $45.00 (53%)  [████████████████▓░░░░░░░░]   │
│     GPT-3.5:      $30.50 (36%)  [████████████░░░░░░░░░░░░░]   │
│     Claude:       $10.00 (11%)  [████░░░░░░░░░░░░░░░░░░░░░]   │
│                                                                 │
│  ⚠️ Alerts                                                     │
│     • alice@company.com exceeded individual budget ($20)       │
│     • Approaching daily budget limit (85%)                     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("\n🎯 COST TRACKING FOR LLM APPLICATIONS")
    print("💰 Monitor and optimize LLM costs\n")

    demo_cost_calculation()
    demo_cost_tracking()
    demo_budget_alerts()
    demo_cost_optimization()
    demo_cost_analytics_dashboard()

    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("  ✅ Track costs per user/endpoint/model")
    print("  ✅ Set daily/monthly budgets")
    print("  ✅ Alert when 80% of budget used")
    print("  ✅ Use cheaper models for simple tasks")
    print("  ✅ Cache frequent queries")
    print("  ✅ Rate limit users")
    print("  ✅ Prompt engineering for conciseness")
    print("  ✅ Monitor trends weekly")

    print("\n" + "="*70)
    print("🚀 IMPLEMENTATION:")
    print("="*70)
    print("  1. Install tiktoken for token counting")
    print("  2. Log every LLM call with cost")
    print("  3. Aggregate by user/model/day")
    print("  4. Set up budget alerts (email/Slack)")
    print("  5. Create cost dashboard (Grafana)")
    print("  6. Review and optimize weekly")
