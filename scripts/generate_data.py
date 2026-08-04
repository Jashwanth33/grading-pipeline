import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_grading_dataset(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "test_pass_rate": np.clip(np.random.beta(5, 2, n_samples), 0, 1),
        "cyclomatic_complexity": np.random.exponential(5, n_samples),
        "num_functions": np.random.poisson(8, n_samples),
        "lines_of_code": np.random.lognormal(6, 0.8, n_samples).astype(int),
        "runtime_ms": np.random.exponential(200, n_samples),
        "memory_usage_mb": np.random.exponential(50, n_samples),
        "num_failed_tests": np.random.poisson(2, n_samples),
        "num_warnings": np.random.poisson(3, n_samples),
        "lint_score": np.clip(np.random.beta(7, 2, n_samples), 0, 1),
        "documentation_score": np.clip(np.random.beta(3, 3, n_samples), 0, 1),
    }
    df = pd.DataFrame(data)
    score = (
        df["test_pass_rate"] * 0.3
        + df["lint_score"] * 0.2
        + df["documentation_score"] * 0.15
        - df["cyclomatic_complexity"] / 30 * 0.1
        - df["num_failed_tests"] / 10 * 0.1
        - df["num_warnings"] / 10 * 0.1
        - df["runtime_ms"] / 2000 * 0.05
    )
    conditions = [
        score >= 0.55, score >= 0.35, score >= 0.15,
    ]
    labels = ["excellent", "good", "needs_improvement"]
    df["quality_label"] = np.select(conditions, labels, default="poor")
    mask = np.random.random(n_samples) < 0.05
    for col in df.columns:
        df.loc[mask, col] = np.nan
    return df


def generate_doubt_dataset(n_samples: int = 1500) -> pd.DataFrame:
    np.random.seed(42)
    topics = ["Python", "Data Structures", "Algorithms", "SQL", "Web Development"]
    urgencies = ["low", "medium", "high", "critical"]

    questions = {
        "Python": [
            "How do I use decorators in Python?",
            "What is the difference between list and tuple?",
            "How to handle exceptions properly?",
            "What are generators and how to use them?",
            "How does the GIL work?",
            "Can you explain list comprehensions?",
            "What is the purpose of __init__?",
            "How to read and write files in Python?",
        ],
        "Data Structures": [
            "When should I use a hash map vs tree map?",
            "How does a stack differ from a queue?",
            "Explain how linked lists work",
            "What is a heap and when to use it?",
            "How does a BST maintain ordering?",
            "What are tries used for?",
        ],
        "Algorithms": [
            "How does merge sort work?",
            "What is the time complexity of quicksort?",
            "Explain dynamic programming with an example",
            "How to detect cycles in a graph?",
            "What is binary search and when to use it?",
            "Explain BFS vs DFS traversal",
        ],
        "SQL": [
            "How do JOINs work in SQL?",
            "What is the difference between WHERE and HAVING?",
            "How to optimize slow queries?",
            "What are indexes and when to use them?",
            "Explain subqueries vs CTEs",
        ],
        "Web Development": [
            "How does HTTP differ from HTTPS?",
            "What is REST API design?",
            "How do cookies and sessions work?",
            "Explain CORS and how to handle it",
            "What is middleware in web frameworks?",
        ],
    }
    records = []
    for _ in range(n_samples):
        topic = np.random.choice(topics, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        q = np.random.choice(questions[topic])
        urgency = np.random.choice(urgencies, p=[0.15, 0.4, 0.3, 0.15])
        records.append({"question": q, "topic": topic, "urgency": urgency})
    return pd.DataFrame(records)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    grading_df = generate_grading_dataset(2000)
    grading_path = data_dir / "grading_dataset.csv"
    grading_df.to_csv(grading_path, index=False)
    print(f"Grading dataset: {grading_path} ({grading_df.shape})")

    doubt_df = generate_doubt_dataset(1500)
    doubt_path = data_dir / "doubt_dataset.csv"
    doubt_df.to_csv(doubt_path, index=False)
    print(f"Doubt dataset: {doubt_path} ({doubt_df.shape})")
