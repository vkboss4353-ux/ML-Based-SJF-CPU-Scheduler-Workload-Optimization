import pandas as pd
import joblib
from pathlib import Path
from collections import deque

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "workloads.csv"
MODEL_FILE = BASE_DIR / "models" / "best_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load workload dataset and trained ML model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

FEATURES = [
    "instructions",
    "memory_mb",
    "io_operations",
    "priority"
]

# Predict burst time using the trained ML model
df["predicted_burst"] = model.predict(df[FEATURES])
df["predicted_burst"] = df["predicted_burst"].clip(lower=0.1)


def schedule_fcfs(data):
    data = data.sort_values(
        ["arrival_time", "process_id"]
    ).copy()

    time = 0
    completion = {}

    for _, p in data.iterrows():
        start = max(time, p["arrival_time"])
        time = start + p["predicted_burst"]
        completion[p["process_id"]] = time

    return completion


def schedule_sjf(data):
    remaining = data.to_dict("records")
    time = 0
    completion = {}
    remaining.sort(key=lambda p: p["arrival_time"])

    while remaining:
        available = [
            p for p in remaining
            if p["arrival_time"] <= time
        ]

        if not available:
            time = remaining[0]["arrival_time"]
            continue

        process = min(
            available,
            key=lambda p: p["predicted_burst"]
        )

        time += process["predicted_burst"]
        completion[process["process_id"]] = time
        remaining.remove(process)

    return completion


def schedule_priority(data):
    remaining = data.to_dict("records")
    remaining.sort(key=lambda p: p["arrival_time"])

    time = 0
    completion = {}

    while remaining:
        available = [
            p for p in remaining
            if p["arrival_time"] <= time
        ]

        if not available:
            time = remaining[0]["arrival_time"]
            continue

        process = min(
            available,
            key=lambda p: p["priority"]
        )

        time += process["predicted_burst"]
        completion[process["process_id"]] = time
        remaining.remove(process)

    return completion



def schedule_round_robin(data, quantum=5):
    processes = data.to_dict("records")
    processes.sort(key=lambda p: p["arrival_time"])

    remaining = {
        p["process_id"]: float(p["predicted_burst"])
        for p in processes
    }

    arrivals = {
        p["process_id"]: float(p["arrival_time"])
        for p in processes
    }

    queue = deque()
    completed = {}
    time = 0
    i = 0

    while i < len(processes) or queue:
        if not queue and i < len(processes):
            time = max(time, arrivals[processes[i]["process_id"]])

            while (
                i < len(processes)
                and arrivals[processes[i]["process_id"]] <= time
            ):
                queue.append(processes[i]["process_id"])
                i += 1

        pid = queue.popleft()

        run_time = min(quantum, remaining[pid])
        time += run_time
        remaining[pid] -= run_time

        while (
            i < len(processes)
            and arrivals[processes[i]["process_id"]] <= time
        ):
            queue.append(processes[i]["process_id"])
            i += 1

        if remaining[pid] <= 1e-9:
            completed[pid] = time
        else:
            queue.append(pid)

    return completed



def evaluate(data, completion, algorithm):
    result = data.copy()
    result["completion_time"] = result["process_id"].map(completion)

    result["turnaround_time"] = (
        result["completion_time"] - result["arrival_time"]
    )

    result["waiting_time"] = (
        result["turnaround_time"] - result["predicted_burst"]
    )

    result["algorithm"] = algorithm
    return result


def recommend_algorithm(summary, objective="Balanced"):
    summary = summary.copy()

    if objective == "Waiting Time":
        best_score = summary["Average_Waiting_Time"].min()
        winners = summary[
            summary["Average_Waiting_Time"] == best_score
        ]["algorithm"].tolist()

    elif objective == "Turnaround Time":
        best_score = summary["Average_Turnaround_Time"].min()
        winners = summary[
            summary["Average_Turnaround_Time"] == best_score
        ]["algorithm"].tolist()

    else:
        # Normalize both metrics so they are comparable
        waiting_range = (
            summary["Average_Waiting_Time"].max()
            - summary["Average_Waiting_Time"].min()
        )
        turnaround_range = (
            summary["Average_Turnaround_Time"].max()
            - summary["Average_Turnaround_Time"].min()
        )

        if waiting_range == 0:
            summary["Waiting_Score"] = 0
        else:
            summary["Waiting_Score"] = (
                (summary["Average_Waiting_Time"]
                 - summary["Average_Waiting_Time"].min())
                / waiting_range
            )

        if turnaround_range == 0:
            summary["Turnaround_Score"] = 0
        else:
            summary["Turnaround_Score"] = (
                (summary["Average_Turnaround_Time"]
                 - summary["Average_Turnaround_Time"].min())
                / turnaround_range
            )

        summary["Balanced_Score"] = (
            summary["Waiting_Score"]
            + summary["Turnaround_Score"]
        ) / 2

        best_score = summary["Balanced_Score"].min()
        winners = summary[
            summary["Balanced_Score"] == best_score
        ]["algorithm"].tolist()

    return winners
if __name__ == "__main__":

    # Run algorithms on a small sample first
    sample = df.head(100).copy()

    algorithms = {
        "FCFS": schedule_fcfs(sample),
        "SJF": schedule_sjf(sample),
        "Priority": schedule_priority(sample),
        "Round Robin": schedule_round_robin(sample, quantum=5)
    }

    all_results = []

    for name, completion in algorithms.items():
        result = evaluate(sample, completion, name)

        if name == "Round Robin":
            print("\nRound Robin - First 10 Processes:")
            print(
                result[
                    [
                        "process_id",
                        "arrival_time",
                        "predicted_burst",
                        "completion_time",
                        "turnaround_time",
                        "waiting_time"
                    ]
                ].head(10).to_string(index=False)
            )

        print(f"\n{name}")
        print(
            f"Minimum waiting time: "
            f"{result['waiting_time'].min():.2f}"
        )
        print(
            f"Average waiting time: "
            f"{result['waiting_time'].mean():.2f}"
        )
        print(
            f"Average turnaround time: "
            f"{result['turnaround_time'].mean():.2f}"
        )

        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)

    combined.to_csv(
        OUTPUT_DIR / "scheduling_results.csv",
        index=False
    )

    print("\nScheduling results saved to outputs/scheduling_results.csv")

    summary = combined.groupby("algorithm").agg(
        Average_Waiting_Time=("waiting_time", "mean"),
        Average_Turnaround_Time=("turnaround_time", "mean")
    ).reset_index()

    print("\nAlgorithm Comparison:")
    print(summary.round(2).to_string(index=False))

    for objective in ["Waiting Time", "Turnaround Time", "Balanced"]:
        winners = recommend_algorithm(summary, objective)
        print(
            f"\nBest algorithm for {objective}: "
            f"{', '.join(winners)}"
        )




# Run algorithms on a small sample first
