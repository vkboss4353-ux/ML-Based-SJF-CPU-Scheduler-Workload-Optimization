
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["instructions", "memory_mb", "io_operations", "priority"]
ALGORITHMS = ["FCFS", "SJF", "Priority", "Round Robin"]

st.set_page_config(
    page_title="ML CPU Scheduler",
    page_icon="⚙️",
    layout="wide"
)

st.title("ML-Based CPU Scheduler & Workload Optimization")
st.write(
    "Predict CPU burst times using machine learning, compare scheduling "
    "algorithms, and recommend an algorithm based on performance."
)

# -------------------- Load Files --------------------

data_file = BASE_DIR / "data" / "workloads.csv"
metrics_file = BASE_DIR / "outputs" / "model_metrics.csv"
results_file = BASE_DIR / "outputs" / "scheduling_results.csv"
model_file = BASE_DIR / "models" / "best_model.pkl"

data = pd.read_csv(data_file) if data_file.exists() else pd.DataFrame()
metrics = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()

try:
    model = joblib.load(model_file) if model_file.exists() else None
except Exception as error:
    model = None
    st.error(f"Could not load trained model: {error}")

st.sidebar.header("Project Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "ML Model Comparison",
        "CPU Scheduling",
        "Burst Time Prediction",
        "Algorithm Recommendation"
    ]
)


# -------------------- Prediction --------------------

def add_predictions(frame):
    """Predict burst times for a workload."""
    frame = frame.copy()

    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    frame["predicted_burst"] = model.predict(frame[FEATURES])
    frame["predicted_burst"] = frame["predicted_burst"].clip(lower=0.1)

    return frame


# -------------------- Scheduling Algorithms --------------------

def simulate_schedule(frame, algorithm, quantum=5):
    """Run a scheduling simulation using predicted burst times."""

    processes = []

    for idx, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        processes.append({
            "process_id": str(row.get("process_id", f"P{idx + 1}")),
            "arrival_time": max(0.0, float(row["arrival_time"])),
            "burst_time": max(0.1, float(row["predicted_burst"])),
            "priority": int(row.get("priority", 5)),
            "index": idx
        })

    n = len(processes)
    if n == 0:
        raise ValueError("No processes available to schedule.")

    completion = {}
    time = 0.0

    # FCFS, non-preemptive SJF, and non-preemptive Priority
    if algorithm in ["FCFS", "SJF", "Priority"]:
        pending = processes.copy()

        while pending:
            arrived = [
                p for p in pending
                if p["arrival_time"] <= time
            ]

            if not arrived:
                time = min(p["arrival_time"] for p in pending)
                arrived = [
                    p for p in pending
                    if p["arrival_time"] <= time
                ]

            if algorithm == "FCFS":
                chosen = min(
                    arrived,
                    key=lambda p: (p["arrival_time"], p["index"])
                )
            elif algorithm == "SJF":
                chosen = min(
                    arrived,
                    key=lambda p: (
                        p["burst_time"],
                        p["arrival_time"],
                        p["index"]
                    )
                )
            else:
                # Priority 1 means highest priority.
                chosen = min(
                    arrived,
                    key=lambda p: (
                        p["priority"],
                        p["arrival_time"],
                        p["index"]
                    )
                )

            time += chosen["burst_time"]
            completion[chosen["index"]] = time
            pending.remove(chosen)

    # Round Robin: preemptive time slicing
    elif algorithm == "Round Robin":
        pending = sorted(
            processes,
            key=lambda p: (p["arrival_time"], p["index"])
        )

        remaining = {
            p["index"]: p["burst_time"]
            for p in processes
        }

        ready = []
        next_process = 0

        while len(completion) < n:
            while (
                next_process < n
                and pending[next_process]["arrival_time"] <= time
            ):
                ready.append(pending[next_process])
                next_process += 1

            if not ready:
                if next_process < n:
                    time = max(
                        time,
                        pending[next_process]["arrival_time"]
                    )
                    continue
                break

            current = ready.pop(0)
            pid = current["index"]

            run_time = min(float(quantum), remaining[pid])
            time += run_time
            remaining[pid] -= run_time

            # Add newly arrived processes before re-queuing current process.
            while (
                next_process < n
                and pending[next_process]["arrival_time"] <= time
            ):
                ready.append(pending[next_process])
                next_process += 1

            if remaining[pid] > 1e-9:
                ready.append(current)
            else:
                completion[pid] = time

    else:
        raise ValueError(f"Unknown scheduling algorithm: {algorithm}")

    output = []

    for p in processes:
        finish = completion[p["index"]]
        turnaround = finish - p["arrival_time"]
        waiting = turnaround - p["burst_time"]

        output.append({
            "process_id": p["process_id"],
            "arrival_time": p["arrival_time"],
            "predicted_burst": p["burst_time"],
            "priority": p["priority"],
            "completion_time": finish,
            "waiting_time": max(0.0, waiting),
            "turnaround_time": turnaround,
            "algorithm": algorithm
        })

    return pd.DataFrame(output)


def compare_algorithms(frame, quantum=5):
    all_results = []

    for algorithm in ALGORITHMS:
        result = simulate_schedule(frame, algorithm, quantum)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)

    summary = (
        combined.groupby("algorithm")
        .agg(
            Average_Waiting_Time=("waiting_time", "mean"),
            Average_Turnaround_Time=("turnaround_time", "mean")
        )
        .reindex(ALGORITHMS)
        .reset_index()
    )

    return combined, summary


# -------------------- Recommendation & Debug Metrics --------------------

def build_recommendation_scores(summary):
    """Calculate normalized balanced scores and metric ranks."""
    summary = summary.copy()

    wait = summary["Average_Waiting_Time"]
    turn = summary["Average_Turnaround_Time"]

    wait_range = wait.max() - wait.min()
    turn_range = turn.max() - turn.min()

    summary["Waiting_Rank"] = wait.rank(method="min").astype(int)
    summary["Turnaround_Rank"] = turn.rank(method="min").astype(int)

    if wait_range > 0:
        wait_norm = (wait - wait.min()) / wait_range
    else:
        wait_norm = pd.Series(0.0, index=summary.index)

    if turn_range > 0:
        turn_norm = (turn - turn.min()) / turn_range
    else:
        turn_norm = pd.Series(0.0, index=summary.index)

    summary["Balanced_Score"] = (wait_norm + turn_norm) / 2

    return summary



def recommend_algorithm(summary, objective):
    """Select the algorithm with the best score for the chosen objective."""

    if objective == "Minimum Waiting Time":
        metric = "Average_Waiting_Time"
        ascending = True

    elif objective == "Minimum Turnaround Time":
        metric = "Average_Turnaround_Time"
        ascending = True

    else:
        metric = "Balanced_Score"
        ascending = True

    # Select the best algorithm; use waiting time and algorithm name
    # as deterministic tie-breakers.
    candidates = summary.sort_values(
        [metric, "Average_Waiting_Time", "algorithm"],
        ascending=[ascending, True, True]
    )

    return candidates.iloc[0]["algorithm"]



# -------------------- Overview --------------------

if page == "Overview":
    st.header("Workload Overview")

    if data.empty:
        st.error("Dataset not found. Check data/workloads.csv.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processes", f"{len(data):,}")
        col2.metric("ML Models Evaluated", len(metrics))
        col3.metric("Input Features", len(FEATURES))

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        if "burst_time" in data.columns:
            st.subheader("Burst Time Distribution")
            st.bar_chart(
                data["burst_time"].value_counts(
                    bins=20, sort=False
                ).sort_index()
            )


# -------------------- ML Model Comparison --------------------

elif page == "ML Model Comparison":
    st.header("Machine Learning Model Evaluation")

    if metrics.empty:
        st.warning("Metrics not found. Run python train_model.py first.")
    else:
        st.dataframe(metrics.round(4), use_container_width=True)

        if "Model" in metrics.columns:
            for column, title in [
                ("MAE", "Mean Absolute Error (MAE)"),
                ("RMSE", "Root Mean Squared Error (RMSE)"),
                ("R2", "R-squared Score")
            ]:
                if column in metrics.columns:
                    st.subheader(title)
                    st.bar_chart(metrics.set_index("Model")[column])

        st.caption(
            "Lower MAE and RMSE are better; higher R² is generally better. "
            "Results are based on the generated workload dataset."
        )


# -------------------- Saved Scheduling Results --------------------

elif page == "CPU Scheduling":
    st.header("CPU Scheduling Performance")

    if results_file.exists():
        results = pd.read_csv(results_file)
        required = {
            "algorithm", "waiting_time", "turnaround_time"
        }

        if required.issubset(results.columns):
            summary = (
                results.groupby("algorithm")
                .agg(
                    Average_Waiting_Time=("waiting_time", "mean"),
                    Average_Turnaround_Time=("turnaround_time", "mean")
                )
                .reset_index()
            )

            st.subheader("Algorithm Comparison")
            st.dataframe(summary.round(3), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Average Waiting Time")
                st.bar_chart(
                    summary.set_index("algorithm")["Average_Waiting_Time"]
                )

            with col2:
                st.subheader("Average Turnaround Time")
                st.bar_chart(
                    summary.set_index("algorithm")["Average_Turnaround_Time"]
                )

            selected = st.selectbox(
                "Inspect an algorithm",
                summary["algorithm"].tolist()
            )

            st.subheader(f"{selected} — Process Results")
            st.dataframe(
                results[results["algorithm"] == selected].head(20),
                use_container_width=True
            )
        else:
            st.error("Scheduling results CSV is missing required columns.")
    else:
        st.warning(
            "Scheduling results not found. Run python scheduler.py "
            "from the project root first."
        )


# -------------------- Single Process Prediction --------------------

elif page == "Burst Time Prediction":
    st.header("Predict CPU Burst Time")
    st.write("Enter process characteristics to estimate its CPU burst time.")

    if model is None:
        st.error("Trained model not found. Run python train_model.py first.")
    else:
        with st.form("prediction_form"):
            instructions = st.number_input(
                "Instructions", min_value=0, value=50000, step=1000
            )
            memory_mb = st.number_input(
                "Memory (MB)", min_value=0, value=256, step=16
            )
            io_operations = st.number_input(
                "I/O Operations", min_value=0, value=100, step=10
            )
            priority = st.number_input(
                "Priority (1 = highest, 10 = lowest)",
                min_value=1, max_value=10, value=2, step=1
            )

            submitted = st.form_submit_button("Predict Burst Time")

        if submitted:
            input_data = pd.DataFrame([{
                "instructions": instructions,
                "memory_mb": memory_mb,
                "io_operations": io_operations,
                "priority": priority
            }], columns=FEATURES)

            try:
                prediction = max(
                    0.1, float(model.predict(input_data)[0])
                )

                st.success(
                    f"Predicted CPU Burst Time: {prediction:.2f}"
                )

                st.caption(
                    "This is an estimate from a model trained on synthetic "
                    "workloads, not a measured CPU runtime."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# -------------------- Algorithm Recommendation --------------------

elif page == "Algorithm Recommendation":
    st.header("Scheduling Algorithm Recommendation")

    st.write(
        "The model predicts burst times for a group of processes. "
        "The four algorithms are then simulated on the same workload."
    )

    if model is None:
        st.error("Trained model not found. Run python train_model.py first.")

    elif data.empty:
        st.error("Dataset not found. Check data/workloads.csv.")

    elif not all(
        col in data.columns for col in FEATURES + ["arrival_time"]
    ):
        st.error(
            "Dataset must contain arrival_time and all model feature columns."
        )

    else:
        objective = st.selectbox(
            "Optimization objective",
            [
                "Minimum Waiting Time",
                "Minimum Turnaround Time",
                "Balanced"
            ]
        )

        max_processes = min(len(data), 500)
        default_count = min(100, max_processes)
        min_processes = min(10, max_processes)

        process_count = st.slider(
            "Number of processes to simulate",
            min_value=min_processes,
            max_value=max_processes,
            value=default_count,
            step=10 if max_processes >= 100 else 1
        )

        quantum = st.number_input(
            "Round Robin time quantum",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

        if st.button("Compare & Recommend", type="primary"):
            try:
                workload = data.head(process_count).copy()
                workload = add_predictions(workload)

                combined, summary = compare_algorithms(
                    workload, int(quantum)
                )
                scored = build_recommendation_scores(summary)
                best = recommend_algorithm(scored, objective)

                st.subheader("Recommended Scheduling Algorithm")
                st.success(best)

                st.write(f"**Selected objective:** {objective}")

                if objective == "Minimum Waiting Time":
                    st.write(
                        f"Average waiting time: "
                        f"{scored.loc[scored['algorithm'] == best, 'Average_Waiting_Time'].iloc[0]:.3f}"
                    )
                elif objective == "Minimum Turnaround Time":
                    st.write(
                        f"Average turnaround time: "
                        f"{scored.loc[scored['algorithm'] == best, 'Average_Turnaround_Time'].iloc[0]:.3f}"
                    )
                else:
                    st.write(
                        f"Balanced score: "
                        f"{scored.loc[scored['algorithm'] == best, 'Balanced_Score'].iloc[0]:.3f}"
                    )

                st.caption(
                    "Recommendation applies only to the selected workload "
                    "and simulation settings. It is not a universal best "
                    "algorithm for every real CPU workload."
                )

                st.subheader("Performance Comparison")
                st.dataframe(
                    scored.round(4),
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Average Waiting Time")
                    st.bar_chart(
                        scored.set_index("algorithm")["Average_Waiting_Time"]
                    )

                with col2:
                    st.subheader("Average Turnaround Time")
                    st.bar_chart(
                        scored.set_index("algorithm")[
                            "Average_Turnaround_Time"
                        ]
                    )

                st.subheader("Debug: Why was this algorithm selected?")
                if objective == "Minimum Waiting Time":
                    st.write(
                        "Lowest average waiting time among the four "
                        "simulated algorithms."
                    )
                elif objective == "Minimum Turnaround Time":
                    st.write(
                        "Lowest average turnaround time among the four "
                        "simulated algorithms."
                    )
                else:
                    st.write(
                        "Lowest average of normalized waiting-time and "
                        "turnaround-time scores."
                    )

                st.subheader("Process-Level Results")
                selected_results = combined[
                    combined["algorithm"] == best
                ]

                st.dataframe(
                    selected_results.head(30),
                    use_container_width=True
                )

                st.subheader("Predicted Workload Preview")
                preview_columns = [
                    "arrival_time", "instructions", "memory_mb",
                    "io_operations", "priority", "predicted_burst"
                ]

                if "process_id" in workload.columns:
                    preview_columns.insert(0, "process_id")

                st.dataframe(
                    workload[preview_columns].head(20),
                    use_container_width=True
                )

                # Make the diagnostic values easy to copy without a screenshot.
                st.subheader("Debug: Exact metric values")
                st.code(
                    scored[
                        [
                            "algorithm",
                            "Average_Waiting_Time",
                            "Average_Turnaround_Time",
                            "Balanced_Score",
                            "Waiting_Rank",
                            "Turnaround_Rank"
                        ]
                    ].to_string(index=False),
                    language="text"
                )

            except Exception as error:
                st.error(f"Scheduling comparison failed: {error}")


st.divider()
st.caption("Academic project prototype | Synthetic workload dataset")


