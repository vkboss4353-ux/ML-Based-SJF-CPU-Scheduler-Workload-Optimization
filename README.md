# ML-Based SJF CPU Scheduler & Workload Optimization

## 📌 Project Overview

This project combines **Machine Learning and Operating Systems** to predict CPU burst time and analyze the performance of different CPU scheduling algorithms.

Traditional Shortest Job First (SJF) scheduling requires the CPU burst time of each process to be known in advance. Since the actual burst time is generally unknown before execution, this project uses machine learning models to estimate burst time from workload characteristics and compares scheduling performance.

The project includes a machine learning pipeline and an interactive Streamlit application for exploring predictions and scheduling results.

## 🎯 Objectives

- Predict CPU burst time using workload features.
- Train and evaluate Linear Regression and Random Forest models.
- Compare prediction accuracy using MAE, RMSE, and R².
- Simulate FCFS, SJF, Priority Scheduling, and Round Robin.
- Compare waiting time and turnaround time across scheduling algorithms.
- Visualize model predictions and scheduling performance.

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization & UI:** Streamlit
- **Concepts:** CPU Scheduling, Regression, Model Evaluation, Operating Systems

## 📂 Dataset

The project uses a synthetic workload dataset containing **10,000 records**.

Features include:

- `arrival_time` — Time at which a process enters the ready queue.
- `instructions` — Approximate workload instruction count.
- `memory_mb` — Memory associated with the workload.
- `io_operations` — Number of I/O operations.
- `priority` — Process priority.
- `burst_time` — Target value representing CPU execution time.

The dataset is generated for experimentation and model training.

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression is used as a baseline model to learn the relationship between workload characteristics and CPU burst time.

### 2. Random Forest Regressor

Random Forest combines predictions from multiple decision trees. It can capture nonlinear relationships between workload features and CPU burst time.

### Model Evaluation

The models are evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors more strongly |
| R² Score | Measures how much variation in the target is explained by the model |

### Model Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.1831 | 1.4953 | 0.9472 |
| Random Forest | 1.2669 | 1.6036 | 0.9393 |

Based on these experimental results, Linear Regression achieved lower prediction error and a higher R² score than Random Forest on this dataset.

## ⚙️ CPU Scheduling Algorithms

The project compares four scheduling algorithms:

- **FCFS (First Come, First Served):** Processes are executed in arrival order.
- **SJF (Shortest Job First):** The available process with the shortest burst time is selected first.
- **Priority Scheduling:** Processes are selected based on priority.
- **Round Robin:** Processes receive CPU time in turns according to a fixed time quantum.

The scheduler evaluates metrics such as:

- Average Waiting Time
- Average Turnaround Time

The implementation's scheduling results depend on its assumptions, workload, and simulation logic.

## 📊 Experimental Results

The following waiting-time and turnaround-time values were obtained from the project's scheduling simulation.

| Algorithm | Average Waiting Time | Average Turnaround Time |
|---|---:|---:|
| FCFS | 927.48 | 946.48 |
| SJF | 750.30 | 769.29 |
| Priority Scheduling | 925.92 | 944.92 |
| Round Robin | 1479.61 | 1498.61 |

In this experiment, SJF achieved the lowest average waiting and turnaround times among the four algorithms.

*Note: These are project-specific simulation results, not guaranteed performance improvements for every workload.*

## 🖥️ Streamlit Application

The application provides an interactive interface to explore:

- CPU burst-time predictions.
- Machine learning model performance.
- Workload characteristics.
- Scheduling algorithm comparisons.
- Waiting-time and turnaround-time results.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vkboss4353-ux/ML-Based-SJF-CPU-Scheduler-Workload-Optimization.git
```

### 2. Open the project folder

```bash
cd ML-Based-SJF-CPU-Scheduler-Workload-Optimization
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit
```

### 4. Run the application

```bash
streamlit run app.py
```

Make sure the filenames and entry point match your actual repository. If your Streamlit application uses a different filename, replace `app.py` with that filename.

## 📁 Project Structure

```text
ML-Based-SJF-CPU-Scheduler-Workload-Optimization/
├── app.py
├── data/
│   └── workloads.csv
├── models/
│   └── best_model.pkl
├── outputs/
├── requirements.txt
└── README.md
```

*The structure above is illustrative; adjust it to match the files in your repository.*

## 💡 Key Learnings

- Understood CPU scheduling algorithms and their trade-offs.
- Learned how regression models can estimate CPU burst time.
- Compared machine learning models using multiple evaluation metrics.
- Explored how predicted burst times can support scheduling decisions.
- Practiced building an interactive machine learning application using Streamlit.

## 🔮 Future Improvements

- Train and evaluate models on real-world workload traces.
- Add more advanced prediction models.
- Explore preemptive scheduling using Shortest Remaining Time First.
- Test model performance across different workload distributions.
- Improve visualizations and compare scheduling outcomes interactively.

## 👨‍💻 Author

**Ram Vishal Manoj**

GitHub: [vkboss4353-ux](https://github.com/vkboss4353-ux)
