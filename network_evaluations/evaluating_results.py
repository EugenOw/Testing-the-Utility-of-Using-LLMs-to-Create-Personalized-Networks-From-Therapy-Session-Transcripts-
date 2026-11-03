import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score



def plot_voting_results(result_counter):
    questions = ['Question 1', 'Question 2', 'Question 3']

    baseline_counts = []
    majority_counts = []

    for q in questions:
        total = result_counter[q]['Baseline'] + result_counter[q]['Majority Vote']
        if total == 0:
            baseline_pct = 0
            majority_pct = 0
        else:
            baseline_pct = (result_counter[q]['Baseline'] / total) * 100
            majority_pct = (result_counter[q]['Majority Vote'] / total) * 100
        baseline_counts.append(baseline_pct)
        majority_counts.append(majority_pct)

    x = np.arange(len(questions))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    bars2 = ax.bar(x, baseline_counts, width, label='Baseline', color='lightcoral')
    bars1 = ax.bar(x, majority_counts, width, bottom=baseline_counts, label='Our', color='mediumseagreen')

    ax.set_ylabel('Selection Percentage (%)')
    ax.set_title('Baseline vs Our')
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with percentages
    for i in range(len(questions)):
        if baseline_counts[i] > 0:
            ax.text(x[i], baseline_counts[i] / 2, f'{baseline_counts[i]:.1f}%', ha='center', va='center', fontsize=9, color='black')
        if majority_counts[i] > 0:
            ax.text(x[i], baseline_counts[i] + majority_counts[i] / 2, f'{majority_counts[i]:.1f}%', ha='center', va='center', fontsize=9, color='black')

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    plt.savefig("voting_results_new.pdf", bbox_inches='tight')
    plt.close()

def plot_agreement(agreement_percentages):
    questions = list(agreement_percentages.keys())
    agreement_values = [
        (agree / total) * 100 if total > 0 else 0
        for agree, total in agreement_percentages.values()
    ]

    fig, ax = plt.subplots()
    ax.bar(questions, agreement_values, color='steelblue')

    ax.set_ylabel('Agreement (%)')
    ax.set_title('Evaluator Agreement per Question')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("agreement_results.pdf")
    plt.close()


# === CONFIGURATION ===
excel_files = [
    "Evaluation Results/EV1_Pairwise_Network_Evaluation.xlsx",
    "Evaluation Results/EV2_Pariwise_Network_Evaluation.xlsx",
    "Evaluation Results/EV3_Pairwise_Network_Evaluation.xlsx"
]
json_file = "session_evaluation.json"

# === LOAD SESSION NETWORK MAPPING ===
with open(json_file, "r") as f:
    session_map = {entry["Session"]: entry for entry in json.load(f)}

# === INITIATE RESULT AND AGREEMENT COUNTERS ===
questions = ['Question 1', 'Question 2', 'Question 3']
result_counter = {q: defaultdict(int) for q in questions}
result_counter['Total'] = defaultdict(int)

# Store all answers per session for agreement checking
session_answers = defaultdict(lambda: defaultdict(dict))  # {session_id: {evaluator: {q: answer}}}
evaluator_names = []

# === PROCESS EACH EXCEL FILE ===
for file_index, excel_file in enumerate(excel_files):
    if not os.path.exists(excel_file):
        print(f"Warning: {excel_file} not found, skipping.")
        continue

    evaluator_name = f"Evaluator_{file_index+1}"
    evaluator_names.append(evaluator_name)

    df = pd.read_excel(excel_file, header=2)  # row 3 = header

    for _, row in df.iterrows():
        session_id = row[0]
        if session_id not in session_map:
            continue

        mapping = session_map[session_id]
        net_a_label = "Baseline" if mapping['Baseline'] == "Network A" else "Majority Vote"
        net_b_label = "Baseline" if mapping['Baseline'] == "Network B" else "Majority Vote"

        for i, question in enumerate(questions):
            choice = row[i + 1]  # B, C, D -> i+1
            session_answers[session_id][evaluator_name][question] = choice

            print(f"Session: {session_id}, Evaluator: {evaluator_name}, Question: {question}, Choice: {choice}")

            if choice == "Network A":
                selected = net_a_label
            elif choice == "Network B":
                selected = net_b_label
            else:
                continue  # skip invalid/missing

            result_counter[question][selected] += 1
            result_counter['Total'][selected] += 1


# === DEBUG FILE: Sessions where Baseline beat Majority Vote ===
debug_lines = []
for session_id, evaluator_data in session_answers.items():
    for question in questions:
        baseline_count = 0
        majority_count = 0

        for evaluator in evaluator_data:
            choice = evaluator_data[evaluator].get(question)
            if not choice:
                continue

            mapping = session_map[session_id]
            net_a_label = "Baseline" if mapping['Baseline'] == "Network A" else "Majority Vote"
            net_b_label = "Baseline" if mapping['Baseline'] == "Network B" else "Majority Vote"

            if choice == "Network A":
                selected = net_a_label
            elif choice == "Network B":
                selected = net_b_label
            else:
                continue

            if selected == "Baseline":
                baseline_count += 1
            elif selected == "Majority Vote":
                majority_count += 1

        if baseline_count > majority_count:
            debug_lines.append(
                f"Session: {session_id}, Question: {question}, "
                f"Baseline: {baseline_count}, Majority Vote: {majority_count}"
            )

# Write to debug file
with open("baseline_wins_debug.txt", "w") as f:
    if debug_lines:
        f.write("Sessions where Baseline beat Majority Vote:\n")
        f.write("\n".join(debug_lines))
    else:
        f.write("No sessions where Baseline beat Majority Vote.\n")


# === AGREEMENT CALCULATION ===
agreement_count = {q: 0 for q in questions}
agreement_total = {q: 0 for q in questions}

for session_id, evaluator_data in session_answers.items():
    evaluators = list(evaluator_data.keys())
    if len(evaluators) < 2:
        continue

    e1, e2 = evaluators[:2]  # compare first 2 evaluators only (can extend to full pairwise if needed)
    for question in questions:
        ans1 = evaluator_data[e1].get(question)
        ans2 = evaluator_data[e2].get(question)

        if ans1 and ans2:
            agreement_total[question] += 1
            if ans1 == ans2:
                agreement_count[question] += 1

# === DISPLAY AGGREGATED RESULTS ===
print("\n=== Aggregated Selections ===")
for key in questions + ['Total']:
    print(f"\n{key}:")
    print(f"  Baseline: {result_counter[key]['Baseline']}")
    print(f"  Majority Vote: {result_counter[key]['Majority Vote']}")

print("\nFinal Vote Counts:")
for question in questions:
    print(f"{question} - Baseline: {result_counter[question]['Baseline']}, Majority Vote: {result_counter[question]['Majority Vote']}")


# === DISPLAY AGREEMENT STATS ===
print("\n=== Agreement Between Evaluators (First Two Compared) ===")
for question in questions:
    total = agreement_total[question]
    agree = agreement_count[question]
    if total > 0:
        percent = (agree / total) * 100
        print(f"{question}: {agree}/{total} agreements ({percent:.1f}%)")
    else:
        print(f"{question}: No overlapping sessions for comparison")

# === PLOT AGREEMENT RESULTS ===
agreement_percentages = {
    q: (agreement_count[q], agreement_total[q]) for q in questions
}
plot_agreement(agreement_percentages)

plot_voting_results(result_counter)


# === COHEN'S KAPPA BETWEEN FIRST TWO EVALUATORS ===
print("\n=== Cohen's Kappa Between First Two Evaluators ===")

# Map answers to integers
label_map = {"Network A": 0, "Network B": 1}

for question in questions:
    y_true = []
    y_pred = []

    for session_id, evaluator_data in session_answers.items():
        evaluators = list(evaluator_data.keys())
        if len(evaluators) < 2:
            continue

        e1, e2 = evaluators[:2]
        ans1 = evaluator_data[e1].get(question)
        ans2 = evaluator_data[e2].get(question)

        if ans1 in label_map and ans2 in label_map:
            y_true.append(label_map[ans1])
            y_pred.append(label_map[ans2])

    if y_true and y_pred:
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"{question}: Cohen's Kappa = {kappa:.3f}")
    else:
        print(f"{question}: Not enough data to compute Cohen's Kappa")


# === BUILD AGREEMENT SUMMARY TABLE ===
from sklearn.metrics import cohen_kappa_score

summary_data = []
label_map = {"Network A": 0, "Network B": 1}

for question in questions:
    total = agreement_total[question]
    agree = agreement_count[question]
    percent = (agree / total) * 100 if total > 0 else 0

    y_true = []
    y_pred = []

    for session_id, evaluator_data in session_answers.items():
        evaluators = list(evaluator_data.keys())
        if len(evaluators) < 2:
            continue

        e1, e2 = evaluators[:2]
        ans1 = evaluator_data[e1].get(question)
        ans2 = evaluator_data[e2].get(question)

        if ans1 in label_map and ans2 in label_map:
            y_true.append(label_map[ans1])
            y_pred.append(label_map[ans2])

    kappa = cohen_kappa_score(y_true, y_pred) if y_true and y_pred else None

    summary_data.append({
        "Question": question,
        "Agreement Count": agree,
        "Total Comparisons": total,
        "Agreement %": round(percent, 1),
        "Cohen's Kappa": round(kappa, 3) if kappa is not None else "N/A"
    })

# Convert to DataFrame and save
agreement_df = pd.DataFrame(summary_data)
print(agreement_df)
agreement_df.to_csv("agreement_summary.csv", index=False)
print("\nAgreement summary saved to 'agreement_summary.csv'")


