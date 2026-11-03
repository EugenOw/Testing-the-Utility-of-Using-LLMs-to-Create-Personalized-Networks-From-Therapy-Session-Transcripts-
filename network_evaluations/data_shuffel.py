
import pandas as pd
import random
import json
from collections import defaultdict# Required in your environment
import matplotlib.pyplot as plt

# Patienten-Daten
session_counts = {
    "P11": 15,
    "P12": 15,
    "P13": 15,
    "P14": 15,
    "P15": 6,
    "P16": 10
}

# Netzwerk-Optionen
networks = ["Network A", "Network B"]

# Ergebnisliste
session_data = []

# Funktion zum zufälligen, aber gleichmäßigen Shuffling
def random_network_order():
    order = networks[:]
    random.shuffle(order)
    return order

# Generieren der Daten
for patient_id, count in session_counts.items():
    for i in range(1, count + 1):
        majority, baseline = random_network_order()
        session_info = {
            "Session": f"{patient_id}_S{str(i).zfill(2)}",
            "Majority Vote": majority,
            "Baseline": baseline
        }
        session_data.append(session_info)

# Ausgabe als JSON-Datei (optional)
with open("session_evaluation.json", "w") as f:
    json.dump(session_data, f, indent=2)

# Optional: Ausgabe zur Kontrolle
print(json.dumps(session_data[:5], indent=2))  # Zeigt nur die ersten 5


evaluators = ["EV1", "EV3", "EV2"]
assigned_sessions = defaultdict(list)

# Schritt 1: Zufällige Auswahl von 20 Sessions mit Overlap
all_session_ids = [s["Session"] for s in session_data]
overlap_sessions = random.sample(all_session_ids, 20)

# Schritt 2: Jeder Session eine*n Evaluator zuweisen (random verteilt)
remaining_sessions = list(all_session_ids)
random.shuffle(remaining_sessions)

# Temporärer Zähler pro Evaluator
evaluator_counts = defaultdict(int)

for session_id in remaining_sessions:
    if session_id in overlap_sessions:
        # Doppelt zuweisen
        evs = random.sample(evaluators, 2)
        for ev in evs:
            assigned_sessions[ev].append(session_id)
            evaluator_counts[ev] += 1
    else:
        # Einfach zuweisen, wähle den Evaluator mit der niedrigsten Anzahl
        ev = min(evaluators, key=lambda e: evaluator_counts[e])
        assigned_sessions[ev].append(session_id)
        evaluator_counts[ev] += 1

# Schritt 3: Ausgabe strukturieren
evaluator_assignments = []
for evaluator, sessions in assigned_sessions.items():
    for session_id in sessions:
        evaluator_assignments.append({
            "Evaluator": evaluator,
            "Session": session_id
        })

# Optional: Ausgabe als JSON-Datei
with open("evaluator_assignment.json", "w") as f:
    json.dump(evaluator_assignments, f, indent=2)

# Kontrolle: Anzahl pro Evaluator und Overlap-Zählung
for ev in evaluators:
    print(f"{ev}: {len(assigned_sessions[ev])} Sessions")

overlap_check = [s for s in all_session_ids if sum(s in assigned_sessions[ev] for ev in evaluators) > 1]
print(f"Anzahl Overlap-Sessions: {len(overlap_check)}")



# Lade Session-Daten
with open("session_evaluation.json") as f:
    session_data = json.load(f)

evaluators = ["EV1", "EV3", "EV2"]
session_assignments = defaultdict(list)

# 1. Zufällige Auswahl von 20 Overlap-Sessions
all_session_ids = [s["Session"] for s in session_data]
overlap_sessions = set(random.sample(all_session_ids, 20))

# 2. Initialisierung
remaining_sessions = list(all_session_ids)
random.shuffle(remaining_sessions)

evaluator_counts = defaultdict(int)

for session_id in remaining_sessions:
    if session_id in overlap_sessions:
        # Zwei verschiedene Evaluatoren zuweisen
        evs = random.sample(evaluators, 2)
        for ev in evs:
            session_assignments[session_id].append(ev)
            evaluator_counts[ev] += 1
    else:
        # Eine*n Evaluator zuweisen – möglichst ausgeglichen
        ev = min(evaluators, key=lambda e: evaluator_counts[e])
        session_assignments[session_id].append(ev)
        evaluator_counts[ev] += 1

# 3. Umwandlung in Tabelle
rows = []
for session_id, ev_list in session_assignments.items():
    ev1 = ev_list[0]
    ev2 = ev_list[1] if len(ev_list) > 1 else ""
    rows.append({
        "Session": session_id,
        "Evaluator 1": ev1,
        "Evaluator 2": ev2
    })

df = pd.DataFrame(rows)

# 4. Sortieren nach Session für Übersichtlichkeit
df = df.sort_values(by="Session").reset_index(drop=True)

# 5. Zeige Tabelle in deiner Umgebung
print(df)
df.to_csv("evaluator_assignments.csv", index=False)

fig, ax = plt.subplots(figsize=(10, len(df) * 0.25))  # Height scales with number of rows
ax.axis('off')

# Draw table
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.2)

plt.title("Evaluator Assignment per Session", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("evaluator_assignment_table.pdf", dpi=300)
plt.show()

