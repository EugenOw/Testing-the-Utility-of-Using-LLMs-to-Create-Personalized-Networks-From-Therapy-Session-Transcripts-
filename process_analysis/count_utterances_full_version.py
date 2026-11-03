import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_ttml_utterances(directory_path):
    ns = {
        "ttml": "http://www.w3.org/ns/ttml",
        "ttm": "http://www.w3.org/ns/ttml#metadata"
    }

    # Structure: {P11: {session: {'Client': x, 'Therapist': y}}}
    person_session_counts = defaultdict(lambda: defaultdict(lambda: {'Client': 0, 'Therapist': 0}))
    person_totals = defaultdict(lambda: {'Client': 0, 'Therapist': 0})
    session_totals = []

    bad_files_log = os.path.join(directory_path, "bad_files.txt")

    # Clear bad files log
    with open(bad_files_log, "w", encoding="utf-8") as f:
        f.write("")

    for filename in os.listdir(directory_path):
        if not filename.endswith(".ttml"):
            continue

        filepath = os.path.join(directory_path, filename)
        print(f"Processing file: {filename}")

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"❌ Skipping malformed file: {filename}")
            with open(bad_files_log, "a", encoding="utf-8") as f:
                f.write(f"{filename} - {e}\n")
            continue

        session_id = filename.split('.')[0]  # e.g., P11_S01
        person_id = session_id.split('_')[0]  # e.g., P11

        session_counts = {'Client': 0, 'Therapist': 0}

        for p in root.findall(".//ttml:body//ttml:div//ttml:p", ns):
            agent = p.attrib.get("{" + ns["ttm"] + "}agent")
            if agent in ["Client", "Therapist"]:
                session_counts[agent] += 1

        person_session_counts[person_id][session_id] = session_counts
        person_totals[person_id]['Client'] += session_counts['Client']
        person_totals[person_id]['Therapist'] += session_counts['Therapist']
        session_totals.append(session_counts)

    # === Output Results ===
    print("\n=== Utterances per Session ===")
    for person, sessions in sorted(person_session_counts.items()):
        for session, counts in sorted(sessions.items()):
            print(f"{session}: Client {counts['Client']}, Therapist {counts['Therapist']}")

    print("\n=== Average Utterances per Person ===")
    for person, totals in sorted(person_totals.items()):
        num_sessions = len(person_session_counts[person])
        avg_client = totals['Client'] / num_sessions if num_sessions else 0
        avg_therapist = totals['Therapist'] / num_sessions if num_sessions else 0
        print(f"{person}: Client {avg_client:.2f}, Therapist {avg_therapist:.2f} over {num_sessions} sessions")

    total_sessions = len(session_totals)
    if total_sessions:
        avg_all_clients = sum(s['Client'] for s in session_totals) / total_sessions
        avg_all_therapists = sum(s['Therapist'] for s in session_totals) / total_sessions
    else:
        avg_all_clients = avg_all_therapists = 0

    print(f"\n=== Overall Averages per Session ===")
    print(f"Client: {avg_all_clients:.2f} utterances")
    print(f"Therapist: {avg_all_therapists:.2f} utterances")

    print("\n=== Total Utterances per Person ===")
    for person, totals in sorted(person_totals.items()):
        print(f"{person}: Client {totals['Client']} utterances, Therapist {totals['Therapist']} utterances")

    # === Grand Total Utterances ===
    total_clients = sum(totals['Client'] for totals in person_totals.values())
    total_therapists = sum(totals['Therapist'] for totals in person_totals.values())
    total_all = total_clients + total_therapists

    print("\n=== Grand Total Utterances Across All Sessions ===")
    print(f"Client utterances: {total_clients}")
    print(f"Therapist utterances: {total_therapists}")
    print(f"Total utterances (Client + Therapist): {total_all}")

    print(f"\n Done. Bad files (if any) logged in: {bad_files_log}")


parse_ttml_utterances("data/full_sessions")
