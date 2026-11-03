import json
import csv
import random
from collections import OrderedDict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_csv(path, data):
    if not data:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return
    fieldnames = list(data[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            flat = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()}
            writer.writerow(flat)

def dedupe_by_key(items, key="utterance"):
    seen = set()
    out = []
    for x in items:
        k = x.get(key)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def add_ids(items, start_id=1):
    """Add an 'id' key to each item as the first field."""
    out = []
    for i, x in enumerate(items, start=start_id):
        # Ensure 'id' is first key using OrderedDict
        new_item = OrderedDict()
        new_item["id"] = i
        for k, v in x.items():
            new_item[k] = v
        out.append(new_item)
    return out

def make_splits(items, key="utterance", seed=42):
    rng = random.Random(seed)

    # 1) Deduplicate
    items = dedupe_by_key(items, key=key)

    if len(items) != 500:
        raise ValueError(f"Expected 500 unique items after dedup; got {len(items)}.")

    # 2) Partition by label
    trues  = [x for x in items if bool(x.get("process"))]
    falses = [x for x in items if not bool(x.get("process"))]

    if len(trues) < 50 or len(falses) < 50:
        raise ValueError(f"Need at least 50 true and 50 false for train; "
                         f"have {len(trues)} true, {len(falses)} false.")

    # 3) Sample train
    train_true  = rng.sample(trues, 50)
    train_false = rng.sample(falses, 50)
    train = train_true + train_false

    train_keys = {x[key] for x in train}
    test = [x for x in items if x[key] not in train_keys]

    # 4) Shuffle
    rng.shuffle(train)
    rng.shuffle(test)

    # 5) Checks
    assert len(train) == 100
    assert len(test) == 400
    assert {x[key] for x in train}.isdisjoint({x[key] for x in test})
    assert sum(1 for x in train if x.get("process")) == 50

    return train, test

if __name__ == "__main__":
    input_path = "mini_session_processes_GT.json"
    train_json_path = "train.json"
    test_json_path  = "test.json"
    train_csv_path  = "train.csv"
    test_csv_path   = "test.csv"
    seed = 1337

    data = load_json(input_path)
    train, test = make_splits(data, key="utterance", seed=seed)

    # Add IDs (train IDs: 1–100, test IDs: 1–400)
    train = add_ids(train, start_id=1)
    test  = add_ids(test,  start_id=1)

    # Save JSON
    save_json(train_json_path, train)
    save_json(test_json_path, test)

    # Save CSV
    save_csv(train_csv_path, train)
    save_csv(test_csv_path, test)

    print("Done.")
    print(f"Train: {len(train)} items → {train_json_path}, {train_csv_path}")
    print(f"Test:  {len(test)} items → {test_json_path}, {test_csv_path}")
