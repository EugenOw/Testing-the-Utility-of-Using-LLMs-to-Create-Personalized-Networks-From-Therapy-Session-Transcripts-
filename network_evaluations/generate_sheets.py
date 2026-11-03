import json
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

# Load the JSON file
with open("evaluator_assignment.json", "r") as f:
    data = json.load(f)

# Group sessions by evaluator
evaluator_sessions = defaultdict(list)
for entry in data:
    evaluator_sessions[entry["Evaluator"]].append(entry["Session"])

# Dropdown setup
dropdown_values = ["Network A", "Network B"]
dropdown_formula = '"' + ",".join(dropdown_values) + '"'

# Questions and Comments
questions = [
    "Which Network has better clinically meaningful themes?",
    "Which Network has better connection?",
    "Which network helps you understand this patient better and could assist you more in conducting therapy sessions?"
]
comment_headers = ["Network A", "Network B"]

# Generate Excel files
for evaluator, sessions in evaluator_sessions.items():
    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluation"

    # Merge and write headers
    ws.merge_cells('A1:A2')
    ws['A1'] = "Session ID"
    ws['A1'].font = Font(bold=True)
    ws['A1'].alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells(start_row=1, start_column=2, end_row=1, end_column=4)
    ws.cell(row=1, column=2).value = "Questions"
    ws.cell(row=1, column=2).font = Font(bold=True)
    ws.cell(row=1, column=2).alignment = Alignment(horizontal="center")

    ws.merge_cells(start_row=1, start_column=5, end_row=1, end_column=6)
    ws.cell(row=1, column=5).value = "Comments (Optional)"
    ws.cell(row=1, column=5).font = Font(bold=True)
    ws.cell(row=1, column=5).alignment = Alignment(horizontal="center")

    # Write second header row (row 2)
    headers = questions + comment_headers
    for col, header in enumerate(headers, start=2):  # Start from column B
        ws.cell(row=2, column=col).value = header

    # Fill in session rows (starting from row 3)
    for row_idx, session in enumerate(sessions, start=3):
        ws.cell(row=row_idx, column=1).value = session  # Session ID
        for col in range(2, 5):  # Columns B-D (questions)
            cell = ws.cell(row=row_idx, column=col)
            dv = DataValidation(type="list", formula1=dropdown_formula, showDropDown=True)
            ws.add_data_validation(dv)
            dv.add(cell)

    # Save Excel file
    wb.save(f"{evaluator}.xlsx")

print("✅ All evaluator files with sessions combined into one sheet.")
