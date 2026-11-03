import os
import json
import re

def clean_json_string(raw_string):
    # Remove markdown formatting like ```json\n, ``` etc.
    cleaned = re.sub(r'^```(?:json)?\n?', '', raw_string.strip())
    cleaned = re.sub(r'\n?```$', '', cleaned)

    # Optionally unescape escaped characters
    try:
        cleaned = bytes(cleaned, "utf-8").decode("unicode_escape")
    except Exception as e:
        print(f"Decoding error: {e}")

    return cleaned

def parse_and_clean_json_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as file:
                try:
                    # Load the wrapper JSON
                    file_content = json.load(file)
                    raw_cluster = file_content.get("clusters", "")

                    # Clean and parse the embedded JSON string
                    cleaned_json_str = clean_json_string(raw_cluster)
                    parsed_json = json.loads(cleaned_json_str)

                    # Write cleaned JSON to output folder
                    output_path = os.path.join(output_folder, filename)
                    with open(output_path, 'w', encoding='utf-8') as out_file:
                        json.dump(parsed_json, out_file, indent=2, ensure_ascii=False)

                    print(f"Processed: {filename}")

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error in {filename}: {e}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Example usage
input_folder = 'output/P13'
output_folder = 'output_cleaned/P13'

parse_and_clean_json_files(input_folder, output_folder)
