import json
import os
import textwrap
from collections import Counter

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive Process Network</title>
    <script src="https://unpkg.com/cytoscape@3.21.1/dist/cytoscape.min.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden;
        }}
        #cy {{
            width: 100vw;
            height: 100vh;
            display: block;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            border: 1px solid #ccc;
            padding: 10px;
            font-family: sans-serif;
            font-size: 14px;
            z-index: 10;
        }}
        .tooltip {{
            position: absolute;
            background: #fefefe;
            border: 1px solid #aaa;
            padding: 10px;
            font-family: sans-serif;
            font-size: 13px;
            max-width: 300px;
            z-index: 20;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            cursor: default;
        }}
        .tooltip .close-btn {{
            float: right;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>
    <div class="legend">
        <b>Legend</b><br>
        <span style="color:#e74c3c;">Red</span>: Excitatory<br>
        <span style="color:#3498db;">Blue</span>: Inhibitory<br>
        <hr style="border-top: 1px dashed #aaa; margin: 4px 0;">
        Dashed line = Weak<br>
        Solid thin = Moderate<br>
        Solid thick = Strong<br>
        <hr>
        <b>Dimensions</b><br>
        🔴 Cognition<br>
        🟠 Attention<br>
        🟡 Motivation<br>
        🟢 (Overt) Behaviour<br>
        🔵 Sense of Self<br>
        🟣 Context/Moderators<br>
        🟤 Affect<br>
        ⚪ Sociocultural<br>
        ⚫ Biophysiological
    </div>
    <script>
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            layout: {{ name: 'cose' }},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-wrap': 'wrap',
                        'text-max-width': 100,
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'background-color': '#ffffff',
                        'border-color': '#000000',
                        'border-width': 1,
                        'color': '#000',
                        'font-size': 10,
                        'width': 'label',
                        'height': 'label',
                        'padding': 10
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 1.2,
                        'label': '',
                        'font-size': 9,
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10,
                        'line-color': 'data(color)',
                        'target-arrow-color': 'data(color)',
                        'width': 'data(width)',
                        'line-style': 'data(lineStyle)'
                    }}
                }}
            ],
            elements: {elements_json}
        }});

        let tooltipCounter = 0;

        cy.on('tap', 'edge', function(evt) {{
            var edge = evt.target;
            var pos = evt.renderedPosition;
            var tooltipId = "tooltip-" + (++tooltipCounter);

            var tooltip = document.createElement("div");
            tooltip.className = "tooltip";
            tooltip.id = tooltipId;
            tooltip.style.left = (pos.x + 20) + "px";
            tooltip.style.top = (pos.y + 20) + "px";

            var closeBtn = document.createElement("span");
            closeBtn.className = "close-btn";
            closeBtn.innerHTML = "&times;";
            closeBtn.onclick = function () {{
                tooltip.remove();
            }};

            tooltip.innerHTML = "";
            tooltip.appendChild(closeBtn);
            var content = document.createElement("div");
            content.innerHTML = edge.data("info");
            tooltip.appendChild(content);

            document.body.appendChild(tooltip);
            makeDraggable(tooltip);
        }});

        function makeDraggable(elmnt) {{
            let offset = {{ x: 0, y: 0 }};
            let isDragging = false;

            elmnt.addEventListener("mousedown", function (e) {{
                if (e.target.classList.contains("close-btn")) return;
                isDragging = true;
                offset.x = e.clientX - elmnt.offsetLeft;
                offset.y = e.clientY - elmnt.offsetTop;
                elmnt.style.cursor = "move";
                e.preventDefault();
            }});

            document.addEventListener("mousemove", function (e) {{
                if (isDragging) {{
                    elmnt.style.left = (e.clientX - offset.x) + "px";
                    elmnt.style.top = (e.clientY - offset.y) + "px";
                }}
            }});

            document.addEventListener("mouseup", function () {{
                isDragging = false;
                elmnt.style.cursor = "default";
            }});
        }}
    </script>
</body>
</html>
"""

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_elements(json_data, node_info):
    nodes = {}
    edges = []

    # Support both with and without "clusters"
    if "clusters" in json_data:
        edge_data = json_data["clusters"]["theme_relationships"]
        classified = json_data["clusters"]["classified_processes"]
    else:
        edge_data = json_data["theme_relationships"]
        classified = json_data["classified_processes"]

    theme_titles = {
        theme_key: theme_data.get("Title", theme_key)
        for theme_key, theme_data in classified.items()
    }

    emoji_map = {
        "Cognition": "🔴",
        "Attention": "🟠",
        "Motivation": "🟡",
        "(Overt) Behaviour": "🟢",
        "Sense of Self": "🔵",
        "Context/Moderators": "🟣",
        "Affect": "🟤",
        "Sociocultural": "⚪",
        "Biophysiological": "⚫"
    }

    dimension_counter = Counter()
    for entry in node_info:
        for dim in entry.get("dimensions", []):
            dimension_counter[dim] += 1

    node_dimensions = {}
    for entry in node_info:
        node_text = entry["input_processes"][0]
        dims = entry.get("dimensions", [])
        count = entry.get("num_processes", 0)
        dots = " ".join(emoji_map.get(dim, "") for dim in dims[:3])
        freq = f"Frequency: {count}" if count else ""
        node_dimensions[node_text] = (dots, freq)

    for entry in edge_data:
        input1, input2 = entry["input_themes"]
        connection = entry["connection"][0]

        for node in [input1, input2]:
            if node not in nodes:
                dotline, freq_text = node_dimensions.get(node, ("", ""))
                display_label = theme_titles.get(node, node)
                label_parts = [display_label]
                if dotline:
                    label_parts.append(dotline)
                if freq_text:
                    label_parts.append(freq_text)
                label = "\n".join(label_parts)
                nodes[node] = {
                    "data": {"id": node, "label": label}
                }

        if connection == 1:
            rel_type = entry["relationship_type"][0].lower()
            strength = entry["strength_of_relationship"][0].lower()
            explanation = entry["explanation"]
            pair_label = entry.get("pair_label", f"{input1}-{input2}")


            if strength != "strong":
                continue

            color = "#3498db" if rel_type == "excitatory" else "#e74c3c"
            width_map = {"weak": 1, "moderate": 2, "strong": 4}
            style_map = {"weak": "dashed", "moderate": "solid", "strong": "solid"}
            width = width_map.get(strength, 2)
            line_style = style_map.get(strength, "solid")

            html_text = (
                f"<b>Type:</b> {rel_type.title()}<br>"
                f"<b>Strength:</b> {strength.title()}<br>"
                f"<b>Explanation:</b><br>{textwrap.fill(explanation, width=60)}"
            )

            edges.append({
                "data": {
                    "id": f"{input1}_{input2}_{pair_label}",
                    "source": input1,
                    "target": input2,
                    "pair_label": str(pair_label),
                    "info": html_text,
                    "color": color,
                    "width": width,
                    "lineStyle": line_style
                }
            })

    # Filter out unconnected nodes
    connected_ids = set()
    for edge in edges:
        connected_ids.add(edge["data"]["source"])
        connected_ids.add(edge["data"]["target"])

    filtered_nodes = [node for node_id, node in nodes.items() if node_id in connected_ids]

    return filtered_nodes + edges


def write_html(elements, output_path):
    html_content = TEMPLATE.format(elements_json=json.dumps(elements))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ HTML graph saved to:", output_path)

def generate_all_networks(root_id):
    linkage_dir = f"output/{root_id}"
    transformed_dir = "output_data/transformed"
    output_dir = f"output_data/finalized_networks/{root_id}"

    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(linkage_dir):
        if fname.startswith("cluster_" + root_id):
            parts = fname.split("_")
            session_id = parts[2].replace(".json", "")
            base_name = f"{root_id}_{session_id}"

            print("Base Name:", base_name)
            edge_path = os.path.join(linkage_dir, fname)
            node_path = os.path.join(transformed_dir, f"{base_name}_dimensions.json")
            output_path = os.path.join(output_dir, f"{base_name}_network.html")

            print("Edge_path:", edge_path)
            print("Node_path:", node_path)

            if not os.path.exists(node_path):
                print(f"⚠️ Skipping {base_name}: Node file not found: {node_path}")
                continue

            edge_data = load_json(edge_path)
            node_info = load_json(node_path)
            elements = build_elements(edge_data, node_info)
            write_html(elements, output_path)

generate_all_networks("P13")
