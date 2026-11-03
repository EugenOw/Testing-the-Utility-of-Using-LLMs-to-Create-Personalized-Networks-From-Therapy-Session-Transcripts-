import json
import os
import textwrap

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
        <span style="color:#e74c3c;">Red</span>: Inhibitory<br>
        <span style="color:#3498db;">Blue</span>: Excitatory<br>
        <hr>
        <b>Dimensions</b><br>
        🔵 Cognition<br>
        🟢 Sense of Self<br>
        🔴 (Overt) Behaviour<br>
        🟣 Affect<br>
        🟠 Context/Moderators<br>
        🟡 Attention<br>
        ⚪ Biophysiological<br>
        🟤 Motivation<br>
        ⚫ Sociocultural
    </div>
    <script>
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            layout: {{ name: 'cose' }},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'shape': 'roundrectangle',
                        'label': 'data(label)',
                        'text-wrap': 'wrap',
                        'text-max-width': 120,
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'background-color': '#ffffff',
                        'border-color': '#000000',
                        'border-width': 1,
                        'color': '#000',
                        'font-size': 14,
                        'padding': '20px',
                        'width': 'label',
                        'height': 'label'
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

def build_elements(edge_data, node_info):
    nodes = {}
    edges = []

    emoji_map = {
        "Cognition": "🔵",
        "Sense of Self": "🟢",
        "(Overt) Behaviour": "🔴",
        "Affect": "🟣",
        "Context/Moderators": "🟠",
        "Attention": "🟡",
        "Biophysiological": "⚪",
        "Motivation": "🟤",
        "Sociocultural": "⚫"
    }

    node_dimensions = {}
    for entry in node_info:
        node_text = entry["input_processes"][0]
        dims = entry.get("dimensions", [])
        count = entry.get("num_processes", 0)
        dots = " ".join(emoji_map.get(dim, "") for dim in dims[:3])
        freq = f"w = {count}" if count else ""
        node_dimensions[node_text] = (dots, freq)

    if isinstance(edge_data, dict) and "relationships" in edge_data:
        edge_data = edge_data["relationships"]

    for entry in edge_data:
        input1, input2 = entry["input_processes"]
        connection = entry["connection"][0]

        if connection == 1:
            rel_type = entry["relationship_type"][0].lower()
            strength = entry["strength_of_relationship"][0].lower()

            if strength != "strong":
                continue

            explanation = entry["explanation"]
            pair_label = entry["pair_label"]
            color = "#3498db" if rel_type == "excitatory" else "#e74c3c"
            width = 4
            line_style = "solid"

            html_text = textwrap.fill(explanation, width=60)

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

            for node in [input1, input2]:
                if node not in nodes:
                    dotline, freq_text = node_dimensions.get(node, ("", ""))
                    label_parts = [node]
                    if dotline:
                        label_parts.append(dotline)
                    if freq_text:
                        label_parts.append(freq_text)
                    label = "\n".join(label_parts)
                    nodes[node] = {
                        "data": {"id": node, "label": label}
                    }

    return list(nodes.values()) + edges

def write_html(elements, output_path):
    html_content = TEMPLATE.format(elements_json=json.dumps(elements))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ HTML graph saved to:", output_path)

def generate_all_networks(root_id):
    linkage_dir = f"output_data/finalized_linkage/{root_id}"
    transformed_dir = "output_data/transformed"
    output_dir = f"output_data/finalized_networks/{root_id}"

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(linkage_dir):
        if fname.startswith(root_id) and fname.endswith("_final.json"):
            parts = fname.split("_")
            session_id = parts[1]
            base_name = f"{root_id}_{session_id}"

            edge_path = os.path.join(linkage_dir, fname)
            node_path = os.path.join(transformed_dir, f"{base_name}_dimensions.json")
            output_path = os.path.join(output_dir, f"{base_name}_network.html")

            if not os.path.exists(node_path):
                print(f"⚠️ Skipping {base_name}: Node file not found: {node_path}")
                continue

            edge_data = load_json(edge_path)
            node_info = load_json(node_path)
            elements = build_elements(edge_data, node_info)
            write_html(elements, output_path)

# Run for a specific root ID
generate_all_networks("P16")
