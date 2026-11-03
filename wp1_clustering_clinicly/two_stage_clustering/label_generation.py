def prompt_construct_generate_labels(dialogue_data, process_data):
    # json_format_example = '{\n    "labels": ["label", "label", ...]\n}'

    prompt_str = ""
    for dialogue in dialogue_data:
        speaker = dialogue["speaker"]
        text = dialogue["text"]
        prompt_str += f"{speaker}: {text}\n"

        # Count the occurrences of each cluster
    cluster_counts = {}
    for process in process_data:
        cluster_label = process["semantic_cluster"]
        if cluster_label in cluster_counts:
            cluster_counts[cluster_label] += 1
        else:
            cluster_counts[cluster_label] = 1

    # Include only one process per cluster in the formatted output
    included_clusters = set()
    formatted_data_process_data = ""
    for process in process_data:
        cluster_label = process["semantic_cluster"]
        if cluster_label not in included_clusters:
            process_text = process["process"]
            frequency = cluster_counts[cluster_label]
            formatted_entry = f"""
        - {{utterance: \"{process_text}\", frequency: {frequency}}}

        """
            formatted_data_process_data += formatted_entry
            included_clusters.add(cluster_label)

    prompt = f"""

You are a clinical psychologist analyzing a therapy session transcript.

**Task:** Generate **new, clinically meaningful themes** in a clear, one-sentence description.

**Definitions:**
    - **Process**: A dynamic psychological mechanism that influences emotions, thoughts, and behavior, affecting mental health and therapy outcomes.
    - **Clinically meaningful theme**: A theme represents a meaningful pattern across psychological processes that reflects underlying functions or dynamics within the client's experience.

**Input:** 
    - A transcript excerpt from a therapy session (for context only)
    - List of psychological processes.

**Your Goal:**
    1. Carefully read the transcript excerpt to understand the emotional and interpersonal context.
    2. Generate new, specific, and clinically meaningful themes based on the processes provided.
    3. Focus on psychological functions, conflicts, or transformations **not just naming topics**.
    4. Avoid vague labels, lists of processes, or surface-level rephrasing.


 **Guidelines for Writing Themes:**
    1. Each theme must be a complete, short, concise statement.
    2. Avoid generalities, labels that are not clinically informative.
    3. The short sentence must convey a specific psychological function, conflict, or transformation linking the processes in the cluster
    3. Return output only as a list in this format: [theme1, theme2, ..., themen]

**Output format:**
Return only a list like this: `["theme 1", "theme 2", "theme 3", ...]`
- No extra explanations, no bullet points, no quotes from the transcript.


**Examples of Themes:**
(Note: These examples are for understanding **style only**,  you must NOT reuse or adapt them!)
- "Self confidence tied to competence in work role" -> good
- "Professional Networking and Collaboration" -> bad
- "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others" -> good
- "Emotional regulation and self-compassion" -> bad
- "Evaluating thoughts to identify any cognitive distortions" -> good
- "Breathing and Behavioral Patterns" -> bad
- "Assessing impact of reframed/alternate thought after identifying cognitive distortion" -> good
- "Therapy and Reflection" -> bad
- "The client's struggle to articulate their feelings is a manifestation of the unexpressed emotions from their past." -> bad
- "Decision Making and Consequences" -> bad
- "Recognizing the role of thoughts and behaviors in emotional experiences" -> good
- "Reflection and Understanding" -> bad
- "Emotional Intimacy and Relationships" -> bad
- "Difficulty with focus and concentration due to trauma and ADHD-like symptoms" -> good
- "Tension between value of achievement and family obligations" -> good
- "Exploring the Impact of Surprises on Relationships" -> bad
- "The client's self-blame is a coping mechanism for feelings of overhelm and anxiety" -> bad

**Do not reuse, these theme examples**

**Transcript:**
{prompt_str}

**Listed Process:**
{formatted_data_process_data}

**Important:**
- If you cannot generate clinically meaningful themes from the input, **do not** return placeholders.
- Instead, try your best to infer psychologically plausible themes from the material.
- Do not generate themes that begin with "The client's" or "I." 
- Each theme should reflect a single, distinct aspect of psychological functioning or experience. Avoid embedding links, comparisons, or connections between different themes within a single theme label.
- Generate only as many themes as are necessary to capture the core psychological dynamics. Fewer, well-differentiated themes are better than many overlapping ones.

Now, generate a concise list of new, clinically meaningful themes, with no additional explanations or modifications to the provided data. Stick to the Guidelines and dont reuse the examples:
"""
    return prompt

