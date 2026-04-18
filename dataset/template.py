PROMPT_TEMPLATE = """You are an autonomous driving scene analyst for a cultural driving benchmark.

Inputs:
1. Three front-view frames from the ego vehicle in chronological order (1 Hz sampling). The last frame is the current frame.
2. Detected objects with bounding boxes (JSON): <objects>
Note: Bounding boxes are automatically generated and may contain errors. Use them only as rough references and rely mainly on the image.
3. Country where the scene occurs: <country>
4. Traffic handbook for that country (rules indexed as S1, S2, S3...): <handbook>

Rules:
- Use only the traffic rules from the provided handbook.
- Do NOT invent rules.
- All reasoning must be consistent with the handbook.

Tasks:

Step1 Scene Description:
Briefly describe the scene including environment, road layout, and important infrastructure (signals, crosswalks, lane markings, signs).

Step2 Dynamic Objects:
Identify important traffic participants and describe their relative positions and likely short-term behavior.

Step3 QA Generation:

Generate four QA pairs with categories:
perception, prediction, planning, regional.

General requirements:
- Each QA must use a different question_type: yes/no, true/false, multiple_choice, one-sentence explanation.
- Questions must depend on BOTH the image and the country-specific traffic rules.
- The answer should require looking at the image and applying the local traffic rules.
- Do NOT mention the country name or rule IDs in the question text.
- Questions must NOT reveal key scene facts. For example, avoid: "What should the ego vehicle do when pedestrians are already crossing?" Instead ask: "What should the ego vehicle do before entering this crossing area?"

Category requirements:

Perception:
Ask about visible objects or infrastructure that may reflect country-specific road design or traffic control.

Prediction:
Ask about the likely short-term behavior of a traffic participant, where the reasoning depends on the scene and local rules.

Planning:
Ask about the correct action of the ego vehicle according to the traffic rules in this road environment.

Regional:
Test region-specific driving conventions or infrastructure.  
The question must start with "In this country" or "Which country".

Each QA must include:
- question
- answer
- options (only for multiple_choice)
- question_type
- question_category
- rule_reference (list of handbook rule IDs)

Return the result in JSON format only.
{
    "Scene Description": "",
    "Dynamic Objects": "",
    "QA": [],
}
"""

qa_generation_prompt = """
You are an autonomous driving QA generator.

========================
INPUT
========================
1. Three front-view frames (chronological, 2 Hz; last = current)
2. Country: <country>
3. Traffic handbook (rules S1, S2, ...): <handbook>
4. Structured scene state: <state>

========================
GOAL
========================
Generate EXACTLY 4 QA pairs:
- perception
- prediction
- planning
- region

Each QA MUST be culturally discriminative:
→ The correct answer must vary across countries.

========================
GLOBAL REQUIREMENTS
========================
1. Vision + Rule Binding
   - Each QA MUST require BOTH:
     (a) visual evidence
     (b) traffic rules

2. No Leakage
   - Do NOT reveal key scene conditions in the question
   - Do NOT mention country names or rule IDs

3. Cultural Validity (CRITICAL)
   - If answers are identical across all countries → INVALID

4. Legal Reasoning (ENCOURAGED)
   - Questions MAY involve legality (legal / illegal / allowed / prohibited)
   - Must still depend on scene + local rules

========================
CATEGORY DEFINITIONS
========================
Perception:
- Ask the attribute (color, type, ...) of some strong regional_clues (e.g., taxi color, license plate, signage, symbols)
- Avoid generic or globally shared features

Prediction:
- Target: OTHER agent (NOT ego)
- Ask short-term behavior
- Must depend on scene + local rules
- May incorporate legality to influence behavior prediction

Planning:
- Target: EGO action
- Must involve a concrete maneuver (e.g., lane change, turn)
- Legal/illegal framing is strongly encouraged
- Must depend on country-specific rules

Region:
- Identify the country
- MUST start with:
  "Which country" OR "In which country"

========================
OPTIONS
========================
Each QA must include 4 options:
A. Correct in THIS country
B. Correct in a DIFFERENT country
C. Common but WRONG heuristic
D. Clearly incorrect

Note that you cannot assume any element which cannot be observerd in the scene!

========================
QUESTION QUALITY RULES
========================
GOOD:
- Requires visual reasoning
- Does NOT expose key conditions
- Answer changes across countries

BAD:
- Reveals scene conditions explicitly
- Pure rule recall (no visual grounding)
- Universal answers (same globally)

FORBIDDEN PHRASING (leakage examples):
- "when facing a green light"
- "given the traffic light state"
- "even if the light is ..."
- "while the signal is ..."

========================
COUNTERFACTUAL VERIFICATION
========================
For EACH QA, simulate answers for:
["CN", "US", "UK", "JP", "SG", "IND"]

Return:
- answers: country → option
- result:
    PASS → answers NOT identical
    FAIL → all identical

========================
OUTPUT FORMAT (STRICT JSON)
========================
{
  "QA": [
    {
      "question": "" (don't be too long),
      "answer": "A|B|C|D",
      "options": [],
      "question_type": "multiple_choice",
      "question_category": "perception|prediction|planning|region",
      "rule_reference": [],
      "counterfactual_verification": {
        "answers": {
          "CN": "A|B|C|D",
          "US": "A|B|C|D",
          "UK": "A|B|C|D",
          "JP": "A|B|C|D",
          "SG": "A|B|C|D",
          "IND": "A|B|C|D"
        },
        "result": "PASS|FAIL"
      },
      "explanation": ""
    }
  ]
}
"""

state_extraction_prompt = """You are an autonomous driving scene parser.

Inputs:
1. Three front-view frames from the ego vehicle in chronological order (2 Hz sampling). The last frame is the current frame.
2. Detected objects with bounding boxes (JSON): <objects>
3. Country where the scene occurs: <country>
4. Traffic handbook for that country (rules indexed as S1, S2, S3...): <handbook>

Important:
- Bounding boxes may be inaccurate; use them only as rough references.
- Rely primarily on visual evidence from the images.
- You may read the handbook but MUST NOT apply or infer traffic rules.
- This step is ONLY for extracting observable scene state.
- Do NOT reason about right-of-way, legality, or decisions.

Task:
Extract a structured scene state from the current frame.

Schema:

1. road_layout:
- intersection_type: ["signalized_intersection","unsignalized_intersection","non_intersection"]
- road_type: ["single_lane_road","multi_lane_road","intersection","merge","roundabout","unknown"]
- lane_count: integer
- has_turn_lanes: boolean
- has_median: boolean
- has_crosswalk: boolean
- has_stop_line: boolean

2. traffic_controls:
All controls MUST include spatial reference and affected target.

- traffic_lights: list of {
    "state": ["red","yellow","green","unknown"],
    "controls": ["ego_lane"],
    "relative_position": ["front"]
}

- lane_markings: list of {
    "type": ["solid","dashed","double_solid","double_dashed","unknown"],
    "relative_position": ["ego_lane_center","left_of_ego","right_of_ego","intersection_area","unknown"],
    "function": ["lane_divider","lane_boundary","no_lane_change","guide","unknown"]
}

- lane_arrows: list of {
    "direction": ["left","right","straight","left_right","unknown"],
    "applies_to": ["ego_lane","left_lane","right_lane","unknown"],
    "relative_position": ["front","front_left","front_right","unknown"]
}

- crosswalks: list of {
    "relative_position": ["front","left","right","unknown"],
    "aligned_with": ["ego_path","cross_traffic","unknown"]
}

- signs: list of {
    "type": string,
    "relative_position": ["front","left","right","unknown"]
}

3. dynamic_agents:
For each important agent:
- category: ["car","truck","bus","motorcycle","bicycle","pedestrian","traffic_police","construction_worker","animal","other_vehicle"]
- refined bounding box: [x1, y1, x2, y2]
- relative_position: ["front","front_left","front_right","left","right","rear","unknown"]
- motion_state: ["moving","slowing","stopped","unknown"]
- lane_relation: ["same_lane","adjacent_lane","opposite_lane","off_road","unknown"]
- salient_state: ["near_crosswalk","near_curb","oncoming","ahead_of_ego","crossing_path","merging","turning","waiting","parked"]

Only include agents relevant to driving decisions.

4. interaction_cues:
- type: ["lead_vehicle","crossing_conflict","oncoming_conflict","cyclist_conflict","merging_conflict","none"]
- description: short phrase based ONLY on observable spatial relations

5. regional_clues:
- type: ["sign_shape","text_language","driving_side","lane_marking_style","traffic_light_style","road_infrastructure","vehicle_style","license_plate","agent_distribution","other"]
- description: short phrase

Guidelines:
- License_plate: describe visible color/format.
- Agent_distribution: include only if visually significant.
- Driving_side: based on vehicle/lane alignment.
- Do NOT infer country or rules.

Constraints:
- Use ONLY visible evidence.
- No rule application or reasoning.

Output (STRICT JSON ONLY):
{
  "road_layout": {},
  "traffic_controls": {},
  "dynamic_agents": [],
  "interaction_cues": [],
  "regional_clues": []
}
"""