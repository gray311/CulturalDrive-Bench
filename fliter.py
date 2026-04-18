import os
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
import boto3
import mimetypes


bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.environ["AWS_REGION"]
)

model_id = "qwen.qwen3-vl-235b-a22b"

REWRITE_PROMPT = """
You are rewriting questions for an autonomous driving benchmark.

Your goal:
Rewrite the question so that it requires visual perception and reasoning from the image.

You are given:
- Question
- Answer
- Options (may be empty)

--------------------------------
CRITICAL REQUIREMENT:

The rewritten question MUST be answerable by the ORIGINAL answer.

- If the original answer is "Yes/No", the rewritten question MUST remain a Yes/No question.
- If the original question is multiple-choice, the rewritten question MUST remain compatible with the same options.
- DO NOT change the answer type or format.
- DO NOT create a question that would require a different answer.

--------------------------------
CORE REQUIREMENTS:

1. The question MUST rely on visual evidence in the scene.
2. It may involve implicit regional reasoning.
3. It should not directly reveal the answer.

--------------------------------
STRICT RULES:

1. DO NOT include:
   - explicit signal states (e.g., "the light is red")
   - answer hints or conclusions
   - wording that directly reveals the answer

2. REMOVE rule-specific terminology such as:
   - "right-of-way"
   - "two-stage turn"
   - "must stop because of law"

3. The question MUST:
   - remain neutral
   - preserve the original intent
   - be concise and natural

4. If options are provided:
   - ensure the rewritten question aligns with the options
   - DO NOT leak the correct answer

--------------------------------
FEW-SHOT EXAMPLES:

Example 1:
Question: Is the motorcyclist required to perform a two-stage turn at this intersection?
Answer: No
Options: []
Rewritten: Does the motorcyclist need to follow any special turning behavior at this intersection?

Example 2:
Question: The traffic light is red; therefore, all vehicles must remain stopped until it turns green.
Answer: Yes
Options: []
Rewritten: Should the vehicles remain stationary at this moment?

Example 3:
Question: What should the ego vehicle do before entering the intersection when the light turns green?
Answer: Slow down and check for pedestrians
Options: []
Rewritten: What should the ego vehicle do before entering the intersection?

Example 4:
Question: In this country, which of the following is a distinctive feature of the STOP sign?
Answer: Red octagon
Options: ["Red octagon", "Blue circle", "Yellow triangle"]
Rewritten: In this country, which of the following best matches the STOP sign visible in the scene?

Example 5:
Question: Is there a visible pedestrian crossing or designated area where pedestrians have right-of-way in this scene?
Answer: Yes
Options: []
Rewritten: Is there a visible pedestrian crossing or designated pedestrian area in this scene?

--------------------------------
Now rewrite the following:

Question: {question}
Answer: {answer}
Options: {options}

Only output the rewritten question.
Rewritten:
""".strip()


def call_qwen(prompt: str, max_tokens: int = 512) -> str:
    response = bedrock.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.2,
            "topP": 0.9
        }
    )

    content = response["output"]["message"]["content"]
    texts = [block["text"] for block in content if "text" in block]
    return "\n".join(texts).strip()





if __name__ == "__main__":

    with open("./traffic_handbook/CultureDrive/CultureDrive_Benchmark_Mini.json", "r") as f:
        data = json.load(f)

    print(data.keys())

    from tqdm import tqdm

    for country in tqdm(data.keys(), desc="Countries"):
        for i, line in tqdm(enumerate(data[country]), total=len(data[country])):
            raw_question = line["question"]
            prompt = REWRITE_PROMPT.format(
                question=line['question'],
                answer=line['answer'],
                options=json.dumps(line['options'], ensure_ascii=False)
            )

            output = call_qwen(prompt)

            output = output.strip().split("\n")[0].strip()
            output = output.removeprefix("Rewritten:").strip()


            data[country][i]['raw_question'] = raw_question
            data[country][i]['question'] = output



    with open("./traffic_handbook/CultureDrive/CultureDrive_Benchmark_Mini_v1.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

