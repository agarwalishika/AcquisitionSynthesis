BASIC = lambda question, answer: f"""You are generating training data for a math reasoner model. Create a new, diverse sample in the same JSON format. Output only valid JSON.
Example 1: 
{{
    "question": "{question}",
    "answer": "{answer}"
}}

{{
    "question": "<enter question here>",
    "answer": "<enter the correct answer here>"
}}
"""

MULTIPLE_ICL = lambda question, answer:f"""You are generating training data for a math reasoner model. Create a new, diverse sample in the same JSON format. Output only valid JSON.

Example 1:
{{
    "question": "{question[0]}",
    "answer": "{answer[0]}"
}}

Example 2:
{{
    "question": "{question[1]}",
    "answer": "{answer[1]}"
}}

Example 3:
{{
    "question": "{question[2]}",
    "answer": "{answer[2]}"
}}

Now generate a new sample that is different from the examples above:
{{
    "question": "<enter question here>",
    "answer": "<enter the correct answer here>"
}}
"""

DETAILED_INSTRUCTION = lambda question, answer: f"""You are generating training data for a mathematical reasoning model. Problems should require multi-step reasoning across domains such as algebra, number theory, geometry, combinatorics, and probability. The answer must include a complete chain-of-thought derivation, not just a final numerical result.

Create a new, diverse sample in the same JSON format. Output only valid JSON.

Example:
{{
    "question": "{question}",
    "answer": "{answer}"
}}

Generate a novel math problem that requires at least 2 logical steps to solve. The answer should walk through the reasoning before stating the final result.
{{
    "question": "<enter question here>",
    "answer": "<step-by-step reasoning leading to the correct answer>"
}}
"""

HTML_TAGS = lambda question, answer: f"""You are generating training data for a math reasoner model. Create a new, diverse sample using the HTML tag format shown below. Output only the tags, no other text.

Example:
<sample>
  <question>{question}</question>
  <answer>{answer}</answer>
</sample>

Now generate a new, original math problem:
<sample>
  <question><!-- enter question here --></question>
  <answer><!-- enter the correct answer here --></answer>
</sample>
"""

NEG_POS_EXAMPLE = lambda question, answer: f"""You are generating training data for a math reasoner model. The model needs problems that require genuine multi-step mathematical reasoning.

Here is a GOOD example (detailed reasoning, non-trivial):
{{
    "question": "{question}",
    "answer": "{answer}"
}}

Here is a BAD example (too simple, no reasoning shown):
{{
    "question": "What is 2 + 2?",
    "answer": "4"
}}

Generate a new sample that resembles the GOOD example: a challenging problem with a thorough step-by-step solution. Output only valid JSON.
{{
    "question": "<enter question here>",
    "answer": "<enter detailed step-by-step solution here>"
}}
"""

PERSONA_ADOPTED = lambda question, answer: f"""You are a math professor writing practice problems for a graduate-level problem set. Each problem should be self-contained, precisely worded, and solvable with a clear logical argument. Avoid trivial arithmetic.

Example:
{{
    "question": "{question}",
    "answer": "{answer}"
}}

Write one new problem and its complete solution. Output only valid JSON.
{{
    "question": "<enter question here>",
    "answer": "<enter the correct answer here>"
}}
"""

COMBINED_METAMATH = lambda question, answer: f"""You are a math professor generating training data for a mathematical reasoning model. Problems should be self-contained, precisely worded, and require multi-step reasoning across domains such as algebra, number theory, geometry, combinatorics, and probability. Avoid trivial arithmetic. The answer must include a complete chain-of-thought derivation, not just a final numerical result.

Here is a GOOD example (detailed reasoning, non-trivial):
{{
    "question": "{question[0]}",
    "answer": "{answer[0]}"
}}

Here is another GOOD example (detailed reasoning, non-trivial):
{{
    "question": "{question[1]}",
    "answer": "{answer[1]}"
}}

Here is a BAD example (too simple, no reasoning shown):
{{
    "question": "What is 2 + 2?",
    "answer": "4"
}}

Here is another BAD example (not math related):
{{
    "question": "Which state is the fastest growing?",
    "answer": "Texas"
}}

Generate a new, original math problem that resembles the GOOD example: a challenging problem requiring at least 2 logical steps with a thorough step-by-step solution. Output only valid JSON.
{{
    "question": "<enter question here>",
    "answer": "<step-by-step reasoning leading to the correct answer>"
}}
"""

MATH_PROMPT = lambda question, reasoning, answer: f"""Generate a NEW, ORIGINAL math problem that is AS DIFFICULT than the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (difficulty calibration only — do not reproduce):
<question>
"{question}"
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated problem:
- Requires non-trivial reasoning steps (no single-step shortcuts)
- Draws from: number theory, combinatorics, algebra, geometry, or probability
- Is self-contained and precisely stated
- Reasoning includes a complete step-by-step derivation
- Answer includes just the final result

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap your question, reasoning, and answer in the following special tokens:
<question> Insert your question here. </question>
<reasoning> Insert the thinking and general reasoning here. </reasoning>
<answer> Insert your short answer here. </answer>"""

MEDICAL_PROMPT = lambda question, reasoning, answer: f"""Generate a NEW, ORIGINAL medical multiple-choice question that is AS DIFFICULT as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (difficulty calibration only — do not reproduce):
<question>
"{question}"
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated problem:
- Requires non-trivial clinical or biomedical reasoning (no single-step lookups)
- Draws from: pharmacology, pathophysiology, clinical medicine, anatomy, or biochemistry
- Includes exactly 4 answer choices labeled A, B, C, D — only one is correct
- Plausible distractors that reflect common misconceptions or near-miss diagnoses
- Is self-contained and precisely stated
- Reasoning includes a complete step-by-step clinical justification for the correct choice and why distractors are wrong
- Answer includes the correct letter and the full text of the chosen option

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap your question, reasoning, and answer in the following special tokens:
<question> Insert your question and answer choices (A/B/C/D) here. </question>
<reasoning> Insert the step-by-step clinical reasoning here, including why each distractor is incorrect. </reasoning>
<answer> Insert the correct letter and its full answer text, e.g. "A: Metformin". </answer>"""

LMARENA_PROMPT = lambda question, answer: f"""Generate a NEW, ORIGINAL instruction-following task that is AS COMPLEX as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (complexity calibration only — do not reproduce):
<question>
"{question}"
</question>

<answer>
{answer}
</answer>

Requirements for your generated task:
- Instruction imposes at least as many explicit constraints as the reference (e.g. format, length, style, content restrictions, conditional logic)
- Constraints are specific and verifiable — a reader can check whether the response satisfies each one
- Draws from: open-ended knowledge tasks (Alpaca-style), conversational requests (LMArena-style), or format-constrained tasks (IFEval-style)
- Instruction is self-contained and unambiguous
- Reasoning walks through how each constraint is satisfied, step by step
- Answer is a complete response that fully obeys every constraint in the instruction

IMPORTANT: generate a (question, answer) pair; wrap your question and answer in the following special tokens:
<question> Insert your instruction here. </question>
<answer> Insert the complete response that satisfies all constraints. </answer>"""

COMBINED_ALPACA = lambda question, answer: f"""You are generating training data for an instruction-following model. Instructions should be clear, self-contained, and require a substantive, helpful response. The answer must directly and completely address the instruction.

Here is a GOOD example (clear instruction, complete and helpful response):
{{
    "question": "{question[0]}",
    "answer": "{answer[0]}"
}}

Here is another GOOD example (clear instruction, complete and helpful response):
{{
    "question": "{question[1]}",
    "answer": "{answer[1]}"
}}

Here is a BAD example (instruction is trivial, response is unhelpful):
{{
    "question": "Say hi.",
    "answer": "Hi."
}}

Here is another BAD example (response ignores the instruction):
{{
    "question": "Explain how photosynthesis works.",
    "answer": "I don't know."
}}

Generate a new, original instruction-response pair that resembles the GOOD examples: a clear, specific instruction with a thorough and directly useful response. Output only valid JSON.
{{
    "question": "<enter instruction here>",
    "answer": "<complete, helpful response that fully addresses the instruction>"
}}
"""

VERY_DETAILED_INSTRUCTION = lambda question, answer: f"""You are an expert training data generator.

You are given a list of examples in JSON format:
[
    {{"question": <example question>, "answer": <example answer including reasoning>}},
    ...
]

Your task is to generate exactly one new training example in the same JSON format:
{{"question": <generated question>, "answer": <generated answer with reasoning>}}

## Requirements

1. **Domain Consistency**
   The generated example must be in the same domain as the provided examples (e.g., math, logic, coding). Do not change domains.

2. **Semantic Similarity with Novelty**
   The new example must:
   - Target similar underlying skills and concepts as the provided examples
   - Be structurally and contextually different
   - Not reuse phrasing, numbers, entities, or surface patterns from the input

   Avoid trivial transformations such as:
   - Changing only numbers
   - Rewording the same scenario
   - Reusing the same problem template

   Instead, create a complementary problem that exercises the same skills in a meaningfully different way.

3. **Difficulty Matching (or Higher)**
   Let k be the average number of meaningful reasoning steps in the provided examples. A "meaningful reasoning step" is one that introduces new information or advances the solution.

   Your generated example must:
   - Require at least k meaningful reasoning steps
   - Avoid filler or redundant steps
   - Ensure each step contributes to solving the problem

4. **Reasoning Quality**
   The reasoning must:
   - Be clear, logically ordered, and complete
   - Reflect the actual steps needed to solve the problem
   - Not include irrelevant or repetitive statements

5. **Output Format (Strict)**
   Output only a valid JSON object with this structure:
   {{"question": ..., "answer": ..., "reasoning": ...}}
   Do not include any text outside the JSON.

Generate a training example for the following sample:
[{{"question": "{question}", "answer": "{answer}"}}]"""



COMBINED_DETAILED = lambda questions, answers: f"""You are an expert training data generator.
You will be given four examples:
- two GOOD examples that demonstrate the desired style
- two BAD examples that demonstrate what to avoid
Your task is to generate exactly ONE new training example.
Each example is in this format:
{{"question": <question>, "answer": <answer that includes reasoning>}}

You must generate exactly one new example in the same format:
{{"question": <generated question>, "answer": <answer with reasoning>}}

Requirements:

1. Domain consistency
The generated example must stay in the same domain as the GOOD examples (for example, math, logic, coding, science, legal reasoning, etc.).
Do not switch domains.

2. Skill consistency with novelty
The generated example must:
- test the same underlying skills and concepts as the GOOD examples
- be meaningfully different in structure, context, and wording
- avoid copying or lightly modifying the GOOD examples

Do NOT make a trivial variant such as:
- changing only names or numbers
- rewording the same scenario
- reusing the same template with superficial edits

Instead, create a new but comparable problem that exercises the same skills in a genuinely different way.

3. Difficulty matching or exceeding the GOOD examples
Infer the approximate reasoning depth required by the GOOD examples.
Your generated example must require at least as many meaningful reasoning steps.

A meaningful reasoning step is one that:
- introduces new information,
- makes a non-trivial inference, or
- advances the solution toward the final answer.

Do NOT inflate difficulty with filler, repetition, or unnecessary exposition.

4. Reasoning quality
The reasoning must be:
- clear
- logically ordered
- complete
- relevant to the question
- free of repetition and irrelevant filler

Every step in the reasoning must contribute to solving the problem.

5. Avoid BAD-example behavior
Your generated example must NOT resemble the BAD examples in the following ways:
- too easy
- answerable in one obvious step
- missing reasoning
- shallow, generic, or underspecified

6. Output format (strict)
Output exactly one valid JSON object with this structure:
{{"question": "...", "answer": "..."}}

Do not output:
- markdown
- code fences
- explanations
- commentary
- multiple JSON objects

GOOD examples:
1.
{{"question": "{questions[0]}", "answer": "{answers[0][0]}", "reasoning": "{answers[0][1]}"}}

2.
{{"question": "{questions[1]}", "answer": "{answers[1][0]}", "reasoning": "{answers[1][1]}"}}

BAD examples:
1.
{{"question": "What is 2 + 2?", "answer": "4", "reasoning": ""}}
This is bad because it is too simple and provides no reasoning.

2.
{{"question": "Which state is the fastest growing?", "answer": "Texas", "reasoning": ""}}
This is bad because it is shallow, underspecified, and does not require meaningful reasoning.

Now generate exactly one new example as a valid JSON object only in the format:
{{"question": <your generated question>, "answer": <your generated answer>}}.
"""