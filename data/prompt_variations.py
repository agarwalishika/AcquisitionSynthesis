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
- IMPORTANT: Reasoning steps rotate through languages in this exact order: English, Spanish, French, Portuguese, Italian, Arabic — one language per step, cycling back if there are more steps than languages
- Answer includes just the final result

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap your question, reasoning, and answer in the following special tokens:
<question> Insert your (English) question here. </question>
<reasoning> Insert the thinking and general reasoning here (each step in the next language in the rotation: English → Spanish → French → Portuguese → Italian → Arabic → English → ...). </reasoning>
<answer> Insert your short (English) answer here. </answer>"""






STEM_PROMPT = lambda question, reasoning, answer: f"""Generate a NEW, ORIGINAL STEM multiple-choice question (MCQA) that is AS DIFFICULT as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (difficulty calibration only — do not reproduce):
<question>
{question}
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated question:
- Draws from STEM domains: physics, chemistry, biology, computer science, engineering, or mathematics
- Requires non-trivial conceptual or quantitative reasoning (no single-step lookups or trivial recall)
- Has exactly 4 answer choices labeled (A), (B), (C), (D) — only one is correct
- Distractors are plausible and reflect common misconceptions or near-miss reasoning errors
- Is self-contained, unambiguous, and precisely stated
- Reasoning walks through the correct derivation/justification step by step and explains why each distractor is wrong
- IMPORTANT: Reasoning steps rotate through languages in this exact order: English, Spanish, French, Portuguese, Italian, Arabic — one language per step, cycling back if there are more steps than languages
- Answer is the correct letter only, e.g. "(B)"

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap them in the following special tokens:
<question> Insert your (English) question stem followed by the four answer choices (A)–(D). </question>
<reasoning> Insert the step-by-step reasoning (each step in the next language in the rotation: English → Spanish → French → Portuguese → Italian → Arabic → English → ...), including why each distractor is incorrect. </reasoning>
<answer> Insert only the correct letter (English), e.g. "(A)". </answer>"""







CHAT_PROMPT = lambda question, reasoning, answer: f"""Generate a NEW, ORIGINAL instruction-following task that is AS COMPLEX as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (complexity calibration only — do not reproduce):
<question>
"{question}"
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated task:
- Instruction imposes at least as many explicit constraints as the reference (e.g. format, length, style, content restrictions, conditional logic)
- Constraints are specific and verifiable — a reader can check whether the response satisfies each one
- Draws from: open-ended knowledge tasks (Alpaca-style), conversational requests (LMArena-style), or format-constrained tasks (IFEval-style)
- Instruction is self-contained and unambiguous
- Reasoning walks through how each constraint is satisfied, step by step
- IMPORTANT: Reasoning steps rotate through languages in this exact order: English, Spanish, French, Portuguese, Italian, Arabic — one language per step, cycling back if there are more steps than languages
- Answer is a complete response that fully obeys every constraint in the instruction

IMPORTANT: generate a (question, answer) pair; wrap your question and answer in the following special tokens:
<question> Insert your instruction here. </question>
<reasoning> Insert the step-by-step reasoning (each step in the next language in the rotation: English → Spanish → French → Portuguese → Italian → Arabic → English → ...). </reasoning>
<answer> Insert the complete response that satisfies all constraints. </answer>"""
