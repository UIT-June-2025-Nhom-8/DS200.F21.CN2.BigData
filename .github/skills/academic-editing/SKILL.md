---
name: academic-editing
description: 'Edit and improve academic writing for the DS200 paper. Use for: proofread, fix grammar, improve clarity, strengthen argument, rewrite sentence, academic English, paraphrase, check structure, tighten abstract, fix passive voice, improve transitions, refine contribution statement, respond to reviewer, revise manuscript.'
argument-hint: "What to edit — e.g. 'Proofread the abstract', 'Rewrite this paragraph more formally', 'Respond to reviewer: the dataset is too small'"
---

# Academic Editing

## When to Use

- Proofreading any section for grammar, spelling, and punctuation
- Improving clarity, concision, or academic register of existing text
- Reviewing section structure for scope violations
- Refining the contribution statement for parallelism and specificity
- Drafting structured responses to reviewer comments

## Constraints

- **All edited output must be in Vietnamese.** When the user provides Vietnamese text, edit it in Vietnamese. When the user provides English text, translate and edit it into academic Vietnamese.
- Never add citations not in the project's reference list
- Never invent claims, results, or numbers
- Work one section per invocation unless a full pass is explicitly requested
- Only edit files under `reports/`

## Procedure

### 1. Read Full Context

Read the entire section — not just the highlighted paragraph — before suggesting edits, to maintain coherence.

### 2. Identify Issues

Scan for: grammar errors, passive overuse, vague quantifiers, structural scope violations, missing transitions, weak contribution bullets.

### 3. Apply Editing Mode

**Proofreading** — grammar, spelling, punctuation, subject-verb agreement. Flag ambiguous sentences rather than auto-fixing.

**Clarity & Concision:**

- Remove redundant openers: "In this paper, we..." → drop the opener
- Prefer active voice
- Break sentences >35 words
- Replace "many/several/a lot" with specific numbers from the data

**Academic Register:**

| Informal                        | Academic                                            |
| ------------------------------- | --------------------------------------------------- |
| "We used a big dataset"         | "We curated a large-scale dataset comprising..."    |
| "The model works well"          | "The model achieves competitive performance on..."  |
| "We tried different settings"   | "We conducted a systematic ablation study..."       |
| "As we can see from the figure" | "As illustrated in Figure X,..."                    |
| "Our method is better"          | "Our approach outperforms the baseline by X% on..." |

**Structural Review — per section:**

- **Abstract**: problem → method → results → conclusion. ≤250 words. No citations.
- **Introduction**: hook → gap → contributions. No results data.
- **Related Work**: theme-grouped; each paragraph ends with differentiation statement.
- **Methodology**: no results; past tense for experiments, present for model description.
- **Experiments**: every table has a caption + in-text pointer; every figure cited in text.
- **Discussion**: state limitations explicitly — do not hide weaknesses.
- **Conclusion**: no new information, no new citations.

**Contribution Statement** — enforce parallel structure:

```
We make the following contributions:
(i) We propose [X], which [does Y].
(ii) We conduct [Z] to demonstrate [outcome].
(iii) We release [artifact] to support [goal].
```

**Reviewer Response:**

```
**Reviewer Comment:** [quote]

**Response:** We thank the reviewer for this observation.
[Acknowledge] → [Explain what was done or defend the original] → [Pointer: "See Section X, paragraph Y."]
```

### 4. Output Format

```
### [Section Name] — Edit Pass

**Issues found:**
- [issue 1]
- [issue 2]

**Revised text:**
> [new version]

**Changes explained:**
- [Change]: [reason]
```

Always show before/after. Never silently overwrite without a diff.
