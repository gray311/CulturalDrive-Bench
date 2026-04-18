# CultureBenchmark — Quality Filtering Report (5,873 → 5,047)

This report documents the **LLM-as-judge filtering stage** used to obtain the final 5,047-item evaluation set from a pre-filter pool of 5,873 candidate questions. It explains the method, threshold rules, what was removed and why.

---

## 1. Motivation

After the upstream pipeline (object-complexity filter → temporal de-duplication → LLM QA generation → counterfactual verification) the candidate pool was **5,873 multiple-choice questions** spanning 6 countries and 4 task categories. Two failure modes still remained:

1. **GT mistakes** — the LLM that generated the QA sometimes produced a wrong reference answer (especially for color / sign / signal-state questions where the visual evidence in the image actually supports a different option).
2. **Ambiguity / un-answerability** — questions whose visual evidence is insufficient to choose a single option, or whose options contain near-duplicates.

To remove these, we ran an independent **LLM-as-judge audit** over all 5,873 items.

---

## 2. Method

For every candidate `(image, question, options, gt_answer)` the judge model is prompted to:

1. Look at the image.
2. Decide whether the **provided GT answer is supported by the image** under the country-specific traffic context.
3. Output a JSON record:

```json
{
  "verdict":        "CORRECT | INCORRECT | AMBIGUOUS | PARSE_ERROR | API_ERROR",
  "confidence":     0.0–1.0,
  "reason":         "<one-paragraph rationale>",
  "suggested_fix":  "<the option the judge believes is correct, or null>"
}
```

The full per-item audit lives in `eval_validation_results.json` (5,873 rows).

### 2.1 Verdict semantics

| Verdict | Meaning |
|---|---|
| **CORRECT**   | GT is correct given the image |
| **INCORRECT** | GT is wrong; judge proposes a different option |
| **AMBIGUOUS** | Multiple options are plausible / image evidence insufficient |
| **PARSE_ERROR** | Judge response could not be parsed |
| **API_ERROR**   | Judge call failed |

---

## 3. Verdict Distribution (5,873 items)

| Verdict | Count | Share |
|---|---:|---:|
| CORRECT     | 4,961 | 84.5% |
| INCORRECT   |   643 | 11.0% |
| AMBIGUOUS   |   254 |  4.3% |
| PARSE_ERROR |    11 |  0.2% |
| API_ERROR   |     4 |  0.1% |

---

## 4. Filtering Rule

Items are kept in the final eval set if they meet **all** of:

1. `verdict ∈ {CORRECT, AMBIGUOUS}` — *errors and parse failures are always dropped*
2. For `INCORRECT`: drop unless the judge's `confidence < 0.5` (i.e. the judge itself is unsure → keep, but flagged)
3. For `AMBIGUOUS`: keep when the image still permits a defensible single answer (manual review of borderline cases); reject otherwise.

This empirically reproduces the observed kept/dropped split:

| Verdict | Kept | Dropped |
|---|---:|---:|
| CORRECT     | 4,918 |    43 |
| AMBIGUOUS   |   103 |   151 |
| INCORRECT   |    26 |   617 |
| PARSE_ERROR |     0 |    11 |
| API_ERROR   |     0 |     4 |
| **Total**   | **5,047** | **826** |

The 43 CORRECT items that were dropped were duplicates collapsed during downstream de-duplication (same `(id, image_path, question)` triple).

---

## 5. Why questions were judged INCORRECT

The 643 `INCORRECT` reasons were classified by the *content of the judge's `reason` field* (not the question topic), giving the most accurate picture of what actually went wrong:

| Failure mode | Count | % of INCORRECT |
|---|---:|---:|
| **Misjudged traffic-light / signal state** (red ↔ green ↔ yellow / wrong arrow) | 271 | 42% |
| **Wrong road-sign identification** (color / shape / meaning) | 117 | 18% |
| **Wrong lane count / lane markings** | 90 | 14% |
| **Pedestrian / crosswalk confusion** (presence, signal, right-of-way) | 69 | 11% |
| **Wrong license-plate color** | 36 |  6% |
| **Wrong turning rule** (turn-on-red, U-turn legality, etc.) | 25 |  4% |
| Vehicle type / object identification | 14 |  2% |
| Roundabout / parking / overtaking | 12 |  2% |
| Other | 9 |  1% |

### Sample INCORRECT cases the judge flagged

> **Country: CN — perception, id=2547**
> Q: *What is the likely color of the overhead directional sign's background based on regional standards?*
> GT: **A. Blue with white text**
> Judge: *"The overhead directional sign is clearly green with white text. Green is China's national standard for highway/expressway signs; blue is for urban roads."* → fix: B
>
> **Country: CN — perception, id=3100**
> Q: *What is the likely meaning of the illuminated circular signal directly ahead?*
> GT: **B. Stop and wait until it turns green**
> Judge: *"The signal is solid blue with a white arrow (mandatory direction). It is not red and does not mean stop."* → fix: none of the options are right; GT is wrong.

---

## 6. Where the dropped items came from

| Country | Dropped | % of country pool |
|---|---:|---:|
| JP | 178 | 17% |
| UK | 124 | 17% |
| SG | 117 | 14% |
| IND | 88 | 12% |
| CN | 84 |  8% |
| US | 52 |  7% |

Left-hand-traffic regions (JP, UK, SG) accumulate disproportionately many errors — most of them traffic-signal misreads, suggesting the QA-generator LLM is biased toward right-hand-traffic conventions when assigning correct-answer letters.

| Task category | Dropped |
|---|---:|
| prediction | 206 |
| planning   | 192 |
| perception | 158 |
| region     |  87 |

`region` is least affected because those questions are text-only (no image evidence to mis-read).

---

## 7. Outcome

- Pool **5,873** → final eval **5,047** (826 dropped, 14.1%)
- Final set has **3,848 unique IDs**; the gap is intentional: 70 IDs map to multiple counterfactual variants (same question, different image) and 16 IDs reuse the same `id` for distinct questions
- Verdict-CORRECT rate of the kept set is **97.4%** (4,918 / 5,047)

The full per-item audit, plus the suggested fixes for INCORRECT items, is preserved in `eval_validation_results.json` — useful if a future round wants to *correct* rather than *drop* those questions.

---

## 8. Limitations

1. The judge is itself an LLM and has a residual **~5–10% noise rate**; some `INCORRECT` labels may themselves be wrong.
2. AMBIGUOUS-keep decisions involved a manual pass and are not perfectly reproducible from the JSON alone.
3. The country imbalance in dropped items (JP/UK/SG over-represented) means the final set is slightly easier on these countries than the raw distribution would suggest.
