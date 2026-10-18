
import csv
import json
import os
import re
import time
import requests

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "sarcasm_500k.csv"
OUTPUT_JSONL = "sarcasm_500k_refined.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

# 100 rows constitute one outer batch.
BATCH_SIZE = 100

# Each model request handles 10 texts.
# Try 20, 25, or 50 after testing quality and memory usage.
REQUEST_SIZE = 10

# Change a label only when confidence meets this threshold.
CONFIDENCE_THRESHOLD = 0.80

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a careful text classification and dataset auditing assistant.

Your task is to audit sarcasm labels, not merely agree with them.

Label definitions:
0 = Not sarcastic: literal, sincere, neutral, or ordinary text.
1 = Sarcastic: intended irony, mock praise, ridicule, or a meaning
that contrasts with the literal wording.

Rules:
- Evaluate the meaning of each text independently.
- Do not assume every question, exclamation, or negative sentence
  is sarcastic.
- Do not assume positive wording is sincere.
- Consider textual clues, but do not invent missing context.
- Short, ambiguous, or context-dependent texts should receive
  low confidence.
- Compare your decision with the original label.
- Never change a label just to force a correction.
- Confidence must be a number between 0 and 1.
- Keep reasons short and explain the strongest textual evidence.
- Return valid JSON only, with no Markdown or extra commentary.

Required output format:
{
  "results": [
    {
      "id": 0,
      "predicted_label": 0,
      "confidence": 0.95,
      "reason": "The statement is literal and contains no clear irony."
    }
  ]
}

Return exactly one result for every input ID.
Do not omit IDs or invent IDs.
"""

# ============================================================
# HELPERS
# ============================================================

def parse_json_response(content):
    """Parse JSON, including responses surrounded by code fences."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    return json.loads(content)


def call_qwen(items):
    """Send one group of texts to the local Ollama model."""

    user_prompt = {
        "instructions": (
            "Audit every text. The original label is supplied "
            "for comparison, not as a hint to copy."
        ),
        "items": items,
    }

    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    user_prompt, ensure_ascii=False
                ),
            },
        ],
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            content = data["message"]["content"]
            result = parse_json_response(content)

            if isinstance(result, list):
                results = result
            else:
                results = result.get("results")

            if not isinstance(results, list):
                raise ValueError("Missing results array")

            # Validate completeness and prevent duplicate IDs.
            expected_ids = {item["id"] for item in items}
            by_id = {}

            for item in results:
                if not isinstance(item, dict):
                    raise ValueError("Invalid result item")

                row_id = item.get("id")

                if row_id not in expected_ids:
                    raise ValueError(f"Unexpected ID: {row_id}")

                if row_id in by_id:
                    raise ValueError(f"Duplicate ID: {row_id}")

                label = item.get("predicted_label")
                confidence = item.get("confidence")
                reason = item.get("reason")

                if (
                    isinstance(label, bool)
                    or label not in (0, 1)
                ):
                    raise ValueError(f"Invalid label for {row_id}")

                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not 0 <= confidence <= 1
                ):
                    raise ValueError(
                        f"Invalid confidence for {row_id}"
                    )

                if not isinstance(reason, str):
                    raise ValueError(f"Invalid reason for {row_id}")

                by_id[row_id] = item

            if set(by_id) != expected_ids:
                raise ValueError("Model omitted one or more IDs")

            return by_id

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            print(
                f"Request attempt {attempt + 1}/{MAX_RETRIES} "
                f"failed: {exc}"
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Qwen request failed after retries: {last_error}"
    )


def write_jsonl(file, record):
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


# ============================================================
# RESUMABLE PROCESSING
# ============================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. "
            "Place it beside this script."
        )

    # Read the source CSV without changing it.
    with open(INPUT_CSV, "r", encoding="utf-8-sig",
              newline="") as f:
        reader = csv.DictReader(f)

        required = {"text", "sarcasm_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV must contain text and sarcasm_label columns"
            )

        rows = []

        for row_index, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            raw_label = (row.get("sarcasm_label") or "").strip()

            if not text or raw_label not in ("0", "1"):
                rows.append({
                    "row_index": row_index,
                    "text": text,
                    "original_label": (
                        int(raw_label)
                        if raw_label in ("0", "1")
                        else None
                    ),
                    "invalid": True,
                })
            else:
                rows.append({
                    "row_index": row_index,
                    "text": text,
                    "original_label": int(raw_label),
                    "invalid": False,
                })

    total = len(rows)

    # Recover previously completed row indexes.
    # model_error records are deliberately retried on the next run.
    completed = set()

    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("status") in (
                        "confirmed",
                        "corrected",
                        "needs_review",
                        "invalid_input",
                    ):
                        completed.add(record["row_index"])
                except (json.JSONDecodeError, KeyError):
                    continue

    remaining = [r for r in rows if r["row_index"] not in completed]

    corrected_count = 0
    reviewed_count = 0
    processed_count = 0

    print(f"Total rows: {total:,}")
    print(f"Already completed: {len(completed):,}")
    print(f"Remaining: {len(remaining):,}")
    print(f"Outer batch size: {BATCH_SIZE}")
    print(f"Texts per model request: {REQUEST_SIZE}")

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as output:
        for outer_start in range(0, len(remaining), BATCH_SIZE):
            outer_batch = remaining[
                outer_start:outer_start + BATCH_SIZE
            ]

            print(
                f"\nBatch {outer_start // BATCH_SIZE + 1}: "
                f"{len(outer_batch)} rows"
            )

            for start in range(0, len(outer_batch), REQUEST_SIZE):
                group = outer_batch[start:start + REQUEST_SIZE]

                # Invalid rows are retained for inspection.
                valid_group = [r for r in group if not r["invalid"]]

                for r in group:
                    if r["invalid"]:
                        write_jsonl(output, {
                            "row_index": r["row_index"],
                            "text": r["text"],
                            "original_label": r["original_label"],
                            "reviewed_label": r["original_label"],
                            "status": "invalid_input",
                            "reason": "Empty text or invalid label",
                        })
                        reviewed_count += 1

                if not valid_group:
                    continue

                items = [
                    {
                        "id": r["row_index"],
                        "text": r["text"],
                        "original_label": r["original_label"],
                    }
                    for r in valid_group
                ]

                try:
                    predictions = call_qwen(items)

                    # Write each validated result immediately.
                    for r in valid_group:
                        prediction = predictions[r["row_index"]]
                        predicted = prediction["predicted_label"]
                        confidence = float(prediction["confidence"])
                        original = r["original_label"]

                        # Low confidence means keep the original
                        # and flag the row instead of guessing.
                        if confidence < CONFIDENCE_THRESHOLD:
                            reviewed_label = original
                            status = "needs_review"

                        elif predicted != original:
                            reviewed_label = predicted
                            status = "corrected"
                            corrected_count += 1

                        else:
                            reviewed_label = original
                            status = "confirmed"

                        write_jsonl(output, {
                            "row_index": r["row_index"],
                            "text": r["text"],
                            "original_label": original,
                            "model_label": predicted,
                            "reviewed_label": reviewed_label,
                            "changed": reviewed_label != original,
                            "confidence": confidence,
                            "reason": prediction["reason"],
                            "status": status,
                        })

                        processed_count += 1

                except RuntimeError as exc:
                    # Do not mark errors as completed. The next run
                    # will retry these rows.
                    print(f"Batch request failed: {exc}")

                    for r in valid_group:
                        write_jsonl(output, {
                            "row_index": r["row_index"],
                            "text": r["text"],
                            "original_label": r["original_label"],
                            "status": "model_error",
                            "reason": str(exc),
                        })

                if processed_count and processed_count % 100 == 0:
                    print(
                        f"Reviewed this run: {processed_count:,} | "
                        f"Label changes: {corrected_count:,} | "
                        f"Need review: {reviewed_count:,}"
                    )

    print("\nProcessing pass finished.")
    print(f"Rows in source: {total:,}")
    print(f"Successfully reviewed this run: {processed_count:,}")
    print(f"Labels changed this run: {corrected_count:,}")
    print(f"Output: {OUTPUT_JSONL}")
    print(
        "Rerun the script to retry failed requests and continue "
        "processing unfinished rows."
    )


if __name__ == "__main__":
    main()
