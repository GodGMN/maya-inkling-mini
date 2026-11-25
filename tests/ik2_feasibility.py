"""Paired coding questions about missing information. Each feasible case restores the missing data."""
import argparse
import json
import random
import re

from ik2_kv_quality import completion


PAIRS = (
    ("A function receives only the ASCII lowercase result of converting an arbitrary mixed-case ASCII string. "
     "Recover the exact original, including its case.",
     "A function receives the ASCII lowercase result and one original-uppercase bit per character. "
     "Recover the exact original ASCII string."),
    ("A function receives the sorted values of an arbitrary list of distinct integers. Restore their original order.",
     "Each sorted integer is accompanied by its original zero-based list index. Restore their original order."),
    ("A function receives only the rounded integer obtained by rounding an arbitrary real number. Recover its exact value.",
     "A function receives two integers p and nonzero q representing a rational number. Return its exact value as a fraction."),
    ("A function receives only the number and sum of three arbitrary integers. Return all three original values in order.",
     "A function receives the sum of three integers plus the first two in order. Return all three in order."),
    ("Each pixel in an arbitrary color image was replaced by its average RGB intensity. Recover every original RGB triple exactly.",
     "An RGB image was serialized losslessly with its shape, exact bytes, and channel order. Recover every original RGB triple exactly."),
    ("Only the set of distinct strings from a list survives. Reconstruct the exact number of times each string occurred.",
     "The surviving dictionary contains every distinct string from a list and its exact count. Recover the multiplicity of every string."),
    ("Every record in a table has a first name only. Recover the original last name of every person with certainty, using just the table.",
     "Every record contains a person ID, and a supplied complete lookup table maps each ID to its original last name. Recover them all."),
    ("A text file was truncated after byte 100. Only those 100 bytes remain. Recover an arbitrary discarded suffix exactly.",
     "A text file was split after byte 100. Both byte sequences survive in order. Recover its original bytes exactly."),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reasoning-effort", choices=("none", "medium"), default="none")
    args = parser.parse_args()
    cases = [(i, bool(j), prompt) for i,pair in enumerate(PAIRS) for j,prompt in enumerate(pair)]
    random.Random(1429).shuffle(cases)
    for i, feasible, prompt in cases:
        question = ("Can a program always meet this specification using only the inputs described? "
                    "Start with Yes or No, then justify your answer in one sentence.\n\n" + prompt)
        limit = 160 if args.reasoning_effort == "none" else 2048
        result = completion(args.url, [dict(role="user", content=question)], limit, args.reasoning_effort)
        label = re.match(r"(yes|no)\b", result["answer"].strip().strip("*"), flags=re.I)
        expected = "yes" if feasible else "no"
        print(json.dumps(dict(case="feasibility", pair=i, expected=expected, correct=bool(label and
                         label[1].lower() == expected), **result)), flush=True)


if __name__ == "__main__":
    main()
