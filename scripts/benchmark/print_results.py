import argparse
from collections import Counter
import json
import re

import numpy as np


SUCCESS_EMOJIS = ["✅️✅️✅️✅️✅️✅️", "✅️"]

parser = argparse.ArgumentParser(description="Prints tactic benchmark results.")
parser.add_argument("results_files", nargs="+", help="The path to the results files.")
parser.add_argument("--plot", action="store_true", help="Plot results.")
parser.add_argument("--ablate", type=str, default=None, help="Ablate for a 'setting', assuming filenames are of the form hammer_k1[-setting].json.")
args = parser.parse_args()

results_files: list[str] = args.results_files

cumulative_solved_decls = set()
cumulative_all_decls = set()
decls_tested_by_all: set[str] | None = None  # some declarations (currently) aren't tested because e.g. they are `def`

counters: list[tuple[str, Counter, int]] = []

# first pass: set cumulative_all_decls and decls_tested_by_all
for results_file in results_files:
    this_cumulative_tested_decls = set()

    with open(results_file) as f:
        results: dict = json.load(f)

    for decl_name, entry in results.items():
        if "result_emoji" in entry and entry["result_emoji"] is not None:
            emoji = entry["result_emoji"]
        elif (match := re.search("^([💥️❌️✅️]+) ", entry.get("result_output", ""), flags=re.MULTILINE)):
            emoji = match.group(1)
        elif "TimeoutExpired" in entry.get("result_output", ""):
            emoji = "⏰"
        else:
            emoji = "not tested"

        if emoji != "not tested":
            this_cumulative_tested_decls.add(decl_name)
        cumulative_all_decls.add(decl_name)

    if decls_tested_by_all is None:
        decls_tested_by_all = this_cumulative_tested_decls
    else:
        decls_tested_by_all &= this_cumulative_tested_decls

for results_file in results_files:
    print(results_file)
    with open(results_file) as f:
        results: dict = json.load(f)

    emojis = []
    gains = set()
    losses = set()
    loss_emojis = []
    for decl_name, entry in results.items():
        # print(*entry.keys())
        # print(entry["result_output"])
        if "result_emoji" in entry and entry["result_emoji"] is not None:
            emoji = entry["result_emoji"]
        elif (match := re.search("^([💥️❌️✅️]+) ", entry.get("result_output", ""), flags=re.MULTILINE)):
            emoji = match.group(1)
        elif "TimeoutExpired" in entry.get("result_output", ""):
            emoji = "⏰"
        else:
            emoji = "not tested"

        if not decl_name in decls_tested_by_all:
            continue

        emojis.append(emoji)

        if emoji in SUCCESS_EMOJIS:
            if decl_name not in cumulative_solved_decls:
                cumulative_solved_decls.add(decl_name)
                gains.add(decl_name)
        elif decl_name in cumulative_solved_decls:
            losses.add(decl_name)
            loss_emojis.append(emoji)

    counter = Counter(emojis)

    num_solved = sum(counter[success_emoji] for success_emoji in SUCCESS_EMOJIS)
    num_decls = counter.total()
    num_tested = num_decls - counter["not tested"]
    num_without_bugs = num_tested - counter["💥️💥️💥️💥️💥️💥️"] - counter["⏰"]

    print(counter)
    # print(f"Number of solved theorems: {num_solved:4d} ({num_solved / num_decls:.4%})")
    print(f"Number of solved theorems: {num_solved} ({num_solved / num_tested:.4%} out of {num_tested} tested)")
    # print(f"Percentage solved (out of runs without bugs): {num_solved / num_without_bugs:.4%}")
    print(f"Gains: {len(gains):4d}, losses: {len(losses):4d} ({Counter(loss_emojis)})")
    print(f"Current cumulative solved theorems: {len(cumulative_solved_decls):4d}")
    print()

    counters.append((results_file, counter, len(cumulative_solved_decls)))

print(f"Total number of theorems: {len(cumulative_all_decls)}")
print(f"Cumulative percentage solved: {len(cumulative_solved_decls) / len(decls_tested_by_all):.4%} (out of {len(decls_tested_by_all)} tested)")  # type: ignore

if args.ablate is not None:
    counters_map = {results_file: counter for (results_file, counter, _) in counters}
    num_solved_default = []
    num_solved_with_setting = []
    diffs = []
    setting_string = f"-{args.ablate}"
    for results_file, counter, _ in counters:
        if setting_string in results_file and (results_file_without_setting := results_file.replace(setting_string, "", 1)) in counters_map:
            counter_without_setting = counters_map[results_file_without_setting]
            num_solved_with_setting.append(sum(counter[success_emoji] for success_emoji in SUCCESS_EMOJIS))
            num_solved_default.append(sum(counter_without_setting[success_emoji] for success_emoji in SUCCESS_EMOJIS))
            diffs.append(num_solved_with_setting[-1] - num_solved_default[-1])
    print(f"Average performance increase by using {args.ablate}: {np.mean(diffs)} ± {np.std(diffs)}")

if args.plot:
    import matplotlib.pyplot as plt
    import numpy as np

    # Data parsed from the results
    # k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    result_emojis = {
        "✅️": ("Solved", "#008000"),
        "❌️": ("Failed", "#5F9EA0"),
        "💥️": ("Uncaught error", "#800080"),

        "✅️✅️✅️✅️✅️✅️": ("Solved", "#44ce1b"),
        "✅️✅️✅️✅️💥️❌️": ("Reconstruction error", "#bbdb44"),
        "✅️✅️✅️💥️❌️❌️": ("Zipperposition fail", "#f7e379"),
        "✅️✅️💥️❌️❌️❌️": ("TPTP error", "#f2a134"),
        "💥️💥️💥️💥️💥️💥️": ("Uncaught error", "#800080"),
        "⏰": ("Uncaught timeout", "#9e9e9e"),
        "✅️💥️❌️❌️❌️❌️": ("Preprocessing error", "#e51f1f"),
        "not tested": ("N/A", "#607d8b"),
    }

    # Preparing the bar positions and labels
    y_pos = np.arange(len(counters))[::-1]
    bar_width = 0.85

    # Stacking the bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    cumulative = np.zeros(len(counters))

    for i, (result_emoji, (label, color)) in enumerate(result_emojis.items()):
        counts = [counter[result_emoji] for _, counter, _ in counters]
        if result_emoji == "not tested":
            continue
        # if result_emoji != "✅️💥️❌️❌️❌️❌️":
        #     continue
        ax.barh(
            y_pos,
            counts,
            left=cumulative,
            height=bar_width,
            label=label,
            color=color,
        )
        cumulative += counts

    # plot cumulative pass rate
    cumulative_pass = [cumulative for _, _, cumulative in counters]
    ax.plot(cumulative_pass, y_pos)

    # Labels and styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        result_file.split("/")[-1].split(".")[0].replace("exact", "exact?")
        for result_file, _, _ in counters
    ])
    ax.set_xlabel("Number of Theorems")
    ax.set_title("Theorem Proving Results by k Value")
    ax.legend(title="Result Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Show the plot
    plt.show()
