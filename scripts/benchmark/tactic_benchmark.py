from typing import Any, Optional, Literal
import json
import subprocess
import os
import shutil
import re
import concurrent.futures
import uuid

import tqdm
import argparse

SCRATCH_DIR = "/data/user_data/thomaszh/tmp-premises"

parser = argparse.ArgumentParser(description="Run tactic benchmark with premises.")
parser.add_argument("--ntp_toolkit_path", type=str, default="/home/thomaszh/ntp-toolkit", help="Path to the ntp-toolkit repository containing a tactic_benchmark script.")
parser.add_argument("--decl_names_file", type=str, default="eval_decls_jan16.json", help="File containing declaration names for benchmark.")
parser.add_argument("--premises_file", type=str, default=None, help="File containing retrieved premises (default: use ground truth premises).")
parser.add_argument("--out_dir", type=str, default="results", help="Output directory for results.")
parser.add_argument("--timeout", type=int, default=300, help="Timeout for each benchmark run in seconds.")
parser.add_argument("--benchmark_type", type=str, default="simp_all_with_premises", help="Type of benchmark to run.")
parser.add_argument("--k", type=int, default=8, help="Number of top premises to use.")
parser.add_argument("--max_workers", type=int, default=8, help="Number of workers for running the benchmark.")
parser.add_argument("--rerank", action="store_true", help="Use reranked premises.")
parser.add_argument("--pred_simp_all_hint", action="store_true", help="Use prediction of simp_all_hint (otherwise use notInSimpAll).")
parser.add_argument("--temp_premises_dir", type=str, default=None, help="A temporary directory that simulates Examples/Mathlib/TrainingDataWithPremises, if premises_file is given")
parser.add_argument("--tag_suffix", type=str, default=None, help="Suffix to the output json file name")

args = parser.parse_args()

decl_names_for_benchmark_file: str = args.decl_names_file
premises_file: str = args.premises_file
out_dir: str = args.out_dir
timeout: int = args.timeout
k: int = args.k
benchmark_type: str = args.benchmark_type
max_workers: int = args.max_workers
ntp_toolkit_path: str = args.ntp_toolkit_path
rerank: bool = args.rerank
pred_simp_all_hint: bool = args.pred_simp_all_hint
temp_premises_dir: str = args.temp_premises_dir or os.path.join(SCRATCH_DIR, f"premises-{uuid.uuid4()}")

uses_premises = "hammer" in benchmark_type or "premise" in benchmark_type
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, (benchmark_type + f"_k{k}" if uses_premises else benchmark_type))
if rerank:
    out_file += "-rerank"
if pred_simp_all_hint:
    out_file += "-pred_simp_all_hint"
if args.tag_suffix is not None:
    out_file += f"-{args.tag_suffix}"
out_file += ".json"

with open(decl_names_for_benchmark_file) as f:
    decl_names_for_benchmark = json.load(f)

results = {d["decl_name"]: {} for d in decl_names_for_benchmark}

# Build `results` mapping declaration name to premises and hints
if premises_file is not None:
    with open(premises_file) as f:
        premises_raw = json.load(f)
        if isinstance(premises_raw, dict):
            premises_raw = premises_raw["dot"]
        # (before nov 20) for each decl, there are multiple states corresponding to the decl (now only one)
        # we assume the first state encountered in the file is the "root" initial state
        for ps_entry in premises_raw:
            decl_name = ps_entry["decl_name"]
            if "premises" not in results[decl_name]:
                premises = ps_entry["premises"]
                # take names of top k premises
                rank_key = "rerank_score" if rerank else "score"
                topk_premises = [p for p in sorted(premises, key=lambda p: p[rank_key], reverse=True)[:k]]
                results[decl_name]["premises"] = [p["corpus_id"] for p in topk_premises]
                results[decl_name]["hints"] = [p.get("simp_all_hint", "notInSimpAll") for p in topk_premises]
else:
    # Use ground truth premises
    for entry in decl_names_for_benchmark:
        decl_name = entry["decl_name"]
        premises = results[decl_name]["premises"] = entry["gt_premises"]
        results[decl_name]["hints"] = [entry["gt_hints"][p] for p in premises]

not_found_decl_names = []
for decl_name in results:
    if "premises" not in results[decl_name]:
        print(f"warning: premises for {decl_name} not found")
        not_found_decl_names.append(decl_name)
decl_names_for_benchmark = [e for e in decl_names_for_benchmark if e["decl_name"] not in not_found_decl_names]

# build Examples/Mathlib/TrainingDataWithPremises-like directory but with retrieved premises
shutil.rmtree(temp_premises_dir, ignore_errors=True)
os.makedirs(temp_premises_dir, exist_ok=True)
for entry in decl_names_for_benchmark:
    decl_name = entry["decl_name"]
    module = entry["module"]
    serialized_premises = []
    for premise, hint in zip(results[decl_name]["premises"], results[decl_name]["hints"]):
        if not pred_simp_all_hint:
            hint = "notInSimpAll"
        serialized_premises.append(f"({premise}, {hint})")
    with open(os.path.join(temp_premises_dir, f"{module}.jsonl"), "a") as f:
        json.dump({"declName": decl_name, "declHammerRecommendation": serialized_premises}, f)  # NOTE: upstream might change name for simp_all
        f.write("\n")

def run_benchmark(entry: dict, print_emoji: bool = False) -> dict[str, str]:
    decl_name = entry["decl_name"]
    module = entry["module"]
    result_data = {"decl_name": decl_name, "module": module}
    command = [
        "lake", "exe", "tactic_benchmark",
        module, decl_name,
        os.path.abspath(temp_premises_dir),
        benchmark_type
    ]
    result_data["command"] = " ".join(command)
    try:
        result = subprocess.run(
            command,
            cwd=ntp_toolkit_path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout
        )
        result_output = "\n".join(
            line for line in result.stdout.splitlines()
            if not any(line.startswith(prefix) for prefix in ["note:", "warning:", "error:", "⚠", "✖", "✔"])
        )
        # print(result_output)
        match = re.search(r"^([❌️💥️✅️]+) ", result.stdout, flags=re.MULTILINE)
        result_emoji = match.group(1) if match else None
    except subprocess.TimeoutExpired as e:
        result_emoji = "⏰"
        result_output = str(e)

    result_data["result_emoji"] = result_emoji
    result_data["result_output"] = result_output

    if print_emoji:
        print(result_emoji)
    return result_data


subprocess.run(
    ["lake", "build", "tactic_benchmark"],
    cwd=ntp_toolkit_path,
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_benchmark, entry): entry for entry in decl_names_for_benchmark}
    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(decl_names_for_benchmark)):
        result_data = future.result()
        results[result_data["decl_name"]].update(result_data)

with open(out_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {out_file}")

# Sometimes timeout tactics leave zombie threads (TODO)
subprocess.run(
    ["killall", "tactic_benchmark"],
    check=False,
)
shutil.rmtree(temp_premises_dir, ignore_errors=True)
