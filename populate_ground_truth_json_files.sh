#!/bin/bash

cat modulesToCollect.txt | while read line
do
	echo "About to process $line"
	stdbuf -o0 lake exe training_data_with_premises $line > ground_truth_json_files/$line.jsonl
done

# stdbuf -o0 lake exe training_data_with_premises Mathlib.Data.Nat.Basic > ground_truth_json_files/Mathlib.Data.Nat.Basic.jsonl
echo "Done"
