# ntp-toolkit

The neural theorem proving toolkit transforms Lean repositories into datasets for training and evaluating machine learning models.

<img width="900" alt="ntp-toolkit" src="https://github.com/user-attachments/assets/61441106-722c-4187-a505-c0d438760582">


The toolkit is originally a fork of Kim Morrison's [lean-training-data](https://github.com/semorrison/lean-training-data) and developed in [miniCTX](https://cmu-l3.github.io/minictx/). 



## Running extraction
To run the full pipeline on all repositories in a config file in the `configs` directory:
```
python scripts/extract_repos.py --cwd {filepath_of_this_repo} --config {filepath_of_config_file} [flags to indicate which processes to run]
```

The flags that can be set to indicate which processes to run are:
- `--training_data`: This outputs to the `TacticPrediction` directory
- `--full_proof_training_data`: This outputs to the `FullProof` directory
- `--premises`: This outputs to the `Premises` directory
- `--state_comments`: This outputs to the `StateComments` directory
- `--full_proof_training_data_states`: This outputs to the `FullProofWithStates` directory
- `--training_data_with_premises`: This outputs to the `TrainingDataWithPremises` directory
- `--add_imports`: This outputs to the `WithImports` directory (and is a prerequisite for performing hammer evaluations)

At least one of the above flags must be set in order for the script to run (but there should be no issue with setting multiple or even all of the above flags)

On a Macbook Pro (M3 Max, 14 CPU) it takes around 2 hours to run the extractions on mathlib.


To run a tool individually, use `lake exe <tool>`. \
The `run_pipeline.py` script uses Python to call tools in this way and organize the resulting files.



#### Extraction tools:
### `training_data`

This produces a `.jsonl` file where each line is an example of the following form:
```json
{
   "state": "{tactic state}",
   "nextTactic" : "{pretty-printed next tactic}",
   "srcUpToTactic" : "{source code in the file up to the tactic invocation}",
   "declUpToTactic" : "{source code in the declaration up to the tactic invocation}",
   "decl": "{declaration without proof (e.g., statement of a theorem)}",
   "declId": "{unique identifier of the declaration}"
}
```

### `full_proof_training_data`

This produces a `.jsonl` file where each line is an example of the following form:
```json
{
   "srcUpToDecl":"{source code in the file up to the declaration}",
   "decl": "{declaration without proof (e.g., statement of a theorem)}",
   "declId": "{unique identifier of the declaration}",
   "proof":"{proof}"
}
```

### `state_comments`

This produces Lean source files with proof states interleaved as comments after each tactic.

### `training_data_with_premises`

This produces a `.jsonl` file where each line contains all of the information in `training_data` plus the following fields:
```json
{
   ...
   "declName": "{The name of the current declaration}",
   "nextTacticIsSimpOrRwVariant": "{A boolean indicating whether the next tactic is one of several `simp` or `rw` variants}",
   "nextTacticTermPremises": "{Prop-typed constants that appear in the proof term generated by the tactic}",
   "nextTacticExplicitConstants": "{Constants that appear in tactic syntax}",
   "nextTacticExplicitPremises": "{Prop-typed constants that appear in tactic syntax}",
   "nextTacticExplicitLocalHypotheses": "{Local context Prop-typed hypotheses that appear in tactic syntax}",
   "nextTacticAllPremises": "{The union of nextTacticTermPremises and nextTacticExplicitPremises}",
   "nextTacticHammerRecommendation": "{Best guess of which constants would help a hammer based on the constants that appear in the next tactic}",
   "declTermPremises": "{Prop-typed constants that appear in the overall declaration's proof term}",
   "declExplicitConstants": "{Constants that appear in the tactic syntax for any tactic used to create the overall declaration's proof}",
   "declExplicitPremises": "{Prop-typed constants that appear in the tactic syntax for any tactic used to create the overall declaration's proof}",
   "declAllPremises": "{The union of declTermPremises and declExplicitPremises}",
   "declHammerRecommendation": "{The union of each nextTacticHammerRecommendation for every tactic used to create the overall declaration's proof}",
   "termPremisesToEndOfGoal": "{Prop-typed constants that appear in the sum of all proof terms used to close the current goal}",
   "explicitConstantsToEndOfGoal": "{Constants that appear in the tactic syntax for any tactic used to close the current goal}",
   "explicitPremisesToEndOfGoal": "{Prop-typed constants that appear in the tactic syntax for any tactic used to close the current goal}",
   "allPremisesToEndOfGoal": "{The union of termPremisesToEndOfGoal and explicitPremisesToEndOfGoal}",
   "hammerRecommendationToEndOfGoal": "{The union of each nextTacticHammerRecommendation for every tactic used to close the current goal}"
}
```

Note: Of the above additional fields, all `...ToEndOfGoal` fields are potentially approximations. They currently appear to be accurate for the patterns I have tested, but it's hard to check whether there may be edge cases for which they are inaccurate.

## Running instruction tuning data generation
After extraction, you can generate various forms of (prompt, completion) examples for fine-tuning language models.

To do so, run:
```
python scripts/instruction_tuning.py --prompt context_state_tactic
```
See `python scripts/instruction_tuning.py -h` for other options for `--prompt` or other settings.

The prompt includes a natural language description of the task, commonly referred to as an "instruction" (hence the name instruction tuning data).

## Hammer evaluation (preliminary)

Before experimenting with the current hammer evaluation tool, make sure you are able to invoke the `hammer` tactic successfully by installing [zipperposition](https://github.com/sneeuwballen/zipperposition) and running the following Lean code:
```
import Hammer
example {p q r : Prop} (hp : p) (hq : q) (hr : r) : p ∧ q := by
  hammer [*]
```

This code should prove the goal and yield the following suggestion:
```
Try this:
  apply @Classical.byContradiction
  intro negGoal
  duper [hp, hq, negGoal] {preprocessing := full}
```

For the purposes of experimenting with a hammer evaluation, `scripts/tactic_benchmark.lean` implements a function called `hammerBenchmarkFromModule` which takes in the name of a module (e.g. `` `Mathlib.Data.Set.Basic``), the path of the `WithImports` directory (generated by running the extraction script with the `--add_imports` flag enabled), and the path of the `TrainingDataWithPremises` directory (generated by running the extraction script with the `--training_data_with_premises` flag enabled).

An example of an invocation of this function (which takes ~5 minutes on my laptop) would be:
```
#eval hammerBenchmarkFromModule `Mathlib.Data.Set.Basic "Examples/Mathlib/WithImports" "Examples/Mathlib/TrainingDataWithPremises"
```

This function outputs every (Prop-valued) declaration in the given module along a string of five icons indicating the outcome of attempting to use the `hammer` tactic to prove said declaration (using the premises indicated by the JSON file in the `TrainingDataWithPremisesDirectory`). The interpretation of the icons is as follows:
- 💥❌❌❌❌❌ indicates that the given declaration has no entry in the JSON file associated with the current module. This primarily occurs when the declaration was proven without entering tactic mode in the original file (meaning the data extraction script did not collect the ground truth for this declaration).
- ✅💥❌❌❌❌ indicates that the hammer tactic encountered an error before beginning the procedure to translate to the TPTP format (this can happen when the `simp_all` preprocessing step encounters an error).
- ✅✅💥❌❌❌ indicates that the given declaration does have an entry in the JSON file but could not be translated to the TPTP format (usually because the declaration itself or one of the premises used to prove it are outside the scope of the current translation procedure).
- ✅✅✅💥❌❌ indicates that the given declaration could be translated to the TPTP format but that the external prover (currently Zipperposition) was unable to solve the goal. This generally occurs when the translation did not preserve enough information (e.g. because the translation did not unfold some necessary constant).
- ✅✅✅✅💥❌ indicates that the given declaration was successfully translated and the external prover successfully proved the goal, but Duper was unable to reconstruct the external prover's proof.
- ✅✅✅✅✅💥 indicates that both the external prover and Duper were able to prove the goal, but some error occurred in the process of applying the proof to the goal.
- ✅✅✅✅✅✅ indicates that the `hammer` tactic was fully successful in proving the goal.
- 💥💥💥💥💥💥 indicates there was some unknown error that does not fit into any of the above categories.

*Note: Currently, there are some inconsistencies between how the `hammer` tactic actually performs and how the evaluation indicates that it performs. These inconsistencies are still being debugged and resolved, but the current known inconsistencies concern distinguishing between the last three cases. So the tool should still be reliable in determining whether a goal can be translated to TPTP and proven by an external prover.*

Another way to use `hammerBenchmarkFromModule` is to replace the path to the `TrainingDataWithPremisesDirectory` directory with the path to a directory of JSON files containing hammer recommendations generated by a relevance filter. The following invariants are expected of any directory used in this way:
- The naming convention of files in the directory matches the naming convention used by the `TrainingDataWithPremisesDirectory` directory.
- Every JSON entry in each file contains a `declName` field (indicating the full global name of the declaration) and a `declHammerRecommendation` field (indicating the list of premises that the `hammer` tactic should attempt to use). These are the only fields that are necessary for the purpose of this current `hammer` evaluation tool.

## Other setup docs from `lean-training-data`

You may find these useful during setup.

* Install [`elan`](https://github.com/leanprover/elan) by running

```shell
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```
* Run `lake exe cache get` (this downloads precompiled binaries for `Mathlib`).
* Run `lake build`
* Run `lake exe <tool>`, where `<tool>` is one of the programs documented below.


## Projects

Projects that use or build upon `ntp-toolkit`:
- [miniCTX: Neural Theorem Proving with (Long-)Contexts](https://cmu-l3.github.io/minictx/)
- [ImProver: Agent-Based Automated Proof Optimization](https://arxiv.org/abs/2410.04753)

Submit a PR to add your project or paper here!


## Citation

The toolkit is originally a fork of Kim Morrison's [lean-training-data](https://github.com/semorrison/lean-training-data):

```bibtex
@misc{lean-training-data,
  author = {Kim Morrison},
  title = {lean-training-data},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/semorrison/lean-training-data}},
}
```

The `ntp-toolkit` was initially developed in [miniCTX](https://cmu-l3.github.io/minictx/):
```bibtex
@misc{hu2024minictxneuraltheoremproving,
      title={miniCTX: Neural Theorem Proving with (Long-)Contexts}, 
      author={Jiewen Hu and Thomas Zhu and Sean Welleck},
      year={2024},
      eprint={2408.03350},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.03350}, 
}
```
