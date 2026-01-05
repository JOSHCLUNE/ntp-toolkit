import Lake
open Lake DSL

package «lean-training-data» {
  moreLeanArgs := #[
    "-Dlinter.unusedVariables=false",
    -- for supporting more lean versions, some usages in the code are deprecated
    -- a macro TODO is to make the code more future proof (e.g. name change of HashMap)
    "-Dlinter.deprecated=false"
  ]
}

require «doc-gen4» from git "https://github.com/leanprover/doc-gen4.git" @ "v4.22.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.22.0"

require QuerySMT from git
  "https://github.com/JOSHCLUNE/LeanSMTParser.git" @ "ed2a49e778b3d39a6d752e94954697d6e666731b"

-- require smt from git "https://github.com/ufmg-smite/lean-smt.git" @ "a79af6cf74b9c4ad3bfb755813fa856f7f41fa9e"

@[default_target]
lean_lib TrainingData where

lean_lib temp where

lean_lib Examples where

@[default_target]
lean_exe training_data where
  root := `scripts.training_data
  supportInterpreter := true

@[default_target]
lean_exe full_proof_training_data where
  root := `scripts.full_proof_training_data
  supportInterpreter := true

@[default_target]
lean_exe state_comments where
  root := `scripts.state_comments
  supportInterpreter := true

@[default_target]
lean_exe premises where
  root := `scripts.premises
  supportInterpreter := true

@[default_target]
lean_exe training_data_with_premises where
  root := `scripts.training_data_with_premises
  supportInterpreter := true

@[default_target]
lean_exe add_imports where
  root := `scripts.add_imports
  supportInterpreter := true

@[default_target]
lean_exe all_modules where
  root := `scripts.all_modules
  supportInterpreter := true

@[default_target]
lean_exe declarations where
  root := `scripts.declarations
  supportInterpreter := true

@[default_target]
lean_exe imports where
  root := `scripts.imports
  supportInterpreter := true

@[default_target]
lean_exe update_hammer_blacklist where
  root := `scripts.update_hammer_blacklist
  supportInterpreter := true

@[default_target]
lean_exe add_premises where
  root := `scripts.add_premises
  supportInterpreter := true

-- @[default_target]
-- lean_lib scripts.collect_minimized_json_files

-- The below version of tactic_benchmark is for when querySMT is imported
@[default_target]
lean_exe tactic_benchmark where
  root := `scripts.tactic_benchmark
  supportInterpreter := true

-- The below version of tactic_benchmark is for when lean-smt is imported
-- @[default_target]
-- lean_exe tactic_benchmark where
--   root := `scripts.tactic_benchmark2
--   supportInterpreter := true
