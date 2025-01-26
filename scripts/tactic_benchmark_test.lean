import Scripts.Tactic_benchmark

open Lean Core Elab IO Meta Term Tactic -- All the monads!

def withImportsDir := "Examples/Mathlib/WithImports"
def jsonDir := "Examples/Mathlib/TrainingDataWithPremises"

-- #eval Command.liftTermElabM $ hammerCoreBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir jsonDir 10
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir (useHammer 10) TacType.Hammer
-- **NOTE** `useAesopHammer` should be given `TacType.General` because `aesop` does not output the special error messages that enable more fine-grained error interpretation
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir (useAesopHammer 10) TacType.General
-- #eval Command.liftTermElabM $ aesopHammerCoreBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir jsonDir 10

------------------------------------------------------------------------------------------------------------------------
-- Log of issues to investigate

/- This test succceeds in lean-auto's evaluation (and running `auto` by hand succeeds)

This test can be handled by `hammer`'s `simp_all` preprocessing, so it succeeds when `simp_all` preprocessing is enabled.
When `hammer`'s `simp_all` preprocessing is disabled, returns the following:
  About to use hammer for Mathlib.Vector.mem_cons_of_mem in module Mathlib.Data.Vector.Mem
    (recommendation: #[(Mathlib.Vector.mem_cons_iff, notInSimpAll)])
    ✅️✅️✅️💥️❌️❌️ Mathlib.Vector.mem_cons_of_mem (2.697926s) (13650348 heartbeats)

This corresponds to successfully translating to TPTP but having the external prover fail. This outcome does not technically contradict
lean-auto's result, but it merits investigation as I don't see why the external prover should fail when Duper can succeed (and Duper does
succeed when we test by hand, even with portfolioInstance set to 1) **TODO** Investigate what's going on here

#eval Command.liftTermElabM $ hammerCoreBenchmarkAtDecl `Mathlib.Data.Vector.Mem `Mathlib.Vector.mem_cons_of_mem withImportsDir jsonDir 10
#eval Command.liftTermElabM $ hammerCoreBenchmarkAtDecl `Mathlib.Data.Vector.Mem `Mathlib.Vector.mem_cons_of_mem withImportsDir jsonDir 10 false
-/
