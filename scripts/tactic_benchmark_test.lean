import Scripts.Tactic_benchmark
import Smt
import Smt.Real
import Smt.Auto

open Lean Core Elab IO Meta Term Tactic -- All the monads!
-- Set this option to modify the premise selection server URL when using `hammer` locally
-- set_option hammer.premiseSelection.apiUrl "http://52.206.70.13/retrieve"

def withImportsDir := "Examples/Mathlib/WithImports"
def jsonDir := "/Users/joshClune/Desktop/ntp-toolkit/Examples/Mathlib/TrainingDataWithPremises"

-- #eval Command.liftTermElabM $ hammerCoreBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir jsonDir 10
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir (useHammer 10 "http://52.206.70.13/retrieve") TacType.Hammer
-- **NOTE** `useAesopHammer` should be given `TacType.General` because `aesop` does not output the special error messages that enable more fine-grained error interpretation
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Order.Heyting.Basic `hnot_le_iff_codisjoint_left withImportsDir (useAesopHammer 10 "http://52.206.70.13/retrieve") TacType.General
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

------------------------------------------------------------------------------------------------------------------------

-- #eval Command.liftTermElabM $ querySMTBenchmarkFromModule `Mathlib.Data.Int.Defs withImportsDir jsonDir 1
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Data.Int.Defs `Int.fakeTheorem withImportsDir useSimpAllWithSelector TacType.General
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Data.Int.Defs `Int.fakeTheorem withImportsDir useAesopWithSelector TacType.General
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Data.Int.Defs `Int.fakeTheorem withImportsDir (useHammer 10 "http://52.206.70.13/retrieve" 16) TacType.Hammer
-- #eval Command.liftTermElabM $ tacticBenchmarkAtDecl `Mathlib.Data.Int.Defs `Int.fakeTheorem withImportsDir (useAesopHammer 10 "http://52.206.70.13/retrieve" 16) TacType.General

-- **TODO** Test script calls

------------------------------------------------------------------------------------------------------------------------

/-
set_option auto.native true

set_option trace.smt.solve true in
example [Nonempty U] {f : U → U → U} {a b c d : U}
  (h0 : a = b) (h1 : c = d) (h2 : p1 ∧ True) (h3 : (¬ p1) ∨ (p2 ∧ p3))
  (h4 : (¬ p3) ∨ (¬ (f a c = f b d))) : False := by
  auto [h0, h1, h2, h3, h4]

theorem test1 : ∀ x : Int, ∃ y : Int, x ≤ y := by
  auto

set_option trace.smt.solve true in
theorem test1a (x : Int) (h : ∀ y : Int, x ≤ y) : False := by
  auto [*]

theorem test1b (x y : Int) : x ≤ y ∨ y < x := by
  auto

theorem test2 (P : Int × Int → Prop) (h : ∀ x : Int, ∀ y : Int, P (x, y)) :
  ∃ z : Int × Int, P z := by
  auto

structure myStructure where
  field1 : Int
  field2 : Int

open myStructure

example : myStructure.mk 0 (1 + 1) = myStructure.mk 0 2 := by
  auto

example : { field1 := 0, field2 := 0 : myStructure } ≠ ⟨0, 1⟩ := by
  auto [myStructure.mk.injEq]

example (p : Prop) (h : p) : p := by
  auto [*]

example (l : List Int) (contains : List Int → Int → Prop)
  (h1 : ∀ x : Int, contains l x → x ≥ 0)
  (h2 : ∃ x : Int, ∃ y : Int, contains l x ∧ contains l y ∧ x + y < 0) : False := by
  auto [*]

inductive myProd (t1 t2 : Type _)
| mk : t1 → t2 → myProd t1 t2

open myProd

example (sum : myProd Int Int → Int)
  (hSum : ∀ x : Int, ∀ y : Int, sum (mk x y) = x + y)
  (x : myProd Int Int) : ∃ y : myProd Int Int, sum x < sum y := by
  smt

example {α : Type u} (x : α) (t : Tree α) : t ≠ Tree.node x t t := by
  auto [*]

example (x z : Int) (hxz : x + z < 2) (f : Int → Int)
  (hz : 0 < z) (hx : 0 ≤ x) : ∀ y : Int, f (x + y) = f y := by
  auto [*]
-/

/-
#eval Command.liftTermElabM $ autoSMTWithPremisesBenchmarkAtDecl `Mathlib.Data.Int.Defs `myFakeTheorem withImportsDir jsonDir
#eval Command.liftTermElabM $ autoSMTWithPremisesBenchmarkAtDecl `Mathlib.Data.Int.Defs `myFakeTheorem2 withImportsDir jsonDir
-/

#eval Command.liftTermElabM $ autoSMTWithPremisesBenchmarkAtDecl `Mathlib.Data.Int.Defs `Int.le_rfl withImportsDir jsonDir
