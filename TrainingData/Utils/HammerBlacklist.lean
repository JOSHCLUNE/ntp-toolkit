import Init.Data.Array.QSort
import Init.Data.Array.BinSearch

/-- Currently, this blacklist consists of theorems from:
    - Init.SimpLemmas.lean
    - Init.Prelude.lean
    - Init.Core.lean
    - Init.PropLemmas.lean
    - Init.Classical.lean
    - Mathlib.Logic.Basic.lean
    - Batteries.Logic.lean -/
def hammerRecommendationBlackList : Array String := #[
  -- Init.SimpLemmas.lean
  "of_eq_true",
  "of_eq_false",
  "eq_true",
  "eq_false",
  "eq_false'",
  "eq_true_of_decide",
  "eq_false_of_decide",
  "eq_self",
  "implies_congr",
  "iff_congr",
  "implies_dep_congr_ctx",
  "implies_congr_ctx",
  "forall_congr",
  "forall_prop_domain_congr",
  "let_congr",
  "let_val_congr",
  "let_body_congr",
  -- Intentionally omitting `ite_congr`
  "Eq.mpr_prop",
  "Eq.mpr_not",
  -- Intentionally omitting `dite_congr`
  "ne_eq",
  "ite_true",
  "ite_false",
  -- Intentionally omitting `dite_true` and `dite_false`, `ite_cond` facts, `dite_cond` facts, and `ite_self`
  "and_true",
  "true_and",
  "and_false",
  "false_and",
  "and_self",
  "and_not_self",
  "not_and_self",
  "and_imp",
  "not_and",
  "or_self",
  "or_true",
  "true_or",
  "or_false",
  "false_or",
  "iff_self",
  "iff_true",
  "true_iff",
  "iff_false",
  "false_iff",
  "false_implies",
  "forall_false",
  "implies_true",
  "true_implies",
  "not_false_eq_true",
  "not_true_eq_false",
  "not_iff_self",
  "and_congr_right",
  "and_congr_left",
  "and_assoc",
  "and_self_left",
  "and_self_right",
  "and_congr_right_iff",
  "and_congr_left_iff",
  "and_iff_left_of_imp",
  "and_iff_right_of_imp",
  "and_iff_left_iff_imp",
  "and_iff_right_iff_imp",
  "iff_self_and",
  "iff_and_self",
  "Or.imp",
  "Or.imp_left",
  "Or.imp_right",
  "or_assoc",
  "or_self_left",
  "or_self_right",
  "or_iff_right_of_imp",
  "or_iff_left_of_imp",
  "or_iff_left_iff_imp",
  "or_iff_right_iff_imp",
  "iff_self_or",
  "iff_or_self",
  "Bool.or_false",
  "Bool.or_true",
  "Bool.false_or",
  "Bool.true_or",
  "Bool.or_self",
  "Bool.or_eq_true",
  "Bool.and_false",
  "Bool.and_true",
  "Bool.false_and",
  "Bool.true_and",
  "Bool.and_self",
  "Bool.and_eq_true",
  "Bool.and_assoc",
  "Bool.or_assoc",
  "Bool.not_not",
  "Bool.not_true",
  "Bool.not_false",
  "beq_true",
  "beq_false",
  "Bool.not_eq_eq_eq_not",
  "Bool.not_eq_not",
  "Bool.not_not_eq",
  "Bool.not_eq_true'",
  "Bool.not_eq_false'",
  "Bool.not_eq_true", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  "Bool.not_eq_false", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  /- Intentionally omitting the following because Duper lacks built-in knowledge about the `Decidable` type class:
    - `decide_eq_true_eq`
    - `decide_eq_false_iff_not`
    - `decide_not`
    - `not_decide_eq_true` -/
  -- Intentionally omitting `heq_eq_eq` the following because Duper lacks built-in knowledge about `HEq`
  -- Intentionally omitting `cond_true` and `cond_false` because Duper lacks built-in knowledge about `cond`
  /- Intentionally omitting the following because Duper lacks built-in knowledge about `LawfulBEq` and `DecidableEq`:
    - `beq_self_eq_true`
    - `beq_self_eq_true'`
    - `bne_self_eq_false`
    - `bne_self_eq_false'`
    - `decide_False`
    - `decide_True`
    - `bne_iff_ne`
    - `beq_eq_false_iff_ne`
    - `bne_eq_false_iff_eq` -/
  "Bool.beq_to_eq",
  "Bool.not_beq_to_not_eq",
  -- Intentionally omitting `Nat.le_zero_eq` because Duper lacks built-in knowledge about `Nat`
  -- Init.Prelude.lean
  "False.elim",
  "absurd",
  "Eq.ndrec",
  "rfl",
  "id_eq",
  "Eq.subst",
  "Eq.symm",
  "Eq.trans",
  "congrArg",
  "congr",
  "congrFun",
  "And.intro",
  "And.left",
  "And.right",
  "Or.inl",
  "Or.inr",
  "Or.intro_left",
  "Or.intro_right",
  "Or.elim",
  "Or.resolve_left",
  "Or.resolve_right",
  "Or.neg_resolve_left",
  "Or.neg_resolve_right",
  "eq_false_of_ne_true",
  "eq_true_of_ne_false",
  "ne_false_of_eq_true",
  "ne_true_of_eq_false",
  -- Init.Core.lean
  "Eq.ndrecOn",
  "Iff.intro",
  "Iff.mp",
  "Iff.mpr",
  "Exists.intro",
  "mt",
  "not_false",
  "not_not_intro",
  "Eq.mp",
  "Eq.mpr",
  "Eq.substr",
  "Ne.intro",
  "Ne.elim",
  "Ne.irrefl",
  "Ne.symm",
  "ne_comm",
  "false_of_ne",
  "ne_false_of_self",
  "ne_true_of_not",
  "true_ne_false",
  "false_ne_true",
  "Bool.of_not_eq_true",
  "Bool.of_not_eq_false",
  -- Intentionally omitting `ne_of_beq_false` and `beq_false_of_ne` because Duper lacks built-in knowledge of `LawfulBEq`
  "iff_iff_implies_and_implies",
  "Iff.refl",
  "Iff.rfl",
  "Iff.of_eq",
  "Iff.trans",
  "Eq.comm",
  "eq_comm",
  "Iff.symm",
  "Iff.comm",
  "iff_comm",
  "And.symm",
  "And.comm",
  "and_comm",
  "Or.symm",
  "Or.comm",
  "or_comm",
  "Exists.elim",
  "nonempty_of_exists",
  "propext",
  "Eq.propIntro",
  "Not.elim",
  "And.elim",
  "Iff.elim",
  "Iff.subst",
  "Not.intro",
  "Not.imp",
  "not_congr",
  "not_not_not",
  "iff_of_true",
  "iff_of_false",
  "iff_true_left",
  "iff_true_right",
  "iff_false_left",
  "iff_false_right",
  "of_iff_true",
  "iff_true_intro",
  "not_of_iff_false",
  "iff_false_intro",
  "not_iff_false_intro",
  "not_true",
  "not_false_iff",
  "Eq.to_iff",
  "iff_of_eq",
  "neq_of_not_iff",
  "iff_iff_eq",
  "eq_iff_iff",
  "eq_self_iff_true",
  "ne_self_iff_false",
  "false_of_true_iff_false",
  "false_of_true_eq_false",
  "true_eq_false_of_false",
  "iff_def",
  "iff_def'",
  "true_iff_false",
  "false_iff_true",
  "iff_not_self",
  "not_not_of_not_imp",
  "not_of_not_imp",
  "imp_not_self",
  "imp_intro",
  "imp_imp_imp",
  "imp_iff_right",
  "imp_true_iff",
  "false_imp_iff",
  "true_imp_iff",
  "imp_self",
  "imp_false",
  "imp.swap",
  "imp_not_comm",
  "imp_congr_left",
  "imp_congr_right",
  "imp_congr_ctx",
  "imp_congr",
  "imp_iff_not",
  -- Init.PropLemmas.lean
  "not_not_em",
  "and_self_iff",
  "and_not_self_iff",
  "not_and_self_iff",
  "And.imp",
  "And.imp_left",
  "And.imp_right",
  "and_congr",
  "and_congr_left'",
  "and_congr_right'",
  "not_and_of_not_left",
  "not_and_of_not_right",
  "and_congr_right_eq",
  "and_congr_left_eq",
  "and_left_comm",
  "and_right_comm",
  "and_rotate",
  "and_and_and_comm",
  "and_and_left",
  "and_and_right",
  "and_iff_left",
  "and_iff_right",
  "or_self_iff",
  "not_or_intro",
  "or_congr",
  "or_congr_left",
  "or_congr_right",
  "or_left_comm",
  "or_right_comm",
  "or_or_or_comm",
  "or_or_distrib_left",
  "or_or_distrib_right",
  "or_rotate",
  "or_iff_left",
  "or_iff_right",
  "not_imp_of_and_not",
  "imp_and",
  "not_and'",
  "and_or_left",
  "or_and_right",
  "or_and_left",
  "and_or_right",
  "or_imp",
  "not_or",
  "not_and_of_not_or_not",
  "forall_imp",
  -- Intentionally omitting `forall_exists_index` because it quantifies over proofs
  "Exists.imp",
  "Exists.imp'",
  "exists_imp",
  -- Intentionally omitting `exists₂_imp` because it quantifies over proofs
  "exists_const",
  -- Intentionally omitting `exists_prop_congr` and `exists_prop_of_true` because they quantify over proofs
  -- Intentionally omitting `exists_true_left` because it reasons about proof irrelevance
  "forall_congr'",
  "exists_congr",
  /- Intentionally omitting the following because although Duper should be able to solve them, it seems to take a long time to do so
    - `forall₂_congr`
    - `exists₂_congr`
    ...
    - `forall₅_congr`
    - `exists₅_congr` -/
  "not_exists",
  "forall_not_of_not_exists",
  "not_exists_of_forall_not",
  "forall_and",
  "exists_or",
  "exists_false",
  "forall_const",
  "not_forall_of_exists_not",
  "forall_eq",
  "forall_eq'",
  "exists_eq",
  "exists_eq'",
  "exists_eq_left",
  "exists_eq_right",
  "exists_and_left",
  "exists_and_right",
  "exists_eq_left'",
  "exists_eq_right'",
  "forall_eq_or_imp",
  "exists_eq_or_imp",
  "exists_eq_right_right",
  "exists_eq_right_right'",
  "exists_or_eq_left",
  "exists_or_eq_right",
  "exists_or_eq_left'",
  "exists_or_eq_right'",
  -- Intentionally omitting `exists_prop'` and `exists_prop` because they quantify over proofs
  "exists_apply_eq_apply",
  -- Intentionally omitting `forall_prop_of_true` because it quantifies over proofs
  "forall_comm", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  "exists_comm", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  "forall_apply_eq_imp_iff", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  "forall_eq_apply_imp_iff", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  "forall_apply_eq_imp_iff₂", -- **NOTE** Duper doesn't solve this currently (see Tests/temp2.lean example in Duper repo)
  -- Intentionally omitting `forall_prop_of_false` because it quantifies over proofs
  /- The following several `Decidable` facts (up to `Decidable.or_congr_right'`) are known to Duper despite its lack of
     built-in `Decidable` reasoning because they follow from Duper's classical assumptions -/
  "Decidable.not_not",
  "Decidable.em",
  "Decidable.by_contra",
  "Or.by_cases",
  "Or.by_cases'",
  "Decidable.of_not_imp",
  "Decidable.not_imp_symm",
  "Decidable.not_imp_comm",
  "Decidable.not_imp_self",
  "Decidable.or_iff_not_imp_left",
  "Decidable.or_iff_not_imp_right",
  "Decidable.not_imp_not",
  "Decidable.not_or_of_imp",
  "Decidable.imp_iff_not_or",
  "Decidable.imp_iff_or_not",
  "Decidable.imp_or",
  "Decidable.imp_or'",
  "Decidable.not_imp_iff_and_not",
  "Decidable.peirce",
  "peirce'",
  "Decidable.not_iff_not",
  "Decidable.not_iff_comm",
  "Decidable.not_iff",
  "Decidable.iff_not_comm",
  "Decidable.iff_iff_and_or_not_and_not",
  "Decidable.iff_iff_not_or_and_or_not",
  "Decidable.not_and_not_right",
  "Decidable.not_and_iff_or_not_not",
  "Decidable.not_and_iff_or_not_not'",
  "Decidable.or_iff_not_and_not",
  "Decidable.and_iff_not_or_not",
  "Decidable.imp_iff_right_iff",
  "Decidable.imp_iff_left_iff",
  "Decidable.and_or_imp",
  "Decidable.or_congr_left'",
  "Decidable.or_congr_right'",
  "Decidable.not_forall",
  "Decidable.not_forall_not",
  "Decidable.not_exists_not",
  -- Init.Classical.lean
  "Classical.em",
  "Classical.propComplete",
  "Classical.byCases",
  "Classical.byContradiction",
  "Classical.not_not",
  "Classical.not_forall",
  "Classical.not_forall_not",
  "Classical.not_exists_not",
  "Classical.forall_or_exists_not",
  "Classical.exists_or_forall_not",
  "Classical.or_iff_not_imp_left",
  "Classical.or_iff_not_imp_right",
  "Classical.not_imp_iff_and_not",
  "Classical.not_and_iff_or_not_not",
  "Classical.not_iff",
  "Classical.imp_iff_left_iff",
  "Classical.imp_iff_right_iff",
  "Classical.and_or_imp",
  "Classical.not_imp",
  "Classical.imp_and_neg_imp_iff",
  -- Mathlib.Logic.Basic.lean
  "eq_iff_eq_cancel_left",
  "eq_iff_eq_cancel_right",
  "ne_and_eq_iff_right",
  "Iff.imp",
  "imp_iff_right_iff",
  "and_or_imp",
  "Function.mt",
  "dec_em",
  "dec_em'",
  "em",
  "em'",
  "or_not",
  "Decidable.eq_or_ne",
  "Decidable.ne_or_eq",
  "eq_or_ne",
  "ne_or_eq",
  "by_contradiction",
  "by_cases",
  "by_contra",
  "of_not_not",
  "not_ne_iff",
  "of_not_imp",
  "Not.decidable_imp_symm",
  "Not.imp_symm",
  "not_imp_comm",
  "not_imp_self",
  "Imp.swap",
  "Iff.not",
  "Iff.not_left",
  "Iff.not_right",
  "Iff.ne",
  "Iff.ne_left",
  "Iff.ne_right",
  "Iff.and",
  "And.rotate",
  "and_symm_right",
  "and_symm_left",
  "Iff.or",
  "Or.rotate",
  "Or.elim3",
  "Or.imp3",
  "not_or_of_imp",
  "Decidable.or_not_of_imp",
  "or_not_of_imp",
  "imp_iff_not_or",
  "imp_iff_or_not",
  "not_imp_not",
  "imp_and_neg_imp_iff",
  "Function.mtr",
  "or_congr_left'",
  "or_congr_right'",
  "Iff.iff",
  "imp_or",
  "imp_or'",
  "not_imp",
  "peirce",
  "not_iff_not",
  "not_iff_comm",
  "not_iff",
  "iff_not_comm",
  "iff_iff_and_or_not_and_not",
  "iff_iff_not_or_and_or_not",
  "not_and_not_right",
  "not_and_or",
  "or_iff_not_and_not",
  "and_iff_not_or_not",
  "forall_cond_comm",
  "forall_mem_comm",
  "ne_of_eq_of_ne",
  "ne_of_ne_of_eq",
  "Eq.trans_ne",
  "Ne.transe_eq",
  "not_forall_not",
  "forall_or_exists_not",
  "exists_or_forall_not",
  "forall_true_iff",
  "forall_true_iff'",
  "forall₂_true_iff",
  "forall₃_true_iff",
  "Decidable.and_forall_ne",
  "and_forall_ne",
  "Ne.ne_or_ne",
  "exists_apply_eq_apply'",
  "exists_apply_eq_apply2",
  "exists_apply_eq_apply2'",
  "exists_apply_eq_apply3",
  "exists_apply_eq_apply3'",
  "exists_apply_eq",
  "forall_apply_eq_imp_iff'",
  "forall_eq_apply_imp_iff'",
  "forall_or_of_or_forall",
  "Decidable.forall_or_left",
  "forall_or_left",
  "Decidable.forall_or_right",
  "forall_or_right",
  "Exists.fst",
  "Exists.snd",
  "Prop.exists_iff",
  "Prop.forall_iff",
  "imp_congr_eq",
  "imp_congr_ctx_eq",
  "eq_true_intro",
  "eq_false_intro",
  "Iff.eq",
  "iff_eq_eq",
  -- Batteries.Logic.lean
  "Decidable.exists_not_of_not_forall",
  "exists_not_of_not_forall"
]

def sortedHammerRecommendationBlackList : Array String :=
  hammerRecommendationBlackList.qsort (fun s1 s2 => s1 < s2)

def isBlackListed (s : String) : Bool :=
  (sortedHammerRecommendationBlackList.binSearch s (fun s1 s2 => s1 < s2)).isSome
