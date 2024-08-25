import QuerySMT
import Aesop

def p := True
def q := True
axiom p_eq_true : p = True
axiom q_eq_true : q = True

theorem list_eq_self1 (l : List α) : l = l := by
  cases l
  . have : p = True := by exact p_eq_true
    rfl
  . have : q = True := q_eq_true
    rfl

theorem list_eq_self2 (α : Type) (l : List α) : l = l := by
  exact list_eq_self1 l

theorem test1 (x : Nat) : x = x := by rfl

theorem test2 (x : α) : x = x := by rfl

theorem test3 (α : Type _) (x : α) : x = x := by rfl

theorem test4 (α : Type) (x : α) : x = x := by rfl

theorem test5 (α : Type) (f : α → Prop) (hf : ∀ x : α, f x = True) (x : α) : f x := by
  duper [*]

theorem zero_eq_zero : 0 = 0 := by omega

theorem querySMTTest (x y z : Int) : x ≤ y → y ≤ z → x ≤ z := by
  intros h0 h1
  apply @Classical.byContradiction
  intro negGoal
  have smtLemma0 : ¬x ≤ z → x + -Int.ofNat 1 * z ≥ Int.ofNat 1 := by simp; omega
  have smtLemma1 : y ≤ z → ¬y + -Int.ofNat 1 * z ≥ Int.ofNat 1 := by simp; omega
  have smtLemma2 : x ≤ y → ¬x + -Int.ofNat 1 * y ≥ Int.ofNat 1 := by simp; omega
  have smtLemma3 :
    (x + -Int.ofNat 1 * y ≥ Int.ofNat 1 ∨ y + -Int.ofNat 1 * z ≥ Int.ofNat 1) ∨ ¬x + -Int.ofNat 1 * z ≥ Int.ofNat 1 :=
    by simp; omega
  duper [h0, h1, negGoal, smtLemma0, smtLemma1, smtLemma2, smtLemma3]
