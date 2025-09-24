-- import Auto.EvaluateAuto.ConstAnalysis
-- import Auto.EvaluateAuto.EnvAnalysis
-- import Auto.EvaluateAuto.NameArr
-- import Auto.Tactic
import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import Mathlib

open Lean Meta Elab IO

def Name.getConstsOfModules (modules : List Name) : CoreM (Array (Name × Name)) := do
  let mut constNames := #[]
  for module in modules do
    let mFile ← findOLean module
    unless (← mFile.pathExists) do
      throwError s!"object file '{mFile}' of module {module} does not exist"
    let (mod, _) ← readModuleData mFile
    constNames := constNames ++ mod.constNames.map (fun n => (n, module))
  return constNames

def intModules : List Name := [
  `Mathlib.Data.Int.AbsoluteValue,
  `Mathlib.Data.Int.Associated,
  `Mathlib.Data.Int.Basic,
  `Mathlib.Data.Int.Bitwise,
  `Mathlib.Data.Int.CardIntervalMod,
  `Mathlib.Data.Int.Cast.Basic,
  `Mathlib.Data.Int.Cast.Defs,
  `Mathlib.Data.Int.Cast.Field,
  `Mathlib.Data.Int.Cast.Lemmas,
  `Mathlib.Data.Int.Cast.Pi,
  `Mathlib.Data.Int.Cast.Prod,
  `Mathlib.Data.Int.CharZero,
  `Mathlib.Data.Int.ConditionallyCompleteOrder,
  `Mathlib.Data.Int.DivMod,
  `Mathlib.Data.Int.GCD,
  `Mathlib.Data.Int.Init,
  `Mathlib.Data.Int.Interval,
  `Mathlib.Data.Int.LeastGreatest,
  `Mathlib.Data.Int.Lemmas,
  `Mathlib.Data.Int.Log,
  `Mathlib.Data.Int.ModEq,
  `Mathlib.Data.Int.NatAbs,
  `Mathlib.Data.Int.NatPrime,
  `Mathlib.Data.Int.Notation,
  `Mathlib.Data.Int.Order.Basic,
  `Mathlib.Data.Int.Order.Lemmas,
  `Mathlib.Data.Int.Order.Units,
  `Mathlib.Data.Int.Range,
  `Mathlib.Data.Int.Sqrt,
  `Mathlib.Data.Int.Star,
  `Mathlib.Data.Int.SuccPred,
  `Mathlib.Data.Int.WithZero,
  `Init.Data.Int.Basic,
  `Init.Data.Int.Bitwise,
  `Init.Data.Int.Compare,
  `Init.Data.Int.DivMod,
  `Init.Data.Int.Gcd,
  `Init.Data.Int.Lemmas,
  `Init.Data.Int.LemmasAux,
  `Init.Data.Int.Order,
  `Init.Data.Int.Pow,
  `Init.Data.Int.Cooper,
  `Init.Data.Int.Linear,
  `Init.Data.Int.OfNat,
  `Init.Data.Int.DivMod.Basic,
  `Init.Data.Int.DivMod.Bootstrap,
  `Init.Data.Int.DivMod.Lemmas,
  `Init.Data.Int.Bitwise.Basic,
  `Init.Data.Int.Bitwise.Lemmas,
  `Batteries.Data.Int
]

def listModules : List Name := [
  `Mathlib.Data.List.AList,
  `Mathlib.Data.List.Basic,
  `Mathlib.Data.List.Chain,
  `Mathlib.Data.List.ChainOfFn,
  `Mathlib.Data.List.Count,
  `Mathlib.Data.List.Cycle,
  `Mathlib.Data.List.Dedup,
  `Mathlib.Data.List.Defs,
  `Mathlib.Data.List.Destutter,
  `Mathlib.Data.List.DropRight,
  `Mathlib.Data.List.Duplicate,
  `Mathlib.Data.List.Enum,
  `Mathlib.Data.List.FinRange,
  `Mathlib.Data.List.Flatten,
  `Mathlib.Data.List.Forall2,
  `Mathlib.Data.List.GetD,
  `Mathlib.Data.List.Indexes,
  `Mathlib.Data.List.Induction,
  `Mathlib.Data.List.Infix,
  `Mathlib.Data.List.InsertIdx,
  `Mathlib.Data.List.InsertNth,
  `Mathlib.Data.List.Intervals,
  `Mathlib.Data.List.Iterate,
  `Mathlib.Data.List.Lattice,
  `Mathlib.Data.List.Lemmas,
  `Mathlib.Data.List.Lex,
  `Mathlib.Data.List.Lookmap,
  `Mathlib.Data.List.Map2,
  `Mathlib.Data.List.MinMax,
  `Mathlib.Data.List.ModifyLast,
  `Mathlib.Data.List.Monad,
  `Mathlib.Data.List.NatAntidiagonal,
  `Mathlib.Data.List.Nodup,
  `Mathlib.Data.List.NodupEquivFin,
  `Mathlib.Data.List.OfFn,
  `Mathlib.Data.List.Pairwise,
  `Mathlib.Data.List.Palindrome,
  `Mathlib.Data.List.Perm.Basic,
  `Mathlib.Data.List.Perm.Lattice,
  `Mathlib.Data.List.Perm.Subperm,
  `Mathlib.Data.List.Permutation,
  `Mathlib.Data.List.Pi,
  `Mathlib.Data.List.Prime,
  `Mathlib.Data.List.ProdSigma,
  `Mathlib.Data.List.Range,
  `Mathlib.Data.List.ReduceOption,
  `Mathlib.Data.List.Rotate,
  `Mathlib.Data.List.Scan,
  `Mathlib.Data.List.Sections,
  `Mathlib.Data.List.Shortlex,
  `Mathlib.Data.List.Sigma,
  `Mathlib.Data.List.Sort,
  `Mathlib.Data.List.SplitBy,
  `Mathlib.Data.List.SplitLengths,
  `Mathlib.Data.List.SplitOn,
  `Mathlib.Data.List.Sublists,
  `Mathlib.Data.List.Sym,
  `Mathlib.Data.List.TFAE,
  `Mathlib.Data.List.TakeDrop,
  `Mathlib.Data.List.TakeWhile,
  `Mathlib.Data.List.ToFinsupp,
  `Mathlib.Data.List.Triplewise,
  `Mathlib.Data.List.Zip,
  `Init.Data.List.Attach,
  `Init.Data.List.Basic,
  `Init.Data.List.BasicAux,
  `Init.Data.List.Control,
  `Init.Data.List.Count,
  `Init.Data.List.Erase,
  `Init.Data.List.Find,
  `Init.Data.List.Impl,
  `Init.Data.List.Lemmas,
  `Init.Data.List.MinMax,
  `Init.Data.List.Monadic,
  `Init.Data.List.Nat,
  `Init.Data.List.Notation,
  `Init.Data.List.Pairwise,
  `Init.Data.List.Sublist,
  `Init.Data.List.TakeDrop,
  `Init.Data.List.Zip,
  `Init.Data.List.Perm,
  `Init.Data.List.Sort,
  `Init.Data.List.ToArray,
  `Init.Data.List.ToArrayImpl,
  `Init.Data.List.MapIdx,
  `Init.Data.List.OfFn,
  `Init.Data.List.FinRange,
  `Init.Data.List.Lex,
  `Init.Data.List.Nat.Basic,
  `Init.Data.List.Nat.Pairwise,
  `Init.Data.List.Nat.Range,
  `Init.Data.List.Nat.Sublist,
  `Init.Data.List.Nat.TakeDrop,
  `Init.Data.List.Nat.Count,
  `Init.Data.List.Nat.Erase,
  `Init.Data.List.Nat.Find,
  `Init.Data.List.Nat.BEq,
  `Init.Data.List.Nat.Modify,
  `Init.Data.List.Nat.InsertIdx,
  `Init.Data.List.Nat.Perm,
  `Init.Data.List.Sort.Basic,
  `Init.Data.List.Sort.Impl,
  `Init.Data.List.Sort.Lemmas,
  `Batteries.Data.List.ArrayMap,
  `Batteries.Data.List.Basic,
  `Batteries.Data.List.Count,
  `Batteries.Data.List.Init.Lemmas,
  `Batteries.Data.List.Lemmas,
  `Batteries.Data.List.Matcher,
  `Batteries.Data.List.Monadic,
  `Batteries.Data.List.OfFn,
  `Batteries.Data.List.Pairwise,
  `Batteries.Data.List.Perm
]

def natModules : List Name := [
  `Mathlib.Data.Nat.Basic,
  `Mathlib.Data.Nat.BinaryRec,
  `Mathlib.Data.Nat.BitIndices,
  `Mathlib.Data.Nat.Bits,
  `Mathlib.Data.Nat.Bitwise,
  `Mathlib.Data.Nat.Cast.Basic,
  `Mathlib.Data.Nat.Cast.Commute,
  `Mathlib.Data.Nat.Cast.Defs,
  `Mathlib.Data.Nat.Cast.Field,
  `Mathlib.Data.Nat.Cast.NeZero,
  `Mathlib.Data.Nat.Cast.Order.Basic,
  `Mathlib.Data.Nat.Cast.Order.Field,
  `Mathlib.Data.Nat.Cast.Order.Ring,
  `Mathlib.Data.Nat.Cast.Prod,
  `Mathlib.Data.Nat.Cast.SetInterval,
  `Mathlib.Data.Nat.Cast.Synonym,
  `Mathlib.Data.Nat.Cast.WithTop,
  `Mathlib.Data.Nat.ChineseRemainder,
  `Mathlib.Data.Nat.Choose.Basic,
  `Mathlib.Data.Nat.Choose.Bounds,
  `Mathlib.Data.Nat.Choose.Cast,
  `Mathlib.Data.Nat.Choose.Central,
  `Mathlib.Data.Nat.Choose.Dvd,
  `Mathlib.Data.Nat.Choose.Factorization,
  `Mathlib.Data.Nat.Choose.Lucas,
  `Mathlib.Data.Nat.Choose.Mul,
  `Mathlib.Data.Nat.Choose.Multinomial,
  `Mathlib.Data.Nat.Choose.Sum,
  `Mathlib.Data.Nat.Choose.Vandermonde,
  `Mathlib.Data.Nat.Count,
  `Mathlib.Data.Nat.Digits.Defs,
  `Mathlib.Data.Nat.Digits.Div,
  `Mathlib.Data.Nat.Digits.Lemmas,
  `Mathlib.Data.Nat.Dist,
  `Mathlib.Data.Nat.EvenOddRec,
  `Mathlib.Data.Nat.Factorial.Basic,
  `Mathlib.Data.Nat.Factorial.BigOperators,
  `Mathlib.Data.Nat.Factorial.Cast,
  `Mathlib.Data.Nat.Factorial.DoubleFactorial,
  `Mathlib.Data.Nat.Factorial.NatCast,
  `Mathlib.Data.Nat.Factorial.SuperFactorial,
  `Mathlib.Data.Nat.Factorization.Basic,
  `Mathlib.Data.Nat.Factorization.Defs,
  `Mathlib.Data.Nat.Factorization.Induction,
  `Mathlib.Data.Nat.Factorization.LCM,
  `Mathlib.Data.Nat.Factorization.PrimePow,
  `Mathlib.Data.Nat.Factorization.Root,
  `Mathlib.Data.Nat.Factors,
  `Mathlib.Data.Nat.Fib.Basic,
  `Mathlib.Data.Nat.Fib.Zeckendorf,
  `Mathlib.Data.Nat.Find,
  `Mathlib.Data.Nat.GCD.Basic,
  `Mathlib.Data.Nat.GCD.BigOperators,
  `Mathlib.Data.Nat.Hyperoperation,
  `Mathlib.Data.Nat.Init,
  `Mathlib.Data.Nat.Lattice,
  `Mathlib.Data.Nat.Log,
  `Mathlib.Data.Nat.MaxPowDiv,
  `Mathlib.Data.Nat.ModEq,
  `Mathlib.Data.Nat.Multiplicity,
  `Mathlib.Data.Nat.Notation,
  `Mathlib.Data.Nat.Nth,
  `Mathlib.Data.Nat.Order.Lemmas,
  `Mathlib.Data.Nat.PSub,
  `Mathlib.Data.Nat.Pairing,
  `Mathlib.Data.Nat.PartENat,
  `Mathlib.Data.Nat.Periodic,
  `Mathlib.Data.Nat.PowModTotient,
  `Mathlib.Data.Nat.Prime.Basic,
  `Mathlib.Data.Nat.Prime.Defs,
  `Mathlib.Data.Nat.Prime.Factorial,
  `Mathlib.Data.Nat.Prime.Infinite,
  `Mathlib.Data.Nat.Prime.Int,
  `Mathlib.Data.Nat.Prime.Nth,
  `Mathlib.Data.Nat.Prime.Pow,
  `Mathlib.Data.Nat.PrimeFin,
  `Mathlib.Data.Nat.Set,
  `Mathlib.Data.Nat.Size,
  `Mathlib.Data.Nat.Sqrt,
  `Mathlib.Data.Nat.Squarefree,
  `Mathlib.Data.Nat.SuccPred,
  `Mathlib.Data.Nat.Totient,
  `Mathlib.Data.Nat.Upto,
  `Mathlib.Data.Nat.WithBot,
  `Init.Data.Nat.Basic,
  `Init.Data.Nat.Div,
  `Init.Data.Nat.Dvd,
  `Init.Data.Nat.Gcd,
  `Init.Data.Nat.MinMax,
  `Init.Data.Nat.Bitwise,
  `Init.Data.Nat.Control,
  `Init.Data.Nat.Log2,
  `Init.Data.Nat.Power2,
  `Init.Data.Nat.Linear,
  `Init.Data.Nat.SOM,
  `Init.Data.Nat.Lemmas,
  `Init.Data.Nat.Mod,
  `Init.Data.Nat.Lcm,
  `Init.Data.Nat.Compare,
  `Init.Data.Nat.Simproc,
  `Init.Data.Nat.Fold,
  `Init.Data.Nat.Div.Basic,
  `Init.Data.Nat.Div.Lemmas,
  `Init.Data.Nat.Bitwise.Basic,
  `Init.Data.Nat.Bitwise.Lemmas,
  `Batteries.Data.Nat.Basic,
  `Batteries.Data.Nat.Bisect,
  `Batteries.Data.Nat.Digits,
  `Batteries.Data.Nat.Gcd,
  `Batteries.Data.Nat.Lemmas
]

def nameToJson (n mod : Name) : IO (Option Json) := do
  let json ←
    try
      let fileName := s!"ground_truth_json_files/{mod}.jsonl"
      let jsonObjects ← IO.FS.lines fileName
      IO.ofExcept $ jsonObjects.mapM Json.parse
    catch e =>
      IO.println s!"Error with {n} (mod: {mod}): {e}"
      return none
  -- Find `declHammerRecommendation` corresponding to current `ci`
  let mut ciEntry := Json.null
  for jsonEntry in json do
    let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
    let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
    if curDeclName == n.toString then
      ciEntry := jsonEntry
      IO.println s!"Found jsonEntry for {n} (mod: {mod})"
      break
  if ciEntry.isNull then
    IO.println s!"No jsonEntry found for {n} (mod: {mod})"
    return none
  let recommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
  let recommendation ← IO.ofExcept $ recommendation.getArr?
  let recommendation ← IO.ofExcept $ recommendation.mapM Json.getStr?

  let recommendation : Array String ←
    recommendation.mapM (fun x => do
      let [name, _] := x.splitOn ","
        | IO.println s!"Unable to parse hammerRecommendation {x}"; pure ""
      pure $ name.drop 1
    )

  let gt_premises := Json.arr $ recommendation.map (fun x => Json.str x)

  let mut gtHints := Json.mkObj []
  for r in recommendation do
    gtHints := gtHints.setObjVal! r "notInSimpAll"

  let json := Json.mkObj [
    ("decl_name", n.toString),
    ("module", mod.toString),
    ("gt_premises", gt_premises),
    ("gt_hints", gtHints)
  ]

  return some json

def storeNames (ns : Array (Name × Name)) (fname : String) : IO Unit := do
  let json ← ns.filterMapM (fun n => nameToJson n.1 n.2)
  /-
  let json := json.take 512
  if json.size < 512 then
    IO.println s!"Not enough names for {fname}"
  -/
  IO.println "Done!"
  -- IO.println s!"json.size: {json.size}"
  let fd ← IO.FS.Handle.mk fname .write
  fd.putStr (Json.pretty (Json.arr json))

/-- Taken from Auto/EvaluateAuto/ConstAnalysis.lean -/
def Name.isTheorem (name : Name) : CoreM Bool := do
  let .some ci := (← getEnv).find? name
    | throwError "Name.isTheorem :: Cannot find name {name}"
  let .thmInfo _ := ci
    | return false
  return true

/-- Taken from Auto/EvaluateAuto/ConstAnalysis.lean -/
def Name.isHumanTheorem (name : Name) : CoreM Bool := do
  let hasDeclRange := (← Lean.findDeclarationRanges? name).isSome
  let isTheorem ← Name.isTheorem name
  let notProjFn := !(← Lean.isProjectionFn name)
  return hasDeclRange && isTheorem && notProjFn

def runStoreNames := @id (CoreM _) do
  let all ← Name.getConstsOfModules intModules
  let all ← all.filterM (fun (n, _) => do pure (← Name.isHumanTheorem n))
  storeNames all s!"minimized_json_files/IntModules.json"

-- #eval runStoreNames
