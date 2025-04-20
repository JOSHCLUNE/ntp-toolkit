import Auto.EvaluateAuto.ConstAnalysis
import Auto.EvaluateAuto.EnvAnalysis
import Auto.EvaluateAuto.NameArr
import Auto.Tactic
import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import Mathlib

open Lean Meta Elab Auto EvalAuto IO

def Name.getConstsOfModules (modules : List Name) : CoreM (Array (Name × Name)) := do
  let mut constNames := #[]
  for module in modules do
    let mFile ← findOLean module
    unless (← mFile.pathExists) do
      throwError s!"object file '{mFile}' of module {module} does not exist"
    let (mod, _) ← readModuleData mFile
    constNames := constNames ++ mod.constNames.map (fun n => (n, module))
  return constNames

/-
#eval @id (CoreM _) do
   let all ← Name.getConstsOfModules numberTheoryModules
   let all ← all.filterM (fun (n, _) => do pure (← Name.isHumanTheorem n))
   let n := 600
   storeNames (← Array.randPick all n) s!"NumberTheoryNames{n}.json"
-/

def numberTheoryModules : List Name := [
  `Mathlib.NumberTheory.ADEInequality,
  `Mathlib.NumberTheory.AbelSummation,
  `Mathlib.NumberTheory.ArithmeticFunction,
  `Mathlib.NumberTheory.Basic,
  `Mathlib.NumberTheory.Bernoulli,
  `Mathlib.NumberTheory.BernoulliPolynomials,
  `Mathlib.NumberTheory.Bertrand,
  `Mathlib.NumberTheory.ClassNumber.AdmissibleAbs,
  `Mathlib.NumberTheory.ClassNumber.AdmissibleAbsoluteValue,
  `Mathlib.NumberTheory.ClassNumber.AdmissibleCardPowDegree,
  `Mathlib.NumberTheory.ClassNumber.Finite,
  `Mathlib.NumberTheory.ClassNumber.FunctionField,
  `Mathlib.NumberTheory.Cyclotomic.Basic,
  `Mathlib.NumberTheory.Cyclotomic.CyclotomicCharacter,
  `Mathlib.NumberTheory.Cyclotomic.Discriminant,
  `Mathlib.NumberTheory.Cyclotomic.Embeddings,
  `Mathlib.NumberTheory.Cyclotomic.Gal,
  `Mathlib.NumberTheory.Cyclotomic.PID,
  `Mathlib.NumberTheory.Cyclotomic.PrimitiveRoots,
  `Mathlib.NumberTheory.Cyclotomic.Rat,
  `Mathlib.NumberTheory.Cyclotomic.Three,
  `Mathlib.NumberTheory.Dioph,
  `Mathlib.NumberTheory.DiophantineApproximation.Basic,
  `Mathlib.NumberTheory.DiophantineApproximation.ContinuedFractions,
  `Mathlib.NumberTheory.DirichletCharacter.Basic,
  `Mathlib.NumberTheory.DirichletCharacter.Bounds,
  `Mathlib.NumberTheory.DirichletCharacter.GaussSum,
  `Mathlib.NumberTheory.DirichletCharacter.Orthogonality,
  `Mathlib.NumberTheory.Divisors,
  `Mathlib.NumberTheory.EllipticDivisibilitySequence,
  `Mathlib.NumberTheory.EulerProduct.Basic,
  `Mathlib.NumberTheory.EulerProduct.DirichletLSeries,
  `Mathlib.NumberTheory.EulerProduct.ExpLog,
  `Mathlib.NumberTheory.FLT.Basic,
  `Mathlib.NumberTheory.FLT.Four,
  `Mathlib.NumberTheory.FLT.MasonStothers,
  `Mathlib.NumberTheory.FLT.Three,
  `Mathlib.NumberTheory.FactorisationProperties,
  `Mathlib.NumberTheory.Fermat,
  `Mathlib.NumberTheory.FermatPsp,
  `Mathlib.NumberTheory.FrobeniusNumber,
  `Mathlib.NumberTheory.FunctionField,
  `Mathlib.NumberTheory.GaussSum,
  `Mathlib.NumberTheory.Harmonic.Bounds,
  `Mathlib.NumberTheory.Harmonic.Defs,
  `Mathlib.NumberTheory.Harmonic.EulerMascheroni,
  `Mathlib.NumberTheory.Harmonic.GammaDeriv,
  `Mathlib.NumberTheory.Harmonic.Int,
  `Mathlib.NumberTheory.Harmonic.ZetaAsymp,
  `Mathlib.NumberTheory.JacobiSum.Basic,
  `Mathlib.NumberTheory.KummerDedekind,
  `Mathlib.NumberTheory.LSeries.AbstractFuncEq,
  `Mathlib.NumberTheory.LSeries.Basic,
  `Mathlib.NumberTheory.LSeries.Convergence,
  `Mathlib.NumberTheory.LSeries.Convolution,
  `Mathlib.NumberTheory.LSeries.Deriv,
  `Mathlib.NumberTheory.LSeries.Dirichlet,
  `Mathlib.NumberTheory.LSeries.DirichletContinuation,
  `Mathlib.NumberTheory.LSeries.HurwitzZeta,
  `Mathlib.NumberTheory.LSeries.HurwitzZetaEven,
  `Mathlib.NumberTheory.LSeries.HurwitzZetaOdd,
  `Mathlib.NumberTheory.LSeries.HurwitzZetaValues,
  `Mathlib.NumberTheory.LSeries.Injectivity,
  `Mathlib.NumberTheory.LSeries.Linearity,
  `Mathlib.NumberTheory.LSeries.MellinEqDirichlet,
  `Mathlib.NumberTheory.LSeries.Nonvanishing,
  `Mathlib.NumberTheory.LSeries.Positivity,
  `Mathlib.NumberTheory.LSeries.PrimesInAP,
  `Mathlib.NumberTheory.LSeries.RiemannZeta,
  `Mathlib.NumberTheory.LSeries.ZMod,
  `Mathlib.NumberTheory.LegendreSymbol.AddCharacter,
  `Mathlib.NumberTheory.LegendreSymbol.Basic,
  `Mathlib.NumberTheory.LegendreSymbol.GaussEisensteinLemmas,
  `Mathlib.NumberTheory.LegendreSymbol.JacobiSymbol,
  `Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic,
  `Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.GaussSum,
  `Mathlib.NumberTheory.LegendreSymbol.QuadraticReciprocity,
  `Mathlib.NumberTheory.LegendreSymbol.ZModChar,
  `Mathlib.NumberTheory.LucasLehmer,
  `Mathlib.NumberTheory.LucasPrimality,
  `Mathlib.NumberTheory.MaricaSchoenheim,
  `Mathlib.NumberTheory.Modular,
  `Mathlib.NumberTheory.ModularForms.Basic,
  `Mathlib.NumberTheory.ModularForms.CongruenceSubgroups,
  `Mathlib.NumberTheory.ModularForms.EisensteinSeries.Basic,
  `Mathlib.NumberTheory.ModularForms.EisensteinSeries.Defs,
  `Mathlib.NumberTheory.ModularForms.EisensteinSeries.IsBoundedAtImInfty,
  `Mathlib.NumberTheory.ModularForms.EisensteinSeries.MDifferentiable,
  `Mathlib.NumberTheory.ModularForms.EisensteinSeries.UniformConvergence,
  `Mathlib.NumberTheory.ModularForms.Identities,
  `Mathlib.NumberTheory.ModularForms.JacobiTheta.Bounds,
  `Mathlib.NumberTheory.ModularForms.JacobiTheta.Manifold,
  `Mathlib.NumberTheory.ModularForms.JacobiTheta.OneVariable,
  `Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable,
  `Mathlib.NumberTheory.ModularForms.LevelOne,
  `Mathlib.NumberTheory.ModularForms.QExpansion,
  `Mathlib.NumberTheory.ModularForms.SlashActions,
  `Mathlib.NumberTheory.ModularForms.SlashInvariantForms,
  `Mathlib.NumberTheory.MulChar.Basic,
  `Mathlib.NumberTheory.MulChar.Duality,
  `Mathlib.NumberTheory.MulChar.Lemmas,
  `Mathlib.NumberTheory.Multiplicity,
  `Mathlib.NumberTheory.NumberField.AdeleRing,
  `Mathlib.NumberTheory.NumberField.Basic,
  `Mathlib.NumberTheory.NumberField.CanonicalEmbedding.Basic,
  `Mathlib.NumberTheory.NumberField.CanonicalEmbedding.ConvexBody,
  `Mathlib.NumberTheory.NumberField.CanonicalEmbedding.FundamentalCone,
  `Mathlib.NumberTheory.NumberField.ClassNumber,
  `Mathlib.NumberTheory.NumberField.Completion,
  `Mathlib.NumberTheory.NumberField.Discriminant.Basic,
  `Mathlib.NumberTheory.NumberField.Discriminant.Defs,
  `Mathlib.NumberTheory.NumberField.Embeddings,
  `Mathlib.NumberTheory.NumberField.EquivReindex,
  `Mathlib.NumberTheory.NumberField.FinitePlaces,
  `Mathlib.NumberTheory.NumberField.FractionalIdeal,
  `Mathlib.NumberTheory.NumberField.House,
  `Mathlib.NumberTheory.NumberField.Norm,
  `Mathlib.NumberTheory.NumberField.ProductFormula,
  `Mathlib.NumberTheory.NumberField.Units.Basic,
  `Mathlib.NumberTheory.NumberField.Units.DirichletTheorem,
  `Mathlib.NumberTheory.NumberField.Units.Regulator,
  `Mathlib.NumberTheory.Ostrowski,
  `Mathlib.NumberTheory.Padics.Hensel,
  `Mathlib.NumberTheory.Padics.MahlerBasis,
  `Mathlib.NumberTheory.Padics.PadicIntegers,
  `Mathlib.NumberTheory.Padics.PadicNorm,
  `Mathlib.NumberTheory.Padics.PadicNumbers,
  `Mathlib.NumberTheory.Padics.PadicVal.Basic,
  `Mathlib.NumberTheory.Padics.PadicVal.Defs,
  `Mathlib.NumberTheory.Padics.ProperSpace,
  `Mathlib.NumberTheory.Padics.RingHoms,
  `Mathlib.NumberTheory.Pell,
  `Mathlib.NumberTheory.PellMatiyasevic,
  `Mathlib.NumberTheory.PrimeCounting,
  `Mathlib.NumberTheory.PrimesCongruentOne,
  `Mathlib.NumberTheory.Primorial,
  `Mathlib.NumberTheory.PythagoreanTriples,
  `Mathlib.NumberTheory.RamificationInertia.Basic,
  `Mathlib.NumberTheory.Rayleigh,
  `Mathlib.NumberTheory.SiegelsLemma,
  `Mathlib.NumberTheory.SmoothNumbers,
  `Mathlib.NumberTheory.SumFourSquares,
  `Mathlib.NumberTheory.SumPrimeReciprocals,
  `Mathlib.NumberTheory.SumTwoSquares,
  `Mathlib.NumberTheory.Transcendental.Lindemann.Init.AnalyticalPart,
  `Mathlib.NumberTheory.Transcendental.Liouville.Basic,
  `Mathlib.NumberTheory.Transcendental.Liouville.LiouvilleNumber,
  `Mathlib.NumberTheory.Transcendental.Liouville.LiouvilleWith,
  `Mathlib.NumberTheory.Transcendental.Liouville.Measure,
  `Mathlib.NumberTheory.Transcendental.Liouville.Residual,
  `Mathlib.NumberTheory.VonMangoldt,
  `Mathlib.NumberTheory.WellApproximable,
  `Mathlib.NumberTheory.Wilson,
  `Mathlib.NumberTheory.ZetaValues,
  `Mathlib.NumberTheory.Zsqrtd.Basic,
  `Mathlib.NumberTheory.Zsqrtd.GaussianInt,
  `Mathlib.NumberTheory.Zsqrtd.QuadraticReciprocity,
  `Mathlib.NumberTheory.Zsqrtd.ToReal
]

def intModules : List Name := [
  `Mathlib.Data.Int.AbsoluteValue,
  `Mathlib.Data.Int.Associated,
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
  `Mathlib.Data.Int.Defs,
  `Mathlib.Data.Int.DivMod,
  `Mathlib.Data.Int.GCD,
  `Mathlib.Data.Int.Interval,
  `Mathlib.Data.Int.LeastGreatest,
  `Mathlib.Data.Int.Lemmas,
  `Mathlib.Data.Int.Log,
  `Mathlib.Data.Int.ModEq,
  `Mathlib.Data.Int.NatPrime,
  `Mathlib.Data.Int.Notation,
  `Mathlib.Data.Int.Order.Basic,
  `Mathlib.Data.Int.Order.Lemmas,
  `Mathlib.Data.Int.Order.Units,
  `Mathlib.Data.Int.Range,
  `Mathlib.Data.Int.Sqrt,
  `Mathlib.Data.Int.Star,
  `Mathlib.Data.Int.SuccPred,
  `Mathlib.Data.Int.WithZero
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
  `Mathlib.Data.List.EditDistance.Bounds,
  `Mathlib.Data.List.EditDistance.Defs,
  `Mathlib.Data.List.EditDistance.Estimator,
  `Mathlib.Data.List.Enum,
  `Mathlib.Data.List.FinRange,
  `Mathlib.Data.List.Flatten,
  `Mathlib.Data.List.Forall2,
  `Mathlib.Data.List.GetD,
  `Mathlib.Data.List.Indexes,
  `Mathlib.Data.List.Infix,
  `Mathlib.Data.List.InsertIdx,
  `Mathlib.Data.List.InsertNth,
  `Mathlib.Data.List.Intervals,
  `Mathlib.Data.List.Iterate,
  `Mathlib.Data.List.Lattice,
  `Mathlib.Data.List.Lemmas,
  `Mathlib.Data.List.Lex,
  `Mathlib.Data.List.Map2,
  `Mathlib.Data.List.MinMax,
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
  `Mathlib.Data.List.Sections,
  `Mathlib.Data.List.Sigma,
  `Mathlib.Data.List.Sort,
  `Mathlib.Data.List.SplitBy,
  `Mathlib.Data.List.SplitLengths,
  `Mathlib.Data.List.Sublists,
  `Mathlib.Data.List.Sym,
  `Mathlib.Data.List.TFAE,
  `Mathlib.Data.List.ToFinsupp,
  `Mathlib.Data.List.Zip
]

def natModules : List Name := [
  /-
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
  `Mathlib.Data.Nat.Defs,
  `Mathlib.Data.Nat.Digits,
  `Mathlib.Data.Nat.Dist,
  `Mathlib.Data.Nat.EvenOddRec,
  `Mathlib.Data.Nat.Factorial.Basic,
  `Mathlib.Data.Nat.Factorial.BigOperators,
  `Mathlib.Data.Nat.Factorial.Cast,
  `Mathlib.Data.Nat.Factorial.DoubleFactorial,
  `Mathlib.Data.Nat.Factorial.SuperFactorial,
  -- End of part 1
  -/
  /-
  `Mathlib.Data.Nat.Factorization.Basic,
  `Mathlib.Data.Nat.Factorization.Defs,
  `Mathlib.Data.Nat.Factorization.Induction,
  `Mathlib.Data.Nat.Factorization.PrimePow,
  `Mathlib.Data.Nat.Factorization.Root,
  `Mathlib.Data.Nat.Factors,
  `Mathlib.Data.Nat.Fib.Basic,
  `Mathlib.Data.Nat.Fib.Zeckendorf,
  `Mathlib.Data.Nat.Find,
  `Mathlib.Data.Nat.GCD.Basic,
  `Mathlib.Data.Nat.GCD.BigOperators,
  `Mathlib.Data.Nat.Hyperoperation,
  `Mathlib.Data.Nat.Lattice,
  `Mathlib.Data.Nat.Log,
  `Mathlib.Data.Nat.MaxPowDiv,
  `Mathlib.Data.Nat.ModEq,
  `Mathlib.Data.Nat.Multiplicity,
  `Mathlib.Data.Nat.Notation,
  `Mathlib.Data.Nat.Nth,
  -/
  -- End of part 2
  `Mathlib.Data.Nat.Order.Lemmas,
  `Mathlib.Data.Nat.PSub,
  `Mathlib.Data.Nat.Pairing,
  `Mathlib.Data.Nat.PartENat,
  `Mathlib.Data.Nat.Periodic,
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
  `Mathlib.Data.Nat.WithBot
]

def getNonMathlibConstants (knownModules : List Name) : CoreM (Array Name) := do
  let env ← getEnv
  let modules := env.allImportedModuleNames
  let mut constNames := #[]
  for module in modules do
    if !knownModules.contains module then
      let mFile ← findOLean module
      unless (← mFile.pathExists) do
        throwError s!"object file '{mFile}' of module {module} does not exist"
      let (mod, _) ← readModuleData mFile
      constNames := constNames ++ mod.constNames
  return constNames

/-
#eval @id (CoreM _) do
  let all ← getNonMathlibConstants numberTheoryModules
  IO.println all
-/

def nameToJson (n mod : Name) : IO (Option Json) := do
  let json ←
    try
      let fileName := (← findJSONFile mod "/Users/joshClune/Desktop/TrainingDataWithPremises").toString
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

def storeNames (ns : Array (Name × Name)) (dirName : String) : IO Unit := do
  let json ← ns.filterMapM (fun n => nameToJson n.1 n.2)
  -- IO.println s!"json.size: {json.size}"
  IO.println "Done computing jsons"
  for jsonEntry in json do
    let declName ← IO.ofExcept $ jsonEntry.getObjVal? "decl_name"
    let declName ← IO.ofExcept $ declName.getStr?
    let declNameInDir := dirName ++ s!"{declName}" ++ ".json"
    let fd ← IO.FS.Handle.mk declNameInDir .write
    fd.putStr (Json.pretty jsonEntry)
  IO.println "Done recording results"

/-
#eval @id (CoreM _) do
  let all ← Name.getConstsOfModules natModules
  let all ← all.filterM (fun (n, _) => do pure (← Name.isHumanTheorem n))
  storeNames all s!"NatNamesPart3/"
  -- let n := 900
  -- storeNames (← Array.randPick all n) s!"ListNames{n}.json"
-/
