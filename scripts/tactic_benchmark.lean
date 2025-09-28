import TrainingData.Frontend
import TrainingData.Utils.SimpAllHint
import TrainingData.Utils.TheoremPrettyPrinting
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.ToExpr
import Aesop
import Lean.Util.Trace
import Duper
import QuerySMT
import Cli

-- This version of tactic_benchmark.lean is compatible with a lakefile that imports querySMT

open Lean Core Elab IO Meta Term Tactic SimpAllHint TheoremPrettyPrinting PremiseSelection Duper Auto

set_option autoImplicit true

def useAesop : TacticM Unit := do evalTactic (← `(tactic| aesop))
def useExact? : TacticM Unit := do evalTactic (← `(tactic| exact?))
def useRfl : TacticM Unit := do evalTactic (← `(tactic| intros; rfl))
def useSimpAll : TacticM Unit := do evalTactic (← `(tactic| intros; simp_all))
def useOmega : TacticM Unit := do evalTactic (← `(tactic| intros; omega))

def useDuper (hammerRecommendation : Array String) (externalProverTimeout : Nat) : TacticM Unit := do
  withOptions (fun o => o.set ``duper.maxSaturationTime externalProverTimeout) do
    let hammerRecommendation : Array Ident ←
      hammerRecommendation.mapM (fun x => do
        let [name, _] := x.splitOn ","
          | throwError "{decl_name%} :: Unable to parse hammerRecommendation {x}"
        let name := name.drop 1 -- Remove leading left parenthesis
        pure (mkIdent name.toName)
      )
    evalTactic (← `(tactic| duper [*, $hammerRecommendation,*]))

def useAuto (hammerRecommendation : Array String) (smtBackend : Bool) : TacticM Unit := do
  let autoOptions :=
    if smtBackend then
      fun o =>
        let o := o.set ``auto.tptp false
        let o := o.set ``auto.smt true
        let o := o.set ``auto.smt.trust true
        let o := o.set ``auto.smt.solver.name "cvc5"
        let o := o.set ``auto.native false
        let o := o.set ``auto.smt.dumpHints false
        let o := o.set ``auto.mono.ignoreNonQuasiHigherOrder true
        let o := o.set ``auto.smt.ignoreUnusableFacts true
        let o := o.set ``duper.ignoreUnusableFacts true
        o
    else
      fun o =>
        let o := o.set ``auto.tptp false
        let o := o.set ``auto.smt false
        let o := o.set ``auto.native true
        o
  withOptions autoOptions do
    let (_, newGoal) ← (← getMainGoal).intros
    let [nngoal] ← newGoal.apply (.const ``Classical.byContradiction [])
      | throwError "{decl_name%} :: Unexpected result after applying Classical.byContradiction"
    let (_, absurd) ← MVarId.intro1 nngoal
    replaceMainGoal [absurd]
    withMainContext do
      let hammerRecommendation : Array Ident ←
        hammerRecommendation.mapM (fun x => do
          let [name, _] := x.splitOn ","
            | throwError "{decl_name%} :: Unable to parse hammerRecommendation {x}"
          let name := name.drop 1 -- Remove leading left parenthesis
          pure (mkIdent name.toName)
        )
      let formulas ← collectAssumptions hammerRecommendation true #[] -- `goalDecls` can be safely set to `#[]` because `withAllLCtx` is set to `true`
      let lemmas ← formulasToAutoLemmas formulas (includeInSetOfSupport := true)
      let lemmas ← lemmas.mapM (m:=MetaM) (Auto.unfoldConstAndPreprocessLemma #[])
      let inhFacts ← Auto.Inhabitation.getInhFactsFromLCtx
      let proof ← runAuto `fake_decl lemmas inhFacts
      absurd.assign proof

def useSimpAllWithRecommendation (simpAllRecommendation : Array String) : TacticM Unit := do
  let simpAllRecommendation : Array Name := simpAllRecommendation.map String.toName
  let simpAllRecommendation : Array Ident := simpAllRecommendation.map mkIdent
  let simpAllRecommendation : Array Term := simpAllRecommendation.map (fun i => ⟨i.raw⟩)
  dbg_trace "simpAllRecommendation: {simpAllRecommendation}"
  evalTactic (← `(tactic| simp_all [$[$simpAllRecommendation:term],*]))

def useQuerySMT (hammerRecommendation : Array String) (externalProverTimeout : Nat) (ignoreHints : Bool) : TacticM Unit := do
  withOptions (fun o => ((o.set ``auto.tptp.timeout externalProverTimeout).set ``duper.maxSaturationTime externalProverTimeout).set ``querySMT.ignoreHints ignoreHints) do
    let hammerRecommendation : Array Ident ←
      hammerRecommendation.mapM (fun x => do
        let [name, _] := x.splitOn ","
          | throwError "{decl_name%} :: Unable to parse hammerRecommendation {x}"
        let name := name.drop 1 -- Remove leading left parenthesis
        pure (mkIdent name.toName)
      )
    evalTactic (← `(tactic| querySMT [*, $hammerRecommendation,*]))

def useQuerySMTBlind : TacticM Unit := do
  evalTactic (← `(tactic| querySMT [*]))

def useGrindWithRecommendation (hammerRecommendation : Array String) : TacticM Unit := do
  let hammerRecommendation : Array Ident ←
    hammerRecommendation.mapM (fun x => do
      let [name, _] := x.splitOn ","
        | throwError "{decl_name%} :: Unable to parse hammerRecommendation {x}"
      let name := name.drop 1 -- Remove leading left parenthesis
      pure (mkIdent name.toName)
    )
  let mut grindParams : TSyntaxArray `Lean.Parser.Tactic.grindParam := #[]
  for t in hammerRecommendation do
    grindParams := grindParams.push $ ← `(Lean.Parser.Tactic.grindParam| $t:ident)
  evalTactic (← `(tactic| grind [$grindParams,*]))

def useGrind : TacticM Unit := do
  evalTactic (← `(tactic| grind))

def useAesopWithPremises (hammerRecommendation : Array String) : TacticM Unit := do
  let hammerRecommendation : Array Ident ←
    hammerRecommendation.mapM (fun x => do
      let [name, _] := x.splitOn ","
        | throwError "{decl_name%} :: Unable to parse hammerRecommendation {x}"
      let name := name.drop 1 -- Remove leading left parenthesis
      pure (mkIdent name.toName)
    )
  let mut addIdentStxs : TSyntaxArray `Aesop.tactic_clause := #[]
  for t in hammerRecommendation do
    let tFeature ← `(Aesop.feature| $t:ident)
    addIdentStxs := addIdentStxs.push (← `(Aesop.tactic_clause| (add unsafe $tFeature:Aesop.feature)))
  evalTactic (← `(tactic| aesop $addIdentStxs*))

/--
Compile the designated module, and run a monadic function with each new `ConstantInfo`,
with the `Environment` as it was *before* the command which created that declaration.

(Internal declarations according to `Name.isBlackListed` are skipped.)

If `withImportsDir` is provided, then `runAtDecls` uses the version of the file contained in the `WithImports` directory

`tac` takes in a `ConstantInfo` as well as an optional `Nat` indicating how many arguments the theorem takes. This is a hack to get around the fact
that `Info.ofConstantVal'` doesn't have access to the constant itself in the environment `cmd.before`. **TODO** Find a more general solution.
-/
def runAtDecls (mod : Name) (withImportsDir : Option String := none) (tac : ConstantInfo → Option Nat → MetaM (Option α)) : MLList IO (ConstantInfo × α) := do
  let fileName ←
    match withImportsDir with
    | none => pure (← findLean mod).toString
    | some withImportsDir => pure (← findLeanWithImports mod withImportsDir).toString
  let steps :=
    match withImportsDir with
    | none => compileModule' mod
    | some withImportsDir => compileModuleWithImports' mod withImportsDir
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  targets.filterMapM fun (cmd, ci) => do
    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    unless cmd.msgs.isEmpty do
      throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    let options := ({} : KVMap).insert `maxHeartbeats (.ofNat 200000)
    let ctx := { fileName, options, fileMap := default }
    let stateBefore := { env := cmd.before }
    let stateAfter := { env := cmd.after }
    -- Calculate `numArgs`, the number of arguments the theorem `ci` takes
    /- **TODO** This `numArgs` approximation is consistent with the approximation that's performed in training_data_with_premises.lean,
       but it is not entirely accurate to the real number of arguments in the theorem declaration. The issue is that `numArgs` counts
       the number of distinct binders in the theorem declaration, not the actual number of arguments that the theorem takes (so `{a b : Nat}`
       is counted as 1 binder, but the theorem really takes 2 arguments). -/
    let numArgs ←
      Prod.fst <$> (CoreM.toIO · ctx stateAfter) do -- Use `stateAfter` because `Info.ofConstantVal'` needs to access the constant in the environment
        MetaM.run' (ctx := {}) (s := {}) do
          match ci with
          | .thmInfo v =>
            let thmInfo ← Info.ofConstantVal' v.toConstantVal
            return some thmInfo.args.size
          | _ => return none
    -- From `IO` to `CoreM`:
    Prod.fst <$> (CoreM.toIO · ctx stateBefore) do -- Use `stateBefore` to accurately simulate the environment before the declaration was created
      if ← ci.name.isBlackListed then
        pure none
      else
        -- From `CoreM` to `MetaM`:
        MetaM.run' (ctx := {}) (s := {}) do
          match ← tac ci numArgs with
          | some r => pure (ci, r)
          | none => pure none

/-- Like `runAtDecls` but only returns the output for a single declaration. If the declaration cannot be found or if the tactic fails, `none` is returned. -/
def runAtDecl (mod : Name) (declName : Name) (withImportsDir : Option String := none) (tac : ConstantInfo → Option Nat → MetaM (Option α)) : IO (Option (ConstantInfo × α)) := do
  let fileName ←
    match withImportsDir with
    | none => pure (← findLean mod).toString
    | some withImportsDir => pure (← findLeanWithImports mod withImportsDir).toString
  let steps :=
    match withImportsDir with
    | none => compileModule' mod
    | some withImportsDir => compileModuleWithImports' mod withImportsDir
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)
  for (cmd, ci) in targets do
    if ci.name == declName then
      let options := ({} : KVMap).insert `maxHeartbeats (.ofNat 200000)
      let ctx := { fileName, options, fileMap := default }
      let stateBefore := { env := cmd.before }
      let stateAfter := { env := cmd.after }
      -- Calculate `numArgs`, the number of arguments the theorem `ci` takes
      /- **TODO** This `numArgs` approximation is consistent with the approximation that's performed in training_data_with_premises.lean,
         but it is not entirely accurate to the real number of arguments in the theorem declaration. The issue is that `numArgs` counts
         the number of distinct binders in the theorem declaration, not the actual number of arguments that the theorem takes (so `{a b : Nat}`
         is counted as 1 binder, but the theorem really takes 2 arguments). -/
      let numArgs ←
        Prod.fst <$> (CoreM.toIO · ctx stateAfter) do -- Use `stateAfter` because `Info.ofConstantVal'` needs to access the constant in the environment
          MetaM.run' (ctx := {}) (s := {}) do
            match ci with
            | .thmInfo v =>
              let thmInfo ← Info.ofConstantVal' v.toConstantVal
              return some thmInfo.args.size
            | _ => return none
      -- From `IO` to `CoreM`:
      let res ← Prod.fst <$> (CoreM.toIO · ctx stateBefore) do -- Use `stateBefore` to accurately simulate the environment before the declaration was created
        if ← ci.name.isBlackListed then
          IO.eprintln s!"{decl_name%} :: {ci.name} is blacklisted and is therefore not an eligible declaration for runAtDecl"
          return none
        else
          -- From `CoreM` to `MetaM`:
          MetaM.run' (ctx := {}) (s := {}) do
            match ← tac ci numArgs with
            | some r => pure $ some (ci, r)
            | none => pure none
      return res
  IO.eprintln s!"Unable to find declaration {declName} in module {mod}"
  return none

/-- This function is intended to achieve the same result as `runAtDecl`, but instead of running `tac` with exactly the environment produced by the original declaration,
    `runAtAliasDecl` creates an alias for `declName` and runs `tac` on that. This is used for declarations that `tac` itself depends on (because `runAtDecl` only works
    when `tac`'s dependencies have already been imported, attempting to use `runAtDecl` for a declaration that `tac` depends on will result in a circular dependency).

    The temporary file that `runAtAliasDecl` creates imports both `mod` and `tac`. For this, `tacImport` is the string that `runAtAliasDecl` uses to import `tac`. -/
def runAtAliasDecl {α} (mod : Name) (declName : Name) (tacImport : Option String) (tac : ConstantInfo → Option Nat → MetaM (Option α)) : IO (Option (ConstantInfo × α)) := do
  FS.withTempFile $ fun fhandle fpath => do
    let modSource :=
      match tacImport with
      | some tacImport => s!"import {mod}\nimport Batteries.Tactic.Alias\nimport {tacImport}\nalias {declName}__eval := {declName}"
      | none => s!"import {mod}\nimport Batteries.Tactic.Alias\nalias {declName}__eval := {declName}"
    fhandle.putStrLn modSource
    fhandle.flush
    let fileName := fpath.toString
    let steps := Lean.Elab.IO.processInput' modSource none {} fileName true mod
    let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)
    for (cmd, ci) in targets do
      if s!"{ci.name}" == s!"{declName}__eval" then
        let options := ({} : KVMap).insert `maxHeartbeats (.ofNat 200000)
        let ctx := { fileName, options, fileMap := default }
        let stateBefore := { env := cmd.before }
        let stateAfter := { env := cmd.after }
        -- Calculate `numArgs`, the number of arguments the theorem `ci` takes
        /- **TODO** This `numArgs` approximation is consistent with the approximation that's performed in training_data_with_premises.lean,
          but it is not entirely accurate to the real number of arguments in the theorem declaration. The issue is that `numArgs` counts
          the number of distinct binders in the theorem declaration, not the actual number of arguments that the theorem takes (so `{a b : Nat}`
          is counted as 1 binder, but the theorem really takes 2 arguments). -/
        let numArgs ←
          Prod.fst <$> (CoreM.toIO · ctx stateAfter) do -- Use `stateAfter` because `Info.ofConstantVal'` needs to access the constant in the environment
            MetaM.run' (ctx := {}) (s := {}) do
              match ci with
              | .thmInfo v =>
                let thmInfo ← Info.ofConstantVal' v.toConstantVal
                return some thmInfo.args.size
              | _ => return none
        -- From `IO` to `CoreM`:
        let res ← Prod.fst <$> (CoreM.toIO · ctx stateBefore) do -- Use `stateBefore` to accurately simulate the environment before the declaration was created
          if ← ci.name.isBlackListed then
            IO.eprintln s!"{decl_name%} :: {ci.name} is blacklisted and is therefore not an eligible declaration for runAtAliasDecl"
            return none
          else
            -- From `CoreM` to `MetaM`:
            MetaM.run' (ctx := {}) (s := {}) do
              match ← tac ci numArgs with
              | some r => pure $ some (ci, r)
              | none => pure none
        return res
    IO.eprintln s!"Unable to find declaration {declName}__eval in alias temporary file for {mod}"
    return none

inductive GeneralResultType
| success
| failure
| subgoals
| notDefEq
| noJSON
deriving Repr, BEq

instance : ToString GeneralResultType where
  toString := fun
  | .success => "success"
  | .failure => "failure"
  | .subgoals => "subgoals"
  | .notDefEq => "notDefEq"
  | .noJSON => "noJSON"

inductive HammerResultType
| success
| noJSON -- For declarations that the JSON file doesn't have data for (currently, these are declarations that are proven without entering tactic mode)
| simpPreprocessingFailure -- For declarations where `hammer` encounters an error during the simp preprocessing stage
| tptpTranslationFailure -- For declarations that cannot be translated to the external prover's format (currently TPTP's higher-order logic format)
| externalProverFailure -- For declarations that can be successfully translated but cannot be proven by the external prover (currently Zipperposition)
| duperFailure -- For declarations successfully translated and proven by the external prover but return proofs that Duper can't reconstruct
| proofFitFailure -- For declarations successfully proven by Duper's proof reconstruction, but returns proofs that yield some sort of error when applied
| miscFailure -- For hammer failures that don't fall into one of the previous five categories
| subgoals -- For declarations that are partially proven but have remaining subgoals even after the tactic is run
| notDefEq -- For declarations for which a proof is found but the proof is not definitionally equal to the expected proof
deriving Repr, BEq

instance : ToString HammerResultType where
  toString := fun
  | .success => "success"
  | .noJSON => "noJSON"
  | .simpPreprocessingFailure => "simpPreprocessingFailure"
  | .tptpTranslationFailure => "tptpTranslationFailure"
  | .externalProverFailure => "externalProverFailure"
  | .duperFailure => "duperFailure"
  | .proofFitFailure => "proofFitFailure"
  | .miscFailure => "miscFailure"
  | .subgoals => "subgoals"
  | .notDefEq => "notDefEq"

inductive QuerySMTResultType
| success
| noJSON -- For declarations that the JSON file doesn't have data for (currently, this is never thrown because the current querySMT evaluation doesn't search for premises)
| skolemizationFailure -- For declarations where `querySMT` encounters an error during the initial skolemization
| smtTranslationFailure -- For declarations where `querySMT` throws a translation error
| externalProverFailure -- For declarations that can be successfully translated but cannot be proven by the external prover (currently cvc5)
| hintParsingFailure -- For declarations where `querySMT` encounters an error parsing the external prover's hints
| selectorConstructionFailure -- For declarations where `querySMT` encounters an error constructing a datatype's selectors
| duperFailure -- For declarations successfully translated and proven by the external prover but return proofs that Duper can't reconstruct
| proofFitFailure -- For declarations successfully proven by Duper's proof reconstruction, but returns proofs that yield some sort of error when applied
| miscFailure -- For `querySMT` failures that don't fall into one of the previous categories
| subgoals -- For declarations that are partially proven but have remaining subgoals even after the tactic is run
| notDefEq -- For declarations for which a proof is found but the proof is not definitionally equal to the expected proof
deriving Repr, BEq

instance : ToString QuerySMTResultType where
  toString := fun
  | .success => "success"
  | .noJSON => "noJSON"
  | .skolemizationFailure => "skolemizationFailure"
  | .smtTranslationFailure => "smtTranslationFailure"
  | .externalProverFailure => "externalProverFailure"
  | .hintParsingFailure => "hintParsingFailure"
  | .selectorConstructionFailure => "selectorConstructionFailure"
  | .duperFailure => "duperFailure"
  | .proofFitFailure => "proofFitFailure"
  | .miscFailure => "miscFailure"
  | .subgoals => "subgoals"
  | .notDefEq => "notDefEq"

structure GeneralResult where
  type : GeneralResultType
  seconds : Float
  heartbeats : Nat

structure HammerResult where
  type : HammerResultType
  seconds : Float
  heartbeats : Nat

structure QuerySMTResult where
  type : QuerySMTResultType
  seconds : Float
  heartbeats : Nat

inductive TacType
| General
| Hammer
| QuerySMT

inductive ResultType
| GeneralResult : GeneralResultType → ResultType
| HammerResult : HammerResultType → ResultType
| QuerySMTResult : QuerySMTResultType → ResultType

instance : ToString ResultType where
  toString := fun
  | .GeneralResult res => toString res
  | .HammerResult res => toString res
  | .QuerySMTResult res => toString res

structure Result where
  type : ResultType
  seconds : Float
  heartbeats : Nat

open ResultType

def withSeconds [Monad m] [MonadLiftT BaseIO m] (act : m α) : m (α × Float) := do
  let start ← IO.monoNanosNow
  let a ← act
  let stop ← IO.monoNanosNow
  return (a, (stop - start).toFloat / 1000000000)

def runTacticAtAliasDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (tacImport : Option String) (tac : TacticM Unit)
  (tacType : TacType) : IO (Option (ConstantInfo × Result)) := do
  runAtAliasDecl mod declName tacImport fun ci numArgs? => do
    if ! (← decls ci) then return none
    let g ←
      match numArgs? with
      | some numArgs =>
        let g ← mkFreshExprMVar ci.type
        let (_, g) ← g.mvarId!.introNP numArgs -- Introduce universal binders corresponding to arguments of the theorem
        pure g
      | none => return none -- Only run the tactic on theorems
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      tryCatchRuntimeEx
        (TermElabM.run' (do
          let gs ← Tactic.run g tac
          match tacType with
          | .General =>
            match gs with
            | _ :: _ => pure $ GeneralResult .subgoals
            | [] =>
              match ci.value? with
              | none => pure $ GeneralResult .success
              | some v =>
                if ← isProp ci.type then
                  pure $ GeneralResult .success
                else
                match ← try? (isDefEq (Expr.mvar g) v) with
                | none
                  -- In this case we should perhaps return an "uncertain" value.
                  -- The problem is that `v` may contain constants generated by the simplifier
                  -- during elaboration of the original proof,
                  -- and which aren't in the current environment, so we can't really compare `g` and `v`
                | some false => pure $ GeneralResult .notDefEq
                | some true => pure $ GeneralResult .success
          | .Hammer =>
            match gs with
            | [] => pure $ HammerResult .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate `hammer` on Prop declarations
            | _ :: _ => pure $ HammerResult .subgoals
          | .QuerySMT =>
            match gs with
            | [] => pure $ QuerySMTResult .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate `querySMT` on Prop declarations
            | _ :: _ => pure $ QuerySMTResult .subgoals
          )
          (ctx := {declName? := `fakeDecl, errToSorry := false}))
        (fun e => do
          match tacType with
          | .General => pure $ GeneralResult .failure
          | .Hammer => throwError "{decl_name%} :: Current version of tactic_benchmark.lean does not support evaluating the hammer"
          | .QuerySMT =>
            if ← QuerySMT.errorIsSkolemizationError e then pure $ QuerySMTResult .skolemizationFailure
            else if ← QuerySMT.errorIsTranslationError e then pure $ QuerySMTResult .smtTranslationFailure
            else if ← QuerySMT.errorIsSolverError e then pure $ QuerySMTResult .externalProverFailure
            else if ← QuerySMT.errorIsHintParsingError e then pure $ QuerySMTResult .hintParsingFailure
            else if ← QuerySMT.errorIsSelectorConstructionError e then pure $ QuerySMTResult .selectorConstructionFailure
            else if ← QuerySMT.errorIsDuperError e then pure $ QuerySMTResult .duperFailure
            else if ← QuerySMT.errorIsProofFitError e then pure $ QuerySMTResult .proofFitFailure
            else
              dbg_trace "{decl_name%} :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
              pure $ QuerySMTResult .miscFailure
        )
    return some ⟨res, seconds, heartbeats⟩

def runGrindAtAliasDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (jsonDir : String)
  : IO (Option (ConstantInfo × GeneralResult)) := do
  runAtAliasDecl mod declName none fun ci numArgs? => do
    if ! (← decls ci) then return none
    let g ←
      match numArgs? with
      | some numArgs =>
        let g ← mkFreshExprMVar ci.type
        let (_, g) ← g.mvarId!.introNP numArgs -- Introduce universal binders corresponding to arguments of the theorem
        pure g
      | none => return none -- Only run the tactic on theorems
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if s!"{curDeclName}__eval" == s!"{ci.name}" then
        ciEntry := jsonEntry
        dbg_trace "Found jsonEntry for {declName}"
        break
    if ciEntry.isNull then
      return some ⟨.noJSON, 0.0, 0⟩
    let recommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
    let recommendation ← IO.ofExcept $ recommendation.getArr?
    let recommendation ← IO.ofExcept $ recommendation.mapM Json.getStr?
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      try
        TermElabM.run' (do
          dbg_trace "About to use grind for {ci.name} in module {mod} (recommendation: {recommendation})"
          let gs ← Tactic.run g $ useGrindWithRecommendation recommendation
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate on Prop declarations
          | _ :: _ => pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false})
      catch e =>
        dbg_trace "Encountered an error: {← e.toMessageData.toString}"
        pure .failure
    return some ⟨res, seconds, heartbeats⟩

def runAutoAtAliasDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (jsonDir : String)
  (smtBackend : Bool) : IO (Option (ConstantInfo × GeneralResult)) := do
  runAtAliasDecl mod declName "Duper" fun ci numArgs? => do
    if ! (← decls ci) then return none
    let g ←
      match numArgs? with
      | some numArgs =>
        let g ← mkFreshExprMVar ci.type
        let (_, g) ← g.mvarId!.introNP numArgs -- Introduce universal binders corresponding to arguments of the theorem
        pure g
      | none => return none -- Only run the tactic on theorems
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if s!"{curDeclName}__eval" == s!"{ci.name}" then
        ciEntry := jsonEntry
        dbg_trace "Found jsonEntry for {declName}"
        break
    if ciEntry.isNull then
      return some ⟨.noJSON, 0.0, 0⟩
    let recommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
    let recommendation ← IO.ofExcept $ recommendation.getArr?
    let recommendation ← IO.ofExcept $ recommendation.mapM Json.getStr?
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      try
        TermElabM.run' (do
          dbg_trace "About to use Auto with premises for {ci.name} in module {mod} (recommendation: {recommendation})"
          let gs ← Tactic.run g $ useAuto recommendation smtBackend
          dbg_trace "Successfully called Auto"
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate on Prop declarations
          | _ :: _ =>
            dbg_trace "{decl_name%} Subgoals case"
            pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false})
      catch e =>
        dbg_trace "{decl_name%} :: failure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
        pure .failure
    return some ⟨res, seconds, heartbeats⟩

def runQuerySMTAtAliasDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (jsonDir : String) (externalProverTimeout : Nat)
  (ignoreHints : Bool) : IO (Option (ConstantInfo × QuerySMTResult)) := do
  runAtAliasDecl mod declName "QuerySMT" fun ci numArgs? => do
    if ! (← decls ci) then return none
    let g ←
      match numArgs? with
      | some numArgs =>
        let g ← mkFreshExprMVar ci.type
        let (_, g) ← g.mvarId!.introNP numArgs -- Introduce universal binders corresponding to arguments of the theorem
        pure g
      | none => return none -- Only run the tactic on theorems
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if s!"{curDeclName}__eval" == s!"{ci.name}" then
        ciEntry := jsonEntry
        dbg_trace "Found jsonEntry for {declName}"
        break
    if ciEntry.isNull then
      return some ⟨.noJSON, 0.0, 0⟩
    let recommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
    let recommendation ← IO.ofExcept $ recommendation.getArr?
    let recommendation ← IO.ofExcept $ recommendation.mapM Json.getStr?
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      try
        TermElabM.run' (do
          dbg_trace "About to use querySMT with premises for {ci.name} in module {mod} (recommendation: {recommendation})"
          let gs ← Tactic.run g $ useQuerySMT recommendation externalProverTimeout ignoreHints
          dbg_trace "Successfully called querySMT"
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate on Prop declarations
          | _ :: _ =>
            dbg_trace "{decl_name%} Subgoals case"
            pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false})
      catch e =>
        if ← QuerySMT.errorIsSkolemizationError e then pure .skolemizationFailure
        else if ← QuerySMT.errorIsTranslationError e then pure .smtTranslationFailure
        else if ← QuerySMT.errorIsSolverError e then pure .externalProverFailure
        else if ← QuerySMT.errorIsHintParsingError e then pure .hintParsingFailure
        else if ← QuerySMT.errorIsSelectorConstructionError e then pure .selectorConstructionFailure
        else if ← QuerySMT.errorIsDuperError e then pure .duperFailure
        else if ← QuerySMT.errorIsProofFitError e then pure .proofFitFailure
        else
          dbg_trace "{decl_name%} :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
          pure .miscFailure
    return some ⟨res, seconds, heartbeats⟩

open Cli System

/-- Gives a string of 6 emojis indicating the success of the following `hammer` stages:
    - Finding premises relating to the current declaration (this can currently fail when the original declaration
      is proven without entering tactic mode)
    - Successfully performing simp preprocessing without encountering any errors (errors that could arise from `simp_all` not changing
      the tactic state are suppressed)
    - Translating the goal (and all of the premises needed to prove it) to TPTP
    - Receiving a proof from an external prover
    - Reconstructing the external proof with Duper
    - Applying Duper's proof to the goal state -/
def hammerResultTypeToEmojiString (res : HammerResultType) : String :=
  match res with
  | .success => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji
  | .noJSON => bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .simpPreprocessingFailure => checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .tptpTranslationFailure => checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .externalProverFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji
  | .duperFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji
  | .proofFitFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji
  | .miscFailure => bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji
  | .subgoals => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji
  | .notDefEq => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji

/-- Gives a string of 7 emojis indicating the success of the following `querySMT` stages:
    - Successfully performing skolemization
    - Successfully translating the goal to the SMT format
    - Receiving a proof from the external solver
    - Successfully parsing the hints returned by the external solver
    - Successfully constructing the selectors corresponding to the datatypes that appear in the problem
    - Successfully reconstructing the external proof with Duper
    - Applying Duper's proof to the goal state -/
def querySMTResultTypeToEmojiString (res : QuerySMTResultType) : String :=
  match res with
  | .success => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji
  | .noJSON => bombEmoji -- `querySMTBenchmarkFromModule` doesn't yet look for additional premises
  | .skolemizationFailure => bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .smtTranslationFailure => checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .externalProverFailure => checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .hintParsingFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji ++ crossEmoji
  | .selectorConstructionFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji ++ crossEmoji
  | .duperFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji ++ crossEmoji
  | .proofFitFailure => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji
  | .miscFailure => bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji ++ bombEmoji
  | .subgoals => -- First bombEmoji because subgoal likely came from `skolemizeAll`
    bombEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji
  | .notDefEq => checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ checkEmoji ++ bombEmoji

def generalResultTypeToEmojiString (res : GeneralResultType) : String :=
  match res with
  | .success => checkEmoji ++ "(success)"
  | .failure => crossEmoji ++ "(failure)"
  | .noJSON => bombEmoji ++ "(noJSON)"
  | .notDefEq => bombEmoji ++ "(notDefEq)"
  | .subgoals => bombEmoji ++ "(subgoals)"

def resultTypeToEmojiString (res : ResultType) : String :=
  match res with
  | .GeneralResult res => generalResultTypeToEmojiString res
  | .HammerResult res => hammerResultTypeToEmojiString res
  | .QuerySMTResult res => querySMTResultTypeToEmojiString res

def tacticBenchmarkAtAliasDecl (module : ModuleName) (declName : Name) (tac : TacticM Unit) (tacImport : Option String) (tacType : TacType) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let result ← runTacticAtAliasDecl module declName (fun _ => pure true) tacImport tac tacType
  IO.println s!"Testing on alias of {declName} in {module}"
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println <| (resultTypeToEmojiString type) ++ s!"({type}) " ++ ci.name.toString ++
      s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run tactic benchmark at alias decl for {declName} (module: {module})"
    return 0

def grindBenchmarkAtAliasDecl (module : ModuleName) (declName : Name) (jsonDir : String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let result ← runGrindAtAliasDecl module declName (fun ci => try isProp ci.type catch _ => pure false) jsonDir
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $ generalResultTypeToEmojiString type ++ " " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run grind benchmark at alias decl for {declName} (module: {module})"
    return 0

def autoBenchmarkAtAliasDecl (module : ModuleName) (declName : Name) (jsonDir : String) (smtBackend : Bool) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let result ← runAutoAtAliasDecl module declName (fun ci => try isProp ci.type catch _ => pure false) jsonDir smtBackend
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $ (generalResultTypeToEmojiString type) ++ s!"({type}) " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run Auto benchmark at alias decl for {declName} (module: {module})"
    return 0

def querySMTBenchmarkAtAliasDecl (module : ModuleName) (declName : Name) (jsonDir : String) (externalProverTimeout : Nat) (ignoreHints : Bool) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let result ← runQuerySMTAtAliasDecl module declName (fun ci => try isProp ci.type catch _ => pure false) jsonDir externalProverTimeout ignoreHints
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $ (querySMTResultTypeToEmojiString type) ++ s!"({type}) " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run querySMT benchmark at {declName} in module {module}"
    return 0

-- **TODO** Restore `duperBenchmarkAtAliasDecl` and `autoSMTBenchmarkAtAliasDecl`

def tacticBenchmarkMain (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "module" |>.as! ModuleName
  let declName := args.positionalArg! "declName" |>.as! String |>.toName
  let premisesPath := args.positionalArg! "premisesPath" |>.as! String
  let benchmarkType := args.positionalArg! "benchmarkType" |>.as! String
  let externalProverTimeout := args.flag! "externalProverTimeout" |>.as! Nat
  -- let withImportsPath := args.flag! "withImportsPath" |>.as! String

  try
    match benchmarkType with
      | "querySMT" => querySMTBenchmarkAtAliasDecl module declName premisesPath externalProverTimeout false
      | "querySMT_ignoreHints" => querySMTBenchmarkAtAliasDecl module declName premisesPath externalProverTimeout true

      | "grindWithRecommendation" => grindBenchmarkAtAliasDecl module declName premisesPath
      | "grind" => tacticBenchmarkAtAliasDecl module declName useGrind none TacType.General
      | "querySMTBlind" => tacticBenchmarkAtAliasDecl module declName useQuerySMTBlind "QuerySMT" TacType.QuerySMT

      | "autoSMT" => autoBenchmarkAtAliasDecl module declName premisesPath true

      | _ => IO.throwServerError s!"Unknown benchmark type {benchmarkType}"
  catch e =>
    IO.eprintln s!"{benchmarkType} failed with error {e}"
    return (1 : UInt32)

/-- Setting up command line options and help text for `lake exe tactic_benchmark`. -/
def tactic_benchmark : Cmd := `[Cli|
  tactic_benchmark VIA tacticBenchmarkMain; ["0.0.1"]
  "Run a customisable tactic at all declarations in a file."

  FLAGS:
    externalProverTimeout : Nat; "Timeout for the external prover."
    apiUrl : String; "API URL for the premise selection server."
    k : Nat; "Number of premises for the premise retriever in `hammer` to retrieve."
    hammerCoreK : Nat; "(Only relevant for aesop_hammerCore_nosimp_with_premises) Number of premises for the premise retriever in `hammer` to retrieve, as opposed to the number of premises directly input to aesop."
    aesopHammerPriority : Nat; "(Only relevant for aesop_hammerCore_nosimp_with_premises) The priority percentage value for hammer as a rule for aesop."
    aesopPremisePriority : Nat; "(Only relevant for aesop_hammerCore_nosimp_with_premises) The priority percentage value for premises as a rule for aesop."
    withImportsPath : String; "Path to directory with .lean files with added imports like Hammer (e.g. Examples/Mathlib/WithImports)."

  ARGS:
    module : ModuleName; "Lean module to run the tactic on."
    declName : String; "Name of the declaration to run the tactic on."
    premisesPath : String; "Path to the premises, such as Examples/Mathlib/TrainingDataWithPremises."
    benchmarkType : String; "Which type of tactic to run (e.g. hammer, hammerCore, aesop, exact)."

  EXTENSIONS:
    defaultValues! #[
      ("externalProverTimeout", "10"),
      ("apiUrl", "http://52.206.70.13"),
      ("k", "16"),
      ("hammerCoreK", "16"),
      ("aesopHammerPriority", "10"),
      ("aesopPremisePriority", "20"),
      ("withImportsPath", "Examples/Mathlib/WithImports")
    ]
]

/-- `lake exe tactic_benchmark` -/
def main (args : List String) : IO UInt32 :=
  tactic_benchmark.validate args

-- See `scripts/tactic_benchmark.sh` for a script to run this on all of Mathlib.

-- #eval tacticBenchmarkFromModule `temp useDuper
-- #eval tacticBenchmarkFromModule `temp useQuerySMT

-- Note: `tacticBenchmarkFromModule` requires that the tactic we want be imported in the module
