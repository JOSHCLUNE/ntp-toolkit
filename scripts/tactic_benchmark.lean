import TrainingData.Frontend
import TrainingData.Utils.SimpAllHint
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.ToExpr
import Aesop
import Lean.Util.Trace
import Duper
import QuerySMT
import Hammer
import Cli

open Lean Core Elab IO Meta Term Tactic SimpAllHint -- All the monads!

set_option autoImplicit true

def useAesop : TacticM Unit := do evalTactic (← `(tactic| aesop))
def useExact? : TacticM Unit := do evalTactic (← `(tactic| exact?))
def useRfl : TacticM Unit := do evalTactic (← `(tactic| intros; rfl))
def useSimpAll : TacticM Unit := do evalTactic (← `(tactic| intros; simp_all))
def useSimpAllWithRecommendation (simpAllRecommendation : Array String) : TacticM Unit := do
  let simpAllRecommendation : Array Name := simpAllRecommendation.map String.toName
  let simpAllRecommendation : Array Ident := simpAllRecommendation.map mkIdent
  let simpAllRecommendation : Array Term := simpAllRecommendation.map (fun i => ⟨i.raw⟩)
  dbg_trace "simpAllRecommendation: {simpAllRecommendation}"
  evalTactic (← `(tactic| simp_all [$[$simpAllRecommendation:term],*]))
def useOmega : TacticM Unit := do evalTactic (← `(tactic| intros; omega))
def useDuper : TacticM Unit := do evalTactic (← `(tactic| duper [*]))
def useQuerySMT : TacticM Unit := do evalTactic (← `(tactic| querySMT))

def useHammerCore (hammerRecommendation : Array String) (externalProverTimeout : Nat) (withSimpPreprocessing := true) : TacticM Unit := do
  withOptions (fun o => o.set ``auto.tptp.timeout externalProverTimeout) do
    let hammerRecommendation : Array (Term × SimpAllHint) ←
      hammerRecommendation.mapM (fun x => do
        let [name, simpAllHint] := x.splitOn ","
          | throwError "useHammer :: Unable to parse hammerRecommendation {x}"
        let name := name.drop 1 -- Remove leading left parenthesis
        let name := ⟨(mkIdent name.toName).raw⟩
        let simpAllHint := simpAllHint.removeLeadingSpaces
        let simpAllHint := simpAllHint.dropRight 1 -- Removing ending right parenthesis
        let simpAllHint ← parseSimpAllHint simpAllHint
        pure (name, simpAllHint)
      )
    let mut simpLemmas : Array (TSyntax [`Lean.Parser.Tactic.simpErase, `Lean.Parser.Tactic.simpLemma]) := #[]
    let mut coreRecommendation : Array Term := #[]
    for (name, hint) in hammerRecommendation do
      coreRecommendation := coreRecommendation.push name
      match hint with
      | notInSimpAll => pure ()
      | unmodified => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| $name:term)
      | simpErase => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpErase| -$name:term)
      | simpPreOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↓$name:term)
      | simpPostOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↑$name:term)
      | backwardOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ←$name:term)
      | simpPreAndBackward => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↓←$name:term)
      | simpPostAndBackward => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↑←$name:term)
    if withSimpPreprocessing then
      evalTactic (← `(tactic| hammerCore [$simpLemmas,*] [*, $(coreRecommendation),*]))
    else
      evalTactic (← `(tactic| hammerCore [$simpLemmas,*] [*, $(coreRecommendation),*] {simpTarget := no_target}))

def useHammer (externalProverTimeout : Nat) (withSimpPreprocessing := true) : TacticM Unit := do
  withOptions (fun o => o.set ``auto.tptp.timeout externalProverTimeout) do
    if withSimpPreprocessing then
      evalTactic (← `(tactic| hammer))
    else
      evalTactic (← `(tactic| hammer {simpTarget := no_target}))

def useAesopHammer (externalProverTimeout : Nat) (withSimpPreprocessing := true) : TacticM Unit := do
  withOptions (fun o => o.set ``auto.tptp.timeout externalProverTimeout) do
    if withSimpPreprocessing then
      evalTactic (← `(tactic| aesop (add unsafe (by hammer))))
    else
      evalTactic (← `(tactic| aesop (add unsafe (by hammer {simpTarget := no_target}))))

def useAesopHammerCore (hammerRecommendation : Array String) (externalProverTimeout : Nat) (withSimpPreprocessing := false) : TacticM Unit := do
  withOptions (fun o => o.set ``auto.tptp.timeout externalProverTimeout) do
    let hammerRecommendation : Array (Term × SimpAllHint) ←
      hammerRecommendation.mapM (fun x => do
        let [name, simpAllHint] := x.splitOn ","
          | throwError "useHammer :: Unable to parse hammerRecommendation {x}"
        let name := name.drop 1 -- Remove leading left parenthesis
        let name := ⟨(mkIdent name.toName).raw⟩
        let simpAllHint := simpAllHint.removeLeadingSpaces
        let simpAllHint := simpAllHint.dropRight 1 -- Removing ending right parenthesis
        let simpAllHint ← parseSimpAllHint simpAllHint
        pure (name, simpAllHint)
      )
    let mut simpLemmas : Array (TSyntax [`Lean.Parser.Tactic.simpErase, `Lean.Parser.Tactic.simpLemma]) := #[]
    let mut coreRecommendation : Array Term := #[]
    for (name, hint) in hammerRecommendation do
      coreRecommendation := coreRecommendation.push name
      match hint with
      | notInSimpAll => pure ()
      | unmodified => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| $name:term)
      | simpErase => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpErase| -$name:term)
      | simpPreOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↓$name:term)
      | simpPostOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↑$name:term)
      | backwardOnly => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ←$name:term)
      | simpPreAndBackward => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↓←$name:term)
      | simpPostAndBackward => simpLemmas := simpLemmas.push $ ← `(Lean.Parser.Tactic.simpLemma| ↑←$name:term)
    if withSimpPreprocessing then
      evalTactic (← `(tactic| aesop (add unsafe (by hammerCore [$simpLemmas,*] [*, $(coreRecommendation),*]))))
    else
      evalTactic (← `(tactic| aesop (add unsafe (by hammerCore [$simpLemmas,*] [*, $(coreRecommendation),*] {simpTarget := no_target}))))

/--
Compile the designated module, and run a monadic function with each new `ConstantInfo`,
with the `Environment` as it was *before* the command which created that declaration.

(Internal declarations according to `Name.isBlackListed` are skipped.)

If `withImportsDir` is provided, then `runAtDecls` uses the version of the file contained in the `WithImports` directory
-/
def runAtDecls (mod : Name) (withImportsDir : Option String := none) (tac : ConstantInfo → MetaM (Option α)) : MLList IO (ConstantInfo × α) := do
  let fileName ←
    match withImportsDir with
    | none => pure (← findLean mod).toString
    | some withImportsDir => pure (← findLeanWithImports mod "mathlib" withImportsDir).toString
  let steps :=
    match withImportsDir with
    | none => compileModule' mod
    | some withImportsDir => compileModuleWithImports' mod "mathlib" withImportsDir
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  targets.filterMapM fun (cmd, ci) => do
    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    unless cmd.msgs.isEmpty do
      throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    let options := ({} : KVMap).insert `maxHeartbeats (.ofNat 200000)
    let ctx := { fileName, options, fileMap := default }
    let state := { env := cmd.before }
    -- From `IO` to `CoreM`:
    Prod.fst <$> (CoreM.toIO · ctx state) do
      if ← ci.name.isBlackListed then
        pure none
      else
        -- From `CoreM` to `MetaM`:
        MetaM.run' (ctx := {}) (s := {}) do
          match ← tac ci with
          | some r => pure (ci, r)
          | none => pure none

/-- Like `runAtDecls` but only returns the output for a single declaration. If the declaration cannot be found or if the tactic fails, `none` is returned. -/
def runAtDecl (mod : Name) (declName : Name) (withImportsDir : Option String := none) (tac : ConstantInfo → MetaM (Option α)) : IO (Option (ConstantInfo × α)) := do
  let fileName ←
    match withImportsDir with
    | none => pure (← findLean mod).toString
    | some withImportsDir => pure (← findLeanWithImports mod "mathlib" withImportsDir).toString
  let steps :=
    match withImportsDir with
    | none => compileModule' mod
    | some withImportsDir => compileModuleWithImports' mod "mathlib" withImportsDir
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)
  for (cmd, ci) in targets do
    if ci.name == declName then
      let options := ({} : KVMap).insert `maxHeartbeats (.ofNat 200000)
      let ctx := { fileName, options, fileMap := default }
      let state := { env := cmd.before }
      -- From `IO` to `CoreM`:
      let res ← Prod.fst <$> (CoreM.toIO · ctx state) do
        if ← ci.name.isBlackListed then
          IO.eprintln s!"runAtDecl :: {ci.name} is blacklisted and is therefore not an eligible declaration for runAtDecl"
          return none
        else
          -- From `CoreM` to `MetaM`:
          MetaM.run' (ctx := {}) (s := {}) do
            match ← tac ci with
            | some r => pure $ some (ci, r)
            | none => pure none
      return res
  IO.eprintln s!"Unable to find declaration {declName} in module {mod}"
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

/--
Compile the designated module, select declarations satisfying the `decls` predicate,
and run a tactic on the type of each declaration.

If `withImportsPath?` is provided, then `runTacticAtDecls` uses the version of the file contained in the `WithImports` directory
-/
def runTacticAtDecls (mod : Name) (decls : ConstantInfo → CoreM Bool) (withImportsPath? : Option String) (tac : TacticM Unit)
  (tacType : TacType) : MLList IO (ConstantInfo × Result) := do
  runAtDecls mod withImportsPath? fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      tryCatchRuntimeEx
        (TermElabM.run' (do
          let gs ← Tactic.run g.mvarId! tac
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
                match ← try? (isDefEq g v) with
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
            | [] => pure $ HammerResult .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate `querySMT` on Prop declarations
            | _ :: _ => pure $ HammerResult .subgoals
          )
          (ctx := {declName? := `fakeDecl, errToSorry := false}))
        (fun e => do
          match tacType with
          | .General => pure $ GeneralResult .failure
          | .Hammer =>
            if ← Hammer.errorIsSimpPreprocessingError e then pure $ HammerResult .simpPreprocessingFailure
            else if ← Hammer.errorIsTranslationError e then pure $ HammerResult .tptpTranslationFailure
            else if ← Hammer.errorIsExternalSolverError e then pure $ HammerResult .externalProverFailure
            else if ← Hammer.errorIsDuperSolverError e then pure $ HammerResult .duperFailure
            else if ← Hammer.errorIsProofFitError e then pure $ HammerResult .proofFitFailure
            else if "tactic 'simp' failed".isPrefixOf (← e.toMessageData.toString) then pure $ HammerResult .simpPreprocessingFailure
            else if "tactic 'simp_all' failed".isPrefixOf (← e.toMessageData.toString) then pure $ HammerResult .simpPreprocessingFailure
            else
              dbg_trace "{decl_name%} :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
              pure $ HammerResult .miscFailure
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

/-- Like `runTacticAtDecls` but only tests a single declaration (indicated by `declName`). -/
def runTacticAtDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (withImportsPath? : Option String) (tac : TacticM Unit)
  (tacType : TacType) : IO (Option (ConstantInfo × Result)) := do
  runAtDecl mod declName withImportsPath? fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      tryCatchRuntimeEx
        (TermElabM.run' (do
          let gs ← Tactic.run g.mvarId! tac
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
                match ← try? (isDefEq g v) with
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
            | [] => pure $ HammerResult .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate `querySMT` on Prop declarations
            | _ :: _ => pure $ HammerResult .subgoals
          )
          (ctx := {declName? := `fakeDecl, errToSorry := false}))
        (fun e => do
          match tacType with
          | .General => pure $ GeneralResult .failure
          | .Hammer =>
            if ← Hammer.errorIsSimpPreprocessingError e then pure $ HammerResult .simpPreprocessingFailure
            else if ← Hammer.errorIsTranslationError e then pure $ HammerResult .tptpTranslationFailure
            else if ← Hammer.errorIsExternalSolverError e then pure $ HammerResult .externalProverFailure
            else if ← Hammer.errorIsDuperSolverError e then pure $ HammerResult .duperFailure
            else if ← Hammer.errorIsProofFitError e then pure $ HammerResult .proofFitFailure
            else if "tactic 'simp' failed".isPrefixOf (← e.toMessageData.toString) then pure $ HammerResult .simpPreprocessingFailure
            else if "tactic 'simp_all' failed".isPrefixOf (← e.toMessageData.toString) then pure $ HammerResult .simpPreprocessingFailure
            else
              dbg_trace "{decl_name%} :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
              pure $ HammerResult .miscFailure
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

def runHammerCoreAtDecls (mod : Name) (decls : ConstantInfo → MetaM Bool) (withImportsPath : String) (jsonDir : String) (externalProverTimeout : Nat) :
  MLList IO (ConstantInfo × HammerResult) := do
  runAtDecls mod (some withImportsPath) fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod "mathlib" jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if curDeclName == s!"{ci.name}" then
        ciEntry := jsonEntry
        break
    if ciEntry.isNull then
      return some ⟨.noJSON, 0.0, 0⟩
    let hammerRecommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
    let hammerRecommendation ← IO.ofExcept $ hammerRecommendation.getArr?
    let hammerRecommendation ← IO.ofExcept $ hammerRecommendation.mapM Json.getStr?
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      tryCatchRuntimeEx
        (TermElabM.run' (do
          dbg_trace "About to use hammer for {ci.name} in module {mod} (recommendation: {hammerRecommendation})"
          let gs ← Tactic.run g.mvarId! $ useHammerCore hammerRecommendation externalProverTimeout
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate the hammer on Prop declarations
          | _ :: _ => pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false}))
        (fun e => do
          if ← Hammer.errorIsSimpPreprocessingError e then pure .simpPreprocessingFailure
          else if ← Hammer.errorIsTranslationError e then pure .tptpTranslationFailure
          else if ← Hammer.errorIsExternalSolverError e then pure .externalProverFailure
          else if ← Hammer.errorIsDuperSolverError e then pure .duperFailure
          else if ← Hammer.errorIsProofFitError e then pure .proofFitFailure
          else if "tactic 'simp' failed".isPrefixOf (← e.toMessageData.toString) then pure .simpPreprocessingFailure
          else if "tactic 'simp_all' failed".isPrefixOf (← e.toMessageData.toString) then pure .simpPreprocessingFailure
          else
            dbg_trace "runHammerCoreAtDecls :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
            pure .miscFailure
        )
    return some ⟨res, seconds, heartbeats⟩

/-- Like `runHammerCoreAtDecls` but only tests a single declaration (indicated by `declName`). -/
def runHammerCoreAtDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (withImportsPath : String) (jsonDir : String)
  (externalProverTimeout : Nat) (withSimpPreprocessing := true) : IO (Option (ConstantInfo × HammerResult)) := do
  runAtDecl mod declName (some withImportsPath) fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod "mathlib" jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if curDeclName == s!"{ci.name}" then
        ciEntry := jsonEntry
        break
    if ciEntry.isNull then
      return some ⟨.noJSON, 0.0, 0⟩
    let hammerRecommendation ← IO.ofExcept $ ciEntry.getObjVal? "declHammerRecommendation"
    let hammerRecommendation ← IO.ofExcept $ hammerRecommendation.getArr?
    let hammerRecommendation ← IO.ofExcept $ hammerRecommendation.mapM Json.getStr?
    let ((res, heartbeats), seconds) ← withSeconds <| withHeartbeats <|
      tryCatchRuntimeEx
        (TermElabM.run' (do
          dbg_trace "About to use hammer for {ci.name} in module {mod} (recommendation: {hammerRecommendation})"
          let gs ← Tactic.run g.mvarId! $ useHammerCore hammerRecommendation externalProverTimeout withSimpPreprocessing
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate the hammer on Prop declarations
          | _ :: _ => pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false}))
        (fun e => do
          if ← Hammer.errorIsSimpPreprocessingError e then pure .simpPreprocessingFailure
          else if ← Hammer.errorIsTranslationError e then pure .tptpTranslationFailure
          else if ← Hammer.errorIsExternalSolverError e then pure .externalProverFailure
          else if ← Hammer.errorIsDuperSolverError e then pure .duperFailure
          else if ← Hammer.errorIsProofFitError e then pure .proofFitFailure
          else if "tactic 'simp' failed".isPrefixOf (← e.toMessageData.toString) then pure .simpPreprocessingFailure
          else if "tactic 'simp_all' failed".isPrefixOf (← e.toMessageData.toString) then pure .simpPreprocessingFailure
          else
            dbg_trace "runHammerCoreAtDecls :: miscFailure for {ci.name} in module {mod}: {← e.toMessageData.toString}"
            pure .miscFailure
        )
    return some ⟨res, seconds, heartbeats⟩

/-- Like `runHammerCoreAtDecl` but only tests `simp_all` rather than `hammer`. Still uses the `hammerRecommendation` field in
    the JSON file -/
def runSimpAllAtDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (withImportsPath : String) (jsonDir : String) :
  IO (Option (ConstantInfo × GeneralResult)) := do
  runAtDecl mod declName (some withImportsPath) fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod "mathlib" jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if curDeclName == s!"{ci.name}" then
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
          dbg_trace "About to use simp_all for {ci.name} in module {mod} (recommendation: {recommendation})"
          let gs ← Tactic.run g.mvarId! $ useSimpAllWithRecommendation recommendation
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate on Prop declarations
          | _ :: _ => pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false})
      catch e =>
        dbg_trace "Encountered an error: {← e.toMessageData.toString}"
        pure .failure
    return some ⟨res, seconds, heartbeats⟩

/-- Like `runHammerAtDecl` but only tests `aesop` with `hammerCore` rather than just `hammerCore`.
    Still uses the `hammerRecommendation` field in the JSON file -/
def runAesopHammerCoreAtDecl (mod : Name) (declName : Name) (decls : ConstantInfo → MetaM Bool) (withImportsPath : String) (jsonDir : String)
  (externalProverTimeout : Nat) (withSimpPreprocessing : Bool) : IO (Option (ConstantInfo × GeneralResult)) := do
  runAtDecl mod declName (some withImportsPath) fun ci => do
    if ! (← decls ci) then return none
    let g ← mkFreshExprMVar ci.type
    -- Find JSON file corresponding to current `mod`
    let fileName := (← findJSONFile mod "mathlib" jsonDir).toString
    let jsonObjects ← IO.FS.lines fileName
    let json ← IO.ofExcept $ jsonObjects.mapM Json.parse
    -- Find `declHammerRecommendation` corresponding to current `ci`
    let mut ciEntry := Json.null
    for jsonEntry in json do
      let jsonDeclName ← IO.ofExcept $ jsonEntry.getObjVal? "declName"
      let curDeclName ← IO.ofExcept $ jsonDeclName.getStr?
      if curDeclName == s!"{ci.name}" then
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
          dbg_trace "About to use aesop with hammerCore for {ci.name} in module {mod} (recommendation: {recommendation})"
          let gs ← Tactic.run g.mvarId! $ useAesopHammerCore recommendation externalProverTimeout withSimpPreprocessing
          match gs with
          | [] => pure .success -- Don't need to case on whether `ci.type` is a Prop because we only evaluate on Prop declarations
          | _ :: _ => pure .subgoals)
          (ctx := {declName? := `fakeDecl, errToSorry := false})
      catch e =>
        dbg_trace "Encountered an error: {← e.toMessageData.toString}"
        pure .failure
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

def resultTypeToEmojiString (res : ResultType) : String :=
  match res with
  | .GeneralResult res => if res == .success then checkEmoji else crossEmoji
  | .HammerResult res => hammerResultTypeToEmojiString res
  | .QuerySMTResult res => querySMTResultTypeToEmojiString res

def tacticBenchmarkFromModule (module : ModuleName) (withImportsPath? : Option String) (tac : TacticM Unit) (tacType : TacType) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result := runTacticAtDecls module (fun _ => pure true) withImportsPath? tac tacType
  IO.println s!"{module}"
  for (ci, ⟨type, seconds, heartbeats⟩) in result do
    IO.println <| (resultTypeToEmojiString type) ++ " " ++ ci.name.toString ++
      s!" ({seconds}s) ({heartbeats} heartbeats)"
  return 0

def tacticBenchmarkAtDecl (module : ModuleName) (declName : Name) (withImportsPath? : Option String) (tac : TacticM Unit) (tacType : TacType) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result ← runTacticAtDecl module declName (fun _ => pure true) withImportsPath? tac tacType
  IO.println s!"Testing on {declName} in {module}"
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println <| (resultTypeToEmojiString type) ++ " " ++ ci.name.toString ++
      s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run tactic benchmark at {declName} in module {module}"
    return 0

def hammerCoreBenchmarkFromModule (module : ModuleName) (withImportsDir : String) (jsonDir : String) (externalProverTimeout : Nat) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result := runHammerCoreAtDecls module (fun ci => try isProp ci.type catch _ => pure false) withImportsDir jsonDir externalProverTimeout
  IO.println s!"{module}"
  for (ci, ⟨type, seconds, heartbeats⟩) in result do
    IO.println <| (hammerResultTypeToEmojiString type) ++ " " ++ ci.name.toString ++
      s!" ({seconds}s) ({heartbeats} heartbeats)"
  return 0

def hammerCoreBenchmarkAtDecl (module : ModuleName) (declName : Name) (withImportsDir : String) (jsonDir : String) (externalProverTimeout : Nat)
  (withSimpPreprocessing := true) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result ← runHammerCoreAtDecl module declName (fun ci => try isProp ci.type catch _ => pure false) withImportsDir jsonDir externalProverTimeout withSimpPreprocessing
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $ (hammerResultTypeToEmojiString type) ++ " " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run hammer benchmark at {declName} in module {module}"
    return 0

def simpAllBenchmarkAtDecl (module : ModuleName) (declName : Name) (withImportsDir : String) (jsonDir : String) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result ← runSimpAllAtDecl module declName (fun ci => try isProp ci.type catch _ => pure false) withImportsDir jsonDir
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $
      (if type == .success then checkEmoji else if type == .failure then crossEmoji else bombEmoji) ++
      " " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run simpAll benchmark at {declName} in module {module}"
    return 0

def aesopHammerCoreBenchmarkAtDecl (module : ModuleName) (declName : Name) (withImportsDir : String) (jsonDir : String) (externalProverTimeout : Nat)
  (withSimpPreprocessing := false) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let result ← runAesopHammerCoreAtDecl module declName (fun ci => try isProp ci.type catch _ => pure false) withImportsDir jsonDir externalProverTimeout withSimpPreprocessing
  match result with
  | some (ci, ⟨type, seconds, heartbeats⟩) =>
    IO.println $
      (if type == .success then checkEmoji else if type == .failure then crossEmoji else bombEmoji) ++
      " " ++ ci.name.toString ++ s!" ({seconds}s) ({heartbeats} heartbeats)"
    return 0
  | none =>
    IO.println s!"Encountered an issue attempting to run aesopHammerCore benchmark at {declName} in module {module}"
    return 0

def tacticBenchmarkMain (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "module" |>.as! ModuleName
  let declName := args.positionalArg! "declName" |>.as! String |>.toName
  let premisesPath := args.positionalArg! "premisesPath" |>.as! String
  let benchmarkType := args.positionalArg! "benchmarkType" |>.as! String
  let externalProverTimeout :=
    match args.positionalArg? "externalProverTimeout" with
    | some externalProverTimeout => externalProverTimeout.as! Nat
    | none => 10

  try
    match benchmarkType with
      | "duper" => tacticBenchmarkAtDecl module declName (some premisesPath) useDuper TacType.General
      | "querySMT" => tacticBenchmarkAtDecl module declName (some premisesPath) useQuerySMT TacType.QuerySMT
      | "aesop" => tacticBenchmarkAtDecl module declName (some premisesPath) useAesop TacType.General
      | "exact" => tacticBenchmarkAtDecl module declName (some premisesPath) useExact? TacType.General
      | "rfl" => tacticBenchmarkAtDecl module declName (some premisesPath) useRfl TacType.General
      | "simp_all" => tacticBenchmarkAtDecl module declName (some premisesPath) useSimpAll TacType.General
      | "omega" => tacticBenchmarkAtDecl module declName (some premisesPath) useOmega TacType.General
      | "hammer" => tacticBenchmarkAtDecl module declName (some premisesPath) (useHammer externalProverTimeout) TacType.Hammer

      | "simp_all_with_premises" => simpAllBenchmarkAtDecl module declName "Examples/Mathlib/WithImports" premisesPath
      | "hammerCore" => hammerCoreBenchmarkAtDecl module declName "Examples/Mathlib/WithImports" premisesPath 10
      | "hammerCore_without_simp_preprocessing" => hammerCoreBenchmarkAtDecl module declName "Examples/Mathlib/WithImports" premisesPath 10 false

      | _ => IO.throwServerError s!"Unknown benchmark type {benchmarkType}"
  catch e =>
    IO.eprintln s!"{benchmarkType} failed with error {e}"
    return (1 : UInt32)

/-- Setting up command line options and help text for `lake exe tactic_benchmark`. -/
def tactic_benchmark : Cmd := `[Cli|
  tactic_benchmark VIA tacticBenchmarkMain; ["0.0.1"]
  "Run a customisable tactic at all declarations in a file."

  ARGS:
    module : ModuleName; "Lean module to run the tactic on."
    declName : String; "Name of the declaration to run the tactic on."
    premisesPath : String; "Path to the premises, such as Examples/Mathlib/TrainingDataWithPremises."
    benchmarkType : String; "Which type of tactic to run (e.g. hammer, hammerCore, aesop, exact)."
    externalProverTimeout : Nat; "Timeout for the external prover (default 10)."
]

/-- `lake exe tactic_benchmark` -/
def main (args : List String) : IO UInt32 :=
  tactic_benchmark.validate args

-- See `scripts/tactic_benchmark.sh` for a script to run this on all of Mathlib.

-- #eval tacticBenchmarkFromModule `temp useDuper
-- #eval tacticBenchmarkFromModule `temp useQuerySMT

-- Note: `tacticBenchmarkFromModule` requires that the tactic we want be imported in the module
