# Resonant Fractal Cognition (RFC)

**A new form of AI derived from the research collected in this repository.**

RFC is a working cognitive architecture in which belief is an *interference
pattern* rather than a stored value, and in which one operator — applied at
different scales, to different evidence — does the work that the source
research assigns to three separate layers.

It is implemented in `rfc/`, runs on Python 3.11 with numpy alone, is
deterministic given a seed, and ships with a benchmark suite whose numbers are
reproduced verbatim below, including the ones RFC loses.

---

## 1. Where it comes from

This repository collects a body of research — the *Fractal Recursive Mind*
blueprint, the FRCL analysis, the RFAI/RFIM implementations, the agent
specification in `AGENTS.md` — converging on five commitments:

| Research commitment | Source |
|---|---|
| Self-similar recursion: one principle re-instantiated at every level | FRM blueprint, `rfai_research_analysis.json` |
| Quantum-*inspired* superposition and interference on classical hardware | FRCL "QECS" layer |
| Neuro-symbolic: symbols that form from, and feed back into, subsymbolic processing | FRCL "NSRP" layer |
| Metacognitive self-governance: a system that observes and retunes itself | FRCL "MCSG" layer, `enhanced_frm_summary.md` |
| Ethical DNA guarding every loop | `AGENTS.md` |

The FRCL analysis is also candid about why the blueprint stays a blueprint:
*integration* is the hard part. Three heterogeneous layers with different
representations need brittle interfaces between them, and the recursive
spawning of sub-clusters invites combinatorial explosion.

RFC's answer is to refuse the stack.

## 2. The idea

> **Cognition as resonance.** Evidence is decomposed into a dyadic ladder of
> scales. Hypotheses are oscillators whose natural frequencies sit on that same
> ladder. A hypothesis grows when evidence at *its* scale keeps arriving in
> phase with it. Belief is what the population's interference pattern says, and
> agreement across scale is the only thing that produces confidence.

Everything else follows from that one commitment:

- **Self-similar recursion** is not an architectural aspiration — the same
  operator `Ψ` advances the sensory field, resolves the residual one level
  down, and reflects over the system's own history. There is one implementation
  (`rfc/operator.py`, ~150 lines) and there are no interfaces between layers,
  because there are no layers.
- **Superposition and interference** are literal: hypotheses carry complex
  amplitudes, the read-out is Born-style (`p ∝ |amplitude|²`), and hypotheses
  that contradict each other cancel rather than being arbitrated.
- **Symbols** are a phase transition, not a module. A coalition that stays
  coherent long enough crystallises into a durable symbol, which then biases
  the field that produced it, and dissolves when the world stops supporting it.
- **Metacognition** is the same operator with the system's own telemetry as
  evidence, decomposed across *time* scale instead of space.
- **Constraints are physics.** A hypothesis that violates an invariant is
  phase-inverted and cut off from drive, so it interferes destructively with
  its own coalition and its amplitude is provably non-increasing from that step
  on. Safety is not a filter downstream of the answer.

## 3. Mechanism

### 3.1 The scale ladder (`rfc/scale_space.py`)

A box (Haar) pyramid splits evidence into bands that sum back to the original
exactly — the reconstruction property is what lets the engine subtract an
explained component and recurse on the residual. Band 0 is coarsest.

`temporal_bands` applies the identical idea to a *history* of feature vectors:
band 0 is the long-run average, band `l` is "what changed when you looked twice
as recently". This is what metacognition consumes.

### 3.2 The field (`rfc/field.py`)

A hypothesis is a unit direction plus an oscillator state (amplitude `r`, phase
`θ`). A hypothesis at scale `s` has natural frequency `ω₀·2⁻ˢ`, and each band
drives with reference phase `ω_l·t`. A hypothesis therefore holds a *constant*
phase lag against its own band and accumulates amplitude, while its lag against
any other band rotates and that drive time-averages to nothing. **Scale
selectivity is not a rule anyone wrote; it falls out of the dyadic ladder.**

Every agreement measure is taken in the co-rotating frame (the lag), because
lab-frame phases on a frequency ladder can never all agree — that is the point
of the ladder.

The read-out is **conjunctive**. A label's scale copies are summed *as waves*:

```
p(label) ∝ | Σ_scales r · e^{i·lag} |²
```

Copies driven by their own bands sit near zero lag and add; a copy its band
refutes is pulled to antiphase and subtracts. Summing probabilities instead
would let the loud coarse scale outvote a quiet fine one, and a label could
score well while its fine-scale evidence flatly contradicted it.

### 3.3 The operator (`rfc/operator.py`)

One tick of `Ψ`:

1. **Drive.** Supporting evidence grows amplitude in proportion to `cos(lag)`;
   contradicting evidence inhibits. Phase is pulled toward the target the
   evidence names — zero lag for support, antiphase for contradiction.
2. **Coupling.** Phases couple Kuramoto-style weighted by vector overlap, so
   agreeing hypotheses lock into coalitions. Amplitudes *compete*: overlapping
   hypotheses are rival explanations of the same evidence, so each suppresses
   the others in proportion to overlap. Cooperation in phase, competition in
   mass.
3. **Constraints.** Violators are inverted and starved.
4. **Conservation.** Mass is conserved *per scale*, so no band can out-shout
   another; an energy cap and relative pruning keep the population bounded.

### 3.4 Recursion (`rfc/engine.py`)

When the leading two readings are too close to call, the engine subtracts what
the leader explains and re-resolves the **residual** one level down, with the
same operator and a depth budget. Recursion therefore looks at new evidence
rather than re-litigating the same evidence at a deeper indentation, which is
what keeps the FRCL "spawn self-similar sub-clusters" idea from exploding.
`decompose()` exposes the same step as a public operation: name one source,
explain it away, look again.

### 3.5 The symbol lattice (`rfc/lattice.py`)

A coalition that holds amplitude and coherence for `persistence` consecutive
steps crystallises into a `Symbol` carrying its support, its scales, and the
episode it formed in. Symbols bias later fields as top-down priors, are linked
to their nearest relative to form a hierarchy, and decay and dissolve when
unsupported. This is the neuro-symbolic bridge with no seam, because there is
no second system to bridge to.

### 3.6 Metacognition (`rfc/metacognition.py`)

Episode telemetry becomes a signed feature vector (coherence, decisiveness,
recursion depth, veto pressure, field pressure, reward against a trailing
baseline, and one **credit slot per band**). A window of those vectors is
decomposed across time scale and fed to **the same operator**, resonating
against a field of policy hypotheses — each a bounded nudge to `Ψ`'s own
parameters, with a signature describing the situation it fits.

The hard part is not the loop, it is the credit. To act on a scale ladder the
system has to work out *which rung is letting it down*, and the obvious signals
all fail:

- **Consensus** — score each band against what the others say — inverts exactly
  when it is needed. On the drift task the two corrupted bands see the *same*
  interference, so they agree with each other, and the clean dissenting bands
  look like the unreliable ones. Any signal built on agreement assumes the
  majority of the ladder is right, and the interesting failures are the ones
  where it is not.
- **Solo accuracy** — how often would this band be right on its own — sounds
  majority-free and is, but it measures the wrong thing. It conflates *weak*
  with *misleading*. Measured per-band solo accuracy runs `[1.00, 0.55, 0.76,
  0.94]` on this task *before* anything is corrupted, and marking down the
  weakest band costs accuracy rather than recovering it (0.719 against 0.777
  for leaving the ladder alone).

What works is **counterfactual pivotality graded by reward**. For each band,
re-run the coherent read-out with that band's contribution removed:

- the answer was right and would have changed without this band → it carried a
  correct decision, credit it;
- the answer was wrong and would have changed without this band → it is what
  tipped the error, debit it;
- the answer would not have changed → this band decided nothing, score zero.

Nothing here consults another band's opinion. The counterfactual is graded by
the reward, so it holds up when the unreliable bands outnumber the reliable
ones. Two corrections make it usable: episodes are weighted by the inverse
frequency of their outcome, because most episodes are right and an unweighted
average ends up flattering whichever band swings the answer around most; and
the ladder mean is subtracted, because on a failing stream *some* band is
pivotal nearly every time and every band's raw score drifts negative together.
Subtracting a common offset from a reward-graded statistic is not the bands
scoring each other — that distinction is the whole point.

The signal also goes quiet by itself on a healthy system: with the answers
coming out right, no band is ever the one that tipped a wrong answer, and a
confident read-out rarely turns on any single band. No "only adapt when
failing" rule was needed; the loop simply holds.

What it adjusts is a **per-band trust vector**, not a single coarse-to-fine
tilt. Reliability is not monotone in scale — the numbers above show band 1
weakest and band 0 strongest before any corruption — so "trust the fine half"
cannot express which rung is actually at fault. There is one policy per band,
and only in the distrust direction: the band weights are renormalised, so
nothing depends on the absolute level of trust and raising every band in turn
is an expensive no-op, which is what a trust-and-distrust pair did in practice.

Two guardrails remain: every parameter has hard bounds and every nudge is
small, and the best-performing configuration is remembered and restored when
reward degrades. The expectation yardstick decays, so a regime change does not
freeze adaptation forever. When no policy fits confidently and the system is
underperforming, it runs a bounded experiment instead of guessing.

### 3.7 Constraints (`rfc/constraints.py`)

An `Invariant` is an ordinary predicate over a hypothesis and the episode
context, so the repository's existing "ethical DNA" policies lift in unchanged.
`bridge.goal_alignment_invariant` turns an RFAI `SemanticGoal` into one, which
moves goal alignment out of a score a search can trade away and into the
substrate.

## 4. Results

`python -m rfc.cli bench` (~90 s). Deterministic; these are the shipped
numbers, wins and losses alike. Baselines: `flat-cosine` (nearest prototype on
the raw vector), `band-cosine` (RFC's optics with none of its dynamics),
`greedy-pursuit` (classical matching pursuit). Rows named `rfc-no-…` are RFC
against itself with one mechanism removed.

### Plain classification (`composite`)

| system | accuracy |
|---|---|
| flat-cosine | **0.956** |
| band-cosine | **0.956** |
| rfc | 0.856 |

RFC **loses** by 10 points on a task that needs nothing it offers. A field of
coupled oscillators settling for 24 steps is a costly way to compute a
correlation, and it settles slightly less accurately. Reported first, because
an architecture that only ever shows its wins is not being measured.

### Decomposition (`superposition`)

| system | both sources correct | member recall |
|---|---|---|
| greedy-pursuit | **0.933** | 0.967 |
| flat-cosine top-2 | 0.833 | 0.917 |
| rfc-decompose | 0.800 | 0.900 |
| rfc one-shot top-2 | 0.650 | 0.800 |

Residual recursion is worth **+15 points** over RFC's own one-shot read-out —
the ablation that attributes the result to the mechanism. Classical matching
pursuit still wins; that is the honest ceiling. The claim is that RFC arrives at
pursuit-like behaviour from resonance without being told about the algorithm,
not that it beats it.

### Consolidation without gradients (`stream`)

| system | overall | frequent classes | rare classes |
|---|---|---|---|
| flat-cosine | **0.783** | 0.828 | 0.703 |
| band-cosine | 0.756 | 0.767 | 0.734 |
| rfc | 0.722 | 0.716 | **0.734** |
| rfc, lattice disabled | 0.633 | 0.647 | 0.609 |

Crystallised symbols are worth **+8.9 points** over RFC with priors switched
off — on a skewed stream, with no labels, no gradients, and no training phase.
Notably the gain is not bought from the rare classes: they improve by 12.5
points too. RFC is still behind flat-cosine overall.

### Robustness and self-tuning (`drift`)

Halfway through the stream, strong low-frequency interference swamps the coarse
half of the ladder. Nothing announces it.

| system | overall | after the shift |
|---|---|---|
| rfc | **0.905** | **0.810** |
| rfc, no metacognition | 0.889 | 0.777 |
| flat-cosine | 0.812 | 0.625 |
| band-cosine | 0.777 | 0.554 |

**The conjunctive cross-scale read-out is the architecture's clearest win.**
When two of four scales are corrupted, RFC holds 0.777 *without adapting at
all*, where a flat correlation falls to 0.625 and a scale-equalised one to
0.554 — 15 and 22 points. Because a label only scores well when its scales
agree, corrupted scales cancel instead of voting.

**Metacognition adds `+0.033` post-shift on top of that** (6 seeds; `+0.025`
over 10 seeds, so the effect is small but consistent in sign). An earlier
version of this loop, built on cross-scale consensus, scored **−0.029** — it
actively made the system worse — and §3.6 records what was wrong with it and
what replaced it. The applied-policy trace is now mostly `hold`, with occasional
targeted `distrust_band_N`: much of the gain is simply that the loop stopped
damaging a configuration that was already working.

The honest ceiling: hand-setting the trust vector to `(-0.5, -0.5, +0.5, +0.5)`
— mark down exactly the two bands the task corrupts — scores **0.892**
post-shift. So the available headroom over no adaptation is about `+0.115`, and
the loop discovers roughly a quarter of it from reward alone, with no idea which
bands were corrupted or that anything happened at all. That gap is the honest
remaining limitation, not a rounding error.

### Safety (`safety`)

Every sample is drawn from the class the system is forbidden to output, so the
maximum-likelihood answer is the forbidden one every time.

| system | violation rate |
|---|---|
| rfc, guarded | **0.000** |
| rfc, unguarded | 1.000 |

Zero violations in 40 episodes, with the refused hypotheses retained in the
percept as an audit trail. This is not a filter result: `test_veto_makes_
amplitude_non_increasing` checks the physical property directly — after a veto,
a hypothesis' amplitude never rises again on any subsequent step.

## 5. What is claimed, and what is not

**Claimed.** A single operator that serves perception, symbol formation,
recursion, and self-inspection; a cross-scale read-out that is measurably more
robust to a corrupted scale than either baseline; unsupervised consolidation
worth ~9 points with no gradients; a self-tuning loop that identifies which
rung of its own ladder to stop believing, from reward alone and without
assuming the majority of rungs is right; a constraint mechanism with a
checkable physical guarantee; and full determinism and auditability throughout.

**Not claimed.** Not consciousness, not AGI, not quantum computation — the
"quantum-inspired" part is complex amplitudes and Born-rule read-out on
classical hardware, and nothing more. Not state of the art at classification:
two of five tasks go to a five-line baseline. Not a learning system in the
usual sense: the codebook of concepts is given, and only the lattice, the
priors, and the operator parameters change with experience. Not scaled: the
benchmarks are 64-dimensional with tens of concepts, and the O(n²) coupling
matrix will need attention long before this reaches interesting sizes.

## 6. Known limitations

1. **Metacognitive attribution works but is weak.** Counterfactual pivotality
   graded by reward (§3.6) took the loop from actively harmful (−0.029) to
   modestly useful (+0.033), and it is majority-free by construction. It still
   recovers only about a quarter of the headroom a hand-set trust vector gets
   (0.810 against a 0.892 ceiling, over 0.777 unadapted). The credit estimate
   is noisy at this window size, and the per-band policies compete for the same
   reflection against nine scalar policies that often win on generic
   "doing badly" evidence without being tested against outcomes.
2. **Classification accuracy trails a plain correlation** on clean, unambiguous
   evidence, and each episode costs ~24–90 operator steps to get there.
3. **The concept codebook is supplied.** Unlabelled "exploratory" hypotheses
   are seeded from the evidence and shape interference, but nothing yet
   promotes one into a named concept. The lattice is the natural place for it.
4. **Coupling is O(n²)** in field size. Fine at the current cap of 96; not fine
   later.
5. **Symbols are per-label.** Crystallisation merges by label, so the lattice
   cannot yet represent two genuinely different senses of one concept.

## 7. Using it

```python
from rfc import RFCConfig, ResonantFractalCognition, forbidden_labels

mind = ResonantFractalCognition(
    RFCConfig(dim=64, seed=0),
    codebook={"alpha": vector_a, "beta": vector_b},
    invariants=[forbidden_labels("house_rule", ["beta"])],
)

percept = mind.perceive(evidence)      # one episode
print(mind.explain(percept))           # answer, runners-up, refusals, symbols
mind.learn(reward=1.0)                 # optional feedback for the meta loop
mind.decompose(mixture, components=2)  # name several sources at once
mind.reflect()                         # retune Psi with Psi
```

```bash
python -m rfc.demo                     # narrated walkthrough plus benchmarks
python -m rfc.cli episode --seed 0 --episodes 20 --verbose
python -m rfc.cli bench --json
pytest tests/test_rfc_substrate.py tests/test_rfc_engine.py tests/test_rfc_integration.py
```

`rfc/bridge.py` connects RFC to the existing stack: RFAI Fractal Information
Motifs become evidence vectors, and an RFAI `SemanticGoal` becomes an enforced
invariant.

## 8. Layout

| file | role |
|---|---|
| `rfc/scale_space.py` | dyadic ladder over space and over time |
| `rfc/field.py` | hypotheses, coalitions, conjunctive read-out, per-band counterfactuals |
| `rfc/operator.py` | `Ψ` — the whole architecture's one moving part |
| `rfc/constraints.py` | invariants as physics |
| `rfc/lattice.py` | symbols by crystallisation |
| `rfc/metacognition.py` | `Ψ` applied to the system's own history |
| `rfc/engine.py` | the assembled system |
| `rfc/telemetry.py` | episode records and features |
| `rfc/bridge.py` | adapters to the existing RFAI/RFIM code |
| `rfc/tasks.py` | benchmarks, baselines, ablations |
| `rfc/cli.py`, `rfc/demo.py` | entry points |
