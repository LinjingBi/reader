"""
experiment only spec. not used in production.

this baseline spec is for collecting evidence gaps and next step inputs to pump the "chunks db". it should not be used in production.
For **research_briefing / brainstorm_directions / implementation_angle**, the evidence provided is intentionally limited by the metadata we currently store (mostly **summary-level** fields plus `new_observation` and optional `history_reports`).

This is **known to be insufficient** for high-specificity planning (e.g., methods, experiments, limitations, eval protocols, feasibility constraints). The prompt is therefore **designed to bias the planner toward**:

* producing a conservative, evidence-grounded `plan` (avoid false concreteness), and
* using `evidence_gaps` aggressively to request the missing artifacts needed to fulfill the intent.

In other words: **insufficiency is a feature in MVP**—it forces the model to surface evidence gaps instead of hallucinating details.
In oother words: current version of plan intends to do two things at the same time: 1. review the evidence and 2. prepare the next step inputs.
and these two can be used to leverage the maturity of call1/the plan before start writing/call2, or frankly spkeaing, call1 should be called iteratively until it's "matured".


"""
from dataclasses import dataclass
from typing import Mapping, Sequence, List, Dict, Optional
from enum import Enum


BASE_UNIVERSAL = """SYSTEM:
You are a technical report planner.

HARD RULES (anti-hallucination):
- Use ONLY the provided evidence in the input JSON. Do NOT invent methods, results, datasets, metrics, numbers, or implementation details.
- Treat intent_mode as first class evidence: it is a user goal signal and must influence planning decisions.
- If you cannot derive any plan field without making up details:
  1) Fill that field with a conservative placeholder derived from available evidence (usually keywords/themes).
  2) Add an item to evidence_request describing what evidence is needed and why.
- Do NOT output Markdown. Return valid JSON only.
- Output JSON structure is enforced by a typed schema. Focus on semantic correctness and evidence grounding.

TASK:
Given the evidence pack below:
<EVIDENCE_PACK_PLACEHOLDER>

Produce a technical report writing plan. Plan guidance:
<PLAN_GUIDANCE_PLACEHOLDER>

After the plan is complete, review it and produce:
- a list of evidence gaps required to refine the plan
- a list of next step inputs required to write the report
"""



@dataclass(frozen=True)
class FieldGuidance:
    question: str                 # "What question is this field answering?"
    use: Sequence[str]            # allowed evidence keys / signals
    decide: Sequence[str]         # decision rules / examples
    if_blocked: Optional[Sequence[str]] = None    # what to do if you can't derive without invention
    if_unblocked: Optional[Sequence[str]] = None  # what to collect/prepare for next step when you can derive without invention



@dataclass(frozen=True)
class IntentSpec:
    intent_goal: str
    # Guidance keyed by Plan field name (must match pydantic model fields)
    plan_field_guidance: Mapping[str, FieldGuidance]
    # !!! notes should be introduced only when observed real model failure modes
    evidence_gaps_notes: Optional[Sequence[str]] = None
    next_step_inputs_notes: Optional[Sequence[str]] = None


# ----------------------------
# intent specs
# ----------------------------


QUICK_BACKGROUND_SPEC = IntentSpec(
    intent_goal="Fast orientation: what it is / why it matters / what to look at next.",
    plan_field_guidance={
        "depth_mode_final": FieldGuidance(
            question="Relative to history, is this onboarding, continuation, deepening, or reframing?",
            use=["intent_mode", "history_reports (if present)", "new_observation keywords/themes"],
            decide=[
                "Onboard: no history or first-time orientation framing dominates",
                "Continue: history exists and themes align; focus on delta and avoid repeats",
                "Deepen: stable topic, coherent slice emerges; keep conceptual unless richer evidence exists",
                "Restructure: drift/mixed signals; reframe subthreads and add uncertainty",
            ],
            if_blocked=[
                "If history is required but missing, choose Onboard and add an evidence_request item asking for history_reports.",
            ],
            if_unblocked=[
                "Add next_step_inputs to support the writer: request only the minimal history slice needed to execute the chosen depth (e.g., last report summary + covered_bullets/next_targets if the plan is Continue/Deepen/Restructure).",
            ],
        ),
        "declared_level_final": FieldGuidance(
            question="Given the intent and evidence richness, how specific can the report be?",
            use=["intent_mode", "new_observation specificity", "history_reports deltas", "paper-level details ONLY if present"],
            decide=[
                "intro: observation broad/generic OR no history",
                "intermediate: clear keyword structure and/or history supports deltas",
                "deep-dive: ONLY if explicit methods/experiments/limitations or full-text extracts exist",
            ],
            if_blocked=[
                "If deeper specificity is desired but missing: choose intermediate, and add an evidence_gaps item requesting paper sections that enable deep-dive claims (methods/experiments/limitations).",
            ],
            if_unblocked=[
                "Add next_step_inputs that match the chosen level: if intermediate, request Top-1~Top-3 paper summaries (if absent); if deep-dive, request the exact enabling sections (methods/experiments/limitations) for the most central paper_ids.",
            ],

        ),
        "subthreads_final": FieldGuidance(
            question="What 2–4 thematic buckets should organize the report?",
            use=["intent_mode", "key_paper_keywords (preferred)", "new_observation keywords", "history_reports (avoid repeats)"],
            decide=[
                "Create 2–4 subthreads grounded in keywords/themes",
                "paper_ids: include only if provided; otherwise []",
            ],
            if_blocked=[
                "If keywords are too generic to form subthreads: create broader buckets, and add an evidence_gaps item requesting key_paper_keywords or Top-K paper summaries (with paper_ids).",
            ],
            if_unblocked=[
                "Add next_step_inputs that make subthreads writer-ready: request a small per-subthread evidence bundle (e.g., 2–3 representative keywords + 1–2 supporting paper summaries if available).",
            ],

        ),
        "next_targets": FieldGuidance(
            question="What should we collect/verify next to improve understanding?",
            use=["intent_mode", "new_observation gaps/ambiguities", "history_reports next_targets (optional)"],
            decide=[
                "3–8 actionable follow-ups",
                "Include evidence upgrades (e.g., PDF intro+conclusion, methods/limitations) when needed",
            ],
            if_blocked=[
                "If you cannot propose specific next_targets without guessing: keep next_targets generic, and add evidence_gaps items for the missing artifacts that would make them concrete (e.g., intro+conclusion / methods / eval protocol).",
            ],
            if_unblocked=[
                "Translate next_targets into next_step_inputs: for each high-value target, request the artifact needed to execute it in Call 2 (and link it to the target).",
            ],

        ),
        "outline": FieldGuidance(
            question="What section order best fits this intent?",
            use=["intent_mode", "depth_mode_final", "declared_level_final", "subthreads_final", "history_reports (if not Onboard)"],
            decide=[
                "framing/definition → why now → subthreads → synthesis → evidence gaps → next_targets",
                "if not Onboard, add a 'delta vs history' bullet early",
            ],
            if_blocked=[
                "If delta-vs-history is needed but history_reports are missing: omit delta bullets, and add an evidence_gaps item requesting history_reports (compressed metadata).",
            ],
            if_unblocked=[
                "For each outline item that needs evidence (even in quick_background), add next_step_inputs so the writer gets the minimal materials per section (especially for 'what's new', 'why it matters', and 'evidence gaps').",
            ],

        ),
        "skip_or_defer": FieldGuidance(
            question="What must the writer avoid expanding on to prevent speculation or repetition?",
            use=["intent_mode", "known evidence limits", "history_reports covered_bullets (optional)"],
            decide=[
                "Defer numeric superiority claims, dataset specifics, feasibility claims unless explicitly supported",
            ],
            if_blocked=[
                "When uncertain whether something is supported, add an evidence_gaps item describing what evidence would be required to safely include it, and keep it deferred.",
            ],
            if_unblocked=[
                "Convert deferrals into retrieval: add next_step_inputs for the highest-impact deferred items (only if they align with intent_mode), so Call 2 can upgrade them safely.",
            ],

        ),
        "sufficiency": FieldGuidance(
            question="Is the evidence enough to produce a coherent report for this intent without guesswork?",
            use=["intent_mode", "new_observation quality", "ability to derive subthreads/outline", "history availability if not Onboard"],
            decide=[
                "sufficient: clear observation + coherent keywords",
                "borderline: weak/mixed observation or missing history when needed",
                "insufficient: observation missing/thin or too generic to form subthreads",
            ],
        ),
    },
)
RESEARCH_BRIEFING_SPEC = IntentSpec(
    intent_goal="Decision-oriented synthesis (≈10–20 min) grounded in Top-K paper summaries; structured comparison + explicit gaps.",
    plan_field_guidance={
        "depth_mode_final": FieldGuidance(
            question="Relative to history, is this a first briefing, an update, a deeper focus, or a reframe?",
            use=[
                "intent_mode (research_briefing)",
                "history_reports (presence/absence; covered_bullets/next_targets if present)",
                "top_papers cohesion + novelty signals (keywords overlap, summary themes)",
                "new_observation keywords/themes",
            ],
            decide=[
                "Onboard: no history_reports OR first time producing a decision briefing for this topic",
                "Continue: history exists and Top-K aligns; focus on delta and avoid repeats",
                "Deepen: Top-K concentrates on a coherent niche; go deeper within summary-supported bounds",
                "Restructure: Top-K looks mixed/drifting; reframe subthreads and increase uncertainty flags",
            ],
            if_blocked=[
                "If choosing Continue/Deepen/Restructure depends on missing history_reports: choose Onboard conservatively and add an evidence_gaps item requesting history_reports.",
            ],
            if_unblocked=[
                "Add next_step_inputs to support the chosen depth: if Continue, request last report's covered_bullets + next_targets; if Deepen, request Top-1~Top-2 paper intro+conclusion to deepen safely; if Restructure, request a slightly wider Top-K (or cluster membership stats) to confirm mixedness.",
            ],

        ),
        "declared_level_final": FieldGuidance(
            question="Given research_briefing intent and summary-level evidence, how technically specific can we be?",
            use=[
                "intent_mode (bias toward intermediate)",
                "top_papers summary specificity (do summaries mention concrete mechanisms or remain vague?)",
                "explicit methods/experiments/limitations ONLY if present in input",
                "history_reports (optional) for delta framing",
            ],
            decide=[
                "intro: only if summaries are extremely thin/generic",
                "intermediate: default for research_briefing with Top-K summaries",
                "deep-dive: ONLY if explicit methods/experiments/limitations or full-text extracts exist in input",
            ],
            if_blocked=[
                "If deeper specificity would require methods/experiments/limitations: choose intermediate and add evidence_gaps items requesting those sections for the most central paper_ids.",
            ],
            if_unblocked=[
                "Add next_step_inputs that align with level: for intermediate, request missing Top-K paper metadata fields (keywords, clearer summaries) if absent; for deep-dive, request the exact missing sections (methods/experiments/limitations/eval protocol) for the paper_ids you will cite most.",
            ],

        ),
        "subthreads_final": FieldGuidance(
            question="What 2–4 buckets best partition the Top-K papers into coherent sections?",
            use=[
                "intent_mode (paper-grounded partition)",
                "top_papers summaries + keywords",
                "new_observation keywords (optional anchor)",
                "history_reports (optional) to avoid repeating identical buckets unless needed",
            ],
            decide=[
                "Create 2–4 subthreads; each Top-K paper_id should appear in exactly one subthread",
                "Subthread names must reflect shared keywords/themes visible in summaries (no invented claims)",
            ],
            if_blocked=[
                "If Top-K is incoherent or too vague to partition: make broader buckets and add evidence_gaps items requesting richer paper evidence (intro+conclusion or expanded Top-K).",
            ],
            if_unblocked=[
                "Add next_step_inputs per subthread: for each bucket, request 1–2 strongest supporting paper sections (intro+conclusion by default; methods only if required by outline).",
            ],

        ),
        "next_targets": FieldGuidance(
            question="What follow-ups would materially improve confidence or enable deeper decisions next time?",
            use=[
                "intent_mode (decision-oriented next actions)",
                "gaps detected across Top-K summaries",
                "history_reports next_targets (continue or avoid repeats)",
                "new_observation ambiguities",
            ],
            decide=[
                "3–8 actionable items; at least 2 should be evidence upgrades enabling stronger claims",
                "Prefer paper-specific requests (cite paper_id) when possible (methods/experiments/limitations, eval protocol, intro+conclusion)",
            ],
            if_blocked=[
                "If concrete follow-ups would require missing details: keep them generic and add evidence_gaps items specifying the missing artifacts (paper-specific if possible).",
            ],
            if_unblocked=[
                "Turn the top next_targets into next_step_inputs: each target becomes an actionable artifact request for Call 2 (paper_id + section).",
            ],

        ),
        "outline": FieldGuidance(
            question="What section order best supports a decision briefing?",
            use=[
                "intent_mode (decision-first structure)",
                "depth_mode_final + declared_level_final",
                "subthreads_final",
                "history_reports (optional) for delta section",
            ],
            decide=[
                "Decision framing → 1–2 sentence thesis (from evidence) → subthreads → cross-cutting comparisons → risks/gaps → next_targets",
                "If Continue/Deepen/Restructure: include explicit 'delta vs history' early",
            ],
            if_blocked=[
                "If delta framing depends on missing history_reports: omit delta section and add an evidence_gaps item requesting history_reports.",
            ],
            if_unblocked=[
                "Map outline → next_step_inputs: any outline item requiring higher resolution than summary-level must have a paired next_step_input (e.g., comparisons need eval protocol; risks need limitations/failure modes).",
            ],

        ),
        "skip_or_defer": FieldGuidance(
            question="What must the writer avoid asserting given summary-level evidence?",
            use=[
                "intent_mode",
                "absence of explicit quantitative/method details in Top-K",
                "history_reports covered_bullets (optional)",
            ],
            decide=[
                "Defer numeric superiority, benchmark scores, dataset specifics, and feasibility claims unless explicitly supported in input",
            ],
            if_blocked=[
                "When tempted to assert numeric superiority or benchmark wins without explicit evidence: defer it and add an evidence_gaps item stating what would be required to support it.",
            ],
            if_unblocked=[
                "Add next_step_inputs for the top deferred-but-important claims (e.g., request the eval setup section or results table excerpt) so Call 2 can upgrade safely.",
            ],

        ),
        "sufficiency": FieldGuidance(
            question="Is evidence sufficient for a decision-oriented briefing without speculation?",
            use=[
                "intent_mode (higher bar than quick_background)",
                "Top-K thematic cohesion and specificity",
                "presence/quality of new_observation",
                "history availability if depth_mode_final != Onboard",
            ],
            decide=[
                "sufficient: coherent Top-K theme + reasonably specific summaries",
                "borderline: mixed/vague Top-K; must caveat and request evidence",
                "insufficient: missing/empty summaries or highly incoherent evidence",
            ],
        ),
    },
)


BRAINSTORM_DIRECTIONS_SPEC = IntentSpec(
    intent_goal="Novelty hunting + idea generation; be conservative with summary-level evidence and push missing detail into evidence_request.",
    plan_field_guidance={
        "depth_mode_final": FieldGuidance(
            question="Do we brainstorm within a stable theme, or do we first need to reframe because evidence is mixed/drifting?",
            use=[
                "intent_mode (brainstorm_directions)",
                "Top-K cohesion / mixedness signals",
                "new_observation keywords/themes",
                "history_reports (optional) for continuity vs drift",
            ],
            decide=[
                "Onboard: if no history and brainstorming must stay conceptual",
                "Continue: if history exists and theme is stable; brainstorm 'next deltas' and unexplored angles",
                "Deepen: if Top-K/new_observation reveals a coherent niche; brainstorm hypotheses within that niche (no fabricated mechanisms)",
                "Restructure: if mixed; brainstorm axes and alternative framings rather than concrete proposals",
            ],
            if_blocked=[
                "If coherence/drift judgement needs missing Top-K or history: choose Restructure/Onboard conservatively and add evidence_gaps items requesting the missing Top-K summaries and/or history_reports.",
            ],
            if_unblocked=[
                "Add next_step_inputs to unlock better ideation: request intro+conclusion + limitations/failure modes for the most central paper_ids (the minimum evidence to brainstorm without false concreteness).",
            ],

        ),
        "declared_level_final": FieldGuidance(
            question="How specific can brainstorming be without fabricating details?",
            use=[
                "intent_mode (wants specificity, but evidence-gated)",
                "Top-K summary specificity (if present)",
                "explicit methods/experiments/limitations ONLY if present",
            ],
            decide=[
                "intro/intermediate: typical in MVP with summaries; keep ideas as hypotheses/axes",
                "deep-dive: ONLY if explicit method/experiment/limitation evidence is present",
            ],
            if_blocked=[
                "If ideation specificity would require method details: choose intermediate and add evidence_gaps items requesting methods/experiments/limitations for central paper_ids.",
            ],
            if_unblocked=[
                "Add next_step_inputs tuned for ideation: prefer limitations/failure-modes + eval protocol sections (these sharpen hypotheses more than extra paper counts).",
            ],

        ),
        "subthreads_final": FieldGuidance(
            question="What ideation axes should structure brainstorming into 2–4 buckets?",
            use=[
                "intent_mode (axes-based partition)",
                "new_observation keywords + key_paper_keywords",
                "Top-K summaries/keywords (if present)",
                "history_reports (optional) to avoid repeating old axes",
            ],
            decide=[
                "Use axes like: problem framing, representation, learning signal, evaluation, deployment constraints — only when justified by evidence keywords",
                "If Top-K present, assign paper_ids; otherwise paper_ids can be []",
            ],
            if_blocked=[
                "If axes cannot be justified from evidence (keywords too generic): use broad axes and add evidence_gaps items requesting richer summaries or key_paper_keywords.",
            ],
            if_unblocked=[
                "Add next_step_inputs per axis: for each axis/subthread, request 1 enabling artifact that would let the writer propose grounded directions (usually limitations + conclusion, paper-specific).",
            ],

        ),
        "next_targets": FieldGuidance(
            question="What evidence upgrades unlock safe, high-quality brainstorming next?",
            use=[
                "intent_mode (evidence acquisition is central)",
                "gaps across summaries/keywords",
                "history_reports next_targets (optional)",
            ],
            decide=[
                "Make next_targets mostly evidence upgrades: methods, experiments, limitations, eval setup, baselines, failure modes",
                "Prefer paper-specific requests (cite paper_id) for Top-2 most central papers",
            ],
            if_blocked=[
                "If you cannot identify what evidence matters: add evidence_gaps items requesting intro+conclusion + methods/experiments/limitations for the most central paper_ids.",
            ],
            if_unblocked=[
                "Convert next_targets into next_step_inputs where possible: each 'direction' should have an evidence upgrade attached (e.g., 'failure modes for paper X').",
            ],

        ),
        "outline": FieldGuidance(
            question="What brainstorming-friendly section order should the writer follow?",
            use=[
                "intent_mode (idea-space structure)",
                "subthreads_final",
                "depth_mode_final + declared_level_final",
            ],
            decide=[
                "Theme framing → novelty pockets (subthreads) → hypothesis directions (high-level) → risks/unknowns → evidence gaps → next_targets",
            ],
            if_blocked=[
                "If 'novelty pockets' cannot be supported without stronger evidence: keep them abstract and add evidence_gaps items requesting the missing artifacts.",
            ],
            if_unblocked=[
                "Attach next_step_inputs to outline items that imply claims: for each novelty/hypothesis section, request a minimal evidence slice (conclusion + limitation) for the paper_ids it draws from.",
            ],

        ),
        "skip_or_defer": FieldGuidance(
            question="What must be deferred to avoid false concreteness in brainstorming?",
            use=[
                "intent_mode",
                "absence of explicit method/eval evidence",
                "summary-only limitations",
            ],
            decide=[
                "Defer implementation-ready proposals, numeric claims, benchmark-driven 'wins', and feasibility assertions unless explicitly supported",
            ],
            if_blocked=[
                "If an idea would become implementation-ready without explicit feasibility evidence: defer it and add an evidence_gaps item requesting feasibility-critical artifacts (methods/eval/constraints).",
            ],
            if_unblocked=[
                "If evidence is strong enough to propose semi-concrete directions, add next_step_inputs to gather the missing feasibility checks (datasets, compute constraints, eval protocol) before escalating to implementation claims.",
            ],

        ),
        "sufficiency": FieldGuidance(
            question="Is evidence sufficient to produce non-trivial brainstorming directions without invention?",
            use=[
                "intent_mode (very high bar)",
                "Top-K specificity/coherence (if present)",
                "new_observation specificity",
            ],
            decide=[
                "sufficient: unusually detailed and coherent summaries",
                "borderline: typical MVP case; must keep ideas high-level and request missing detail",
                "insufficient: too vague/mixed; brainstorming should be mostly evidence_request",
            ],
        ),
    },
)


IMPLEMENTATION_ANGLE_SPEC = IntentSpec(
    intent_goal="Feasibility-gated build/test planning; with summary-only evidence, avoid false concreteness and request implementation-critical details.",
    plan_field_guidance={
        "depth_mode_final": FieldGuidance(
            question="Are we onboarding feasibility, continuing a build thread, deepening into a build plan, or restructuring for feasibility?",
            use=[
                "intent_mode (implementation_angle)",
                "Top-K coherence (is there a buildable core?)",
                "new_observation keywords/themes",
                "history_reports (optional) for ongoing build threads",
            ],
            decide=[
                "Onboard: evidence too thin; focus on feasibility gates/questions rather than concrete plan",
                "Continue: history exists; continue the build/test thread; emphasize deltas and unresolved gates",
                "Deepen: coherent build target emerges; expand into gated plan (still avoid specifics not supported)",
                "Restructure: mixed topic; split into 'buildable now' vs 'needs evidence' subthreads",
            ],
            if_blocked=[
                "If deciding requires missing Top-K or history: choose Onboard/Restructure conservatively and add evidence_gaps items requesting the missing evidence.",
            ],
            if_unblocked=[
                "Add next_step_inputs for feasibility-gated writing: request training recipe + eval protocol + limitations for the central paper_ids the plan will lean on (minimum viable buildability evidence).",
            ],
        ),
        "declared_level_final": FieldGuidance(
            question="How specific can we be about what to build/test given the evidence?",
            use=[
                "intent_mode (wants specificity but evidence-gated)",
                "Top-K summaries (do they mention components, training signals, eval setup?)",
                "explicit methods/experiments/limitations ONLY if present",
            ],
            decide=[
                "intermediate: typical; produce feasibility gates and minimal experiment plan, not concrete recipes",
                "deep-dive: ONLY if explicit method/training/eval details exist in input",
                "intro: if evidence is extremely thin; focus on gate questions",
            ],
            if_blocked=[
                "If concrete planning would require methods/training/eval details: choose intermediate and add evidence_gaps items requesting those artifacts.",
            ],
            if_unblocked=[
                "Add next_step_inputs aligned with the chosen level: if intermediate, request the missing feasibility gates evidence (datasets/metrics/baselines); if deep-dive, request exact training/eval sections + reproducibility notes.",
            ],

        ),
        "subthreads_final": FieldGuidance(
            question="What 2–4 component/gate buckets should structure implementation planning?",
            use=[
                "intent_mode (component/gate buckets)",
                "Top-K keywords/summaries (if present)",
                "new_observation key_paper_keywords/keywords",
            ],
            decide=[
                "Prefer buckets like: Data/Inputs, Model/Representation, Training/Optimization, Evaluation/Baselines, Deployment/Constraints",
                "Assign paper_ids if Top-K present; otherwise []",
            ],
            if_blocked=[
                "If component buckets can’t be justified from evidence: use broader buckets and add evidence_gaps items requesting method/eval details.",
            ],
            if_unblocked=[
                "Add next_step_inputs per component bucket: request the artifact that makes that bucket implementable (Data→dataset spec, Training→training recipe, Eval→protocol/baselines, Constraints→compute/runtime).",
            ],
        ),
        "next_targets": FieldGuidance(
            question="What evidence is needed to turn this into a concrete build/test plan?",
            use=[
                "intent_mode (implementation-critical evidence acquisition)",
                "gaps across summaries (missing datasets, metrics, baselines, training recipe, constraints)",
                "history next_targets (optional)",
            ],
            decide=[
                "Make next_targets mostly concrete evidence requests: training recipe, datasets/splits, eval protocol/baselines, compute/runtime, reproducibility notes, limitations/failure modes",
                "Prefer paper-specific requests (cite paper_id) for central papers",
            ],
            if_blocked=[
                "If you can’t tell what’s missing, add evidence_gaps items requesting intro+conclusion + methods/training/eval + limitations for the most central paper_ids.",
            ],
            if_unblocked=[
                "Convert next_targets into next_step_inputs: each feasibility gate becomes a concrete artifact request (paper_id + section) so Call 2 can produce writer-ready evidence.",
            ],
        ),
        "outline": FieldGuidance(
            question="What feasibility-first section order should the writer follow?",
            use=[
                "intent_mode (feasibility gates first)",
                "subthreads_final",
                "depth_mode_final + declared_level_final",
            ],
            decide=[
                "Target framing → feasibility gates/assumptions → components (subthreads) → minimal experiment plan → risks/unknowns → evidence gaps → next_targets",
            ],
            if_blocked=[
                "If feasibility gates can’t be derived without guessing: keep them generic and add evidence_gaps items requesting missing feasibility evidence.",
            ],
            if_unblocked=[
                "Bind outline to retrieval: every outline section that implies build/test specificity must have next_step_inputs that provide the enabling artifacts (training recipe, eval protocol, constraints).",
            ],
        ),
        "skip_or_defer": FieldGuidance(
            question="What implementation claims must be deferred given summary-only evidence?",
            use=[
                "intent_mode",
                "absence of explicit method/eval details",
            ],
            decide=[
                "Defer hyperparameters, exact architectures, compute budgets, dataset choices, and step-by-step instructions unless explicitly supported",
            ],
            if_blocked=[
                "If you’re about to name exact architectures/hyperparams/compute budgets without explicit support: defer and add evidence_gaps items stating what would be needed to justify them.",
            ],
            if_unblocked=[
                "Request the missing implementation-critical artifacts as next_step_inputs (reproducibility notes, hyperparam tables, ablations) before allowing the writer to get concrete.",
            ],
        ),
        "sufficiency": FieldGuidance(
            question="Is evidence sufficient to recommend concrete build/test actions without invention?",
            use=[
                "intent_mode (very high bar)",
                "Top-K presence of method/eval specificity",
                "new_observation specificity",
            ],
            decide=[
                "sufficient: unusually detailed summaries with concrete method/eval signals",
                "borderline: typical MVP; plan is gate-based and evidence_request-heavy",
                "insufficient: too vague/missing key details; must request implementation-enabling evidence",
            ],
        ),
    },
)
