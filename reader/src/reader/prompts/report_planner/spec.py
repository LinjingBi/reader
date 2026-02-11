from dataclasses import dataclass
from typing import Mapping, Sequence, List, Dict
from enum import Enum


BASE_UNIVERSAL = """SYSTEM:
You are a technical report planner.

HARD RULES (anti-hallucination):
- Use ONLY the provided evidence in the input JSON. Do NOT invent methods, results, datasets, metrics, numbers, or implementation details.
- Treat intent_mode as evidence: it is a user goal signal and must influence planning decisions.
- If you cannot derive any plan field without making up details:
  1) Fill that field with a conservative placeholder derived from available evidence (usually keywords/themes).
  2) Add an item to evidence_request describing what evidence is needed and why.
- Do NOT output Markdown. Return valid JSON only.
- Output JSON structure is enforced by a typed schema. Focus on semantic correctness and evidence grounding.

TASK:
Given a new observation, a history of reports and a user intent for the new observation, produce:
- a conservative, evidence-grounded "plan" that guides report writing
- an optional "evidence_request" listing missing evidence required to fulfill the intent without speculation
"""



@dataclass(frozen=True)
class FieldGuidance:
    question: str                 # "What question is this field answering?"
    use: Sequence[str]            # allowed evidence keys / signals
    decide: Sequence[str]         # decision rules / examples
    if_blocked: Sequence[str]     # what to do if you can't derive without invention


@dataclass(frozen=True)
class IntentSpec:
    intent_goal: str
    evidence_expectation: Sequence[str]
    # Guidance keyed by Plan field name (must match Plan.model_fields)
    plan_field_guidance: Mapping[str, FieldGuidance]
    # Optional: extra guidance for evidence_request itself
    evidence_request_guidance: Sequence[str] = ()


# ----------------------------
# UserIntent enum
# ----------------------------

class UserIntent(str, Enum):
    """User intent options for report generation"""
    QUICK_BACKGROUND = "Quick Background (5-10 min overview)"
    RESEARCH_BRIEFING = "Research Briefing (Decision-oriented)"
    BRAINSTORM_DIRECTIONS = "Brainstorm Directions (Novelty hunting)"
    IMPLEMENTATION_ANGLE = "Implementation Angle (What to build/test)"
    
    @classmethod
    def get_all_display_strings(cls) -> List[str]:
        """Get all display strings as a list for UI selection"""
        return [intent.value for intent in cls]
    
    @classmethod
    def from_display_string(cls, display_string: str) -> "UserIntent":
        """Get UserIntent enum from display string"""
        for intent in cls:
            if intent.value == display_string:
                return intent
        raise ValueError(f"Unknown user intent display string: {display_string}")

# ----------------------------
# intent specs
# ----------------------------
"""
TODO: update the evidence expectation.

For **research_briefing / brainstorm_directions / implementation_angle**, the `evidence_expectation` is intentionally limited by the metadata we currently store (mostly **summary-level** fields plus `new_observation` and optional `history_reports`).

This is **known to be insufficient** for high-specificity planning (e.g., methods, experiments, limitations, eval protocols, feasibility constraints). The prompt is therefore **designed to bias the planner toward**:

* producing a conservative, evidence-grounded `plan` (avoid false concreteness), and
* using `evidence_request` aggressively to request the missing artifacts needed to fulfill the intent.

In other words: **insufficiency is a feature in MVP**—it forces the model to surface evidence gaps instead of hallucinating details.

"""

QUICK_BACKGROUND_SPEC = IntentSpec(
    intent_goal="Fast orientation: what it is / why it matters / what to look at next.",
    evidence_expectation=[
        "intent_mode (required)",
        "new_observation (required): name, summary, keywords, key_paper_keywords (if available)",
        "history_reports (optional): last N compressed report metadata",
        "Paper summaries are NOT assumed unless explicitly present in the input",
    ],
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
                "If deeper specificity is desired but missing, choose intermediate and request the needed paper sections.",
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
                "If keywords are too generic, request key_paper_keywords or Top-K paper summaries.",
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
                "Keep targets generic and add specifics to evidence_request instead of guessing.",
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
                "If delta depends on missing history, omit delta bullet and request history_reports.",
            ],
        ),
        "skip_or_defer": FieldGuidance(
            question="What must the writer avoid expanding on to prevent speculation or repetition?",
            use=["intent_mode", "known evidence limits", "history_reports covered_bullets (optional)"],
            decide=[
                "Defer numeric superiority claims, dataset specifics, feasibility claims unless explicitly supported",
            ],
            if_blocked=[
                "When uncertain, defer rather than speculate; request evidence.",
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
            if_blocked=[
                "If critical evidence is missing, set sufficiency borderline/insufficient and request it.",
            ],
        ),
    },
    evidence_request_guidance=[
        "If present, each item should be a compact need with name + why (tie it to blocked plan fields).",
        "Prefer requesting concrete artifacts: Top-K paper summaries, PDF intro+conclusion, methods/experiments/limitations, eval protocol.",
    ],
)
RESEARCH_BRIEFING_SPEC = IntentSpec(
    intent_goal="Decision-oriented synthesis (≈10–20 min) grounded in Top-K paper summaries; structured comparison + explicit gaps.",
    evidence_expectation=[
        "intent_mode (required)",
        "new_observation (required): name, summary, keywords, key_paper_keywords (if available)",
        "top_papers (required): Top-K (K<=5) papers with paper_id, title, summary, keywords, rank_in_cluster, sim_to_centroid",
        "history_reports (optional): last N compressed report metadata",
        "Paper full-text / methods / experiments are NOT assumed unless explicitly present in the input",
    ],
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
                "If deciding between Continue/Deepen/Restructure depends on history but it's missing: choose Onboard and request history_reports.",
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
                "If deeper specificity would require methods/experiments/limitations, choose intermediate and request those artifacts for the most central paper_ids.",
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
                "If Top-K is incoherent or too vague to partition: make broader buckets and request richer summaries or additional evidence (e.g., intro+conclusion).",
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
                "If you can’t name concrete follow-ups without guessing, keep targets generic and put specifics in evidence_request.",
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
                "If delta requires missing history, omit delta framing and request history_reports.",
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
                "When uncertain, defer and request the specific evidence needed.",
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
            if_blocked=[
                "If key evidence is missing, set borderline/insufficient and request it.",
            ],
        ),
    },
    evidence_request_guidance=[
        "Use evidence_request to request missing artifacts that block specificity (methods/experiments/limitations, eval setup, intro+conclusion).",
        "Prefer paper-specific requests (cite paper_id) and explain why the evidence is needed for planning/writing.",
    ],
)


BRAINSTORM_DIRECTIONS_SPEC = IntentSpec(
    intent_goal="Novelty hunting + idea generation; be conservative with summary-level evidence and push missing detail into evidence_request.",
    evidence_expectation=[
        "intent_mode (required)",
        "new_observation (required): name, summary, keywords, key_paper_keywords (if available)",
        "top_papers (recommended): Top-K (K<=5) papers with paper_id, title, summary, keywords, rank_in_cluster, sim_to_centroid",
        "history_reports (optional): last N compressed report metadata",
        "Methods/experiments/limitations are NOT assumed unless explicitly present; request them if needed to ideate safely",
    ],
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
                "If coherence/drift judgement needs missing Top-K or history: choose Restructure or Onboard conservatively and request the missing evidence.",
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
                "If specificity would require method details, choose intermediate and request methods/experiments/limitations for central paper_ids.",
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
                "If axes cannot be justified by evidence (keywords too generic), request richer paper summaries or key_paper_keywords.",
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
                "If you can’t identify which evidence matters, request: intro+conclusion + methods/experiments/limitations for the most central paper_ids.",
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
                "If you can’t support 'novelty pockets' without stronger evidence, keep them abstract and request evidence in evidence_request.",
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
                "When uncertain, defer and request the missing feasibility/method evidence.",
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
            if_blocked=[
                "If insufficient, mark insufficient and request the enabling evidence (methods/experiments/eval/failure modes).",
            ],
        ),
    },
    evidence_request_guidance=[
        "Brainstorm mode expects evidence_request frequently; prefer requests that unlock novelty safely (methods, eval setup, limitations, failure modes).",
        "Avoid requesting 'more papers' first; request higher-value detail for central papers first.",
    ],
)


IMPLEMENTATION_ANGLE_SPEC = IntentSpec(
    intent_goal="Feasibility-gated build/test planning; with summary-only evidence, avoid false concreteness and request implementation-critical details.",
    evidence_expectation=[
        "intent_mode (required)",
        "new_observation (required): name, summary, keywords, key_paper_keywords (if available)",
        "top_papers (recommended): Top-K (K<=5) papers with paper_id, title, summary, keywords, rank_in_cluster, sim_to_centroid",
        "history_reports (optional): last N compressed report metadata",
        "Implementation details (methods/training recipe/eval protocol) are NOT assumed unless explicitly present; request them when needed",
    ],
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
                "If deciding requires missing Top-K or history, choose Onboard/Restructure conservatively and request the missing evidence.",
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
                "If concrete planning would require methods/training/eval details, choose intermediate and request those artifacts.",
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
                "If component buckets can’t be justified from evidence, use broader buckets and request method/eval details.",
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
                "If you can’t tell what’s missing, request intro+conclusion + methods/training/eval + limitations for the most central paper_ids.",
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
                "If feasibility gates can’t be derived without guessing, keep them generic and request missing evidence.",
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
                "When uncertain, defer and request the missing method/eval evidence.",
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
            if_blocked=[
                "If insufficient, mark insufficient and request methods/training/eval/limitations/failure modes.",
            ],
        ),
    },
    evidence_request_guidance=[
        "Implementation mode expects evidence_request frequently; prioritize feasibility-critical artifacts (methods, training recipe, eval protocol, baselines, constraints, failure modes).",
        "Prefer a few high-value requests over many low-value ones.",
    ],
)

# ----------------------------
# Mapping from UserIntent to IntentSpec
# ----------------------------

USER_INTENT_TO_SPEC: Dict[UserIntent, IntentSpec] = {
    UserIntent.QUICK_BACKGROUND: QUICK_BACKGROUND_SPEC,
    UserIntent.RESEARCH_BRIEFING: RESEARCH_BRIEFING_SPEC,
    UserIntent.BRAINSTORM_DIRECTIONS: BRAINSTORM_DIRECTIONS_SPEC,
    UserIntent.IMPLEMENTATION_ANGLE: IMPLEMENTATION_ANGLE_SPEC,
}


def get_intent_spec(user_intent: UserIntent) -> IntentSpec:
    """Get IntentSpec for a given UserIntent"""
    spec = USER_INTENT_TO_SPEC.get(user_intent)
    if spec is None:
        raise ValueError(f"IntentSpec not found for UserIntent: {user_intent}")
    return spec
