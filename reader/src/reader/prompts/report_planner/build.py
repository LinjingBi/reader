import json
from typing import Dict, List
from enum import Enum
from reader.pipelines.report import LLMReportPlannerPlan
from reader.prompts.report_planner.spec_baseline import (
    BASE_UNIVERSAL,
    FieldGuidance,
    IntentSpec,
    QUICK_BACKGROUND_SPEC,
    RESEARCH_BRIEFING_SPEC,
    BRAINSTORM_DIRECTIONS_SPEC,
    IMPLEMENTATION_ANGLE_SPEC,
)
from reader.adapters.memo import GetReportPlannerMetadataResponse


def _format_guidance(field_name: str, g: FieldGuidance) -> str:
    use_lines = "\n".join([f"- {x}" for x in g.use])
    decide_lines = "\n".join([f"- {x}" for x in g.decide])
    
    parts = [
        f"### plan.{field_name}",
        "",
        f"Q: {g.question}",
        "",
        "Use:",
        use_lines,
        "",
        "Decide:",
        decide_lines,
    ]
    
    if g.if_blocked:
        blocked_lines = "\n".join([f"- {x}" for x in g.if_blocked])
        parts.extend([
            "",
            "If blocked:",
            blocked_lines,
        ])
    
    if g.if_unblocked:
        unblocked_lines = "\n".join([f"- {x}" for x in g.if_unblocked])
        parts.extend([
            "",
            "If unblocked:",
            unblocked_lines,
        ])
    
    return "\n".join(parts)


def build_baseline_planner_prompt(intent_spec: IntentSpec, cluster_metadata: GetReportPlannerMetadataResponse) -> str:
    # Build plan guidance content
    plan_guidance_parts = [
        f"Intent mode:\n- {intent_spec.intent_goal}\n",
    ]

    # Render guidance in stable order, matching Plan model fields order if possible
    for field_name in LLMReportPlannerPlan.model_fields.keys():
        g = intent_spec.plan_field_guidance[field_name]
        plan_guidance_parts.append(_format_guidance(field_name, g))

    if intent_spec.evidence_gaps_notes:
        er_lines = "\n".join([f"- {x}" for x in intent_spec.evidence_gaps_notes])
        plan_guidance_parts.append("### evidence_gaps_notes\n" + er_lines + "\n")

    if intent_spec.next_step_inputs_notes:
        ns_lines = "\n".join([f"- {x}" for x in intent_spec.next_step_inputs_notes])
        plan_guidance_parts.append("### next_step_inputs_notes\n" + ns_lines + "\n")

    plan_guidance = "\n".join(plan_guidance_parts)

    # Process evidence pack
    cluster_dict = cluster_metadata.model_dump()
    
    # Remove top_papers_from_new_observation if empty/none
    if not cluster_metadata.top_papers_from_new_observation:
        cluster_dict.pop('top_papers_from_new_observation', None)
    
    # Set history_reports message if empty/none
    if not cluster_metadata.history_reports:
        cluster_dict['history_reports'] = 'no history, observed for the first time.'
    
    evidence_pack = json.dumps(cluster_dict, indent=2, ensure_ascii=False)

    # Replace placeholders in BASE_UNIVERSAL
    prompt = BASE_UNIVERSAL.replace('<EVIDENCE_PACK_PLACEHOLDER>', evidence_pack)
    prompt = prompt.replace('<PLAN_GUIDANCE_PLACEHOLDER>', plan_guidance)

    return prompt


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
