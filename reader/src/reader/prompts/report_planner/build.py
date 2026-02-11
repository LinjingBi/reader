import json
from reader.pipelines.report import LLMReportPlannerPlan
from reader.prompts.report_planner.spec import BASE_UNIVERSAL, FieldGuidance, IntentSpec
from reader.adapters.memo import GetReportPlannerMetadataResponse


def _format_guidance(field_name: str, g: FieldGuidance) -> str:
    use_lines = "\n".join([f"- {x}" for x in g.use])
    decide_lines = "\n".join([f"- {x}" for x in g.decide])
    blocked_lines = "\n".join([f"- {x}" for x in g.if_blocked])

    return f"""### plan.{field_name}

Q: {g.question}

Use:
{use_lines}

Decide:
{decide_lines}

If blocked:
{blocked_lines}
"""


def build_planner_prompt(intent_spec: IntentSpec, cluster_metadata: GetReportPlannerMetadataResponse) -> str:
    # (2) intent semantics
    evidence_lines = "\n".join([f"- {x}" for x in intent_spec.evidence_expectation])

    sections = [
        BASE_UNIVERSAL,
        "# (2) User intent bias + field semantics",
        f"Intent goal:\n- {intent_spec.intent_goal}\n",
        "Evidence expectation:\n" + evidence_lines + "\n",
    ]

    # Render guidance in stable order, matching Plan model fields order if possible
    for field_name in LLMReportPlannerPlan.model_fields.keys():
        g = intent_spec.plan_field_guidance[field_name]
        sections.append(_format_guidance(field_name, g))

    if intent_spec.evidence_request_guidance:
        er_lines = "\n".join([f"- {x}" for x in intent_spec.evidence_request_guidance])
        sections.append("### evidence_request\n" + er_lines + "\n")

    # (3) input json    
    cluster_json = json.dumps(cluster_metadata.model_dump(), indent=2, ensure_ascii=False)
    
    sections += [
        "# (3) INPUT JSON",
        cluster_json,
    ]

    return "\n".join(sections)
