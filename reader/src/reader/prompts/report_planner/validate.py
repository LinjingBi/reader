# for CI/CD validation
from reader.pipelines.report import LLMReportPlannerPlan
from reader.prompts.report_planner.spec_baseline import IntentSpec


def validate_intent_spec(intent_spec: IntentSpec) -> None:
    plan_fields = set(LLMReportPlannerPlan.model_fields.keys())
    spec_fields = set(intent_spec.plan_field_guidance.keys())

    missing_in_spec = plan_fields - spec_fields
    extra_in_spec = spec_fields - plan_fields

    if missing_in_spec or extra_in_spec:
        raise ValueError(
            "IntentSpec drift detected.\n"
            f"Missing in spec (new Plan fields?): {sorted(missing_in_spec)}\n"
            f"Extra in spec (removed/renamed Plan fields?): {sorted(extra_in_spec)}"
        )
