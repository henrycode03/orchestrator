"""Phase 28R-V source-reference contract/parser alignment tests."""

import pytest

from app.services.planning.input_manifest import build_input_manifest
from app.services.planning.planning_brief_stage import (
    PlanningBriefProviderInput,
    PlanningBriefProviderOutputError,
    build_planning_brief_request,
    canonicalize_planning_brief_candidate,
    parse_planning_brief_candidate,
)
from app.services.planning.provider_contract import (
    BRIEF_SOURCE_REFERENCE_INSTRUCTIONS,
    build_planning_brief_schema_contract,
)


def test_brief_contract_distinguishes_manifest_and_semantic_record_references():
    contract = build_planning_brief_schema_contract()
    fields = contract["top_level"]["fields"]

    assert fields["objective"]["record"]["fields"]["source_refs"][
        "reference_semantics"
    ] == (
        "every value must exactly equal a canonical source_id supplied in the "
        "bounded Input Manifest"
    )
    for collection, field_name in (
        ("constraints", "applies_to_refs"),
        ("acceptance_criteria", "source_requirement_ids"),
        ("implementation_strategy", "requirement_ids"),
        ("implementation_strategy", "constraint_ids"),
        ("validation_strategy", "acceptance_criterion_ids"),
        ("validation_strategy", "requirement_ids"),
        ("unresolved_questions", "temporary_assumption_id"),
    ):
        descriptor = fields[collection]["record"]["fields"][field_name]
        assert descriptor["reference_semantics"] == (
            "semantic record reference using objective or collection[index]; "
            "the application assigns canonical IDs"
        )

    assert contract["source_reference_requirements"]["manifest_authority"] == (
        BRIEF_SOURCE_REFERENCE_INSTRUCTIONS
    )
    assert (
        "manifest source identifiers"
        in contract["source_reference_requirements"]["semantic_record_refs"]
    )


def _candidate(source_id: str, applies_to_ref: str) -> dict:
    return {
        "objective": {"statement": "bounded", "source_refs": [source_id]},
        "background": [],
        "scope": [
            {
                "classification": "in_scope",
                "statement": "bounded",
                "source_refs": [source_id],
            }
        ],
        "requirements": [
            {
                "type": "functional",
                "statement": "bounded",
                "priority": "required",
                "source_refs": [source_id],
            }
        ],
        "constraints": [
            {
                "type": "technical",
                "statement": "bounded",
                "severity": "should",
                "enforcement": "automated",
                "source_refs": [source_id],
                "applies_to_refs": [applies_to_ref],
            }
        ],
        "acceptance_criteria": [
            {
                "statement": "bounded",
                "verification_method": "test",
                "source_requirement_ids": ["requirements[0]"],
                "criticality": "required",
            }
        ],
        "architecture_context": [],
        "interface_contracts": [],
        "implementation_strategy": [
            {
                "statement": "bounded",
                "source_refs": [source_id],
                "requirement_ids": ["requirements[0]"],
                "constraint_ids": ["constraints[0]"],
            }
        ],
        "validation_strategy": [
            {
                "statement": "bounded",
                "source_refs": [source_id],
                "acceptance_criterion_ids": ["acceptance_criteria[0]"],
                "requirement_ids": ["requirements[0]"],
            }
        ],
        "assumptions": [],
        "risks": [],
        "unresolved_questions": [],
        "operator_decisions": [],
    }


def test_canonicalizer_rejects_manifest_source_id_in_record_reference_field():
    manifest = build_input_manifest(
        session_id=28,
        session_generation_id="phase28rv-generation",
        planning_request={"message_id": 1, "content": "bounded"},
        clarification_messages=[],
        project_metadata={"project_id": 28, "name": "bounded"},
        project_rules="bounded",
        repository={"available": False, "workspace": "bounded"},
        runtime_configuration={"provider": "test", "model": "test"},
        stage_configuration={"stages": [{"identifier": "planning_brief"}]},
        manifest_built_at="2026-07-22T00:00:00+00:00",
    )
    source_id = manifest.sources[0].source_id

    with pytest.raises(
        PlanningBriefProviderOutputError,
        match="not a semantic record reference",
    ):
        canonicalize_planning_brief_candidate(
            parse_planning_brief_candidate(_candidate(source_id, source_id)),
            manifest,
        )

    accepted = canonicalize_planning_brief_candidate(
        parse_planning_brief_candidate(_candidate(source_id, "objective")),
        manifest,
    )
    assert accepted.constraints[0].applies_to_refs == ("GOAL-001",)


def test_rendered_brief_prompt_places_manifest_rule_at_provider_boundary():
    request = build_planning_brief_request(
        PlanningBriefProviderInput(
            manifest_id="manifest:phase28rv",
            manifest_hash="a" * 64,
            manifest_schema_version="protocol-v2-input-manifest/1.0",
            sources=(),
            stage_configuration={},
        )
    )
    assert BRIEF_SOURCE_REFERENCE_INSTRUCTIONS in request.prompt
    assert (
        "source_refs and semantic record-reference fields are distinct"
        in request.prompt
    )
