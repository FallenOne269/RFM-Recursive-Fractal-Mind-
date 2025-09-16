
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import asyncio

class ActionType(str, Enum):
    CREATE = "create"
    READ = "read" 
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MODIFY_SYSTEM = "modify_system"

class StakeholderType(str, Enum):
    HUMAN_INDIVIDUAL = "human_individual"
    HUMAN_GROUP = "human_group"
    AI_SYSTEM = "ai_system"
    ENVIRONMENT = "environment"
    FUTURE_GENERATIONS = "future_generations"

class ActionProposal(BaseModel):
    action_id: str
    action_type: ActionType
    description: str
    target_system: str
    parameters: Dict[str, any]
    urgency_level: int = Field(ge=1, le=5)
    reversibility: bool

class StakeholderImpact(BaseModel):
    stakeholder_id: str
    stakeholder_type: StakeholderType
    impact_description: str
    impact_magnitude: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    mitigation_possible: bool

class ValueSystem(BaseModel):
    core_values: List[str]
    value_weights: Dict[str, float]
    cultural_context: str
    adaptation_allowed: bool

class EthicalDNARequest(BaseModel):
    action_proposal: ActionProposal
    context_data: Dict[str, any]
    stakeholder_impact: List[StakeholderImpact]
    value_system: ValueSystem

class EthicalViolation(BaseModel):
    violation_type: str
    severity: str
    description: str
    affected_principles: List[str]
    remediation_suggestions: List[str]

class AlternativeAction(BaseModel):
    action_id: str
    description: str
    ethical_score: float
    trade_offs: List[str]
    implementation_complexity: str

class EthicalDNAResponse(BaseModel):
    ethical_assessment: Dict[str, float]
    violation_flags: List[EthicalViolation]
    alternative_actions: List[AlternativeAction]
    value_alignment_score: float
    reasoning_chain: List[str]
    confidence_level: float

@app.post("/api/v1/ethical/dna", response_model=EthicalDNAResponse)
async def evaluate_ethical_dna(
    request: EthicalDNARequest,
    auth_data: dict = Depends(verify_fractal_token)
) -> EthicalDNAResponse:
    """
    Ethical DNA validation and enforcement endpoint

    Evaluates proposed actions against core ethical principles,
    identifies violations, and suggests alternatives.
    """

    # Core ethical principles evaluation
    ethical_scores = {
        "harm_prevention": await evaluate_harm_prevention(request),
        "autonomy_respect": await evaluate_autonomy_respect(request),
        "fairness": await evaluate_fairness(request),
        "transparency": await evaluate_transparency(request),
        "accountability": await evaluate_accountability(request)
    }

    # Identify violations
    violations = []
    for principle, score in ethical_scores.items():
        if score < 0.3:  # Threshold for violation
            violations.append(EthicalViolation(
                violation_type=principle,
                severity="high" if score < 0.1 else "medium",
                description=f"Action violates {principle} principle",
                affected_principles=[principle],
                remediation_suggestions=[f"Modify action to improve {principle}"]
            ))

    # Generate alternatives if violations exist
    alternatives = []
    if violations:
        alternatives = await generate_alternative_actions(request, ethical_scores)

    # Calculate overall value alignment
    value_alignment = await calculate_value_alignment(request, ethical_scores)

    # Generate reasoning chain
    reasoning = [
        f"Analyzed action: {request.action_proposal.description}",
        f"Evaluated {len(request.stakeholder_impact)} stakeholder impacts",
        f"Applied {len(request.value_system.core_values)} core values",
        f"Generated {len(alternatives)} alternative actions"
    ]

    return EthicalDNAResponse(
        ethical_assessment=ethical_scores,
        violation_flags=violations,
        alternative_actions=alternatives,
        value_alignment_score=value_alignment,
        reasoning_chain=reasoning,
        confidence_level=0.85  # Would be calculated
    )

async def evaluate_harm_prevention(request: EthicalDNARequest) -> float:
    """Evaluate potential for harm prevention"""
    negative_impacts = [
        impact for impact in request.stakeholder_impact 
        if impact.impact_magnitude < 0
    ]
    if not negative_impacts:
        return 1.0

    total_negative = sum(abs(impact.impact_magnitude) for impact in negative_impacts)
    mitigation_possible = sum(
        1 for impact in negative_impacts if impact.mitigation_possible
    ) / len(negative_impacts)

    return max(0.0, 1.0 - (total_negative * 0.5) + (mitigation_possible * 0.3))

async def evaluate_autonomy_respect(request: EthicalDNARequest) -> float:
    """Evaluate respect for stakeholder autonomy"""
    human_impacts = [
        impact for impact in request.stakeholder_impact 
        if impact.stakeholder_type in [StakeholderType.HUMAN_INDIVIDUAL, StakeholderType.HUMAN_GROUP]
    ]

    if not human_impacts:
        return 1.0

    # Check if action preserves human agency
    agency_preservation = 0.8  # Would be calculated based on action details
    consent_level = 0.7  # Would be determined from context

    return (agency_preservation + consent_level) / 2

async def generate_alternative_actions(
    request: EthicalDNARequest, 
    ethical_scores: Dict[str, float]
) -> List[AlternativeAction]:
    """Generate ethically improved alternative actions"""
    alternatives = []

    # Example alternative generation logic
    if ethical_scores["harm_prevention"] < 0.5:
        alternatives.append(AlternativeAction(
            action_id=f"{request.action_proposal.action_id}_safer",
            description=f"Modified {request.action_proposal.description} with additional safeguards",
            ethical_score=0.8,
            trade_offs=["Reduced efficiency", "Increased complexity"],
            implementation_complexity="medium"
        ))

    return alternatives

async def calculate_value_alignment(
    request: EthicalDNARequest, 
    ethical_scores: Dict[str, float]
) -> float:
    """Calculate alignment with core value system"""
    weighted_scores = []
    for value in request.value_system.core_values:
        # Map values to ethical principle scores
        if value in ["safety", "harm_prevention"]:
            weighted_scores.append(ethical_scores["harm_prevention"])
        elif value in ["freedom", "autonomy"]:
            weighted_scores.append(ethical_scores["autonomy_respect"])
        # Add more mappings as needed

    return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.5
