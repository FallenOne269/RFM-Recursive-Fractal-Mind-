"""Orchestrator for the Recursive Fractal Autonomous Intelligence system."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from .utils.validation import (
    ValidationError,
    validate_component_config,
    validate_input_payload,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    """Metadata describing how to load a component."""

    module_path: str
    class_name: str
    output_key: str


DEFAULT_COMPONENTS: Mapping[str, ComponentSpec] = {
    "fractal_engine": ComponentSpec("src.fractal_engine", "FractalEngine", "fractal_output"),
    "swarm_coordinator": ComponentSpec(
        "src.swarm_coordinator", "SwarmCoordinator", "swarm_output"
    ),
    "quantum_processor": ComponentSpec(
        "src.quantum_processor", "QuantumProcessor", "quantum_output"
    ),
    "meta_learner": ComponentSpec("src.meta_learner", "MetaLearner", "meta_output"),
}


class RFAISystem:
    """Coordinate the lifecycle of all RFAI components."""

    def __init__(
        self,
        config: Optional[MutableMapping[str, Any]] = None,
        component_factories: Optional[Mapping[str, Callable[[Dict[str, Any]], Any]]] = None,
    ) -> None:
        self.config = dict(config or {})
        self.component_factories = dict(component_factories or {})
        self.components: Dict[str, Any] = {}
        self.system_state: Dict[str, Any] = {
            "cycles": 0,
            "last_run": None,
            "last_errors": [],
        }
        self.last_outputs: Dict[str, Any] | None = None
        self._initialise_components()

    def _initialise_components(self) -> None:
        for name, spec in DEFAULT_COMPONENTS.items():
            component_cfg = self.config.get(name, {})
            enabled = bool(component_cfg.get("enabled", True))
            raw_config = component_cfg.get("config") or {}
            try:
                validated_config = validate_component_config(name, raw_config)
            except (ValidationError, ValueError) as exc:
                logger.error("Configuration for %s is invalid: %s", name, exc)
                self.components[name] = None
                continue

            if not enabled:
                logger.info("Component %s is disabled via configuration", name)
                self.components[name] = None
                continue

            try:
                factory = self.component_factories.get(name)
                if factory is not None:
                    component = factory(validated_config)
                else:
                    module = importlib.import_module(spec.module_path)
                    component_cls = getattr(module, spec.class_name)
                    component = component_cls(validated_config)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to load component %s", name)
                self.components[name] = None
                continue

            self.components[name] = component

    def _adapter_input(self, component: str, outputs: Dict[str, Any], base_input: Any) -> Any:
        if component == "fractal_engine":
            return base_input
        if component == "swarm_coordinator":
            return outputs.get("fractal_output")
        if component == "quantum_processor":
            return {
                "fractal_output": outputs.get("fractal_output"),
                "swarm_output": outputs.get("swarm_output"),
            }
        if component == "meta_learner":
            return {
                "fractal_output": outputs.get("fractal_output"),
                "swarm_output": outputs.get("swarm_output"),
                "quantum_output": outputs.get("quantum_output"),
            }
        return base_input

    def run_cycle(self, input_data: Any) -> Dict[str, Any]:
        """Execute a full inference cycle across all components."""

        outputs: Dict[str, Any] = {
            "fractal_output": None,
            "swarm_output": None,
            "quantum_output": None,
            "meta_output": None,
            "errors": [],
        }

        try:
            normalised_input = validate_input_payload(input_data)
        except ValueError as exc:
            logger.error("Input validation failed: %s", exc)
            outputs["errors"].append(f"input: {exc}")
            self._update_state(outputs)
            return outputs

        for name, spec in DEFAULT_COMPONENTS.items():
            component = self.components.get(name)
            if component is None:
                continue
            try:
                adapted_input = self._adapter_input(name, outputs, normalised_input)
                result = component.process(adapted_input)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Component %s failed during processing", name)
                outputs["errors"].append(f"{name}: {exc}")
                outputs[spec.output_key] = None
                continue

            outputs[spec.output_key] = result

        self._update_state(outputs)
        return outputs

    def _update_state(self, outputs: Dict[str, Any]) -> None:
        self.system_state["cycles"] += 1
        self.system_state["last_run"] = datetime.now(timezone.utc).isoformat()
        self.system_state["last_errors"] = list(outputs.get("errors", []))
        self.last_outputs = outputs

    def get_status(self) -> Dict[str, Any]:
        """Return orchestrator status information."""

        return {
            "components": {
                name: self.components.get(name) is not None for name in DEFAULT_COMPONENTS
            },
            "cycles": self.system_state["cycles"],
            "last_run": self.system_state["last_run"],
            "last_errors": list(self.system_state["last_errors"]),
        }


__all__ = ["RFAISystem"]
