"""Decorators for automatic workflow step/loop recording."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional

from reader.pipelines.report_generation.workflow_register.register import WorkflowRegister
from reader.pipelines.report_generation.workflow_register.models import WorkflowNodeDef

_workflow_register_var: ContextVar[Optional[WorkflowRegister]] = ContextVar(
    "workflow_register", default=None
)


def with_workflow_register(
    workflow_id: str,
    node_defs: Optional[list] = None,
    cfg_arg: str = "cfg",
    cluster_pk_arg: str = "cluster_pk_hash",
    user_intent_arg: str = "user_intent",
) -> Callable:
    """
    Decorator that sets up WorkflowRegister, runs the workflow, and ensures
    write_trace_to_cache is called on any error (or success).
    
    Node definitions are automatically registered via lazy registration when
    @record_step/@record_loop decorators are called. Pass node_defs only for
    backward compatibility (deprecated).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            cfg = bound.arguments.get(cfg_arg)
            cluster_pk_hash = bound.arguments.get(cluster_pk_arg)
            user_intent = bound.arguments.get(user_intent_arg)
            if cfg is None or cluster_pk_hash is None:
                raise ValueError(
                    f"with_workflow_register requires {cfg_arg} and {cluster_pk_arg} in {fn.__name__} signature"
                )

            cache_path = cfg.cache.report_generation_cache / f"{cluster_pk_hash}.json"
            register = WorkflowRegister(
                workflow_id=workflow_id,
                node_defs=node_defs or [],
                cache_path=cache_path,
                cluster_pk_hash=cluster_pk_hash,
            )
            token = _workflow_register_var.set(register)
            try:
                result = await fn(*args, **kwargs)
                register.write_trace_to_cache(cfg, user_intent)
                return result
            except Exception:
                register.write_trace_to_cache(cfg, user_intent)
                raise
            finally:
                _workflow_register_var.reset(token)

        return wrapper

    return decorator


def record_step(
    node_id: str,
    display_name: Optional[str] = None,
    contains: Optional[list[str]] = None,
    must_rerun_status: Optional[list[str]] = None,
):
    """Decorator that records step output and status to the workflow register."""

    def decorator(fn):
        # Store metadata on function object during decoration
        fn._workflow_node_id = node_id
        fn._workflow_node_kind = "step"
        fn._workflow_node_display_name = display_name or node_id.replace("_", " ").title()
        fn._workflow_node_contains = contains
        fn._workflow_node_must_rerun_status = must_rerun_status or ["error", "not_run"]

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            from reader.pipelines.report_generation.blocks import StepTerminationStatus

            register = _workflow_register_var.get()
            # Lazy registration: register node if not already registered
            if register is not None and node_id not in register.node_defs:
                node_def = WorkflowNodeDef(
                    id=node_id,
                    kind="step",
                    display_name=getattr(fn, "_workflow_node_display_name", node_id.replace("_", " ").title()),
                    contains=getattr(fn, "_workflow_node_contains", None),
                    must_rerun_status=getattr(fn, "_workflow_node_must_rerun_status", ["error", "not_run"]),
                )
                register.add_node_def(node_def)
            try:
                output, status = await fn(*args, **kwargs)
                if register is not None:
                    register.record_step(node_id, status, output)
                return (output, status)
            except Exception:
                if register is not None:
                    register.record_step(node_id, StepTerminationStatus.error, None)
                raise

        return wrapper

    return decorator


def record_loop(
    node_id: str,
    display_name: Optional[str] = None,
    must_rerun_status: Optional[list[str]] = None,
):
    """Decorator that records loop output and status to the workflow register."""

    def decorator(fn):
        # Store metadata on function object during decoration
        fn._workflow_node_id = node_id
        fn._workflow_node_kind = "loop"
        fn._workflow_node_display_name = display_name or node_id.replace("_", " ").title()
        fn._workflow_node_must_rerun_status = must_rerun_status or ["not_run", "error"]

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            from reader.pipelines.report_generation.workflow_register.models import LoopRunStatus

            register = _workflow_register_var.get()
            # Lazy registration: register node if not already registered
            if register is not None and node_id not in register.node_defs:
                node_def = WorkflowNodeDef(
                    id=node_id,
                    kind="loop",
                    display_name=getattr(fn, "_workflow_node_display_name", node_id.replace("_", " ").title()),
                    contains=None,
                    must_rerun_status=getattr(fn, "_workflow_node_must_rerun_status", ["not_run", "error"]),
                )
                register.add_node_def(node_def)
            try:
                output, status = await fn(*args, **kwargs)
                if register is not None:
                    register.record_loop(node_id, status, output)
                return (output, status)
            except Exception:
                if register is not None:
                    register.record_loop(node_id, LoopRunStatus.error, None)
                raise

        return wrapper

    return decorator


def get_workflow_register_var() -> ContextVar[Optional[WorkflowRegister]]:
    """Return the context var for workflow register (for setting/resetting)."""
    return _workflow_register_var
