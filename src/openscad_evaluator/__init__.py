"""openscad_evaluator: an AST evaluator for the OpenSCAD language, producing
Manifold CSG geometry from a parsed AST (see `openscad_lalr_parser`)."""

from openscad_evaluator.evaluator import (
    Evaluator,
    EvalContext,
    EvalError,
    CSGNode,
    ColoredBody,
    ManifoldCache,
    OscObject,
    OscRange,
    CallSiteProfile,
    ProfileResult,
    to_renderable_bodies,
    flatten_csg_tree,
    format_csg_tree,
    resolve_use_scopes,
)

__all__ = [
    "Evaluator",
    "EvalContext",
    "EvalError",
    "CSGNode",
    "ColoredBody",
    "ManifoldCache",
    "OscObject",
    "OscRange",
    "CallSiteProfile",
    "ProfileResult",
    "to_renderable_bodies",
    "flatten_csg_tree",
    "format_csg_tree",
    "resolve_use_scopes",
]
