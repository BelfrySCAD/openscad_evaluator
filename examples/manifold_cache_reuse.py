"""Minimal example: reuse `ManifoldCache` across repeated `evaluate()` calls.

A real editor re-parses the whole script and builds a brand-new `Evaluator`
on every render, so nothing is cached by default. Passing the *same*
`ManifoldCache` instance into successive `Evaluator(manifold_cache=...)`
calls lets `generate_tree()` skip Manifold work for any CSGNode subtree
whose *content* (kind/params/children) is byte-for-byte the same as a
previous render -- exactly what happens when a user edits one part of a
script and re-renders the whole thing: everything under the edit still has
to be regenerated, but everything else is served straight from cache.

Run: python examples/manifold_cache_reuse.py
"""
import time

from openscad_lalr_parser import getASTfromString, build_scopes
from openscad_evaluator import Evaluator, ManifoldCache

# `minkowski()` of two moderately dense spheres is deliberately expensive;
# `translate`'s own offset is the only thing that changes between renders,
# so the minkowski subtree underneath is identical both times.
SCRIPT = """\
translate([{x}, 0, 0])
  minkowski() {{
    sphere(r=6, $fn=48);
    sphere(r=2, $fn=48);
  }}
"""


def render(cache: ManifoldCache, x: int) -> float:
    nodes = getASTfromString(SCRIPT.format(x=x))
    root_scope = build_scopes(nodes)
    ev = Evaluator(manifold_cache=cache)
    start = time.perf_counter()
    bodies, _id_to_node = ev.evaluate(nodes, root_scope)
    elapsed = time.perf_counter() - start
    assert len(bodies) == 1
    return elapsed


def main():
    cache = ManifoldCache()

    cold = render(cache, x=0)
    print(f"cold render (nothing cached yet):       {cold * 1000:6.1f}ms")

    # Same minkowski content, only translate's own offset differs -- translate
    # misses (its own params changed) but the expensive minkowski subtree
    # underneath is served from cache.
    warm = render(cache, x=10)
    print(f"warm render (minkowski subtree cached): {warm * 1000:6.1f}ms")

    assert warm < cold, "expected the cached minkowski to make the second render faster"
    print(f"speedup: {cold / warm:.1f}x")


if __name__ == "__main__":
    main()
