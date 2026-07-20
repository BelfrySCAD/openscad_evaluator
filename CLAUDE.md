# CLAUDE.md — openscad_evaluator

## Build & Test

```bash
# Install in development mode (with DXF import support)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/openscad_evaluator --cov-report=term-missing
```

## Architecture

### Data Flow

```
OpenSCAD AST (from openscad_lalr_parser)
    ↓
Evaluator.evaluate(nodes, root_scope)
    ↓ resolve (AST walk, no Manifold calls)
CSGNode tree (plain data)
    ↓ generate (bottom-up Manifold/CrossSection construction)
ColoredBody list + originalID → AST node table
```

Two-pass, not one: `resolve` walks the whole AST and builds an explicit `CSGNode` tree describing
what to build, with zero Manifold calls; `generate_tree()` then walks that tree bottom-up and does
all the actual Manifold/CrossSection work, checking `ManifoldCache` (content-hash keyed) before
each node so an unchanged subtree — e.g. one untouched by a debugger step or a partial re-render —
reuses its previous result instead of recomputing it.

### Key Files

- `src/openscad_evaluator/evaluator.py` — everything: `Evaluator`, `EvalContext`, built-ins,
  2D/3D geometry, CSG tree, profiling, `ManifoldCache`, font/DXF/3MF import, `resolve_use_scopes`
- `src/openscad_evaluator/_css_colors.py` — static CSS/SVG color-name → RGB table (generated from
  a live Qt install, not transcribed — see the table's own module docstring for how to regenerate
  it if it ever needs updating)
- `src/openscad_evaluator/resources/fonts/` — bundled Liberation Sans fallback font (used when
  `fc-match` isn't available or a `font=` spec can't be resolved)

### Design Patterns

- **Callback injection, not GUI coupling**: `Evaluator.__init__` takes `echo_fn`/`debug_hook`/
  `error_break_fn`/`return_hook`, all optional plain callables. No `QObject`, no signals — a
  caller (like BelfrySCAD's debugger) wires these to its own event system; the evaluator itself
  has zero GUI-toolkit dependencies. See `examples/minimal_debugger.py` for a minimal, runnable
  `debug_hook` integration (tracing, breakpoints, variable overrides via `mods`).
- **`EvalContext`**: `__slots__`-based, threaded through recursive evaluation, carries lexical
  scope + `$`-variable dynamic scope + `let` bindings + color/children state. `child_ctx()`/
  `call_ctx()`/`let_child_ctx()` derive new contexts for different scoping situations.
- **Content-hash geometry cache**: `ManifoldCache` (opt-in, `None` by default) keys on each
  `CSGNode`'s resolved content, letting `generate_tree()` skip recomputing unchanged Manifold work
  across renders/debugger pauses. See `examples/manifold_cache_reuse.py` for a minimal, runnable
  demonstration of the speedup on an unchanged subtree.
- **Profiling**: opt-in (`Evaluator(profile=True)`), self-time + cumulative-time per call site
  (not per declaration), zero overhead when off.

See `docs/evaluator.md` for the full reference: scope processing, assignment order, the complete
built-ins table, 2D/3D geometry handling, error format, `$variables` scoping, `include`/`use`, and
the Manifold provenance / AST-to-geometry-ID mapping API used by BelfrySCAD's WYSIWYG picking.

### Test Organization

- `tests/test_evaluator.py` — the whole test suite: built-ins, scoping, CSG tree, profiling,
  `ManifoldCache`, error handling, real-script regression cases
- `tests/test_examples.py` — runs every script under `examples/` as `__main__`, so their own
  `assert`s double as regression coverage against the examples drifting out of sync with the API
