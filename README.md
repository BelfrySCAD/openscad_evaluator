# openscad_evaluator

An AST evaluator for the OpenSCAD language, producing [Manifold](https://github.com/elalish/manifold) CSG geometry from a parsed AST.

[![Tests](https://github.com/BelfrySCAD/openscad_evaluator/actions/workflows/pytest.yml/badge.svg)](https://github.com/BelfrySCAD/openscad_evaluator/actions/workflows/pytest.yml)
[![PyPI version](https://img.shields.io/pypi/v/openscad-evaluator)](https://pypi.org/project/openscad-evaluator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`openscad_evaluator` takes an OpenSCAD AST — as produced by
[`openscad_lalr_parser`](https://github.com/BelfrySCAD/openscad_lalr_parser) — and walks it to
produce [Manifold](https://github.com/elalish/manifold) meshes: a two-pass **resolve** (AST walk,
no CSG calls) then **generate** (bottom-up Manifold/CrossSection construction) pipeline, with
`$fn`/`$fa`/`$fs`, full built-in coverage (primitives, transforms, boolean ops, 2D geometry,
`text()`, `surface()`, DXF/3MF import, `roof()`), and a content-hash geometry cache
(`ManifoldCache`) so repeated renders/debugger pauses skip unchanged Manifold work.

It's GUI- and toolkit-agnostic: the only way it talks back to a caller is a handful of optional
callback parameters on `Evaluator.__init__` (`echo_fn`, `debug_hook`, `error_break_fn`,
`return_hook`) — no `QObject`, no signals, no direct rendering. This is the evaluator that powers
[BelfrySCAD](https://github.com/BelfrySCAD/BelfrySCAD).

See [`docs/evaluator.md`](docs/evaluator.md) for the full architecture reference: scope
processing, assignment order, the built-ins table, 2D/3D geometry handling, error format,
`$variables` scoping, `include`/`use`, and the Manifold provenance / AST-to-geometry-ID mapping
used for WYSIWYG picking.

## Installation

```bash
pip install openscad-evaluator
```

For DXF import support:

```bash
pip install openscad-evaluator[dxf]
```

### From Source

```bash
git clone https://github.com/BelfrySCAD/openscad_evaluator.git
cd openscad_evaluator
pip install -e ".[dev]"
```

## Quick Start

```python
from openscad_lalr_parser import getASTfromString, build_scopes
from openscad_evaluator import Evaluator

nodes = getASTfromString("cube([10, 10, 10]);")
root_scope = build_scopes(nodes)

ev = Evaluator()
bodies, id_to_node = ev.evaluate(nodes, root_scope)
for body in bodies:
    print(body.body.num_tri(), "triangles")
```

## Examples

Runnable, self-checking scripts under [`examples/`](examples/) go beyond the Quick Start above to
cover the two optional integration points most callers ask about:

- [`examples/minimal_debugger.py`](examples/minimal_debugger.py) — hooking a debugger into
  `Evaluator(debug_hook=...)`: tracing every statement, stopping evaluation at a breakpoint, and
  overriding a variable's value mid-run via the hook's `mods` return value.
- [`examples/manifold_cache_reuse.py`](examples/manifold_cache_reuse.py) — sharing one
  `ManifoldCache` across repeated `evaluate()` calls so an unchanged subtree (e.g. everything but
  the part of the script a user just edited) skips re-running Manifold work on the next render.

```bash
python examples/minimal_debugger.py
python examples/manifold_cache_reuse.py
```

Both are also run under `pytest` (`tests/test_examples.py`) so they fail CI if they drift out of
sync with the real API.

## Development

```bash
uv sync --all-extras
uv run pytest
```
