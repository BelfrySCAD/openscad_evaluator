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

STL/OBJ/OFF/3MF export needs nothing extra -- all four writers are pure Python (`export.py`; 3MF
is just a ZIP of XML, built with the standard library's `zipfile`/`xml.etree.ElementTree`, not the
platform-limited `lib3mf`).

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

A body carries a 3D `body` (Manifold), a 2D `section` (CrossSection), or -- for an open mesh
that doesn't close a solid, which OpenSCAD still draws -- neither, with the triangles in
`raw_mesh`. A renderer or exporter should draw those too; `export.py`'s writers do.

### Language extensions

Shared with [openscad_cpp_evaluator](https://github.com/BelfrySCAD/openscad_cpp_evaluator), and
not part of upstream OpenSCAD:

```openscad
obj = render() { difference() { cube(10, center=true); sphere(4); } };
echo(obj.volume, obj.area, obj.genus, obj.boundingbox);   // measured; nothing is drawn
polyhedron(obj);                                          // and straight back in
polyhedron(spheroid(d=30));                               // any BOSL2 VNF, too
```

`render()` in expression position builds its children's geometry, measures it and discards it.
`render` is therefore a reserved word (openscad_lalr_parser >= 1.2.0), and the braced form is
required. See [`docs/evaluator.md`](docs/evaluator.md) for the object's keys.

## Command Line

Installing the package also installs an `openscad-evaluator` script that evaluates a `.scad` file
and exports the result -- STL, OBJ, OFF, or 3MF, inferred from `-o`'s extension:

```bash
openscad-evaluator model.scad -o model.stl
openscad-evaluator model.scad -o model.3mf
openscad-evaluator model.scad -o model.mesh --format obj   # explicit format overrides the extension
```

`echo()`/warning output goes to stdout; a parse or evaluation error goes to stderr and exits
non-zero.

`--debug` drops into a small, gdb-style interactive debugger instead of running straight through:
set breakpoints and `run`, then `step`/`next`/`finish`/`continue`, `print <name>`, `backtrace`, and
`set <name>=<value>` to override a variable before resuming.

```
$ openscad-evaluator model.scad -o model.stl --debug
Reading symbols from model.scad...
(scad-dbg) break 12
Breakpoint set at model.scad:12
(scad-dbg) run

Breakpoint hit at model.scad:12
     10  module bracket(w) {
     11      h = w * 2;
->   12      cube([w, h, 3]);
     13  }
(scad-dbg) print h
$1 = 20
(scad-dbg) continue
Exported to model.stl
```

See `src/openscad_evaluator/_debug_repl.py` for the full command list (`help` at either prompt also
prints it), and `examples/minimal_debugger.py` for the underlying `debug_hook` protocol if you're
integrating your own debugger rather than using this one.

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
