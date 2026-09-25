# TODO

- Materials support
- Tail Recursion Optimizations — `function acc(n,a=0)=n<=0?a:acc(n-1,a+1); acc(200)` dies with
  "AST too deeply nested"; OpenSCAD handles millions (cpp 968aaf7, 169d440)

## Backports from openscad_cpp_evaluator

Triaged 2026-09-24 against cpp 1dc0538..e7a6859 (1.27.0). Hashes are cpp commits, `#N` are cpp
PRs. Each item below was confirmed to still be wrong here, most by a probe script; items marked
**silent** give wrong output with no warning.

### Bug fixes

`$` variables
- `$parent_modules` inside forwarded children is 1 here and 0 in the C++ port; OpenSCAD
  gives 2 for `module w(){children();} module o(){w() echo($parent_modules);} o(){cube();}`
- A never-set `$var` and `is_undef(zzz)` should be undef with no warning (37539f9, #114)
- `$preview` should be seeded `false`, not undef-with-warning (#100)

Builtin arguments
- `resize(auto=)` ignored (04797d0)
- `offset(2)` means r=2; bare `offset()` means r=1 (04797d0)
- `children([0:1])` crashes (`int()` of an OscRange); expand vectors and ranges (#107)
- `search()` ignores `num_returns_per_match`/`index_col_num` (#146)
- `object()` single-element `[key]` delete entry (#106)
- `font` style ignored in `text()`/`textmetrics()`/`fontmetrics()` (#163, #164); `:style=bold`
  falls back to Regular when `fc-match` is missing, i.e. always on macOS (#171)
- `fill()` missing (#100)
- `version()` is `[2025,1,1]`; `version_num([2019,5,0])` is undef (#98)

Numbers and strings
- Undefined escape sequences (`\q`, `\x80`) should warn "Undefined escape sequence";
  backslash-newline inside a string waits on openscad_lalr_parser, which rejects it
- `-7%3` should be -1, `7%0` nan, `0^-1` inf (#99)
- Degree trig not bit-exact: `sin(45)-cos(45)` is -1.1e-16; port `degree_trig.cc` (#133)
- `[5:1:0]==[5:1:0]` is false (#146)
- `str(function(x) x+1)` should print `function(x) (x + 1)` like the reference (#146)

Diagnostics
- Undeclared / extra arguments are silently dropped in `_bind_args` (#92)
- Numeric builtins don't warn on a non-number ("cos() parameter could not be converted")
  (04797d0); argument/operand warnings in general, `abs(undef)`, `1+"a"` (#99)
- Warnings should name the user's call site and carry TRACE lines, generate-time ones too
  (3e11352)

Meshes
- Polyhedron fan-triangulates faces; reuse `_triangulate_planar_face` (#88)
- Polyhedron vertices cast to float32 (#94)
- Polyhedron always welds coincident vertices, fusing touching shells; keep the weld only if
  the result stays manifold (#105)

Cache
- A `ManifoldCache` hit reuses the first call site's bodies/originalIDs and doesn't refresh
  `id_to_node` (#84, #85)
- A cache hit skips generation, so its warnings vanish on re-render (#186)

Debugger / viewport
- `x=f(y);` stops twice at statement level (#82)
- Step-to-child from `children()` finds nothing; read the context's forwarded children (#83)
- 2D preview slab is 1e-3 thick, reference is 1; translated 2D shape loses its Z (#138)

### Features

Needs openscad_lalr_parser changes first
- Strict-commas mode (#158)

Language extensions
- `children(separate=true)` (#110, #111, #112)
- `$_BELFRYSCAD`, `$_SUPPORTED_FEATURE`, `supported_feature()` (#113, #115)
- `minkowski_difference()` (#101), `sphere(style=)` (#102), `simplify()` (#103)
- `linear_solve()`; explicit undef argument counts as absent (#120, #122, #125)
- `levelset()` from grid or function, 2D contours, clean box cut (#124, #126, #127). Bring #136
  with it: never cache a subtree whose params hold a closure (`_canon` keys by identity)
- Strict mesh check, `repair()`, `import(repair=true)`, export-time check (72ca136);
  `mesh_repair()` (#191)
- `dxf_dim()` / `dxf_cross()` (#100), `list_fonts()` (#163)
- SVG `import()` filtering by `id=`/`class=` (#182)

Export
- Export the implicit top-level union split per colour then per component, with per-triangle
  colour (#93, #80, #81, #157, #165, #177)
- Formats: AMF with a single-sourced format list (#118), PLY, VRML, X3D, ASCII STL, OBJ+`.mtl`,
  SVG (#159), PDF with ruler/page options (#160); sliver stripping and mesh checks

Tools and API
- Coverage: statements, branch arms, bodies, used-file globals, tail-called bodies
  (#169, #170, #175)
- Per-path profiling tree (`ProfileResult.paths`), child-call kind, call columns
  (8e54284, 709da93)
- `idToCallSite` (#180), `evaluate(generate=False)` (#144)
- Cut-face green for uncoloured subtrahends, `keep_minuend_color` (#173)
- Pickable originalID for 2D sections (#192), `flat_preview_height` (#193)
- Real text shaping (kerning, bidi) — optional, large (#96)
