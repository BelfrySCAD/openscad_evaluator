# TODO

- Materials support
- Tail Recursion Optimizations — `function acc(n,a=0)=n<=0?a:acc(n-1,a+1); acc(200)` dies with
  "AST too deeply nested"; OpenSCAD handles millions (cpp 968aaf7, 169d440)

## Backports from openscad_cpp_evaluator

Triaged 2026-09-24 against cpp 1dc0538..e7a6859 (1.27.0). Hashes are cpp commits, `#N` are cpp
PRs. Each item below was confirmed to still be wrong here, most by a probe script; items marked
**silent** give wrong output with no warning.

### Features

Needs openscad_lalr_parser changes first
- Strict-commas mode (#158)

Language extensions
- `children(separate=true)` (#110, #111, #112)
- `$_BELFRYSCAD`, `$_SUPPORTED_FEATURE`, `supported_feature()` (#113, #115)
- `minkowski_difference()` (#101), `sphere(style=)` (#102), `simplify()` (#103)
- `levelset()` from grid or function, 2D contours, clean box cut (#124, #126, #127). Bring #136
  with it: never cache a subtree whose params hold a closure (`_canon` keys by identity), and
  #125's explicit-undef `isovalue` counting as absent
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
