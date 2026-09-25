# TODO

- Materials support
- A missing `import()` file aborts the render (`ERROR: import: [Errno 2] ...`); OpenSCAD warns
  `Can't open import file '...'` and carries on, and in expression position warns `Could not
  read file '...'` and returns undef. openscad_cpp_evaluator aborts too, so fix both together
- SVG import ignores units and the document height: OpenSCAD maps a unitless SVG at 72 dpi
  (25.4/72 mm per unit) and flips Y about the height, so a 100-unit drawing's rect at y=0..10
  lands at y=31.75..35.28 there and y=-10..0 here. openscad_cpp_evaluator does the same as this
  package, so fix both together

## Backports from openscad_cpp_evaluator

Triaged 2026-09-24 against cpp 1dc0538..e7a6859 (1.27.0). Hashes are cpp commits, `#N` are cpp
PRs. Each item below was confirmed to still be wrong here, most by a probe script; items marked
**silent** give wrong output with no warning.

### Features

Needs openscad_lalr_parser changes first
- Strict-commas mode (#158)

Language extensions
- Strict mesh check, `repair()`, `import(repair=true)`, export-time check (72ca136);
  `mesh_repair()` (#191)

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
- Real text shaping (kerning, bidi) — optional, large (#96)
