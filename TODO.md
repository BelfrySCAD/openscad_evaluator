# TODO

- Materials support
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

Tools and API
- Real text shaping (kerning, bidi) — optional, large (#96)
