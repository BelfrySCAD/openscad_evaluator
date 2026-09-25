# TODO

- Materials support

## Backports from openscad_cpp_evaluator

Triaged 2026-09-24 against cpp 1dc0538..e7a6859 (1.27.0). Hashes are cpp commits, `#N` are cpp
PRs. Each item below was confirmed to still be wrong here, most by a probe script; items marked
**silent** give wrong output with no warning.

### Features

Needs openscad_lalr_parser changes first
- Strict-commas mode (#158)

Tools and API
- Real text shaping (kerning, bidi) — optional, large (#96)
