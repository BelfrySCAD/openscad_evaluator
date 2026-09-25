# TODO

- Materials support
- Tail Recursion Optimizations — `function acc(n,a=0)=n<=0?a:acc(n-1,a+1); acc(200)` dies with
  "AST too deeply nested"; OpenSCAD handles millions (cpp 968aaf7, 169d440)

## Backports from openscad_cpp_evaluator

Triaged 2026-09-24 against cpp 1dc0538..e7a6859 (1.27.0). Hashes are cpp commits, `#N` are cpp
PRs. Each item below was confirmed to still be wrong here, most by a probe script; items marked
**silent** give wrong output with no warning.

### Bug fixes

Scoping and values
- **silent** Escaping closures lose their captures: `function mk(x)=function(y) x+y; mk(10)(5)`
  is undef, should be 15. `_eval_function_literal` must capture `ctx.let` (81c260a, 5ceafc6)
- **silent** Globals re-evaluated on every read inside a function: `r=rands(0,1,1); function g()=r;
  g()==g()` is false. Evaluate every file's globals once per run, used files included (#166)
- **silent** A user function doesn't shadow a builtin: `function sin(x)=42; sin(0)` gives 0
  (`_eval_function_call` checks builtins first) (#119)
- **silent** Braced block isn't a scope: `if(true){x=2;}` / `translate(){a=99;}` leak into the
  outer scope, `$fn` too, with a spurious "overwritten" warning (#132)
- **silent** `let(a=1,b=a+1) echo(b)` statement form isn't sequential, gives undef (#134)
- **silent** Multi-variable `for` evaluates every range before binding: `[for(i=[0:2],j=[0:i]) ..]`
  warns "unknown variable 'i'" and yields 3 items not 6. Also list-comp for and
  intersection_for (c7059f9)

`$` variables
- **silent** `children()` forwarding overwrites `$children`/`$parent_modules` with the wrapper's
  own value, breaking BOSL2's `if($children>N) children(N)` (c7059f9)
- A never-set `$var` and `is_undef(zzz)` should be undef with no warning (37539f9, #114)
- `$preview` should be seeded `false`, not undef-with-warning (#100)

CSG
- **silent** One Manifold-invalid operand empties the whole union/difference/intersection;
  filter operands whose `status()` isn't NoError (1d8f674)
- **silent** Empty-operand rule: `intersection(){cube(2);*cube(1);}`,
  `difference(){*cube(10);cube(2);}`, `intersection(){cube(2); if(false)..}` should all be the
  cube. Only a statement that built a CSG node is an operand (#139)
- **silent** `echo("hi") cube(10);` drops the cube (#141)
- 2D shapes lose out-of-plane transforms: `rotate([55,0,25]) square(5)` stays flat (#141)
- `union(){cube(2);square(3);}` crashes (`NoneType + CrossSection`); union, color and transforms
  must carry both dimensions (#145)
- `!` re-roots the tree instead of filtering roles: `translate([50,0,0]) !cube(5)` should land at
  x=50 (#87)
- 2D `minkowski()` produces nothing (#89)

Builtin arguments
- **silent** `cylinder()` positionals are `(h, r1, r2, center)`, `d` overrides `r`, `r2`
  defaults to 1; `cylinder(10,5,2)` currently gets top radius 5 (04797d0)
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
- **silent** String escapes never resolved: `len("a\nb")` is 4, `len("\x41")` is 4 (#86, #150,
  #174). The parser keeps them verbatim on purpose; resolve here
- `-7%3` should be -1, `7%0` nan, `0^-1` inf (#99)
- Degree trig not bit-exact: `sin(45)-cos(45)` is -1.1e-16; port `degree_trig.cc` (#133)
- `[5:1:0]==[5:1:0]` is false (#146)
- `[each "12"]` should split to characters; `[each [0:2]]` should expand the range (#146)
- `str(function(x) x+1)` should print `function(x) (x + 1)` like the reference (#146)

Diagnostics
- Undeclared / extra arguments are silently dropped in `_bind_args` (#92)
- Numeric builtins don't warn on a non-number ("cos() parameter could not be converted")
  (04797d0); argument/operand warnings in general, `abs(undef)`, `1+"a"` (#99)
- Backwards-range warning for `[5:0]` with an implicit step — needs the lalr parser's
  step-written flag first (#108)
- Range of ≥1,000,000 elements: warn "Bad range parameter in for statement: too many elements"
  and iterate zero times (f614e0d); `[for(i=[0:1:1/0]) i]` hangs instead (#147)
- Warnings should name the user's call site and carry TRACE lines, generate-time ones too
  (3e11352)
- Open polyhedron gives no warning; report boundary-edge count and first location (#188, #189)

Meshes
- **silent** Open polyhedron/`import()` vanishes; draw it as a display-only surface (needs a raw
  mesh on `ColoredBody`, kept out of CSG) (57fa5d4); `hull()` of one is empty (#135)
- Polyhedron fan-triangulates faces; reuse `_triangulate_planar_face` (#88)
- Polyhedron vertices cast to float32 (#94)
- Polyhedron always welds coincident vertices, fusing touching shells; keep the weld only if
  the result stays manifold (#105)

Colour
- **silent** 2D union of red and blue squares comes out all red (#161)
- **silent** Colour set via a module's `children()` is lost in a union (#162)

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
- `render()` in expression position returning an object (#104)
- Strict-commas mode (#158)

Language extensions
- `polyhedron(vnf)`: `[verts,faces]` or the render object (#106)
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
