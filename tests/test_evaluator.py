"""
Tests for the openscad_evaluator evaluator.

Each test calls run(src) which parses, scopes, and evaluates OpenSCAD source,
returning (bodies, echo_lines). Geometry tests inspect bounding boxes;
expression tests capture echo output.
"""
import numpy as np
import pytest
from openscad_lalr_parser import getASTfromString, build_scopes

from openscad_evaluator.evaluator import (
    Evaluator, EvalContext, EvalError, _resolve_font, CSGNode, flatten_csg_tree,
    format_csg_tree, _summarize_param, ManifoldCache, _DEFAULT_GEOMETRY_COLOR,
)


def run(src: str):
    """Parse, scope, and evaluate src. Returns (bodies, echo_lines)."""
    echo_lines = []
    nodes = getASTfromString(src, include_comments=False)
    root_scope = build_scopes(nodes)
    ev = Evaluator(echo_fn=lambda msg: echo_lines.append(msg))
    bodies, _ = ev.evaluate(nodes, root_scope)
    return bodies, echo_lines


def _echoes(lines):
    """The ECHO lines alone, for a test about values whose script also warns."""
    return [l for l in lines if not l.startswith("WARNING:")]


def run_tree(src: str):
    """Like run(), but also returns the Evaluator so tests can inspect
    its csg_tree. Returns (bodies, echo_lines, evaluator)."""
    echo_lines = []
    nodes = getASTfromString(src, include_comments=False)
    root_scope = build_scopes(nodes)
    ev = Evaluator(echo_fn=lambda msg: echo_lines.append(msg))
    bodies, _ = ev.evaluate(nodes, root_scope)
    return bodies, echo_lines, ev


def skip_unless_font_installed(font_spec: str, expected_family: str):
    """Skip a test if `font_spec` doesn't resolve (via fc-match) to a font
    actually named `expected_family` on this machine. Arial/Times New
    Roman/STIXGeneral aren't installed on every system (e.g. CI runners) —
    fc-match then silently substitutes a metric-compatible fallback
    (Liberation Sans/Serif), which has different real glyph metrics than
    the tests' hardcoded expected values. Skip rather than assert against
    whatever substitute happens to be installed."""
    resolved = _resolve_font(font_spec)
    family = resolved["family_name"]
    if family.lower() != expected_family.lower():
        pytest.skip(
            f"{expected_family!r} not installed on this system "
            f"(fc-match resolved {font_spec!r} to {family!r} instead)"
        )


def bbox(bodies):
    """Return (xmin,ymin,zmin,xmax,ymax,zmax) union over all non-empty manifold bodies."""
    assert bodies, "no geometry produced"
    bbs = [b.body.bounding_box() for b in bodies if b.body is not None]
    assert bbs, "no 3D geometry produced"
    return (
        min(bb[0] for bb in bbs), min(bb[1] for bb in bbs), min(bb[2] for bb in bbs),
        max(bb[3] for bb in bbs), max(bb[4] for bb in bbs), max(bb[5] for bb in bbs),
    )


def approx(v, rel=1e-4):
    return pytest.approx(v, rel=rel)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class TestExpressions:
    def test_not_equal(self):
        _, lines = run("echo(1 != 2);")
        assert lines == ["ECHO: true"]

    def test_greater_than_or_equal(self):
        _, lines = run("echo(3 >= 3);")
        assert lines == ["ECHO: true"]

    def test_vector_add(self):
        _, lines = run("echo([1,2,3] + [4,5,6]);")
        assert lines == ["ECHO: [5, 7, 9]"]

    def test_vector_subtract(self):
        _, lines = run("echo([5,7,9] - [4,5,6]);")
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_matrix_add(self):
        # `+`/`-` between lists of vectors (matrices) must recurse element-wise
        # per row, not concatenate each row's elements (e.g. `[0,0,0,0] +
        # [1,1,1,1]` must give `[1,1,1,1]`, not `[0,0,0,0,1,1,1,1]`).
        _, lines = run("echo([[0,0,0,0],[0,0,0,0]] + [[1,1,1,1],[2,2,2,2]]);")
        assert lines == ["ECHO: [[1, 1, 1, 1], [2, 2, 2, 2]]"]

    def test_matrix_subtract(self):
        _, lines = run("echo([[5,5,5],[5,5,5]] - [[1,2,3],[4,5,6]]);")
        assert lines == ["ECHO: [[4, 3, 2], [1, 0, -1]]"]

    def test_string_plus_string_is_undef(self):
        # OpenSCAD has no `+` for strings (unlike Python's str.__add__,
        # which would silently concatenate them).
        _, lines = run('echo("ab" + "cd");')
        assert _echoes(lines) == ["ECHO: undef"]

    def test_vector_scale_right(self):
        _, lines = run("echo([1,2,3] * 2);")
        assert lines == ["ECHO: [2, 4, 6]"]

    def test_vector_scale_left(self):
        _, lines = run("echo(3 * [1,2,3]);")
        assert lines == ["ECHO: [3, 6, 9]"]

    def test_unary_minus_vector(self):
        _, lines = run("echo(-[1,2,3]);")
        assert lines == ["ECHO: [-1, -2, -3]"]

    def test_member_x(self):
        _, lines = run("v = [10,20,30]; echo(v.x);")
        assert lines == ["ECHO: 10"]

    def test_member_y(self):
        _, lines = run("v = [10,20,30]; echo(v.y);")
        assert lines == ["ECHO: 20"]

    def test_member_z(self):
        _, lines = run("v = [10,20,30]; echo(v.z);")
        assert lines == ["ECHO: 30"]

    def test_arithmetic(self):
        _, lines = run("echo(2 + 3 * 4);")
        assert lines == ["ECHO: 14"]

    def test_division(self):
        _, lines = run("echo(10 / 4);")
        assert lines == ["ECHO: 2.5"]

    def test_modulo(self):
        _, lines = run("echo(10 % 3);")
        assert lines == ["ECHO: 1"]

    def test_exponent(self):
        _, lines = run("echo(2 ^ 10);")
        assert lines == ["ECHO: 1024"]

    def test_unary_minus(self):
        _, lines = run("echo(-5);")
        assert lines == ["ECHO: -5"]

    def test_comparison(self):
        _, lines = run("echo(3 > 2);")
        assert lines == ["ECHO: true"]

    def test_logical_and(self):
        _, lines = run("echo(true && false);")
        assert lines == ["ECHO: false"]

    def test_logical_or(self):
        _, lines = run("echo(false || true);")
        assert lines == ["ECHO: true"]

    def test_logical_not(self):
        _, lines = run("echo(!true);")
        assert lines == ["ECHO: false"]

    def test_ternary_true(self):
        _, lines = run("echo(1 > 0 ? 42 : 99);")
        assert lines == ["ECHO: 42"]

    def test_ternary_false(self):
        _, lines = run("echo(1 < 0 ? 42 : 99);")
        assert lines == ["ECHO: 99"]

    def test_vector_literal(self):
        _, lines = run("echo([1, 2, 3]);")
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_vector_index(self):
        _, lines = run("v = [10, 20, 30]; echo(v[1]);")
        assert lines == ["ECHO: 20"]

    def test_range(self):
        # Ranges echo as lazy [start : step : end], not expanded
        _, lines = run("echo([1:3]);")
        assert lines == ["ECHO: [1 : 1 : 3]"]

    def test_range_step(self):
        _, lines = run("echo([0:2:6]);")
        assert lines == ["ECHO: [0 : 2 : 6]"]

    def test_range_descending(self):
        _, lines = run("echo([5:-1:3]);")
        assert lines == ["ECHO: [5 : -1 : 3]"]


# ---------------------------------------------------------------------------
# Bitwise/shift operators (real OpenSCAD PR #4833, merged 2025-03-14)
# ---------------------------------------------------------------------------

class TestBitwiseOps:
    def test_basic_operations(self):
        _, lines = run(
            'echo(12 & 5, 12 | 5, 2 << 3, 16 >> 2, ~0, ~-1, ~5);'
        )
        assert lines == ["ECHO: 4, 13, 16, 4, -1, 0, -6"]

    def test_precedence_shift_over_and(self):
        # Shift binds tighter than binary-and/or -- real OpenSCAD's grammar
        # deliberately differs from C to avoid `x & 1 == 0` parsing as
        # `x & (1 == 0)`.
        _, lines = run("echo((3 << 1 & 14) == ((3 << 1) & 14));")
        assert lines == ["ECHO: true"]

    def test_precedence_and_over_or(self):
        _, lines = run("echo((1 & 2 | 3) == ((1 & 2) | 3));")
        assert lines == ["ECHO: true"]

    def test_precedence_or_over_comparison(self):
        _, lines = run("echo((1 | 2 > 3) == ((1 | 2) > 3));")
        assert lines == ["ECHO: true"]

    def test_negative_numbers(self):
        _, lines = run("echo((-1 | 0) == -1, (-1 | 0) == ~0);")
        assert lines == ["ECHO: true, true"]

    def test_fractions_truncate(self):
        _, lines = run("echo(1.4 | 0, 1.6 | 0, -1.4 & 3, -1.6 & 3);")
        assert lines == ["ECHO: 1, 1, 3, 3"]

    def test_shift_overflow_wraps_like_64bit_twos_complement(self):
        _, lines = run("echo(1 << 32 << 32, 1 >> 1);")
        assert lines == ["ECHO: 0, 0"]

    def test_out_of_range_shift_is_undef_and_warns(self):
        _, lines = run(
            "echo(is_undef(1 << 64), is_undef(1 << -1), is_undef(1 >> 64), is_undef(1 >> -1));"
        )
        assert lines == [
            "WARNING: shift too large in file <string>, line 1",
            "WARNING: negative shift in file <string>, line 1",
            "WARNING: shift too large in file <string>, line 1",
            "WARNING: negative shift in file <string>, line 1",
            "ECHO: true, true, true, true",
        ]

    def test_non_numeric_operand_is_undef_and_warns(self):
        _, lines = run('echo(is_undef(1 | "hello"), is_undef(1 & true), is_undef(~"hello"));')
        assert lines == [
            'WARNING: undefined operation (number | string) in file <string>, line 1',
            'WARNING: undefined operation (number & bool) in file <string>, line 1',
            'WARNING: undefined operation (~string) in file <string>, line 1',
            "ECHO: true, true, true",
        ]


# ---------------------------------------------------------------------------
# Variables and scoping
# ---------------------------------------------------------------------------

class TestVariables:
    def test_assignment(self):
        _, lines = run("x = 7; echo(x);")
        assert lines == ["ECHO: 7"]

    def test_undef(self):
        _, lines = run("echo(undef);")
        assert lines == ["ECHO: undef"]

    def test_boolean_literals(self):
        _, lines = run("echo(true, false);")
        assert lines == ["ECHO: true, false"]

    def test_string_literal(self):
        _, lines = run('echo("hello");')
        assert lines == ['ECHO: "hello"']

    def test_computed_assignment(self):
        _, lines = run("a = 3; b = a * 2; echo(b);")
        assert lines == ["ECHO: 6"]

    def test_special_var_assignment(self):
        # $fn at top level goes into dynamic context
        bodies, _ = run("$fn = 8; sphere(r=1);")
        assert bodies

    def test_special_var_lookup(self):
        _, lines = run("$fn = 64; echo($fn);")
        assert lines == ["ECHO: 64"]

    def test_animation_t_defaults_to_zero(self):
        _, lines = run("echo($t);")
        assert lines == ["ECHO: 0"]

    def test_animation_t_set_via_viewport_params(self):
        nodes = getASTfromString("echo($t);", include_comments=False)
        root_scope = build_scopes(nodes)
        echo_lines = []
        ev = Evaluator(echo_fn=lambda msg: echo_lines.append(msg))
        ev.evaluate(nodes, root_scope, {"$t": 0.25})
        assert echo_lines == ["ECHO: 0.25"]


class TestDynExplicit:
    """`EvalContext.dyn_explicit` distinguishes "the script itself assigned
    this $-variable" from "this $-variable is merely present in `dyn`"
    (e.g. $vp* seeded from the current camera via viewport_params, or
    $fn/$fa/$fs/$t/$parent_modules seeded from _DEFAULT_DOLLAR) -- used by
    MainWindow to decide whether a script's $vp* assignment should move
    the viewport camera, vs. leaving a manually-adjusted camera alone."""

    _VP_SEED = {"$vpt": [0.0, 0.0, 0.0], "$vpr": [55.0, 0.0, 25.0], "$vpd": 140.0, "$vpf": 22.5}

    def test_seeded_value_not_explicit(self):
        nodes = getASTfromString("cube(10);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev.evaluate(nodes, root_scope, self._VP_SEED)
        assert "$vpt" in ev._root_ctx.dyn
        assert ev._root_ctx.dyn_explicit == set()

    def test_script_assignment_is_explicit(self):
        nodes = getASTfromString("$vpt = [1, 2, 3]; cube(10);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev.evaluate(nodes, root_scope, self._VP_SEED)
        assert ev._root_ctx.dyn["$vpt"] == [1.0, 2.0, 3.0]
        assert ev._root_ctx.dyn_explicit == {"$vpt"}

    def test_only_the_assigned_name_is_explicit(self):
        nodes = getASTfromString("$vpd = 50; cube(10);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev.evaluate(nodes, root_scope, self._VP_SEED)
        assert ev._root_ctx.dyn_explicit == {"$vpd"}
        assert "$vpt" not in ev._root_ctx.dyn_explicit
        assert "$vpt" in ev._root_ctx.dyn  # still present (seeded), just not explicit

    def test_regular_special_var_assignment_also_tracked(self):
        nodes = getASTfromString("$fn = 64; cube(10);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev.evaluate(nodes, root_scope)
        assert ev._root_ctx.dyn_explicit == {"$fn"}


class TestFormatCsgTree:
    """format_csg_tree/_summarize_param back the Design menu's "Dump CSG
    Tree to Console" command."""

    def test_summarize_param_short_list_shown_in_full(self):
        assert _summarize_param([1, 2, 3]) == "[1, 2, 3]"

    def test_summarize_param_long_list_collapsed(self):
        assert _summarize_param(list(range(20))) == "<list of 20>"

    def test_summarize_param_small_nested_list_shown_in_full(self):
        # A small list of lists (e.g. a handful of polyhedron points) is
        # shown in full, not collapsed just for containing sub-lists --
        # collapsing is purely size-based (item count), so users can
        # actually see e.g. a translate's vector or a small polyhedron's
        # own points instead of an opaque placeholder.
        assert _summarize_param([[0, 0], [1, 0], [1, 1]]) == "[[0, 0], [1, 0], [1, 1]]"

    def test_summarize_param_long_nested_list_collapsed(self):
        assert _summarize_param([[i, i] for i in range(20)]) == "<list of 20>"

    def test_summarize_param_small_dict_shown_in_full(self):
        assert _summarize_param({"a": 1, "b": 2}) == "{'a': 1, 'b': 2}"

    def test_summarize_param_long_dict_collapsed(self):
        assert _summarize_param({str(i): i for i in range(10)}) == "<dict of 10>"

    def test_summarize_param_small_ndarray_shown_in_full(self):
        import numpy as np
        arr = np.array([[1.0, 2.0, 3.0]])
        assert _summarize_param(arr) == "[[1.0, 2.0, 3.0]]"

    def test_summarize_param_large_ndarray_collapsed(self):
        # A large array (e.g. a resolved sphere/imported STL's tessellated
        # verts) collapses by item count like any other list -- also
        # regression coverage for the original bug this was written for:
        # numpy's own repr() can span multiple lines and would otherwise
        # break the one-line-per-node dump format.
        import numpy as np
        arr = np.zeros((50, 3))
        result = _summarize_param(arr)
        assert result == "<list of 50>"
        assert "\n" not in result

    def test_summarize_param_long_string_truncated(self):
        result = _summarize_param("x" * 100)
        assert len(result) <= 40

    def test_format_csg_tree_is_one_line_per_node(self):
        _, _, ev = run_tree("difference() { cube(10); sphere(3); }")
        dump = format_csg_tree(ev.csg_tree)
        lines = dump.split("\n")
        assert len(lines) == 3  # difference, cube, sphere
        for line in lines:
            assert "\n" not in line

    def test_format_csg_tree_indent_offset_for_console_fold_display(self):
        # Regression: the console displays multi-line output with a
        # "<arrow> " prefix (2 display columns) on the first line only,
        # and no prefix on the rest. Without a compensating +1 indent
        # unit for every non-root line, a depth-1 child's own "  " padding
        # would land in the exact same column the root's text starts at
        # (right after the arrow), making it look like a sibling of the
        # root rather than its child. Depth 0 gets no padding (it's the
        # header text handed to the console as-is); depth 1 must start
        # further right than where the root's own text starts.
        _, _, ev = run_tree("union() { cube(10); sphere(3); }")
        dump = format_csg_tree(ev.csg_tree)
        lines = dump.split("\n")
        assert lines[0].startswith("union(")            # depth 0: no padding
        assert lines[1].startswith("    cube(")          # depth 1: 4 spaces
        assert not lines[1].startswith("  cube(")

    def test_format_csg_tree_splices_user_module_call_not_wrapped(self):
        # A user-module call isn't itself geometry -- only the geometry
        # its body produces should appear in the tree, spliced in at the
        # call site, not wrapped in a node named after the module.
        _, _, ev = run_tree("module foo() { cube(1); } foo();")
        dump = format_csg_tree(ev.csg_tree)
        assert "foo" not in dump
        assert dump == "cube(size=[1.0, 1.0, 1.0], center=false)"

    def test_format_csg_tree_omits_body_count(self):
        # A cache-hit ancestor makes generate_tree() skip visiting its
        # descendants entirely (see ManifoldCache), leaving their .bodies
        # at the empty default -- not because they produced no geometry,
        # just because nothing ever populated them on that pass. A body
        # count would be actively misleading in that case, so the dump
        # never shows one at all (structure only, always reliable).
        _, _, ev = run_tree("cube(1);")
        dump = format_csg_tree(ev.csg_tree)
        assert "body" not in dump
        assert "->" not in dump

    def test_format_csg_tree_hides_redundant_bookkeeping_params(self):
        # op/name duplicate the node's own kind; group_sizes is
        # _generate_csg's private re-chunking bookkeeping; color is
        # already represented structurally by color()'s own wrapping
        # CSGNode (same as translate/rotate) when relevant, and just
        # noise (color=None) otherwise.
        _, _, ev = run_tree("difference() { cube(10); translate([1,0,0]) sphere(3); }")
        dump = format_csg_tree(ev.csg_tree)
        assert "op=" not in dump
        assert "name=" not in dump
        assert "group_sizes" not in dump
        assert "color" not in dump

    def test_format_csg_tree_shows_transform_args(self):
        # A transform's resolved "args" param is a _resolve_args()-shaped
        # dict ({0: [1.0, 2.0, 3.0]} for a single positional arg) --
        # rendered as OpenSCAD call syntax directly in the parens
        # (translate([1.0, 2.0, 3.0])), not as a Python dict literal
        # behind an "args=" key (translate(args={0: [1.0, 2.0, 3.0]})),
        # which hid the actual argument value behind a collapsed dict.
        _, _, ev = run_tree("translate([1,2,3]) cube(1);")
        dump = format_csg_tree(ev.csg_tree)
        assert "translate([1.0, 2.0, 3.0])" in dump
        assert "args=" not in dump

    def test_format_csg_tree_hides_sphere_tessellation(self):
        _, _, ev = run_tree("sphere(3);")
        dump = format_csg_tree(ev.csg_tree)
        assert "verts" not in dump
        assert "tris" not in dump

    def test_format_csg_tree_shows_sphere_radius(self):
        # _resolve_sphere computes its radius (normalizing r/d to r, same
        # convention as _resolve_cylinder's r1/r2) but originally never
        # stored it in params -- only the tessellated verts/tris, which
        # the dump hides. Without "r" in params, sphere(...) showed
        # completely empty in the dump.
        _, _, ev = run_tree("sphere(5);")
        assert "r=5.0" in format_csg_tree(ev.csg_tree)
        _, _, ev = run_tree("sphere(d=10);")
        assert "r=5.0" in format_csg_tree(ev.csg_tree)

    def test_format_csg_tree_shows_sphere_fn(self):
        # Same "segs" key/convention as cylinder/circle/offset/text --
        # renamed to "$fn" for display via _DUMP_KEY_RENAMES.
        _, _, ev = run_tree("sphere(r=5, $fn=24);")
        dump = format_csg_tree(ev.csg_tree)
        assert "$fn=24" in dump
        assert "segs" not in dump

    def test_format_csg_tree_renames_segs_to_fn(self):
        # "segs" is every _resolve_X's own internal variable name for the
        # circular-segment count resolved from $fn/$fa/$fs (via _fn()) --
        # shared by cylinder/circle/offset/text. "$fn=" in the dump reads
        # as what it actually represents, not the resolve step's private
        # variable name; the underlying params key stays "segs" (that's
        # what _generate_cylinder etc. actually read).
        _, _, ev = run_tree("cylinder(h=5, r=2);")
        dump = format_csg_tree(ev.csg_tree)
        assert "$fn=" in dump
        assert "segs" not in dump

    def test_summarize_param_bool_is_lowercase(self):
        # OpenSCAD source spells booleans lowercase (true/false); Python's
        # own repr() (True/False) would read as foreign syntax in a dump
        # meant to mirror what the user typed.
        assert _summarize_param(True) == "true"
        assert _summarize_param(False) == "false"

    def test_format_csg_tree_shows_lowercase_booleans(self):
        _, _, ev = run_tree("cube(10, center=true);")
        dump = format_csg_tree(ev.csg_tree)
        assert "center=true" in dump
        assert "True" not in dump

    def test_format_csg_tree_shows_polyhedron_verts(self):
        # Unlike sphere/cylinder's auto-generated tessellation, a
        # polyhedron's verts/faces are the user's own authored content
        # and are worth showing (still subject to the normal size-based
        # collapse for a large polyhedron).
        _, _, ev = run_tree(
            "polyhedron(points=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]]);"
        )
        dump = format_csg_tree(ev.csg_tree)
        assert "verts=" in dump
        assert "0.0, 0.0, 0.0" in dump

    def test_format_csg_tree_empty_tree(self):
        assert format_csg_tree([]) == ""

    def test_format_csg_tree_splices_children_call_not_wrapped(self):
        # children() is a call-site substitution, not geometry -- the tree
        # should show the substituted subtree directly under its caller,
        # not a "children()" node wrapping it.
        _, _, ev = run_tree(
            "module m() { translate([1,0,0]) children(); }\n"
            "m() sphere(1);"
        )
        dump = format_csg_tree(ev.csg_tree)
        assert "children" not in dump
        assert "r=1.0" in dump

    def test_format_csg_tree_groups_multiple_spliced_children_under_union(self):
        # A single children()/user-module call site producing more than one
        # sibling is implicitly one shape there -- group them under a
        # "union()" label in the dump instead of showing N independent-
        # looking flat siblings. (A single sibling still splices flat, with
        # no union wrapper -- see test_format_csg_tree_splices_*_not_wrapped.)
        _, _, ev = run_tree(
            "module m() { translate([1,0,0]) children(); }\n"
            "m() { cube(1); sphere(1); }"
        )
        dump = format_csg_tree(ev.csg_tree)
        assert "union()" in dump
        lines = dump.split("\n")
        union_line = next(l for l in lines if "union(" in l)
        cube_line = next(l for l in lines if "cube(" in l)
        assert len(cube_line) - len(cube_line.lstrip()) > len(union_line) - len(union_line.lstrip())

    def test_multiple_spliced_children_keep_separate_bodies_not_merged(self):
        # The union() grouping in the dump is display-only (is_builtin=False
        # so generate_tree takes the default-concatenation path, not the
        # real _generate_csg boolean merge) -- juxtaposed statements with no
        # explicit combinator must keep separate ColoredBody entries, same
        # as any module body or top-level script, not collapse into one
        # Manifold the way an *explicit* union() call does.
        bodies, _, ev = run_tree(
            "module donut2() { cube(1); sphere(1); }\n"
            "donut2();"
        )
        assert len(bodies) == 2


class TestManifoldCache:
    """Correctness safety net for the incremental-rebuild ManifoldCache:
    proves a cache hit produces output equivalent to (a) the original
    cache-populating run and (b) an entirely separate run with caching
    disabled — so caching is provably a pure optimization, never a
    behavior change. A silently-wrong cache would be worse than the
    "slow but correct" status quo it replaces."""

    @staticmethod
    def _run_with_cache(src: str, cache):
        nodes = getASTfromString(src, include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator(manifold_cache=cache)
        bodies, _ = ev.evaluate(nodes, root_scope)
        return bodies

    @staticmethod
    def _bodies_equivalent(a, b) -> bool:
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x.role != y.role or x.color != y.color:
                return False
            if (x.body is None) != (y.body is None):
                return False
            if x.body is not None and x.body.bounding_box() != y.body.bounding_box():
                return False
        return True

    # Covers: plain primitive, union/difference/intersection, hull,
    # minkowski, a for loop (transparent in the tree -- no wrapping node),
    # if/else, a user module called multiple times with different args
    # (same kind, different params -- must NOT collide in the cache), and
    # children() (whose actual tree-children come from the call site, not
    # the children() call's own AST position -- see CSGNode/_cache_key's
    # documented children() hazard).
    _SCRIPTS = [
        "cube(10);",
        "union() { cube(10); translate([5,0,0]) sphere(3); }",
        "difference() { cube(10); translate([2,2,-1]) cylinder(h=12, r=2); }",
        "intersection() { cube(10); sphere(8); }",
        "hull() { cube(2); translate([10,0,0]) sphere(3); }",
        "minkowski() { cube(2); sphere(1); }",
        "for (i=[0:2]) translate([i*3,0,0]) cube(1);",
        "x = 5; if (x > 3) { cube(x); } else { sphere(1); }",
        "module foo(n) { cube(n); } foo(4); foo(6);",
        "module bar() { children(); } bar() { sphere(2); }",
        "$fn=16; sphere(5);",
    ]

    @pytest.mark.parametrize("src", _SCRIPTS)
    def test_cold_warm_and_disabled_are_equivalent(self, src):
        cache = ManifoldCache()
        cold = self._run_with_cache(src, cache)
        warm = self._run_with_cache(src, cache)       # same cache -> should hit
        disabled = self._run_with_cache(src, None)     # caching off entirely
        assert self._bodies_equivalent(cold, warm)
        assert self._bodies_equivalent(cold, disabled)

    def test_unseeded_rands_never_served_from_cache(self):
        # The one case where "identical output" would be the WRONG
        # assertion -- an unseeded rands() must never be reused, so two
        # independent runs sharing a cache should (overwhelmingly likely)
        # still differ.
        src = "cube([rands(1,10,1)[0], 5, 5]);"
        cache = ManifoldCache()
        first = self._run_with_cache(src, cache)
        second = self._run_with_cache(src, cache)
        assert first[0].body.bounding_box() != second[0].body.bounding_box()

    def test_seeded_rands_is_still_correct_across_cache_states(self):
        # A seeded rands() call is deterministic in isolation, so cold/
        # warm/disabled should agree here even though the node is still
        # (conservatively) tainted uncacheable either way.
        src = "cube([rands(1,10,1,42)[0], 5, 5]);"
        cache = ManifoldCache()
        cold = self._run_with_cache(src, cache)
        warm = self._run_with_cache(src, cache)
        disabled = self._run_with_cache(src, None)
        assert self._bodies_equivalent(cold, warm)
        assert self._bodies_equivalent(cold, disabled)

    def test_repeated_generate_tree_is_a_full_cache_hit(self):
        # Models the debugger's repeated-pause pattern: the SAME Evaluator/
        # csg_tree, generate_tree() called again with the same cache.
        cache = ManifoldCache()
        nodes = getASTfromString("union() { cube(10); sphere(3); }", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator(manifold_cache=cache)
        first, _ = ev.evaluate(nodes, root_scope)
        second = ev.generate_tree(ev.csg_tree)
        assert self._bodies_equivalent(first, second)

    def test_flush_clears_cache(self):
        cache = ManifoldCache()
        self._run_with_cache("cube(10);", cache)
        assert cache._entries
        cache.clear()
        assert not cache._entries

    def test_rands_taint_survives_splicing_single_child(self):
        # Regression: children()/user-module calls splice their resolved
        # subtree directly into the tree instead of getting a wrapping
        # node of their own (_eval_statement). A rands() call in a
        # non-geometry statement (an assignment) during that call's own
        # resolve -- rather than inside the spliced child's own resolve --
        # used to have its taint silently dropped: only the spliced
        # children's *own* uncacheable flags were checked, never whether
        # rands() was called during the splicing call's own resolve.
        # Checking the flag directly (not just end-to-end body output) is
        # necessary here, since jitter also happens to feed into the
        # cube's size, which would already force a cache-key mismatch
        # regardless of whether the flag itself is correct.
        src = """
        module m() {
            jitter = rands(1, 10, 1)[0];
            translate([jitter, 0, 0]) cube(1);
        }
        m();
        """
        _, _, ev = run_tree(src)
        assert len(ev.csg_tree) == 1
        assert ev.csg_tree[0].uncacheable is True

    def test_rands_taint_survives_splicing_union_wrapped(self):
        # Same regression, but with >1 spliced sibling (the union-wrapper
        # branch) -- the wrapper and every spliced child must all end up
        # tainted.
        src = """
        module m() {
            jitter = rands(1, 10, 1)[0];
            translate([jitter, 0, 0]) cube(1);
            sphere(1);
        }
        m();
        """
        _, _, ev = run_tree(src)
        assert len(ev.csg_tree) == 1
        assert ev.csg_tree[0].kind == "union"
        assert ev.csg_tree[0].uncacheable is True
        assert all(c.uncacheable for c in ev.csg_tree[0].children)

    def test_no_rands_call_leaves_spliced_nodes_cacheable(self):
        # Sanity check the fix doesn't over-taint: no rands() anywhere ->
        # every spliced/wrapped node stays cacheable.
        src = """
        module m() {
            translate([1, 0, 0]) cube(1);
            sphere(1);
        }
        m();
        """
        _, _, ev = run_tree(src)
        assert ev.csg_tree[0].uncacheable is False
        assert all(not c.uncacheable for c in ev.csg_tree[0].children)


# ---------------------------------------------------------------------------
# Profiling (Evaluator(profile=True))
# ---------------------------------------------------------------------------

class TestProfiling:
    """Correctness safety net for the opt-in per-call-site profiler (see
    CallSiteProfile/ProfileResult and Evaluator._profile_enter/_profile_exit).
    Timing-based assertions (everywhere except test_self_times_plus_
    unattributed_equal_resolve_time, which is a pure arithmetic identity)
    use large workloads and loose, order-of-magnitude bounds rather than
    tight tolerances, since wall-clock measurements are inherently noisier
    than the rest of this suite -- the goal is to catch a broken self/
    cumulative-time *relationship*, not to pin down exact numbers."""

    @staticmethod
    def _profile(src: str):
        nodes = getASTfromString(src, include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator(profile=True)
        ev.evaluate(nodes, root_scope)
        return ev.profile_result

    @staticmethod
    def _site(result, name: str, call_count: int | None = None):
        """The one CallSiteProfile matching `name` (optionally also
        call_count, to disambiguate a function's outer vs. recursive
        call site when both share a name)."""
        matches = [s for s in result.call_sites if s.name == name
                   and (call_count is None or s.call_count == call_count)]
        assert len(matches) == 1, f"expected exactly one match for {name!r}/{call_count!r}, got {matches}"
        return matches[0]

    def test_profiling_off_by_default(self):
        # Plain Evaluator() (no profile=) must leave profile_result unset --
        # the single most important regression guard for "zero overhead
        # when off" (every profiling code path is behind `if self._profiling`).
        nodes = getASTfromString("cube(1);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev.evaluate(nodes, root_scope)
        assert ev.profile_result is None

    def test_single_call_site_self_time(self):
        src = """
        function busy(n) = len([for (i=[0:n]) i*i]);
        x = busy(20000);
        echo(x);
        """
        result = self._profile(src)
        site = self._site(result, "busy")
        assert site.call_count == 1
        assert site.self_time > 0
        # No nested user calls inside busy() -- cumulative ~= self.
        assert site.cumulative_time == pytest.approx(site.self_time, rel=0.2)

    def test_same_site_looped_aggregates(self):
        src = """
        function busy(n) = len([for (i=[0:n]) i*i]);
        for (i=[0:9]) echo(busy(2000));
        """
        result = self._profile(src)
        site = self._site(result, "busy")
        assert site.call_count == 10
        assert site.self_time > 0

    def test_nested_distinct_call_sites_self_vs_cumulative(self):
        src = """
        function inner(n) = len([for (i=[0:n]) i*i]);
        function outer() = inner(20000);
        echo(outer());
        """
        result = self._profile(src)
        outer = self._site(result, "outer")
        inner = self._site(result, "inner")
        assert outer.self_time < outer.cumulative_time
        assert outer.cumulative_time >= inner.cumulative_time

    def test_recursive_function_not_double_counted(self):
        src = """
        function rec(n) = n <= 0 ? len([for (i=[0:300]) i*i]) : len([for (i=[0:300]) i*i]) + rec(n-1);
        x = rec(30);
        echo(x);
        """
        result = self._profile(src)
        # The recursive call site (rec(n-1), inside rec's own body) --
        # disambiguate from the one-off top-level `x = rec(30)` call site,
        # which shares the name "rec" but has call_count == 1.
        recursive_site = self._site(result, "rec", call_count=30)
        # Without the recursion guard, summing every nested invocation's
        # elapsed time would make cumulative_time balloon to roughly a
        # triangular-number multiple of the real wall time -- this bounds
        # it to (approximately) the real resolve time instead.
        assert recursive_site.cumulative_time <= result.resolve_time * 1.05

    def test_dominant_children_low_self_time(self):
        src = """
        function inner(n) = len([for (i=[0:n]) i*i]);
        function outer() = inner(150000);
        echo(outer());
        """
        result = self._profile(src)
        outer = self._site(result, "outer")
        inner = self._site(result, "inner")
        # outer's own work is a single call dispatch; inner's is a large
        # loop -- a big size gap gives headroom against transient
        # scheduling noise on a fast machine (a genuine self/cumulative
        # accounting bug would show outer near 50%+, not a small fraction).
        assert outer.self_time < 0.3 * inner.self_time
        assert outer.cumulative_time > 0.8 * result.resolve_time

    def test_self_times_plus_unattributed_equal_resolve_time(self):
        src = """
        function inner(n) = len([for (i=[0:n]) i*i]);
        function outer() = inner(5000);
        module box(s) { cube(s); }
        box(3);
        echo(outer());
        """
        result = self._profile(src)
        self_sum = sum(s.self_time for s in result.call_sites)
        assert result.resolve_time == pytest.approx(self_sum + result.unattributed_time, abs=1e-9)


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------

class TestBuiltinFunctions:
    def test_abs(self):
        _, lines = run("echo(abs(-5));")
        assert lines == ["ECHO: 5"]

    def test_sqrt(self):
        _, lines = run("echo(sqrt(4));")
        assert lines == ["ECHO: 2"]

    def test_floor(self):
        _, lines = run("echo(floor(3.9));")
        assert lines == ["ECHO: 3"]

    def test_ceil(self):
        _, lines = run("echo(ceil(3.1));")
        assert lines == ["ECHO: 4"]

    def test_round(self):
        _, lines = run("echo(round(3.5));")
        assert lines == ["ECHO: 4"]

    def test_round_half_away_from_zero(self):
        # OpenSCAD rounds .5 away from zero, unlike Python's round-half-to-even
        # (Python's round(2.5) == 2 and round(-0.5) == 0).
        _, lines = run("echo(round(2.5), round(-2.5), round(0.5), round(-0.5));")
        assert lines == ["ECHO: 3, -3, 1, -1"]

    def test_min(self):
        _, lines = run("echo(min(5, 3, 8));")
        assert lines == ["ECHO: 3"]

    def test_max(self):
        _, lines = run("echo(max(5, 3, 8));")
        assert lines == ["ECHO: 8"]

    def test_min_single_scalar(self):
        # A single non-list argument is returned as-is.
        _, lines = run("echo(min(5));")
        assert lines == ["ECHO: 5"]

    def test_min_max_multiple_vector_args_is_undef(self):
        # Real OpenSCAD only supports a single vector argument (returns
        # min/max of its elements) or multiple scalar arguments; mixing in
        # more than one vector is undef.
        _, lines = run("echo(min([1,5],[3,2]), max([1,5],[3,2]));")
        assert _echoes(lines) == ["ECHO: undef, undef"]

    def test_sin(self):
        _, lines = run("echo(sin(90));")
        assert lines == ["ECHO: 1"]

    def test_cos(self):
        _, lines = run("echo(cos(0));")
        assert lines == ["ECHO: 1"]

    def test_sin_cos_tan_exact_at_90_degree_multiples(self):
        # Real OpenSCAD special-cases exact multiples of 90 degrees to avoid
        # floating-point noise (e.g. cos(90) -> 6.12e-17, tan(90) -> 1.63e+16).
        _, lines = run(
            "echo(sin(180), cos(90), cos(180), tan(90), tan(270), "
            "sin(360), sin(-90), tan(180), sin(450), cos(-270));"
        )
        assert lines == ["ECHO: 0, 0, -1, inf, -inf, 0, -1, 0, 1, 0"]

    def test_sin_cos_tan_near_90_degree_multiple_is_not_special_cased(self):
        _, lines = run("echo(cos(90.0000001));")
        assert float(lines[0].split(": ")[1]) == approx(-1.74533e-9)

    def test_cos_of_infinity_is_nan(self):
        _, lines = run("echo(cos(1/0));")
        assert lines == ["ECHO: nan"]

    def test_len(self):
        _, lines = run("echo(len([1,2,3]));")
        assert lines == ["ECHO: 3"]

    def test_concat(self):
        _, lines = run("echo(concat([1,2],[3,4]));")
        assert lines == ["ECHO: [1, 2, 3, 4]"]

    def test_str_numbers(self):
        _, lines = run('echo(str(1, 2, 3));')
        assert lines == ["ECHO: \"123\""]

    def test_str_string_no_quotes(self):
        _, lines = run('echo(str("hello", 42));')
        assert lines == ['ECHO: "hello42"']

    def test_is_num(self):
        _, lines = run("echo(is_num(3));")
        assert lines == ["ECHO: true"]

    def test_is_list(self):
        _, lines = run("echo(is_list([1,2]));")
        assert lines == ["ECHO: true"]

    def test_is_undef(self):
        _, lines = run("echo(is_undef(undef));")
        assert lines == ["ECHO: true"]

    def test_is_bool(self):
        _, lines = run("echo(is_bool(true));")
        assert lines == ["ECHO: true"]

    def test_is_string(self):
        _, lines = run('echo(is_string("hi"));')
        assert lines == ["ECHO: true"]

    def test_tan(self):
        _, lines = run("echo(tan(45));")
        assert float(lines[0].split(": ")[1]) == approx(1.0)

    def test_asin(self):
        _, lines = run("echo(asin(1));")
        assert float(lines[0].split(": ")[1]) == approx(90.0)

    def test_acos(self):
        _, lines = run("echo(acos(1));")
        assert float(lines[0].split(": ")[1]) == approx(0.0)

    def test_atan(self):
        _, lines = run("echo(atan(1));")
        assert float(lines[0].split(": ")[1]) == approx(45.0)

    def test_atan2(self):
        _, lines = run("echo(atan2(1, 1));")
        assert float(lines[0].split(": ")[1]) == approx(45.0)

    def test_ln(self):
        _, lines = run("echo(ln(1));")
        assert lines == ["ECHO: 0"]

    def test_log(self):
        _, lines = run("echo(log(100));")
        assert lines == ["ECHO: 2"]

    def test_exp(self):
        _, lines = run("echo(exp(0));")
        assert lines == ["ECHO: 1"]

    def test_pow(self):
        _, lines = run("echo(pow(3, 3));")
        assert lines == ["ECHO: 27"]

    def test_pow_zero_negative_exponent(self):
        # 0 ** negative is +inf in OpenSCAD; Python's pow()/math.pow() raise.
        _, lines = run("echo(pow(0, -1));")
        assert lines == ["ECHO: inf"]

    def test_norm(self):
        _, lines = run("echo(norm([3, 4]));")
        assert float(lines[0].split(": ")[1]) == approx(5.0)

    def test_cross(self):
        _, lines = run("echo(cross([1,0,0],[0,1,0]));")
        assert lines == ["ECHO: [0, 0, 1]"]

    def test_cross_2d(self):
        # 2D cross product returns a scalar: a[0]*b[1] - a[1]*b[0]
        _, lines = run("echo(cross([1,2],[3,4]));")
        assert lines == ["ECHO: -2"]

    def test_chr(self):
        _, lines = run("echo(chr(65));")
        assert lines == ['ECHO: "A"']

    def test_chr_vector(self):
        # chr() also accepts a vector of code points, converting and
        # concatenating each one.
        _, lines = run("echo(chr([65,66,67]));")
        assert lines == ['ECHO: "ABC"']

    def test_chr_vector_truncates_floats(self):
        _, lines = run("echo(chr([65.7,66.2]));")
        assert lines == ['ECHO: "AB"']

    def test_chr_empty_vector(self):
        _, lines = run("echo(chr([]));")
        assert lines == ['ECHO: ""']

    def test_ord(self):
        _, lines = run('echo(ord("A"));')
        assert lines == ["ECHO: 65"]

    def test_ord_multichar_uses_first_char(self):
        _, lines = run('echo(ord("ab"));')
        assert lines == ["ECHO: 97"]


# ---------------------------------------------------------------------------
# Math builtins: nan / inf / -inf / undef inputs
#
# _eval_function_call wraps every _math_fns call in a bare
# try/except Exception: return None (evaluator.py, _eval_function_call) --
# so passing undef (Python None) into a lambda that doesn't itself guard
# against it (e.g. abs(None), math.isnan(None)) raises TypeError internally
# and safely degrades to undef output, rather than crashing evaluation.
# Several functions (ceil/floor/round/sqrt/ln/log/sin/cos/tan/asin/acos)
# additionally have explicit nan/inf guards of their own, to avoid Python's
# math.ceil/floor raising OverflowError/ValueError on inf/nan.
# ---------------------------------------------------------------------------

class TestMathBuiltinsNanInfUndef:
    def test_abs(self):
        _, lines = run("echo(abs(1/0), abs(-1/0), abs(0/0), abs(undef));")
        assert _echoes(lines) == ["ECHO: inf, inf, nan, undef"]

    def test_sign(self):
        _, lines = run("echo(sign(1/0), sign(-1/0), sign(0/0), sign(undef));")
        assert _echoes(lines) == ["ECHO: 1, -1, 0, undef"]

    def test_ceil(self):
        _, lines = run("echo(ceil(1/0), ceil(-1/0), ceil(0/0), ceil(undef));")
        assert _echoes(lines) == ["ECHO: inf, -inf, nan, undef"]

    def test_floor(self):
        _, lines = run("echo(floor(1/0), floor(-1/0), floor(0/0), floor(undef));")
        assert _echoes(lines) == ["ECHO: inf, -inf, nan, undef"]

    def test_round(self):
        _, lines = run("echo(round(1/0), round(-1/0), round(0/0), round(undef));")
        assert _echoes(lines) == ["ECHO: inf, -inf, nan, undef"]

    def test_sqrt(self):
        # sqrt of a negative (including -inf) is nan, not a crash.
        _, lines = run("echo(sqrt(1/0), sqrt(-1/0), sqrt(0/0), sqrt(undef));")
        assert _echoes(lines) == ["ECHO: inf, nan, nan, undef"]

    def test_ln(self):
        _, lines = run("echo(ln(1/0), ln(-1/0), ln(0/0), ln(undef));")
        assert _echoes(lines) == ["ECHO: inf, nan, nan, undef"]

    def test_log(self):
        _, lines = run("echo(log(1/0), log(-1/0), log(0/0), log(undef));")
        assert _echoes(lines) == ["ECHO: inf, nan, nan, undef"]

    def test_exp(self):
        # exp(-inf) underflows to 0, not nan.
        _, lines = run("echo(exp(1/0), exp(-1/0), exp(0/0), exp(undef));")
        assert _echoes(lines) == ["ECHO: inf, 0, nan, undef"]

    def test_sin_cos_tan(self):
        # _deg_trig explicitly returns nan for any nan/inf input (its
        # multiple-of-90-degrees table lookup has no meaningful entry
        # for an unbounded angle) rather than falling through to
        # math.sin/cos/tan(radians(inf)), which would raise.
        _, lines = run(
            "echo(sin(1/0), sin(-1/0), sin(0/0), sin(undef));"
            "echo(cos(1/0), cos(-1/0), cos(0/0), cos(undef));"
            "echo(tan(1/0), tan(-1/0), tan(0/0), tan(undef));"
        )
        assert _echoes(lines) == [
            "ECHO: nan, nan, nan, undef",
            "ECHO: nan, nan, nan, undef",
            "ECHO: nan, nan, nan, undef",
        ]

    def test_asin_acos(self):
        # |x| > 1 (true for +-inf) is nan, matching real OpenSCAD's
        # domain check rather than raising ValueError like math.asin/acos.
        _, lines = run(
            "echo(asin(1/0), asin(-1/0), asin(0/0), asin(undef));"
            "echo(acos(1/0), acos(-1/0), acos(0/0), acos(undef));"
        )
        assert _echoes(lines) == [
            "ECHO: nan, nan, nan, undef",
            "ECHO: nan, nan, nan, undef",
        ]

    def test_atan(self):
        # atan has no domain restriction -- +-inf -> +-90 degrees exactly,
        # only nan input produces nan output.
        _, lines = run("echo(atan(1/0), atan(-1/0), atan(0/0), atan(undef));")
        assert _echoes(lines) == ["ECHO: 90, -90, nan, undef"]

    def test_atan2(self):
        _, lines = run(
            "echo(atan2(1/0, 1), atan2(1, 1/0), atan2(0/0, 1), atan2(undef, 1));"
        )
        assert _echoes(lines) == ["ECHO: 90, 0, nan, undef"]

    def test_pow(self):
        _, lines = run(
            "echo(pow(1/0, 2), pow(2, 1/0), pow(0/0, 2), pow(0, -1/0), "
            "pow(-2, 0.5), pow(undef, 2));"
        )
        assert _echoes(lines) == ["ECHO: inf, inf, nan, inf, nan, undef"]

    def test_max_min_nan_position_dependent(self):
        # Regression/documentation: max/min dispatch to Python's own
        # max()/min() (_builtin_minmax), whose nan handling is
        # position-dependent, not propagating-or-ignoring consistently --
        # nan as the *first* candidate is never displaced (every later
        # `x > nan`/`x < nan` comparison is False), but nan appearing
        # *later* is itself skipped over the same way. A change to
        # _builtin_minmax (e.g. switching to numpy) could easily flip
        # this silently, so it's pinned here rather than just assumed.
        _, lines = run(
            "echo(max(0/0, 1), max(1, 0/0));"
            "echo(min(0/0, 1), min(1, 0/0));"
            "echo(max([0/0, 1, 3]), max([1, 3, 0/0]));"
        )
        assert _echoes(lines) == [
            "ECHO: nan, 1",
            "ECHO: nan, 1",
            "ECHO: nan, 3",
        ]

    def test_max_min_inf(self):
        _, lines = run(
            "echo(max(1, 1/0), min(1, -1/0));"
            "echo(max([1, 1/0, 3]));"
        )
        assert _echoes(lines) == ["ECHO: inf, -inf", "ECHO: inf"]

    def test_max_min_undef_in_multi_arg_form_is_undef(self):
        # Mixing undef into the multi-scalar-argument form of max/min
        # isn't a list, so it doesn't hit the "mixing in a vector is
        # undef" guard -- it instead lands in Python's own max()/min(),
        # which raises TypeError comparing None to a number, caught by
        # the outer try/except and surfaced as undef.
        _, lines = run("echo(max(1, undef), min(1, undef));")
        assert _echoes(lines) == ["ECHO: undef, undef"]

    def test_norm(self):
        _, lines = run(
            "echo(norm([1/0, 0]), norm([0/0, 0]), norm([undef, 0]));"
        )
        assert _echoes(lines) == ["ECHO: inf, nan, undef"]

    def test_cross_rejects_non_finite_components(self):
        # Confirmed against real OpenSCAD 2022.08.22: cross() validates
        # every component up front and returns undef (with a WARNING
        # naming the offending value) rather than computing through --
        # inf*0 is nan, not 0, so a naive computation would otherwise
        # produce a mixed finite/nan/inf result instead of a clean undef.
        _, lines = run(
            "echo(cross([1/0,0,0],[0,1,0]));"
            "echo(cross([0/0,0,0],[0,1,0]));"
            "echo(cross([1,0,0],[0,1,0]));"
        )
        assert _echoes(lines) == ["ECHO: undef", "ECHO: undef", "ECHO: [0, 0, 1]"]

    def test_is_num_excludes_nan_but_not_inf(self):
        # is_num(nan) is explicitly false (evaluator.py's is_num lambda
        # has its own `not math.isnan(x)` check) -- but +-inf still counts
        # as "a number" by this predicate, only nan is excluded.
        _, lines = run(
            "echo(is_num(1/0), is_num(-1/0), is_num(0/0), is_num(undef));"
        )
        assert _echoes(lines) == ["ECHO: true, true, false, false"]

    def test_str_formatting(self):
        _, lines = run("echo(str(1/0), str(-1/0), str(0/0), str(undef));")
        assert _echoes(lines) == ['ECHO: "inf", "-inf", "nan", "undef"']

    def test_chr_non_finite_or_undef_returns_empty_string(self):
        # Confirmed against real OpenSCAD 2022.08.22: chr() returns ""
        # (not undef) for a non-finite/undef scalar argument, and silently
        # skips non-finite/undef elements within a list argument rather
        # than failing the whole call.
        _, lines = run(
            "echo(chr(1/0), chr(0/0), chr(undef));"
            "echo(chr([65, 1/0, 66]));"
        )
        assert _echoes(lines) == ['ECHO: "", "", ""', 'ECHO: "AB"']

    def test_ord_undef_is_undef(self):
        # ord() indexes into a string, which raises on None -- caught by
        # the outer try/except, same safety net as every other math
        # builtin here (unlike chr(), ord()'s real-OpenSCAD behavior for
        # undef input is undef, not "").
        _, lines = run("echo(ord(undef));")
        assert _echoes(lines) == ["ECHO: undef"]


# ---------------------------------------------------------------------------
# Math builtins given non-numeric arguments (list / bool / object() / string)
#
# list/object()/string all raise inside the underlying lambda (e.g.
# abs([1,2]), abs(object(a=1)), abs("hi")) and are caught by
# _eval_function_call's generic try/except -> undef, same safety net
# covered by TestMathBuiltinsNanInfUndef.
#
# bool is the interesting case: Python's bool is a subclass of int, so
# every one of these functions would otherwise silently treat true/false
# as 1/0 (abs(true) -> 1, max(true, 1) -> true, norm([true, 0]) -> 1)
# instead of raising -- confirmed against real OpenSCAD 2022.08.22 that
# every one of these must reject a bool argument as a type error (undef),
# same as list/object()/string. _NUMERIC_ONLY_MATH_FNS (evaluator.py)
# explicitly checks for and rejects a bool positional argument (including
# inside a list argument, for max/min/norm/cross) before ever calling the
# underlying function.
# ---------------------------------------------------------------------------

class TestMathBuiltinsNonNumericArgs:
    def test_unary_functions_reject_list_object_string(self):
        fns = ["abs", "sign", "ceil", "floor", "round", "sqrt", "ln", "log", "exp",
               "sin", "cos", "tan", "asin", "acos", "atan"]
        for fn in fns:
            for arg in ("[1,2,3]", 'object(a=1)', '"hi"'):
                _, lines = run(f"echo({fn}({arg}));")
                assert _echoes(lines) == ["ECHO: undef"], f"{fn}({arg})"

    def test_unary_functions_reject_bool(self):
        # Confirmed against real OpenSCAD: abs(true), sign(true),
        # ceil(true), sqrt(true), sin(true), etc. are all undef there,
        # not the Python-bool-is-int-coerced numeric result.
        fns = ["abs", "sign", "ceil", "floor", "round", "sqrt", "ln", "log", "exp",
               "sin", "cos", "tan", "asin", "acos", "atan"]
        for fn in fns:
            _, lines = run(f"echo({fn}(true), {fn}(false));")
            assert _echoes(lines) == ["ECHO: undef, undef"], fn

    def test_atan2_rejects_bool(self):
        _, lines = run("echo(atan2(true, false), atan2(1, true), atan2(true, 1));")
        assert _echoes(lines) == ["ECHO: undef, undef, undef"]

    def test_pow_rejects_bool_in_either_position(self):
        _, lines = run("echo(pow(true, 2), pow(2, true), pow(false, 2));")
        assert _echoes(lines) == ["ECHO: undef, undef, undef"]

    def test_max_min_reject_bool_multi_arg_and_list_forms(self):
        _, lines = run(
            "echo(max(true, 1), max(1, true), min(true, 1));"
            "echo(max([true, 1, 2]));"
        )
        assert _echoes(lines) == ["ECHO: undef, undef, undef", "ECHO: undef"]

    def test_norm_cross_reject_bool_vector_component(self):
        _, lines = run(
            "echo(norm([true, 0]));"
            "echo(cross([true,0,0],[0,1,0]));"
        )
        assert _echoes(lines) == ["ECHO: undef", "ECHO: undef"]

    def test_numeric_only_fns_still_work_normally(self):
        # Sanity check the guard doesn't over-reject legitimate numeric
        # (int/float) arguments.
        _, lines = run(
            "echo(abs(5), sqrt(4), pow(2,3), atan2(1,1));"
            "echo(max(1,2), min(1,2), norm([3,4]), cross([1,0,0],[0,1,0]));"
        )
        assert _echoes(lines) == ["ECHO: 5, 2, 8, 45", "ECHO: 2, 1, 5, [0, 0, 1]"]


# ---------------------------------------------------------------------------
# User-defined functions
# ---------------------------------------------------------------------------

class TestUserFunctions:
    def test_simple_function(self):
        _, lines = run("function double(x) = x * 2; echo(double(5));")
        assert lines == ["ECHO: 10"]

    def test_recursive_function(self):
        src = """
        function fact(n) = n <= 1 ? 1 : n * fact(n - 1);
        echo(fact(5));
        """
        _, lines = run(src)
        assert lines == ["ECHO: 120"]

    def test_function_default_args(self):
        src = "function add(a, b=10) = a + b; echo(add(5));"
        _, lines = run(src)
        assert lines == ["ECHO: 15"]

    def test_undefined_function_warns_and_returns_undef(self):
        # Real OpenSCAD treats a call to an unknown function as a WARNING
        # ("Ignoring unknown function 'X'") and evaluates it to undef,
        # rather than aborting the whole render.
        _, lines = run("echo(nope(1));")
        assert lines[0] == "WARNING: Ignoring unknown function 'nope' in file <string>, line 1"
        assert lines[1] == "ECHO: undef"

    def test_undefined_function_in_nested_call_no_traceback(self):
        src = """
        function outer() = inner();
        echo(outer());
        """
        _, lines = run(src)
        assert lines[0] == ("WARNING: Ignoring unknown function 'inner' in file <string>, line 2, "
                            "from <string>, line 3\nTRACE: called by 'outer' in file <string>, line 3")
        assert lines[1] == "ECHO: undef"


# ---------------------------------------------------------------------------
# Default-parameter scoping: default expressions are evaluated lexically
# against the function/module's own declaration scope, never the caller's,
# and can't see the call's own sibling parameters either -- both verified
# directly against real OpenSCAD (Applications/OpenSCAD.app). Regression
# coverage for a bug where _apply_defaults evaluated defaults against the
# caller's ctx, so a default reading a name the caller shadowed via let()
# picked up the caller's value instead of the function's own lexical one.
# ---------------------------------------------------------------------------

class TestDefaultParamScoping:
    def test_default_ignores_caller_let_shadow(self):
        src = """
        function f(x, y=k) = x + y;
        k = 100;
        function g() = let(k=1) f(1);
        function h() = let(k=2) f(1);
        echo(g());
        echo(h());
        """
        _, lines = run(src)
        assert lines == ["ECHO: 101", "ECHO: 101"]

    def test_default_cannot_see_sibling_param(self):
        _, lines = run("function f(a, b=a*2) = a + b; echo(f(3));")
        assert lines == [
            'WARNING: Ignoring unknown variable "a" in file <string>, line 1',
            "WARNING: undefined operation (undefined * number) in file <string>, line 1",
            "WARNING: undefined operation (number + undefined) in file <string>, line 1\n"
            "TRACE: called by 'f' in file <string>, line 1",
            "ECHO: undef",
        ]

    def test_function_body_already_ignores_caller_let(self):
        # Sanity check: the function BODY (not a default) already correctly
        # resolves free variables lexically -- only defaults had the bug.
        src = """
        function f(x) = x + k;
        k = 100;
        function g() = let(k=1) f(1);
        echo(g());
        """
        _, lines = run(src)
        assert lines == ["ECHO: 101"]

    def test_module_default_ignores_caller_let_shadow(self):
        src = """
        module m(x, y=k) { echo(x + y); }
        k = 100;
        module g() { let(k=1) m(1); }
        g();
        """
        _, lines = run(src)
        assert lines == ["ECHO: 101"]

    def test_explicit_arg_bypasses_default(self):
        _, lines = run("function f(x, y=k) = x + y; k=100; echo(f(1, 5));")
        assert lines == ["ECHO: 6"]


# ---------------------------------------------------------------------------
# $-prefixed function/module parameters: regression coverage for a bug where
# _apply_defaults checked ONLY child_ctx.let for "already bound", regardless
# of a param's $-prefix -- so a $-prefixed param bound into child_ctx.dyn (by
# the bound-argument loop just above _apply_defaults' own call) still looked
# "unbound" to it, and its own no-default branch clobbered the real value
# with a fresh None/undef written into child_ctx.let. Since _eval_identifier
# checks .let before .dyn, that stray None then shadowed the correct .dyn
# value for any bare `$fn`-style reference in the body -- even though
# geometry builtins reading ctx.dyn directly (sphere's own $fn lookup, the
# pre-existing test_module_with_dollar_arg above) never noticed, since they
# never go through _eval_identifier at all. The same bug affected a
# $-prefixed parameter's OWN declared default value too (never reached dyn
# either). Fixed by making _apply_defaults check `bound` (the caller's
# actual argument dict) instead of child_ctx.let/dyn, and route a
# $-prefixed name's result into child_ctx.dyn instead of .let.
# ---------------------------------------------------------------------------

class TestDollarPrefixedParameterBinding:
    def test_function_dollar_param_bound_by_caller_visible_as_identifier(self):
        _, lines = run("function f($fn) = $fn; echo(f($fn=16));")
        assert lines == ["ECHO: 16"]

    def test_function_dollar_param_default_visible_as_identifier(self):
        _, lines = run("function f($fn=16) = $fn; echo(f());")
        assert lines == ["ECHO: 16"]

    def test_function_literal_dollar_param_bound_by_caller_visible_as_identifier(self):
        _, lines = run("g = function($fn) $fn; echo(g($fn=16));")
        assert lines == ["ECHO: 16"]

    def test_module_dollar_param_bound_by_caller_visible_as_identifier(self):
        src = 'module m($fn) { echo($fn); } m($fn=16);'
        _, lines = run(src)
        assert lines == ["ECHO: 16"]

    def test_module_dollar_param_default_visible_as_identifier(self):
        src = 'module m($fn=16) { echo($fn); } m();'
        _, lines = run(src)
        assert lines == ["ECHO: 16"]

    def test_share_dyn_fast_path_does_not_leak_default_into_caller(self):
        # The share_dyn optimization (_eval_user_function/
        # _eval_function_literal) must be False whenever the declaration
        # has ANY $-prefixed parameter, even one the caller didn't supply
        # this call -- otherwise child_ctx.dyn is the SAME dict object as
        # the caller's own ctx.dyn, and _apply_defaults' new dyn-writing
        # branch would corrupt the caller's $fn instead of only this call's
        # private copy. Calling f() a second time with a caller-side $fn
        # already set confirms the caller's own $fn survives unpolluted.
        src = """
        function f($fn=99) = $fn;
        result = let($fn=5) f();
        echo(result);
        echo($fn);
        """
        _, lines = run(src)
        # result: f() gets no $fn argument, so its own default (99) applies
        # -- $fn=5 from the enclosing let() is irrelevant here since f()
        # declares its own $fn parameter, shadowing the dynamically-scoped
        # one entirely for its own body. The trailing echo($fn) confirms
        # the caller's own $fn (root-seeded default 0, untouched by the
        # let() -- which only scopes to evaluating f() itself) survived
        # unpolluted by f()'s private dyn write.
        assert lines == ["ECHO: 99", "ECHO: 0"]

    def test_share_dyn_fast_path_does_not_leak_undeclared_dollar_arg_into_caller(self):
        # _has_dollar_param alone is a purely static property of the
        # DECLARATION -- it misses a $-prefixed *named* call-site argument
        # that doesn't match any declared parameter at all. _bind_args
        # writes ANY named argument into `bound` unconditionally (that's
        # how an undeclared `$fn=64`-style dynamic-scope override works
        # for a call in general), so f() below -- which declares no
        # $-parameter -- still gets "$fn" written into child_ctx.dyn via
        # the bound-argument loop. Before _bound_has_dollar_key was added,
        # share_dyn was True here (no declared $-param), so child_ctx.dyn
        # was the SAME dict as the caller's ctx.dyn, and that write leaked
        # $fn=99 into the caller's own scope, visible to the trailing
        # echo($fn). Found while porting this optimization to the C++
        # port's own evaluator (see its Scoping.
        # DollarArgUndeclaredByCalleeFunctionDoesNotLeakToCallerDyn),
        # reproduced here, and fixed in both places the same way.
        src = """
        function f(x) = x;
        a = f(x=1, $fn=99);
        echo($fn);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 0"]


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_if_true(self):
        src = "if (true) { echo(1); }"
        _, lines = run(src)
        assert lines == ["ECHO: 1"]

    def test_if_false(self):
        src = "if (false) { echo(1); }"
        _, lines = run(src)
        assert lines == []

    def test_if_else(self):
        src = "if (false) { echo(1); } else { echo(2); }"
        _, lines = run(src)
        assert lines == ["ECHO: 2"]

    def test_for_loop(self):
        src = "for (i = [1:3]) { echo(i); }"
        _, lines = run(src)
        assert lines == ["ECHO: 1", "ECHO: 2", "ECHO: 3"]

    def test_for_step(self):
        src = "for (i = [0:2:4]) { echo(i); }"
        _, lines = run(src)
        assert lines == ["ECHO: 0", "ECHO: 2", "ECHO: 4"]

    def test_for_vector(self):
        src = "for (x = [10, 20, 30]) { echo(x); }"
        _, lines = run(src)
        assert lines == ["ECHO: 10", "ECHO: 20", "ECHO: 30"]

    def test_for_string_iterates_chars(self):
        _, lines = run('for (c = "abc") { echo(c); }')
        assert lines == ['ECHO: "a"', 'ECHO: "b"', 'ECHO: "c"']

    def test_for_string_variable_iterates_chars(self):
        _, lines = run('s = "hi"; for (c = s) { echo(c); }')
        assert lines == ['ECHO: "h"', 'ECHO: "i"']


# ---------------------------------------------------------------------------
# List comprehensions
# ---------------------------------------------------------------------------

class TestListComprehensions:
    def test_for_comp(self):
        _, lines = run("echo([for (i=[1:3]) i*2]);")
        assert lines == ["ECHO: [2, 4, 6]"]

    def test_if_comp(self):
        _, lines = run("echo([for (i=[1:5]) if (i % 2 == 1) i]);")
        assert lines == ["ECHO: [1, 3, 5]"]

    def test_each_flat(self):
        _, lines = run("a = [1,2,3]; echo([each a]);")
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_each_nested(self):
        _, lines = run("a = [[1,2,3],[4,5,6]]; b = [each a]; echo(b);")
        assert lines == ["ECHO: [[1, 2, 3], [4, 5, 6]]"]

    def test_listcompif_direct(self):
        # ListCompIf as a direct element (not nested in for)
        _, lines = run("x = [if (true) 1, if (false) 2, if (true) 3]; echo(x);")
        assert lines == ["ECHO: [1, 3]"]

    def test_listcompifelse_direct(self):
        _, lines = run("x = [if (true) 1 else 9, if (false) 2 else 8]; echo(x);")
        assert lines == ["ECHO: [1, 8]"]

    def test_for_each_flatten(self):
        src = """
        function flatten(list) = [for (x=list) each x];
        grid = [[1,2,3],[4,5,6]];
        echo(flatten(grid));
        """
        _, lines = run(src)
        assert lines == ["ECHO: [1, 2, 3, 4, 5, 6]"]

    def test_for_comp_string_iterates_chars(self):
        _, lines = run('echo([for (c = "xyz") c]);')
        assert lines == ['ECHO: ["x", "y", "z"]']

    def test_c_style_for(self):
        # `for (init...; cond; incr...)` — the C-style for in a list
        # comprehension (parsed as `ListCompCFor`), used e.g. by BOSL2's
        # `cumsum()`: [for (a=v[0], i=1; i<=len(v); a = i<len(v)?a+v[i]:a, i=i+1) a]
        src = """
        v = [0, 1, 2, 3];
        echo([for (a = v[0], i = 1; i <= len(v); a = i < len(v) ? a + v[i] : a, i = i + 1) a]);
        """
        _, lines = run(src)
        assert lines == ["ECHO: [0, 1, 3, 6]"]


# ---------------------------------------------------------------------------
# Primitives and geometry
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_cube_default(self):
        bodies, _ = run("cube(1);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(1)  # x size

    def test_cube_sized(self):
        bodies, _ = run("cube([2, 3, 4]);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)
        assert bb[4] - bb[1] == approx(3)
        assert bb[5] - bb[2] == approx(4)

    def test_cube_centered(self):
        bodies, _ = run("cube([4, 4, 4], center=true);")
        bb = bbox(bodies)
        assert bb[0] == approx(-2)
        assert bb[3] == approx(2)

    def test_sphere(self):
        bodies, _ = run("sphere(r=5, $fn=32);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(10, rel=0.02)

    def test_cylinder(self):
        bodies, _ = run("cylinder(h=10, r=3, $fn=32);")
        bb = bbox(bodies)
        assert bb[5] - bb[2] == approx(10, rel=0.01)
        assert bb[3] - bb[0] == approx(6, rel=0.02)

    def test_no_geometry_for_assignment(self):
        bodies, _ = run("x = 5;")
        assert bodies == []

    def test_sphere_diameter(self):
        bodies, _ = run("sphere(d=4, $fn=32);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(4, rel=0.02)

    def test_cylinder_diameter(self):
        bodies, _ = run("cylinder(h=5, d=6, $fn=32);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(6, rel=0.02)

    def test_cylinder_r1_r2(self):
        bodies, _ = run("cylinder(h=10, r1=3, r2=1, $fn=32);")
        bb = bbox(bodies)
        assert bb[5] - bb[2] == approx(10, rel=0.01)
        # base diameter = 6
        assert bb[3] - bb[0] == approx(6, rel=0.02)

    def test_cylinder_d1_d2(self):
        bodies, _ = run("cylinder(h=10, d1=6, d2=2, $fn=32);")
        bb = bbox(bodies)
        assert bb[5] - bb[2] == approx(10, rel=0.01)
        assert bb[3] - bb[0] == approx(6, rel=0.02)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_translate(self):
        bodies, _ = run("translate([10, 0, 0]) cube(1);")
        bb = bbox(bodies)
        assert bb[0] == approx(10)
        assert bb[3] == approx(11)

    def test_scale(self):
        bodies, _ = run("scale([2, 1, 1]) cube(1);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_scale_uniform(self):
        # scalar argument scales all three axes uniformly
        bodies, _ = run("scale(3) cube(1);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)
        assert bb[4] - bb[1] == approx(3)

    def test_rotate(self):
        bodies, _ = run("rotate([0, 0, 90]) translate([5, 0, 0]) cube(1);")
        bb = bbox(bodies)
        # after 90° z-rotation, x extent of translated cube maps to y axis
        assert abs(bb[1]) == approx(5, rel=0.01)

    def test_rotate_axis_angle(self):
        # rotate(90, v=[0,0,1]) is equivalent to rotate([0,0,90])
        bodies_euler, _ = run("rotate([0,0,90]) translate([5,0,0]) cube(1);")
        bodies_axis,  _ = run("rotate(90, v=[0,0,1]) translate([5,0,0]) cube(1);")
        bb_e = bodies_euler[0].body.bounding_box()
        bb_a = bodies_axis[0].body.bounding_box()
        assert bb_a[0] == approx(bb_e[0], rel=0.01)
        assert bb_a[3] == approx(bb_e[3], rel=0.01)


# ---------------------------------------------------------------------------
# color()
# ---------------------------------------------------------------------------

class TestColor:
    def _color(self, bodies):
        assert bodies
        return bodies[0].color

    def test_color_rgb_list(self):
        bodies, _ = run("color([1,0,0]) cube(1);")
        c = self._color(bodies)
        assert c[0] == approx(1.0)
        assert c[1] == approx(0.0)
        assert c[2] == approx(0.0)

    def test_color_rgba_list(self):
        bodies, _ = run("color([0,1,0,0.5]) cube(1);")
        c = self._color(bodies)
        assert c[1] == approx(1.0)
        assert c[3] == approx(0.5)

    def test_color_css_name(self):
        bodies, _ = run('color("red") cube(1);')
        c = self._color(bodies)
        assert c[0] == approx(1.0)
        assert c[1] == approx(0.0)

    def test_color_hex6(self):
        bodies, _ = run('color("#ff0000") cube(1);')
        c = self._color(bodies)
        assert c[0] == approx(1.0)
        assert c[1] == approx(0.0)

    def test_color_hex3(self):
        bodies, _ = run('color("#f00") cube(1);')
        c = self._color(bodies)
        assert c[0] == approx(1.0)
        assert c[1] == approx(0.0)

    def test_color_alpha_arg(self):
        bodies, _ = run('color("blue", alpha=0.25) cube(1);')
        c = self._color(bodies)
        assert c[2] == approx(1.0)
        assert c[3] == approx(0.25)

    def test_color_geometry_preserved(self):
        bodies, _ = run("color([0,0,1]) cube([2,3,4]);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)


# ---------------------------------------------------------------------------
# hull()
# ---------------------------------------------------------------------------

class TestHull:
    def test_hull_two_cubes(self):
        src = "hull() { cube(1); translate([5,0,0]) cube(1); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(6)

    def test_hull_contains_children(self):
        src = "hull() { sphere(r=1, $fn=16); translate([4,0,0]) sphere(r=1, $fn=16); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(6, rel=0.05)


# ---------------------------------------------------------------------------
# Modifiers (#, %, !, *)
# ---------------------------------------------------------------------------

class TestModifiers:
    def test_highlight_produces_geometry(self):
        # # (highlight) produces geometry with role="highlight"
        bodies, _ = run("#cube(2);")
        assert len(bodies) == 1
        assert bodies[0].role == "highlight"
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_showonly_produces_geometry(self):
        # ! (show-only) makes its subtree the whole model
        bodies, _ = run("!cube(3);")
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)

    def test_showonly_filters_others(self):
        # ! drops its siblings
        bodies, _ = run("cube(1); !cube(3);")
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(27)

    def test_showonly_inside_union_keeps_role(self):
        # Nested in a boolean, ! still isolates its subtree: the tree is
        # re-rooted there before generation, so the union never happens.
        bodies, _ = run("union() { !cube(5); translate([10,0,0]) sphere(3); }")
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(5)

    def test_showonly_inside_difference_keeps_role(self):
        bodies, _ = run(
            "difference() { cube(10); !translate([2,2,-1]) cylinder(h=12, r=2); }"
        )
        assert len(bodies) == 1
        bb = bbox(bodies)  # the cylinder alone, not cut from the cube
        assert (bb[2], bb[3], bb[5]) == approx((-1, 4, 11))

    def test_showonly_inside_intersection_keeps_role(self):
        bodies, _ = run("intersection() { cube(10); !sphere(8); }")
        assert len(bodies) == 1
        assert bbox(bodies)[0] == approx(-8, rel=1e-2)  # the sphere alone, not clipped

    def test_showonly_inside_hull_keeps_role(self):
        bodies, _ = run("hull() { !cube(5); translate([10,0,0]) sphere(3); }")
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(125)

    def test_showonly_inside_minkowski_keeps_role(self):
        bodies, _ = run("minkowski() { !cube(5); sphere(1); }")
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(125)

    def test_background_role(self):
        # % (background) produces a ghost body tagged role="background"
        bodies, _ = run("%cube(1);")
        assert len(bodies) == 1
        assert bodies[0].role == "background"

    def test_disable_suppressed(self):
        # * (disable) produces no geometry
        bodies, _ = run("*cube(1);")
        assert bodies == []

    def test_background_with_other_geometry(self):
        # % cube is kept (role="background") alongside normal geometry
        src = "cube(1); %cube([10,10,10]);"
        bodies, _ = run(src)
        assert len(bodies) == 2
        normal = [b for b in bodies if b.role == "normal"]
        bg = [b for b in bodies if b.role == "background"]
        assert len(normal) == 1
        assert len(bg) == 1
        bb = normal[0].body.bounding_box()
        assert bb[3] - bb[0] == approx(1)


# ---------------------------------------------------------------------------
# CSG operations
# ---------------------------------------------------------------------------

class TestCSG:
    def test_union(self):
        src = "union() { cube([2,1,1]); translate([1,0,0]) cube([2,1,1]); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)

    def test_difference(self):
        src = "difference() { cube([4,4,4]); cube([2,2,2]); }"
        bodies, _ = run(src)
        assert bodies  # geometry produced; exact shape is hollow

    def test_intersection(self):
        src = "intersection() { cube([3,3,3]); translate([1,1,1]) cube([3,3,3]); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)


# ---------------------------------------------------------------------------
# More transforms
# ---------------------------------------------------------------------------

class TestMoreTransforms:
    def test_mirror_x(self):
        bodies, _ = run("mirror([1,0,0]) translate([3,0,0]) cube(1);")
        bb = bbox(bodies)
        # cube was at x=[3,4]; after mirroring on YZ plane it lands at x=[-4,-3]
        assert bb[3] <= 0.01

    def test_resize(self):
        bodies, _ = run("resize([6,6,6]) cube(2);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(6, rel=0.01)

    def test_multmatrix_identity(self):
        # identity matrix should leave the cube unchanged
        src = """
        multmatrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            cube(2);
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_multmatrix_translate(self):
        # translation via multmatrix
        src = """
        multmatrix([[1,0,0,5],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            cube(1);
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[0] == approx(5)


# ---------------------------------------------------------------------------
# for loop producing geometry
# ---------------------------------------------------------------------------

class TestForGeometry:
    def test_for_produces_multiple_bodies(self):
        src = "for (i=[0:2:4]) { translate([i,0,0]) cube(1); }"
        bodies, _ = run(src)
        # three cubes (i=0,2,4) produced as separate bodies
        assert len(bodies) == 3
        xmax = max(b.body.bounding_box()[3] for b in bodies)
        assert xmax == approx(5)

    def test_for_vector_geometry(self):
        src = "for (x=[0,10]) { translate([x,0,0]) cube(1); }"
        bodies, _ = run(src)
        assert bodies


# ---------------------------------------------------------------------------
# let blocks
# ---------------------------------------------------------------------------

class TestLetBlocks:
    def test_let_expression(self):
        _, lines = run("echo(let(x=5, y=3) x + y);")
        assert lines == ["ECHO: 8"]

    def test_let_scoping(self):
        _, lines = run("x = 1; echo(let(x=99) x);")
        assert lines == ["ECHO: 99"]

    def test_let_block_geometry(self):
        src = "let(s=3) { cube(s); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)

    def test_let_block_shadowing(self):
        src = "s = 1; let(s=5) { cube(s); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(5)

    def test_let_sequential_binding_reference(self):
        # Later bindings in the same let() can reference earlier ones.
        _, lines = run("echo(let(a=1, b=a+1) b);")
        assert lines == ["ECHO: 2"]


class TestLetDollarVarScoping:
    """$-prefixed let() bindings are special variables -- real OpenSCAD
    scopes them dynamically, so they must stay visible to any function or
    module called from inside the let, not just the let's own body.
    Verified directly against real OpenSCAD (Applications/OpenSCAD.app).
    Regression coverage for a bug where let() always wrote bindings into
    the lexical .let scope, even for $-prefixed names, so a let($fn=...)
    was invisible to anything it called."""

    def test_expr_let_propagates_into_called_function(self):
        src = """
        function f() = $fn;
        $fn = 10;
        function g() = let($fn=99) f();
        echo(g());
        """
        _, lines = run(src)
        assert lines == ["ECHO: 99"]

    def test_statement_let_block_propagates_into_module_and_function(self):
        src = """
        module m() { echo($fn); }
        function f() = $fn;
        $fn = 10;
        let ($fn = 99) {
            m();
            echo(f());
        }
        """
        _, lines = run(src)
        assert lines == ["ECHO: 99", "ECHO: 99"]

    def test_let_dollar_var_does_not_leak_out(self):
        src = """
        function f() = $fn;
        $fn = 10;
        v1 = let($fn=55) f();
        v2 = f();
        echo(v1, v2);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 55, 10"]


# ---------------------------------------------------------------------------
# User-defined modules
# ---------------------------------------------------------------------------

class TestUserModules:
    def test_simple_module(self):
        src = "module box(s) { cube(s); } box(3);"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)

    def test_module_named_args(self):
        src = "module box(w, h) { cube([w, h, 1]); } box(h=3, w=5);"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(5)
        assert bb[4] - bb[1] == approx(3)

    def test_if_else_geometry_true_branch(self):
        src = "if (true) { cube(2); } else { cube(5); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_module_with_children(self):
        src = """
        module twice() { children(); translate([5,0,0]) children(); }
        twice() cube(1);
        """
        bodies, _ = run(src)
        assert bodies  # produces geometry

    def test_children_indexed(self):
        src = """
        module first_only() { children(0); }
        first_only() { cube(2); cube(5); }
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_module_default_param(self):
        src = "module box(s=2) { cube(s); } box();"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)


# ---------------------------------------------------------------------------
# Echo
# ---------------------------------------------------------------------------

class TestEcho:
    def test_echo_named_arg(self):
        _, lines = run("echo(x=42);")
        assert lines == ["ECHO: x = 42"]

    def test_echo_multiple(self):
        _, lines = run("echo(1, 2, 3);")
        assert lines == ["ECHO: 1, 2, 3"]

    def test_echo_in_module(self):
        src = "module m() { echo(99); } m();"
        _, lines = run(src)
        assert lines == ["ECHO: 99"]


# ---------------------------------------------------------------------------
# assert() statement
# ---------------------------------------------------------------------------

class TestAssert:
    def test_assert_statement(self):
        # ModularAssert as a statement produces no geometry and no error
        bodies, _ = run("assert(true); cube(1);")
        assert len(bodies) == 1

    def test_assert_modular_call(self):
        # assert(true) with a child module propagates the child's geometry
        bodies, _ = run("assert(true) cube(1);")
        assert len(bodies) == 1
        assert abs(bodies[0].body.volume() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Unknown / echo-as-module / misc module dispatch
# ---------------------------------------------------------------------------

class TestModuleDispatch:
    def test_echo_as_modular_call_with_children(self):
        # echo() with children runs echo and still draws them, as OpenSCAD does
        bodies, lines = run('echo("hi") cube(1);')
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(1)
        assert "hi" in lines[0]

    def test_unknown_module_skipped(self):
        # Unrecognised module name produces no geometry, no error
        bodies, _ = run("unknownmod() cube(1);")
        assert bodies == []

    def test_module_with_dollar_arg(self):
        # $fn passed as named arg to a user module goes into dyn
        src = "module m($fn=8) { sphere(r=1); } m($fn=16);"
        bodies, _ = run(src)
        assert bodies


# ---------------------------------------------------------------------------
# Primitive edge cases
# ---------------------------------------------------------------------------

class TestPrimitiveEdgeCases:
    def test_sphere_no_args(self):
        # sphere() with no arguments defaults to r=1
        bodies, _ = run("sphere($fn=16);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2, rel=0.05)

    def test_cylinder_no_r(self):
        # cylinder with no r → defaults r1=r2=1
        bodies, _ = run("cylinder(h=5, $fn=16);")
        bb = bbox(bodies)
        assert bb[5] - bb[2] == approx(5, rel=0.01)
        assert bb[3] - bb[0] == approx(2, rel=0.05)

    def test_cylinder_r1_only(self):
        # cylinder with r1 but no r2 → r2 defaults to r1
        bodies, _ = run("cylinder(h=5, r1=3, $fn=16);")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(6, rel=0.05)


# ---------------------------------------------------------------------------
# Transform edge cases
# ---------------------------------------------------------------------------

class TestTransformEdgeCases:
    def test_transform_no_children(self):
        # translate with no children returns no geometry
        bodies, _ = run("translate([1,0,0]);")
        assert bodies == []

    def test_rotate_scalar_no_v(self):
        # rotate(angle) with no axis vector defaults to z-axis
        bodies, _ = run("rotate(90) translate([5,0,0]) cube(1);")
        bb = bbox(bodies)
        # cube was on +x, after 90° z-rotation should land on -y/+y
        assert abs(bb[1]) == approx(5, rel=0.01)

    def test_rotate_zero_axis(self):
        # rotate with a zero-length axis — identity rotation
        bodies, _ = run("rotate(90, v=[0,0,0]) translate([5,0,0]) cube(1);")
        bb = bbox(bodies)
        assert bb[0] == approx(5, rel=0.01)

    def test_translate_scalar_v(self):
        # translate with a scalar (becomes [v, 0, 0])
        bodies, _ = run("translate(5) cube(1);")
        bb = bbox(bodies)
        assert bb[0] == approx(5)

    def test_translate_2d_vector(self):
        # translate with a 2-element vector (z padded to 0)
        bodies, _ = run("translate([3, 4]) cube(1);")
        bb = bbox(bodies)
        assert bb[0] == approx(3)
        assert bb[1] == approx(4)

    def test_multmatrix_3x3(self):
        # multmatrix with 3×3 rows — columns are padded with 0 (no translation)
        src = "multmatrix([[1,0,0],[0,1,0],[0,0,1]]) translate([2,0,0]) cube(1);"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[0] == approx(2)


# ---------------------------------------------------------------------------
# Color edge cases
# ---------------------------------------------------------------------------

class TestColorEdgeCases:
    def test_color_no_children(self):
        # color() with no children produces no geometry
        bodies, _ = run('color("red");')
        assert bodies == []

    def test_color_wrapping_children_call_inside_user_module(self):
        # Regression: _resolve_color built its descendant context via
        # ctx.child_ctx(color=rgba) without re-threading children_nodes/
        # children_caller_ctx, silently severing the deferred children()
        # forwarding chain -- so `color(c) children();` as a user module's
        # body (BOSL2's attachable() does exactly this, e.g. `_color($color)
        # children();`) swallowed the call-site geometry entirely, leaving
        # a colored node with no children and zero output bodies.
        # _resolve_transform never hit this since it evaluates children
        # against the original ctx directly, never deriving a new one.
        src = 'module m(c) { color(c) children(); } m("blue") cube(5);'
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].color == approx((0.0, 0.0, 1.0, 1.0))

    def test_dollar_var_override_wrapping_children_call_inside_user_module(self):
        # Same root cause, different trigger: _resolve_call_args's
        # dyn_overrides branch (any call with a $-prefixed named arg, e.g.
        # $fn=8) also derives a new context via ctx.child_ctx(dyn=...)
        # without re-threading children_nodes/children_caller_ctx.
        src = 'module m() { translate([0,0,0], $fn=8) children(); } m() cube(5);'
        bodies, _ = run(src)
        assert len(bodies) == 1


# ---------------------------------------------------------------------------
# CSG edge cases
# ---------------------------------------------------------------------------

class TestCSGEdgeCases:
    def test_union_no_children(self):
        bodies, _ = run("union();")
        assert bodies == []

    def test_union_single_child(self):
        bodies, _ = run("union() { cube(2); }")
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(2)

    def test_hull_no_children(self):
        bodies, _ = run("hull();")
        assert bodies == []

    def test_children_out_of_range(self):
        # children(idx) where idx >= $children returns no geometry
        src = "module m() { children(10); } m() cube(1);"
        bodies, _ = run(src)
        assert bodies == []

    def test_difference_first_stmt_multiple_bodies_unioned(self):
        # When the first child statement of difference() produces multiple bodies
        # (e.g., a module that emits two shapes), they must be unioned as the
        # positive operand — not treated as sequential subtractors.
        # Two non-overlapping cubes at z=+10 and z=-10; only a tiny cube at
        # the origin is actually subtracted (no overlap with either cube).
        src = """
        module pair() {
            translate([0, 0,  10]) cube([4, 4, 4], center=true);
            translate([0, 0, -10]) cube([4, 4, 4], center=true);
        }
        difference() {
            pair();               // statement 0: TWO bodies
            cube(1, center=true); // statement 1: subtractor at origin
        }
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        # Both cubes must survive: z ranges [8,12] and [-12,-8].
        assert bb[2] == approx(-12)
        assert bb[5] == approx(12)

    def test_union_first_stmt_multiple_bodies(self):
        # union() produces the same result regardless of body grouping, but
        # verify that a multi-body first statement still contributes all bodies.
        src = """
        module pair() {
            translate([0, 0,  5]) cube([2, 2, 2], center=true);
            translate([0, 0, -5]) cube([2, 2, 2], center=true);
        }
        union() { pair(); }
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[2] == approx(-6)
        assert bb[5] == approx(6)

    def test_intersection_empty_first_operand_gives_empty(self):
        # intersection(∅, B) = ∅. When the first child statement of intersection()
        # produces no geometry (an empty loop), the clip body from the second
        # statement must NOT escape as the result.
        src = "intersection() { for (i = [1:0]) cube(10); cube(5); }"
        bodies, _ = run(src)
        assert bodies == []

    def test_difference_empty_first_operand_gives_empty(self):
        # difference(∅, B) = ∅. If the positive operand of difference() is empty,
        # the subtractor must not become the result.
        src = "difference() { for (i = [1:0]) cube(10); cube(5); }"
        bodies, _ = run(src)
        assert bodies == []

    def test_intersection_empty_second_operand_gives_empty(self):
        # intersection(A, ∅) = ∅. If any operand is empty, result must be empty.
        src = "intersection() { cube(5); for (i = [1:0]) cube(10); }"
        bodies, _ = run(src)
        assert bodies == []

    def test_difference_empty_subtractor_leaves_base(self):
        # difference(A, ∅) = A. An empty subtractor is a no-op.
        src = "difference() { cube(4); *cube(10); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(4)


# ---------------------------------------------------------------------------
# for loop body variables
# ---------------------------------------------------------------------------

class TestForBodyVars:
    def test_for_body_variable(self):
        # Variable assigned inside a for body (not the loop var) must be visible to siblings
        src = "for (a=[1:3]) { x = a*2; echo(x); }"
        _, lines = run(src)
        assert lines == ["ECHO: 2", "ECHO: 4", "ECHO: 6"]

    def test_for_body_var_geometry(self):
        # Variable binding in for body used in geometry
        src = "r = 50; for (a=[0:90:270]) { pos = r*[cos(a), sin(a), 0]; translate(pos) cube(2); }"
        bodies, _ = run(src)
        assert len(bodies) == 4

    def test_for_scalar_iterable(self):
        # for over a scalar — treated as [scalar] (single-element sequence)
        src = "for (x = 5) { echo(x); }"
        _, lines = run(src)
        assert lines == ["ECHO: 5"]


# ---------------------------------------------------------------------------
# Expression edge cases
# ---------------------------------------------------------------------------

class TestExpressionEdgeCases:
    def test_division_by_zero(self):
        _, lines = run("echo(1/0);")
        assert lines == ["ECHO: inf"]

    def test_neg_division_by_zero(self):
        _, lines = run("echo(-1/0);")
        assert lines == ["ECHO: -inf"]

    def test_zero_division_by_zero(self):
        _, lines = run("echo(0/0);")
        assert lines == ["ECHO: nan"]

    def test_bool_arithmetic_is_undef(self):
        _, lines = run("echo(true + 1);")
        assert _echoes(lines) == ["ECHO: undef"]

    def test_bool_mul_is_undef(self):
        _, lines = run("echo(true * 5);")
        assert _echoes(lines) == ["ECHO: undef"]

    def test_scalar_times_matrix(self):
        _, lines = run("echo(2 * [[1,2],[3,4]]);")
        assert lines == ["ECHO: [[2, 4], [6, 8]]"]

    def test_matrix_times_scalar(self):
        _, lines = run("echo([[1,2],[3,4]] * 2);")
        assert lines == ["ECHO: [[2, 4], [6, 8]]"]

    def test_vector_dot_product(self):
        _, lines = run("echo([1,2,3] * [4,5,6]);")
        assert lines == ["ECHO: 32"]

    def test_matrix_times_vector(self):
        _, lines = run("echo([[1,0],[0,1]] * [3,4]);")
        assert lines == ["ECHO: [3, 4]"]

    def test_vector_times_matrix(self):
        _, lines = run("echo([3,4] * [[1,0],[0,1]]);")
        assert lines == ["ECHO: [3, 4]"]

    def test_matrix_times_matrix(self):
        _, lines = run("echo([[1,2],[3,4]] * [[1,0],[0,1]]);")
        assert lines == ["ECHO: [[1, 2], [3, 4]]"]

    def test_vector_divided_by_scalar(self):
        _, lines = run("echo([2,4,6] / 2);")
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_undef_comparison_lt(self):
        # Ordering comparisons between mismatched types (here undefined vs.
        # number) warn and evaluate to undef, matching real OpenSCAD.
        _, lines = run("echo(undef < 1);")
        assert lines == ["WARNING: undefined operation (undefined < number) in file <string>, line 1",
                          "ECHO: undef"]

    def test_number_equals_bool_is_false(self):
        # bool is a distinct type from number in OpenSCAD: 1 == true is
        # false, unlike Python where True == 1.
        _, lines = run("echo(1 == true, true == 1, 0 == false);")
        assert lines == ["ECHO: false, false, false"]

    def test_int_equals_float_is_true(self):
        _, lines = run("echo(1 == 1.0);")
        assert lines == ["ECHO: true"]

    def test_list_equality_with_bool_element_is_false(self):
        # [1, true] != [1, 1] even though Python's `1 == True`.
        _, lines = run("echo([1, true] == [1, 1]);")
        assert lines == ["ECHO: false"]

    def test_list_equality_different_lengths(self):
        _, lines = run("echo([1,2] == [1,2,3]);")
        assert lines == ["ECHO: false"]

    def test_bool_greater_than_number_is_undef(self):
        _, lines = run("echo(true > 0);")
        assert lines == ["WARNING: undefined operation (bool > number) in file <string>, line 1",
                          "ECHO: undef"]

    def test_bool_comparison_works(self):
        _, lines = run("echo(true >= false);")
        assert lines == ["ECHO: true"]

    def test_vector_comparison_works(self):
        _, lines = run("echo([1,2] < [3,4]);")
        assert lines == ["ECHO: true"]

    def test_floor_of_nan_is_nan(self):
        _, lines = run("echo(floor(0/0));")
        assert lines == ["ECHO: nan"]

    def test_ceil_of_inf_is_inf(self):
        _, lines = run("echo(ceil(1/0));")
        assert lines == ["ECHO: inf"]

    def test_round_of_nan_is_nan(self):
        _, lines = run("echo(round(0/0));")
        assert lines == ["ECHO: nan"]

    def test_sqrt_negative_is_nan(self):
        _, lines = run("echo(sqrt(-1));")
        assert lines == ["ECHO: nan"]

    def test_ln_zero_is_neg_inf(self):
        _, lines = run("echo(ln(0));")
        assert lines == ["ECHO: -inf"]

    def test_ln_negative_is_nan(self):
        _, lines = run("echo(ln(-1));")
        assert lines == ["ECHO: nan"]

    def test_asin_out_of_range_is_nan(self):
        _, lines = run("echo(asin(2));")
        assert lines == ["ECHO: nan"]

    def test_string_negative_index_is_undef(self):
        _, lines = run('echo("hello"[-1]);')
        assert lines == ["ECHO: undef"]

    def test_index_out_of_bounds(self):
        _, lines = run("echo([1,2,3][10]);")
        assert lines == ["ECHO: undef"]

    def test_index_non_list(self):
        _, lines = run("echo(5[0]);")
        assert lines == ["ECHO: undef"]

    def test_member_not_in_swizzle(self):
        # .w on a 2-element vector is out of range
        _, lines = run("echo([1,2].w);")
        assert lines == ["ECHO: undef"]

    def test_named_arg_to_builtin(self):
        # OpenSCAD maps named args to positional for built-ins
        _, lines = run("echo(abs(x=-3));")
        assert lines == ["ECHO: 3"]

    def test_let_op_in_expression(self):
        _, lines = run("echo(let(a=3, b=4) a + b);")
        assert lines == ["ECHO: 7"]


# ---------------------------------------------------------------------------
# List comprehension edge cases
# ---------------------------------------------------------------------------

class TestListCompEdgeCases:
    def test_listcomp_for_nested_body(self):
        # bracketed sub-comprehension in for body → each iteration yields one list
        _, lines = run("echo([for (i=[1:3]) [for (j=[1:2]) i*j]]);")
        assert lines == ["ECHO: [[1, 2], [2, 4], [3, 6]]"]

    def test_listcomp_if_false_no_else(self):
        # ListCompIf with false condition and no else → item excluded
        _, lines = run("echo([for (i=[1:3]) if (i > 10) i]);")
        assert lines == ["ECHO: []"]

    def test_listcomp_ifelse_false_branch(self):
        # ListCompIfElse, condition false → take false branch
        _, lines = run("echo([for (i=[1:2]) if (i > 1) i*10 else i]);")
        assert lines == ["ECHO: [1, 20]"]

    def test_listcomp_for_undef_iterable(self):
        # for with undef iterable → empty result
        _, lines = run("echo([for (x = undef) x]);")
        assert lines == ["ECHO: []"]

    def test_listcomp_for_scalar_iterable(self):
        # for with scalar iterable → treated as single-element sequence
        _, lines = run("echo([for (x = 5) x]);")
        assert lines == ["ECHO: [5]"]

    def test_listcomp_for_undef_body(self):
        # for body that evaluates to undef → undef is a valid element, not dropped
        _, lines = run("echo([for (i=[1:2]) undef]);")
        assert lines == ["ECHO: [undef, undef]"]

    def test_listcomp_for_undef_body_via_var(self):
        # same via a variable: mirrors test16.scad force_list(undef, 2)
        src = """
        function force_list(value, n=1, fill) =
            is_list(value) ? value :
            is_undef(fill)
              ? [for (i=[1:1:n]) value]
              : [value, for (i=[2:1:n]) fill];
        echo(force_list(undef, 2));
        """
        _, lines = run(src)
        assert lines == ["ECHO: [undef, undef]"]


# ---------------------------------------------------------------------------
# Range edge cases
# ---------------------------------------------------------------------------

class TestRangeEdgeCases:
    def test_range_zero_step(self):
        # [start:0:end] echoes as a lazy range object (iteration yields nothing)
        _, lines = run("echo([1:0:5]);")
        assert lines == ["ECHO: [1 : 0 : 5]"]

    def test_range_zero_step_iteration(self):
        # iterating a zero-step range produces no values -- and, toward a
        # larger end, never ends, which OpenSCAD reports as too many elements
        _, lines = run("echo([for (i=[1:0:5]) i]);")
        assert lines == ["WARNING: Bad range parameter in for statement: too many elements "
                         "(4294967295) in file <string>, line 1", "ECHO: []"]

    def test_range_indexing(self):
        # Indexing a range yields its [start, step, end] components, not its
        # iterated values: `[2:3:11][0]` -> 2, `[1]` -> 3, `[2]` -> 11. This is
        # what BOSL2's `is_range()`/`is_finite()` inspect to detect ranges.
        _, lines = run("r = [2:3:11]; echo(r[0], r[1], r[2]);")
        assert lines == ["ECHO: 2, 3, 11"]


# ---------------------------------------------------------------------------
# Function literal values (`function (params) expr`)
# ---------------------------------------------------------------------------

class TestFunctionLiterals:
    def test_call_stored_function_literal(self):
        _, lines = run("g = function(x) x*2; echo(g(3));")
        assert lines == ["ECHO: 6"]

    def test_function_literal_closure(self):
        # The literal closes over the scope where it was written, not the call site.
        _, lines = run("y = 10; h = function(x) x + y; echo(h(5));")
        assert lines == ["ECHO: 15"]

    def test_function_literal_default_param(self):
        _, lines = run("k = function(x, y=100) x + y; echo(k(1));")
        assert lines == ["ECHO: 101"]

    def test_function_literal_named_arg(self):
        _, lines = run("k = function(x, y=100) x + y; echo(k(1, y=5));")
        assert lines == ["ECHO: 6"]

    def test_pass_function_literal_as_argument(self):
        _, lines = run("function apply(fn, v) = fn(v); echo(apply(function(x) x*x, 4));")
        assert lines == ["ECHO: 16"]


# ---------------------------------------------------------------------------
# Function call edge cases
# ---------------------------------------------------------------------------

class TestFunctionCallEdgeCases:
    def test_call_non_function_variable(self):
        # Calling a variable that is not a function returns undef (no error)
        _, lines = run("x = [1,2,3]; echo(x());")
        assert lines == ["ECHO: undef"]

    def test_missing_param_is_undef(self):
        # Function called with fewer args than params → missing param is undef
        _, lines = run("function f(a, b) = b; echo(f(1));")
        assert lines == ["ECHO: undef"]


# ---------------------------------------------------------------------------
# Color numeric fallback
# ---------------------------------------------------------------------------

class TestColorNumericFallback:
    def test_color_non_string_non_list(self):
        # color() with a non-string, non-list arg falls back to white
        bodies, _ = run("color(42) cube(1);")
        assert bodies  # geometry still produced


# ---------------------------------------------------------------------------
# children() with no children bodies
# ---------------------------------------------------------------------------

class TestChildrenNoChildren:
    def test_children_with_no_children(self):
        # Calling children() inside a module with no children passed
        src = "module m() { children(); } m();"
        bodies, _ = run(src)
        assert bodies == []


# ---------------------------------------------------------------------------
# for loop with undef iterable (modular for, not list comp)
# ---------------------------------------------------------------------------

class TestForUndef:
    def test_for_undef_iterable(self):
        # Modular for with undef iterable produces no geometry
        src = "for (x = undef) { echo(x); }"
        _, lines = run(src)
        assert lines == []


# ---------------------------------------------------------------------------
# Function literal (lambda)
# ---------------------------------------------------------------------------

class TestFunctionLiteral:
    def test_function_literal_stored(self):
        # function literal is stored as a value (calling it is not yet implemented)
        src = "f = function(x) x * 3; echo(is_undef(f));"
        _, lines = run(src)
        # f stores the literal node (not a Python callable) — is_undef returns false
        assert lines == ["ECHO: false"]


# ---------------------------------------------------------------------------
# each in for body (scalar)
# ---------------------------------------------------------------------------

class TestEachInForBody:
    def test_each_scalar_in_for_body(self):
        # each applied to a scalar in for body wraps it in a list
        _, lines = run("echo([for (i=[1:3]) each i]);")
        assert lines == ["ECHO: [1, 2, 3]"]


# ---------------------------------------------------------------------------
# Expression operators: EchoOp, AssertOp
# ---------------------------------------------------------------------------

class TestExpressionOps:
    def test_echo_op_passthrough(self):
        # echo("msg") expr — evaluates to expr, side-effect logs the args
        _, lines = run('x = echo("debug") 5; echo(x);')
        assert lines[-1] == "ECHO: 5"

    def test_echo_op_side_effect(self):
        # the echo side-effect fires before the enclosing expression is used
        _, lines = run('x = echo("side") 42; echo(x);')
        assert 'ECHO: "side"' in lines
        assert "ECHO: 42" in lines

    def test_assert_op_passthrough(self):
        # assert(true) expr — evaluates to expr when condition holds
        _, lines = run('x = assert(true) 5; echo(x);')
        assert lines[-1] == "ECHO: 5"

    def test_assert_op_fails_on_false(self):
        # assert(false) should raise an error
        import pytest
        with pytest.raises(Exception):
            run('x = assert(false) 5; echo(x);')

    def test_assert_op_message(self):
        # assert(false, "msg") — error message included in EvalError
        import pytest
        with pytest.raises(Exception, match="Assertion 'false' failed"):
            run('x = assert(false, "msg") 5; echo(x);')


# ---------------------------------------------------------------------------
# List comprehension: let bindings
# ---------------------------------------------------------------------------

class TestListCompLet:
    def test_let_in_listcomp(self):
        # let inside list comprehension introduces a local binding
        _, lines = run("echo([for (i=[1:3]) let(j=i*2) j]);")
        assert lines == ["ECHO: [2, 4, 6]"]

    def test_let_multiple_bindings(self):
        _, lines = run("echo([for (i=[1:2]) let(a=i+1, b=i*3) [a, b]]);")
        assert lines == ["ECHO: [[2, 3], [3, 6]]"]

    def test_nested_let_in_listcomp(self):
        # let can shadow outer variable
        _, lines = run("x = 10; echo([for (i=[1:2]) let(x=i) x]);")
        assert lines == ["ECHO: [1, 2]"]

    def test_let_in_listcomp_with_if(self):
        # let combined with if filter
        _, lines = run("echo([for (i=[1:4]) let(j=i*2) if (j > 4) j]);")
        assert lines == ["ECHO: [6, 8]"]

    def test_grid_let_comprehension(self):
        # the original bug report: nested for with outer let binding
        _, lines = run("grid = [for(h=[0:2]) [let(b=h) for(a=[0:2]) a+b]]; echo(grid);")
        assert lines == ["ECHO: [[0, 1, 2], [1, 2, 3], [2, 3, 4]]"]

    def test_dollar_var_let_propagates_into_called_function(self):
        # $-prefixed let() bindings are dynamically scoped (see
        # TestLetDollarVarScoping) -- same rule applies inside a list
        # comprehension's let() clause. Verified against real OpenSCAD.
        src = """
        function f() = $fn;
        $fn = 10;
        echo([for (i=[0:1]) let($fn=99+i) f()]);
        """
        _, lines = run(src)
        assert lines == ["ECHO: [99, 100]"]


# ---------------------------------------------------------------------------
# List comprehension: each with nested lists
# ---------------------------------------------------------------------------

class TestListCompEach:
    def test_each_splices_list(self):
        # each over a list splices its elements into the parent
        _, lines = run("echo([each [1,2,3]]);")
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_each_nested_list_not_flattened(self):
        # each over a list of lists keeps sub-lists intact
        _, lines = run("a = [[1,2],[3,4]]; echo([each a]);")
        assert lines == ["ECHO: [[1, 2], [3, 4]]"]

    def test_each_in_for_body(self):
        # each inside a for body splices one level
        _, lines = run("echo([for (i=[[1,2],[3,4]]) each i]);")
        assert lines == ["ECHO: [1, 2, 3, 4]"]

    def test_each_preserves_inner_structure(self):
        # each splices exactly one level — inner nesting is preserved
        # a has one element: [[1,2],[3,4]]; each a yields that element as-is
        _, lines = run("a = [[[1,2],[3,4]]]; echo([each a]);")
        assert lines == ["ECHO: [[[1, 2], [3, 4]]]"]


# ---------------------------------------------------------------------------
# New built-ins: sign, rands, PI, is_function, search, polyhedron
# ---------------------------------------------------------------------------

class TestNewBuiltins:
    def test_sign_positive(self):
        _, lines = run("echo(sign(5));")
        assert lines == ["ECHO: 1"]

    def test_sign_negative(self):
        _, lines = run("echo(sign(-3));")
        assert lines == ["ECHO: -1"]

    def test_sign_zero(self):
        _, lines = run("echo(sign(0));")
        assert lines == ["ECHO: 0"]

    def test_PI_constant(self):
        _, lines = run("echo(PI);")
        assert len(lines) == 1
        assert abs(float(lines[0].replace("ECHO: ", "")) - 3.14159265) < 1e-5

    def test_rands_length(self):
        _, lines = run("v = rands(0, 1, 5); echo(len(v));")
        assert lines == ["ECHO: 5"]

    def test_rands_range(self):
        _, lines = run("v = rands(10, 20, 3, 42); echo(v[0] >= 10 && v[0] <= 20);")
        assert lines == ["ECHO: true"]

    def test_rands_seeded_deterministic(self):
        _, lines1 = run("v = rands(0, 100, 4, 123); echo(v);")
        _, lines2 = run("v = rands(0, 100, 4, 123); echo(v);")
        assert lines1 == lines2

    def test_is_function_true(self):
        _, lines = run("g = function(x) x*2; echo(is_function(g));")
        assert lines == ["ECHO: true"]

    def test_is_function_false_on_named_function_reference(self):
        # Real OpenSCAD: variables and functions live in separate namespaces,
        # so a bare reference to `function f(x) = ...` is an unknown variable
        # (-> undef, with a warning), not a callable value.
        _, lines = run("function f(x) = x*2; echo(is_function(f));")
        assert lines == ["WARNING: Ignoring unknown variable \"f\" in file <string>, line 1",
                          "ECHO: false"]

    def test_is_function_false_on_num(self):
        _, lines = run("echo(is_function(42));")
        assert lines == ["ECHO: false"]

    def test_is_num_excludes_bool(self):
        # bool is not a number in OpenSCAD
        _, lines = run("echo(is_num(true));")
        assert lines == ["ECHO: false"]

    def test_is_num_excludes_nan(self):
        # nan fails is_num() in real OpenSCAD, even though it's a float.
        _, lines = run("echo(is_num(0/0));")
        assert lines == ["ECHO: false"]

    def test_is_num_includes_inf(self):
        # ...but inf/-inf pass.
        _, lines = run("echo(is_num(1/0), is_num(-1/0));")
        assert lines == ["ECHO: true, true"]

    def test_unknown_variable_warns_and_returns_undef(self):
        _, lines = run("echo(totally_undefined_var);")
        assert lines == ["WARNING: Ignoring unknown variable \"totally_undefined_var\" in file <string>, line 1",
                          "ECHO: undef"]

    def test_search_string_single_char(self):
        # String match in string vector → char-by-char; single char in string
        _, lines = run('echo(search("b", "abc"));')
        assert lines == ["ECHO: [1]"]

    def test_search_string_single_char_not_found(self):
        # Not found with num_returns=1 → dropped from outer list → []
        _, lines = run('echo(search("z", "abc"));')
        assert lines == ["ECHO: []"]

    def test_search_list(self):
        _, lines = run('echo(search(["b","a"], ["a","b","c"]));')
        assert lines == ["ECHO: [1, 0]"]

    def test_search_vector_match_direct_equality(self):
        # When the match value is itself a vector, it's compared directly
        # against each whole element of the haystack (not column-indexed) —
        # this is the basis of BOSL2's `in_list(v, [UP,RIGHT,BACK])`.
        _, lines = run('echo(search([[0,0,1]], [[0,0,1],[1,0,0],[0,1,0]]));')
        assert lines == ["ECHO: [0]"]

    def test_search_scalar_match_uses_index_col(self):
        # A scalar match value still compares against vector[i][index_col].
        _, lines = run('echo(search([0,0,1], [[0,0,1],[1,0,0],[0,1,0]]));')
        assert lines == ["ECHO: [0, 0, 1]"]

    def test_search_string_as_char_array(self):
        # Multi-char string: each char searched independently in a string vector
        _, lines = run('echo(search("ba", "abcd"));')
        assert lines == ["ECHO: [1, 0]"]

    def test_search_string_num_returns_zero(self):
        # num_returns=0 → all matches per char
        _, lines = run('echo(search("a", "abcdabcd", 0));')
        assert lines == ["ECHO: [[0, 4]]"]

    def test_search_string_in_string(self):
        # Single char in string vector
        _, lines = run('echo(search("a", "abcdabcd"));')
        assert lines == ["ECHO: [0]"]

    def test_search_numeric_scalar(self):
        # Numeric (non-string) scalar: returns list of up to num_returns matches
        _, lines = run('echo(search(2, [1,2,3,2]));')
        assert lines == ["ECHO: [1]"]

    def test_search_numeric_not_found(self):
        _, lines = run('echo(search(9, [1,2,3]));')
        assert lines == ["ECHO: []"]

    def test_polyhedron_tetrahedron(self):
        # Simple tetrahedron using OpenSCAD's CW-from-outside face winding
        src = """
        polyhedron(
          points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
          faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]
        );
        """
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].body.volume() > 0

    def test_polyhedron_cube_equiv(self):
        # 6-face polyhedron matching a unit cube; faces use OpenSCAD CW-from-outside winding
        src = """
        polyhedron(
          points=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
          faces=[[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]]
        );
        """
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert abs(bodies[0].body.volume() - 1.0) < 0.01

    def test_polyhedron_triangles_alias(self):
        # legacy 'triangles' parameter name should work identically to 'faces'
        src = """
        polyhedron(
          points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
          triangles=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]
        );
        """
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].body.volume() > 0


# ---------------------------------------------------------------------------
# 2D primitives, linear_extrude, rotate_extrude, minkowski
# ---------------------------------------------------------------------------

class Test2DAndExtrusion:
    def test_circle_produces_section(self):
        bodies, _ = run("circle(r=5);")
        assert len(bodies) == 1
        assert bodies[0].section is not None
        assert bodies[0].section.area() > 0

    def test_square_produces_section(self):
        bodies, _ = run("square([3, 4]);")
        assert len(bodies) == 1
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 12.0) < 0.01

    def test_square_centered(self):
        bodies, _ = run("square(2, center=true);")
        bounds = bodies[0].section.bounds()
        assert abs(bounds[0] - (-1.0)) < 1e-6  # min_x
        assert abs(bounds[2] - 1.0) < 1e-6     # max_x

    def test_polygon_triangle(self):
        bodies, _ = run("polygon([[0,0],[1,0],[0,1]]);")
        assert len(bodies) == 1
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 0.5) < 0.01

    def test_polygon_cw_winding_fills(self):
        # polygon() must fill regardless of winding direction (OpenSCAD uses EvenOdd).
        # CW triangle: same area as CCW triangle [[0,0],[1,0],[0,1]].
        bodies, _ = run("polygon([[0,0],[0,1],[1,0]]);")
        assert len(bodies) == 1
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 0.5) < 0.01

    def test_polygon_with_hole(self):
        # outer square minus inner square hole
        src = "polygon(points=[[0,0],[4,0],[4,4],[0,4],[1,1],[3,1],[3,3],[1,3]], paths=[[0,1,2,3],[4,5,6,7]]);"
        bodies, _ = run(src)
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 12.0) < 0.1  # 16 - 4

    def test_linear_extrude_circle(self):
        # Use $fn=64 to get close to analytic volume; 2% tolerance
        src = "linear_extrude(height=5) circle(r=2, $fn=64);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].body is not None
        import math
        expected = math.pi * 4 * 5  # pi*r^2*h
        assert abs(bodies[0].body.volume() - expected) / expected < 0.02

    def test_linear_extrude_center(self):
        src = "linear_extrude(height=4, center=true) square([2,2]);"
        bodies, _ = run(src)
        bb = bodies[0].body.bounding_box()  # (min_x, min_y, min_z, max_x, max_y, max_z)
        assert abs(bb[2] - (-2.0)) < 0.01   # min_z
        assert abs(bb[5] - 2.0) < 0.01      # max_z

    def test_linear_extrude_twist(self):
        src = "linear_extrude(height=10, twist=90, slices=20) square([2,2]);"
        bodies, _ = run(src)
        assert bodies[0].body.volume() > 0

    def test_linear_extrude_scale(self):
        # scale=0 at top → cone shape, volume less than full cylinder
        src = "linear_extrude(height=3, scale=0) circle(r=1);"
        bodies, _ = run(src)
        assert bodies[0].body.volume() > 0

    def test_rotate_extrude_full(self):
        # revolve a 1x1 square at x=2 → torus-like; volume ≈ 2π²Rr² = 2π²*2.5*0.25
        src = "rotate_extrude($fn=64) square([1,1], center=true);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].body is not None
        assert bodies[0].body.volume() > 0

    def test_rotate_extrude_partial(self):
        src = "rotate_extrude(angle=180, $fn=32) square([1,1]);"
        bodies, _ = run(src)
        assert bodies[0].body.volume() > 0

    def test_minkowski_inflates_cube(self):
        # cube + sphere → rounded cube; volume > cube alone
        src = "minkowski() { cube([2,2,2]); sphere(r=0.5, $fn=16); }"
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert bodies[0].body.volume() > 8.0  # more than the original cube

    def test_minkowski_single_child(self):
        # single child — just returns the child unchanged
        src = "minkowski() { cube(2); }"
        bodies, _ = run(src)
        assert abs(bodies[0].body.volume() - 8.0) < 0.01

    def test_roof_square_pyramid(self):
        # roof() over a square produces a hip-roof/pyramid: apex height ==
        # inradius (half the square's side), bbox close to (0,0,0)-(10,10,5).
        # The straight-skeleton path is exact for a square, so this matches
        # the analytic pyramid volume to within float precision.
        src = "roof() square([10,10]);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        bb = bodies[0].body.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 5.0) < 0.5
        expected_vol = 10 * 10 * 5 / 3  # pyramid: base_area * height / 3
        assert abs(bodies[0].body.volume() - expected_vol) / expected_vol < 1e-3

    def test_roof_circle_cone(self):
        # roof() over a circle produces a cone-like solid; apex height ==
        # circle radius.
        src = "roof() circle(5);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        bb = bodies[0].body.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 5.0) < 0.5
        assert bodies[0].body.volume() > 0

    def test_roof_straight_matches_voronoi_for_convex(self):
        # for a convex shape, method="straight" and the default "voronoi"
        # produce equivalent results.
        bodies_v, _ = run("roof() square([10,10]);")
        bodies_s, _ = run('roof(method="straight") square([10,10]);')
        assert bodies_v[0].body.bounding_box() == bodies_s[0].body.bounding_box()
        assert abs(bodies_v[0].body.volume() - bodies_s[0].body.volume()) < 1e-6

    def test_roof_no_children_returns_none(self):
        bodies, _ = run("roof();")
        assert bodies == []

    def test_roof_concave_polygon(self):
        # L-shaped polygon with a reflex corner, both arms 4 units wide —
        # the straight skeleton collapses to a single ridge point at height
        # 2 with no intermediate topology events, so this is exact too.
        src = "roof() polygon([[0,0],[10,0],[10,4],[4,4],[4,10],[0,10]]);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        bb = bodies[0].body.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 2.0) < 1e-3
        expected_vol = 58.0 + 2.0 / 3.0
        assert abs(bodies[0].body.volume() - expected_vol) / expected_vol < 1e-3

    def test_roof_rectangle_ridge(self):
        # An 8x2 rectangle's straight skeleton is a hip roof with a ridge of
        # length 6 at height 1 (half the short side).
        src = "roof() square([8,2]);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        bb = bodies[0].body.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 1.0) < 1e-3
        assert abs(bodies[0].body.volume() - 22.0 / 3.0) < 1e-2

    def test_roof_asymmetric_l_exact(self):
        # Asymmetric L (arms of different widths) — the mitered offset has
        # an intermediate edge-collapse event before fully vanishing, so this
        # isn't "stable" for the tier-1 closed-form path. The tier-2
        # skeleton-graph path (shapely_polyskel) handles it exactly: the
        # skeleton has internal nodes at heights 1 and 2, giving bbox z=2 and
        # an exact volume of 92/3.
        src = "roof() polygon([[0,0],[8,0],[8,4],[2,4],[2,8],[0,8]]);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        bb = bodies[0].body.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 2.0) < 1e-3
        assert abs(bodies[0].body.volume() - 92.0 / 3.0) / (92.0 / 3.0) < 1e-3

    def test_roof_polygon_with_hole(self):
        # A 10x10 square with a 6x6 square hole (frame width = 2). Tier 2
        # handles this exactly via skeletonize() with holes. The max ridge
        # height equals the half-width of the frame = 1.0, and the volume
        # of the roof over the frame is exactly 32.
        src = """
        roof() polygon(
            points=[[0,0],[10,0],[10,10],[0,10],[2,2],[2,8],[8,8],[8,2]],
            paths=[[0,1,2,3],[4,5,6,7]]
        );
        """
        bodies, _ = run(src)
        assert len(bodies) == 1
        b = bodies[0].body
        import manifold3d as m3d
        assert b.status() == m3d.Error.NoError
        bb = b.bounding_box()
        assert bb[2] == 0.0
        assert abs(bb[5] - 1.0) < 1e-3
        assert abs(b.volume() - 32.0) < 0.1

    def test_roof_text_with_holes(self):
        # Glyphs that have counter-holes (like "a" and "g") must be roofed
        # using the hole-aware skeleton path. Verify they produce valid,
        # non-empty geometry.
        bodies, _ = run('roof() text("ag", size=72);')
        assert len(bodies) == 1
        b = bodies[0].body
        import manifold3d as m3d
        assert b.status() == m3d.Error.NoError
        assert not b.is_empty()
        assert b.volume() > 0

    def test_roof_unknown_method_warns(self):
        bodies, echoes = run('roof(method="bogus") square([10,10]);')
        assert any("Unknown roof method 'bogus'" in e for e in echoes)
        assert len(bodies) == 1


# ---------------------------------------------------------------------------
# offset, projection, intersection_for, lookup, $children, ModularAssert
# ---------------------------------------------------------------------------

class TestRemainingBuiltins:
    def test_offset_r_expands(self):
        # round offset of unit square by 1 should have area > 1
        bodies, _ = run("offset(r=1) square([2,2]);")
        assert bodies[0].section is not None
        assert bodies[0].section.area() > 4.0

    def test_offset_negative_shrinks(self):
        bodies, _ = run("offset(r=-0.5) square([4,4]);")
        assert bodies[0].section.area() < 16.0

    def test_offset_delta_square_corners(self):
        bodies, _ = run("offset(delta=1) square([2,2]);")
        assert bodies[0].section.area() > 4.0

    def test_projection_cut_false(self):
        # project a cube → roughly square cross section
        bodies, _ = run("projection() cube([3,4,5]);")
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 12.0) < 0.1

    def test_projection_cut_true(self):
        # cut at z=0 through a cube starting at z=-1 → cross section at z=0
        bodies, _ = run("projection(cut=true) translate([0,0,-1]) cube([3,4,2]);")
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 12.0) < 0.1

    def test_intersection_for(self):
        # intersection of three rotated cubes → rounded shape with less volume than a single cube
        src = "intersection_for(i=[0:2]) rotate([0,0,i*60]) cube([10,2,10], center=true);"
        bodies, _ = run(src)
        assert len(bodies) == 1
        assert 0 < bodies[0].body.volume() < 200

    def test_lookup_interpolates(self):
        _, lines = run("echo(lookup(0.5, [[0,0],[1,10]]));")
        assert lines == ["ECHO: 5"]

    def test_lookup_clamps_low(self):
        _, lines = run("echo(lookup(-1, [[0,0],[1,10]]));")
        assert lines == ["ECHO: 0"]

    def test_lookup_clamps_high(self):
        _, lines = run("echo(lookup(5, [[0,0],[1,10]]));")
        assert lines == ["ECHO: 10"]

    def test_lookup_empty_table_is_undef(self):
        _, lines = run("echo(lookup(5, []));")
        assert lines == ["ECHO: undef"]

    def test_lookup_exact_key_match_no_interpolation(self):
        # Landing exactly on a table key (first, last, or a middle one in
        # a 3+ point table) must return that key's value exactly, not an
        # interpolated approximation from floating-point t=0/t=1.
        _, lines = run(
            "echo(lookup(0, [[0,0],[1,10]]));"
            "echo(lookup(1, [[0,0],[1,10]]));"
            "echo(lookup(1, [[0,0],[1,10],[2,50]]));"
        )
        assert lines == ["ECHO: 0", "ECHO: 10", "ECHO: 10"]

    def test_lookup_multi_segment_picks_correct_bracket(self):
        # A 3-point table must interpolate within whichever segment the
        # key actually falls in, not always the first or last pair.
        _, lines = run(
            "echo(lookup(0.5, [[0,0],[1,10],[2,50]]));"
            "echo(lookup(1.5, [[0,0],[1,10],[2,50]]));"
        )
        assert lines == ["ECHO: 5", "ECHO: 30"]

    def test_lookup_single_entry_table_always_returns_its_value(self):
        # With only one point, key <= pairs[0][0] and key >= pairs[-1][0]
        # are simultaneously true for any query -- both clamp branches
        # agree, so every query returns the sole entry's value.
        _, lines = run("echo(lookup(5, [[5, 99]]), lookup(-5, [[5, 99]]));")
        assert lines == ["ECHO: 99, 99"]

    def test_lookup_unsorted_table_is_sorted_internally(self):
        # _builtin_lookup sorts the table by key before searching, so an
        # out-of-order table still works correctly.
        _, lines = run("echo(lookup(5, [[1,10],[0,0]]));")
        assert lines == ["ECHO: 10"]

    def test_children_count(self):
        src = "module m() { echo($children); } m() { cube(1); sphere(1); }"
        _, lines = run(src)
        assert lines == ["ECHO: 2"]

    def test_children_count_zero(self):
        src = "module m() { echo($children); } m();"
        _, lines = run(src)
        assert lines == ["ECHO: 0"]

    def test_children_count_counts_statements_not_geometries(self):
        # $children counts child *statements* in {}, regardless of how many
        # geometries each one produces — an `if` that yields nothing (or a
        # `children()` forwarding zero bodies) still counts as one child.
        src = "module m() { echo($children); } m() { cube(1); if (false) sphere(1); }"
        _, lines = run(src)
        assert lines == ["ECHO: 2"]

    def test_children_call_counts_even_with_no_bodies_to_forward(self):
        # `children()` is itself one child statement in the calling block,
        # even if the *caller's* own children (forwarded here) is empty.
        src = (
            "module inner() { echo($children); }"
            " module outer() { inner() { cube(1); children(); } }"
            " outer();"
        )
        _, lines = run(src)
        assert lines == ["ECHO: 2"]

    def test_modular_assert_passes(self):
        # assert with true condition — children's geometry passes through
        bodies, _ = run("assert(true) cube(1);")
        assert len(bodies) == 1
        assert abs(bodies[0].body.volume() - 1.0) < 0.01

    def test_modular_assert_fails(self):
        import pytest
        with pytest.raises(Exception, match="Assertion 'false' failed"):
            run("assert(false, \"bad\") cube(1);")

    def test_render_passthrough(self):
        # render() is a display hint — just passes through children
        bodies, _ = run("render() cube(2);")
        assert len(bodies) == 1
        assert abs(bodies[0].body.volume() - 8.0) < 0.01


# ---------------------------------------------------------------------------
# 2D CSG: union, difference, intersection on CrossSection children
# ---------------------------------------------------------------------------

class Test2DCSG:
    def test_2d_union(self):
        bodies, _ = run("union() { square([3,1]); square([1,3]); }")
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 5.0) < 0.01  # 3+3-1 overlap

    def test_2d_difference(self):
        bodies, _ = run("difference() { square([4,4]); square([2,2]); }")
        assert bodies[0].section is not None
        assert abs(bodies[0].section.area() - 12.0) < 0.01  # 16-4

    def test_2d_intersection(self):
        bodies, _ = run("intersection() { square([3,3]); circle(r=2, $fn=64); }")
        assert bodies[0].section is not None
        # intersection of 3x3 square and r=2 circle (area ~12.57) — circle wins in corners
        import math
        assert bodies[0].section.area() < math.pi * 4  # less than full circle

    def test_2d_difference_with_circle(self):
        # square with circle punched out
        bodies, _ = run("difference() { square([4,4], center=true); circle(r=1, $fn=64); }")
        import math
        expected = 16.0 - math.pi
        assert abs(bodies[0].section.area() - expected) / expected < 0.01

    def test_2d_csg_then_extrude(self):
        # 2D boolean then extrude to 3D
        src = "linear_extrude(height=5) difference() { square([4,4]); circle(r=1, $fn=32); }"
        bodies, _ = run(src)
        assert bodies[0].body is not None
        assert bodies[0].body.volume() > 0


# ---------------------------------------------------------------------------
# Error call chain — module errors
# ---------------------------------------------------------------------------

class TestModuleErrorCallChain:
    def test_module_appears_in_chain(self):
        src = """
        module bad() { assert(false, "boom"); }
        bad();
        """
        with pytest.raises(EvalError) as exc_info:
            run(src)
        msg = str(exc_info.value)
        assert "Assertion 'false' failed" in msg
        assert "called by 'bad'" in msg

    def test_nested_modules_in_chain(self):
        src = """
        module inner() { assert(false, "boom"); }
        module outer() { inner(); }
        outer();
        """
        with pytest.raises(EvalError) as exc_info:
            run(src)
        msg = str(exc_info.value)
        assert "called by 'inner'" in msg
        assert "called by 'outer'" in msg

    def test_function_inside_module_in_chain(self):
        src = """
        function bad() = assert(false, "boom") 1;
        module m() { echo(bad()); }
        m();
        """
        with pytest.raises(EvalError) as exc_info:
            run(src)
        msg = str(exc_info.value)
        assert "called by 'bad'" in msg
        assert "called by 'm'" in msg


# ---------------------------------------------------------------------------
# Recursive user modules
# ---------------------------------------------------------------------------

class TestRecursiveModule:
    def test_recursive_module_echo(self):
        src = """
        module countdown(n) {
            if (n > 0) {
                echo(n);
                countdown(n - 1);
            }
        }
        countdown(3);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 3", "ECHO: 2", "ECHO: 1"]

    def test_recursive_module_geometry(self):
        src = """
        module stack(n, h=1) {
            cube([1, 1, h]);
            if (n > 1) { translate([0, 0, h]) stack(n - 1, h); }
        }
        stack(3);
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[5] == approx(3.0)  # max z = 3


# ---------------------------------------------------------------------------
# Scoping: last-wins and hoisting
# ---------------------------------------------------------------------------

class TestScoping:
    def test_last_wins_in_block(self):
        _, lines = run("x = 1; x = 7; echo(x);")
        assert any("WARNING" in l and "x" in l and "overwritten" in l for l in lines)
        assert "ECHO: 7" in lines

    def test_param_self_reference_default_does_not_recurse(self):
        # A parameter with no default, re-assigned via a self-referential
        # expression in the body (the BOSL2 `chamfer = approx(chamfer,0) ?
        # undef : chamfer;` pattern) must resolve to its own (undef) param
        # value on the RHS, not recurse into the body's own assignment.
        src = "module m(x) { x = is_undef(x) ? 5 : x; echo(x); } m();"
        _, lines = run(src)
        assert lines == ["ECHO: 5"]

    def test_param_shadow_reassignment_no_warning(self):
        # A body assignment that shadows a parameter name with the same
        # value-normalization pattern must NOT emit a spurious "was assigned
        # ... but was overwritten" warning — only real double-assignments do.
        src = "module m(x) { x = is_undef(x) ? 5 : x; echo(x); } m();"
        _, lines = run(src)
        assert lines == ["ECHO: 5"]
        assert not any("WARNING" in l for l in lines)

    def test_forward_reference_function(self):
        src = "echo(double(5)); function double(x) = x * 2;"
        _, lines = run(src)
        assert lines == ["ECHO: 10"]

    def test_forward_reference_module(self):
        src = "box(3); module box(s) { cube(s); }"
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(3)

    def test_module_scope_isolates_variable(self):
        src = """
        x = 10;
        module m() { x = 20; echo(x); }
        m();
        echo(x);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 20", "ECHO: 10"]

    def test_nested_module_closes_over_reassigned_outer_local(self):
        # A module nested inside another module's body is a closure over
        # the enclosing call's locals (BOSL2's `cuboid()` defines a nested
        # `module corner_shape()` that reads cuboid's local `edges`, which
        # cuboid reassigns from its own parameter via `edges =
        # _edges(edges, ...)` before calling corner_shape). The inner
        # module must see the REASSIGNED value, not recurse forever trying
        # to resolve the outer assignment's own right-hand side.
        src = """
        module outer(edges=[1,2,3]) {
            edges = [edges[0]+1, edges[1]+1, edges[2]+1];
            module inner() {
                echo(edges);
            }
            inner();
        }
        outer();
        """
        _, lines = run(src)
        assert lines == ["ECHO: [2, 3, 4]"]

    def test_children_forwarded_through_isolated_call_with_same_named_param_still_sees_ancestor_binding(self):
        # A real bug found running a real BOSL2 script (attachable()/
        # trapezoid(), which deep-forwards children() through several more
        # calls while an intermediate ISOLATED call -- attachable() itself,
        # with its own unbound "path"/"h" parameters -- stays on the call
        # stack throughout). `wrapper(path)` below mirrors that shape: an
        # isolated (non-closure) call whose own declared parameter shares a
        # name with an ancestor's variable, forwarding children() straight
        # through past itself via `leaf()`. The deferred child block
        # (`echo(path)`, lexically inside `outer()`) must resolve `path`
        # against OUTER's own binding, not wrapper's own same-named
        # (unbound, undef) parameter -- even though wrapper's own scope is
        # still open (unpopped) the entire time the deferred block runs.
        #
        # This was broken by an earlier trail-storage design (tried first in
        # the C++ port) that judged visibility purely by level-number
        # ordering ("pushed no later than my own level"): wrapper's own
        # undef "path" binding is chronologically *more recent* than
        # outer's real one, so a numeric-only check picked it up as if it
        # were an ancestor, even though wrapper is not an ancestor of the
        # deferred child block at all -- just an unrelated, still-open
        # branch. Fixed (in both ports) by tracking true parent-chain
        # ancestry per level instead of comparing level numbers -- ported
        # here with that fix already in place, not re-derived.
        src = """
        module leaf() { children(0); }
        module wrapper(path) { leaf() children(0); }
        module outer() {
            path = [1, 2, 3];
            wrapper() echo(path);
        }
        outer();
        """
        _, lines = run(src)
        assert lines == ["ECHO: [1, 2, 3]"]


# ---------------------------------------------------------------------------
# Hull 2D
# ---------------------------------------------------------------------------

class TestHull2D:
    def test_hull_two_circles_yields_section(self):
        src = "hull() { circle(r=1, $fn=32); translate([4,0]) circle(r=1, $fn=32); }"
        bodies, _ = run(src)
        assert bodies[0].section is not None
        assert bodies[0].body is None

    def test_hull_2d_larger_than_parts(self):
        import math
        # Hull of two unit circles separated by 5 units, extruded to measure area
        src = "linear_extrude(1) hull() { circle(r=1, $fn=64); translate([5,0]) circle(r=1, $fn=64); }"
        bodies, _ = run(src)
        vol = bodies[0].body.volume()
        # Two separate circles would give ~2*pi ≈ 6.28; hull is strictly larger
        assert vol > 2 * math.pi * 0.95


# ---------------------------------------------------------------------------
# str() and concat() edge cases
# ---------------------------------------------------------------------------

class TestStrEdgeCases:
    def test_str_bool_true(self):
        _, lines = run('echo(str(true));')
        assert lines == ['ECHO: "true"']

    def test_str_bool_false(self):
        _, lines = run('echo(str(false));')
        assert lines == ['ECHO: "false"']

    def test_str_undef(self):
        _, lines = run('echo(str(undef));')
        assert lines == ['ECHO: "undef"']

    def test_str_list(self):
        _, lines = run('echo(str([1, 2, 3]));')
        assert lines == ['ECHO: "[1, 2, 3]"']

    def test_str_multi_arg_concatenates(self):
        _, lines = run('echo(str(1, "+", 2, "=", 3));')
        assert lines == ['ECHO: "1+2=3"']

    def test_concat_two_lists(self):
        _, lines = run('echo(concat([1, 2], [3, 4]));')
        assert lines == ["ECHO: [1, 2, 3, 4]"]

    def test_concat_list_and_scalar(self):
        _, lines = run('echo(concat([1, 2], 3));')
        assert lines == ["ECHO: [1, 2, 3]"]

    def test_concat_three_lists(self):
        _, lines = run('echo(concat([1], [2], [3]));')
        assert lines == ["ECHO: [1, 2, 3]"]


class TestNumberFormatting:
    """`echo()`/`str()` number formatting must match OpenSCAD's output:
    6 significant digits, exponents without a leading zero, and fixed
    notation for exponents in [-5, 5] (one wider than Python's `%g`)."""

    def test_large_exponent_no_leading_zero(self):
        _, lines = run("echo(1000000);")
        assert lines == ["ECHO: 1e+6"]

    def test_small_number_stays_fixed_notation(self):
        _, lines = run("echo(0.00001);")
        assert lines == ["ECHO: 0.00001"]

    def test_small_exponent_no_leading_zero(self):
        _, lines = run("echo(1.23456789e-7);")
        assert lines == ["ECHO: 1.23457e-7"]

    def test_negative_zero(self):
        _, lines = run("echo(-0.0);")
        assert lines == ["ECHO: 0"]


# ---------------------------------------------------------------------------
# Special variables and stub built-ins
# ---------------------------------------------------------------------------

class TestSpecialVariables:
    def test_fa_default(self):
        _, lines = run('echo($fa);')
        assert lines == ["ECHO: 12"]

    def test_fs_default(self):
        _, lines = run('echo($fs);')
        assert lines == ["ECHO: 2"]

    def test_fn_default(self):
        _, lines = run('echo($fn);')
        assert lines == ["ECHO: 0"]

    def test_fn_override_via_named_arg(self):
        # $fn set as named arg on a built-in should not crash
        bodies, _ = run("sphere(r=1, $fn=8);")
        assert bodies[0].body.volume() > 0

    def test_version_returns_list(self):
        _, lines = run('echo(is_list(version()));')
        assert lines == ["ECHO: true"]

    def test_version_num_returns_number(self):
        _, lines = run('echo(is_num(version_num()));')
        assert lines == ["ECHO: true"]

    def test_parent_module_at_toplevel(self):
        # At top level, parent_module() returns undef (no parent)
        _, lines = run('echo(is_undef(parent_module()));')
        assert lines == ["ECHO: true"]


# ---------------------------------------------------------------------------
# breakpoint()
# ---------------------------------------------------------------------------

def _run_with_hook(src: str):
    """Run src with a debug hook attached. Returns (paused_lines, echo_lines)."""
    echo_lines = []
    paused_lines = []

    def hook(line, depth, *, forced=False, expr_level=False, expr_depth=0, origin=None, get_frames=None):
        if forced:
            paused_lines.append(line)
        return ("continue", {})

    nodes = getASTfromString(src, include_comments=False)
    root_scope = build_scopes(nodes)
    ev = Evaluator(echo_fn=lambda msg: echo_lines.append(msg), debug_hook=hook)
    ev.evaluate(nodes, root_scope)
    return paused_lines, echo_lines


class TestBreakpoint:
    def test_unconditional_pauses_in_debug_mode(self):
        paused, _ = _run_with_hook("breakpoint();")
        assert len(paused) == 1

    def test_unconditional_noop_without_hook(self):
        # No exception and no side effects when no debugger is attached.
        bodies, lines = run("breakpoint(); cube(1);")
        assert lines == []
        assert len(bodies) == 1

    def test_true_condition_pauses(self):
        paused, _ = _run_with_hook("breakpoint(true);")
        assert len(paused) == 1

    def test_false_condition_skips(self):
        paused, _ = _run_with_hook("breakpoint(false);")
        assert paused == []

    def test_zero_condition_skips(self):
        paused, _ = _run_with_hook("breakpoint(0);")
        assert paused == []

    def test_nonzero_condition_pauses(self):
        paused, _ = _run_with_hook("breakpoint(1);")
        assert len(paused) == 1

    def test_named_condition_arg(self):
        paused, _ = _run_with_hook("breakpoint(condition=true);")
        assert len(paused) == 1

    def test_named_condition_false_skips(self):
        paused, _ = _run_with_hook("breakpoint(condition=false);")
        assert paused == []

    def test_variable_condition(self):
        paused, _ = _run_with_hook("x = 5; breakpoint(x > 3);")
        assert len(paused) == 1

    def test_variable_condition_false(self):
        paused, _ = _run_with_hook("x = 2; breakpoint(x > 3);")
        assert paused == []

    def test_pauses_at_correct_line(self):
        src = "cube(1);\nbreakpoint();\ncube(2);"
        paused, _ = _run_with_hook(src)
        assert paused == [2]

    def test_multiple_breakpoints(self):
        src = "breakpoint();\nbreakpoint();"
        paused, _ = _run_with_hook(src)
        assert len(paused) == 2

    def test_produces_no_geometry(self):
        bodies, _ = run("breakpoint();")
        assert bodies == []

    def test_does_not_interfere_with_geometry(self):
        paused, _ = _run_with_hook("cube(1); breakpoint(); sphere(1);")
        assert len(paused) == 1

    def test_breakpoint_inside_module(self):
        src = "module foo() { breakpoint(); } foo();"
        paused, _ = _run_with_hook(src)
        assert len(paused) == 1

    def test_conditional_breakpoint_inside_loop(self):
        # Breaks only on iterations where i >= 3 (i = 3, 4 → 2 breaks)
        src = "for (i = [0:4]) { breakpoint(i >= 3); }"
        paused, _ = _run_with_hook(src)
        assert len(paused) == 2


# ---------------------------------------------------------------------------
# object()
# ---------------------------------------------------------------------------

class TestObject:
    def test_basic_creation_and_access(self):
        src = 'o = object(a=1, b="hello", c=[1,2,3]); echo(o.a, o.b, o.c, o["a"]);'
        _, echoes = run(src)
        assert echoes == ['ECHO: 1, "hello", [1, 2, 3], 1']

    def test_nested_object(self):
        src = "o = object(a=1, nested=object(x=10, y=20)); echo(o.nested.x); echo(o.nested);"
        _, echoes = run(src)
        assert echoes == ["ECHO: 10", "ECHO: { x = 10; y = 20; }"]

    def test_empty_object_echo(self):
        _, echoes = run("echo(object());")
        assert echoes == ["ECHO: { }"]

    def test_missing_key_is_undef(self):
        src = 'o = object(a=1); echo(o.nope); echo(o["nope"]); echo(o[0]);'
        _, echoes = run(src)
        assert echoes == ["ECHO: undef", "ECHO: undef", "ECHO: undef"]

    def test_type_predicates(self):
        src = (
            "o = object(a=1);"
            "echo(is_object(o), is_list(o), is_string(o), is_num(o), is_undef(o), is_object(5));"
        )
        _, echoes = run(src)
        assert echoes == ["ECHO: true, false, false, false, false, false"]

    def test_len(self):
        _, echoes = run("echo(len(object(a=1, b=2, c=3)));")
        assert echoes == ["ECHO: 3"]

    def test_has_key_present_and_absent(self):
        src = 'o = object(a=1, b=2); echo(has_key(o, "a"), has_key(o, "nope"));'
        _, echoes = run(src)
        assert echoes == ["ECHO: true, false"]

    def test_has_key_non_object_is_undef(self):
        src = 'echo(has_key(5, "a"), has_key([1,2], "a"), has_key("str", "a"), has_key(undef, "a"));'
        _, echoes = run(src)
        assert echoes == ["ECHO: undef, undef, undef, undef"]

    def test_has_key_on_empty_object(self):
        _, echoes = run('echo(has_key(object(), "a"));')
        assert echoes == ["ECHO: false"]

    def test_equality_is_deep_and_order_sensitive(self):
        src = (
            "echo(object(a=1,b=2) == object(a=1,b=2));"
            "echo(object(a=1,b=2) == object(b=2,a=1));"
            "echo(object(a=1,b=2) != object(b=2,a=1));"
        )
        _, echoes = run(src)
        assert echoes == ["ECHO: true", "ECHO: false", "ECHO: true"]

    def test_str_formatting(self):
        src = 'echo(str(object(a=1, nested=object(x=10, y=20))));'
        _, echoes = run(src)
        assert echoes == ['ECHO: "{ a = 1; nested = { x = 10; y = 20; }; }"']

    def test_for_iterates_over_keys(self):
        src = "for (k = object(z=1, a=2, m=3)) echo(k);"
        _, echoes = run(src)
        assert echoes == ['ECHO: "z"', 'ECHO: "a"', 'ECHO: "m"']

    def test_function_valued_member_is_callable(self):
        src = "f = object(fn=function(x) x*2); echo(f.fn(5));"
        _, echoes = run(src)
        assert echoes == ["ECHO: 10"]

    def test_merge_via_positional_object(self):
        src = (
            "o1 = object(a=1, b=2);"
            "echo(object(o1, c=3));"
            "echo(object(o1, b=99, c=3));"
        )
        _, echoes = run(src)
        assert echoes == [
            "ECHO: { a = 1; b = 2; c = 3; }",
            "ECHO: { a = 1; b = 99; c = 3; }",
        ]

    def test_merge_via_positional_list_of_pairs(self):
        _, echoes = run('echo(object([["x",10],["y",20]]));')
        assert echoes == ["ECHO: { x = 10; y = 20; }"]

    def test_invalid_positional_arg_warns_and_is_undef(self):
        _, echoes = run("echo(object(1,2));")
        assert len(echoes) == 2
        assert "WARNING: object(Argument 0 <number>) An unnamed argument must be either <object> or <list>, it is <number>." in echoes[0]
        assert echoes[1] == "ECHO: undef"

    def test_addition_on_objects_is_undef(self):
        _, echoes = run("echo(object(a=1) + object(b=2));")
        assert _echoes(echoes) == ["ECHO: undef"]


class TestOperandWarnings:
    """Output checked against OpenSCAD 2026.02.01 byte for byte."""

    def test_operators(self):
        _, lines = run('echo(1+"a"); echo(-"a"); echo(true*2); echo(object(a=1)+object(b=2));')
        assert lines == [
            "WARNING: undefined operation (number + string) in file <string>, line 1",
            "ECHO: undef",
            "WARNING: undefined operation (-string) in file <string>, line 1",
            "ECHO: undef",
            "WARNING: undefined operation (bool * number) in file <string>, line 1",
            "ECHO: undef",
            "WARNING: undefined operation (object + object) in file <string>, line 1",
            "ECHO: undef",
        ]

    def test_products(self):
        _, lines = run("echo([1,2]*[1,2,3]); echo([[1,2]]*[1]); echo([1]*[[1,2],[3,4]]);"
                       'echo([[1,2]]*[[1,2]]); echo(["a",1]*[1,2]); echo([[1,"a"]]*[1,2]); echo([]*[]);')
        assert [l.removesuffix(" in file <string>, line 1") for l in lines if l != "ECHO: undef"] == [
            "WARNING: vector*vector requires matching lengths (2 != 3)",
            "WARNING: matrix*vector requires matrix column count to match vector length (2 != 1)",
            "WARNING: vector*matrix requires vector length to match matrix row count (1 != 2)",
            "WARNING: matrix*matrix requires left operand column count to match right operand row count (2 != 1)",
            "WARNING: undefined vector*vector multiplication where first elements are types string and number",
            "WARNING: Matrix must contain only numbers. Problem at row 0, col 1",
            "WARNING: Multiplication is undefined on empty vectors",
        ]

    def test_builtin_parameters(self):
        _, lines = run('echo(cos("a")); echo(max([true,1])); echo(ord(1)); echo(norm([true,0]));'
                       "echo(cross([1,0],[1,0,0])); echo(cross([1/0,0,0],[0,1,0]));")
        assert [l.removesuffix(" in file <string>, line 1") for l in lines if l != "ECHO: undef"] == [
            'WARNING: cos() parameter could not be converted: argument 0: expected number, found string ("a")',
            "WARNING: max() parameter could not be converted: vector element 0: expected number, found bool (true)",
            "WARNING: ord() parameter could not be converted: argument 0: expected string, found number (1)",
            "WARNING: Incorrect arguments to norm()",
            "WARNING: Invalid vector size of parameter for cross()",
            "WARNING: Invalid value (INF) in parameter vector for cross()",
        ]


class TestErosionAndSimplify:
    """minkowski_difference() and simplify(), not in OpenSCAD (cpp #101,
    #103). Erosion volumes match openscad_cpp_evaluator's exactly."""

    def test_erosion_by_a_cube_is_exact(self):
        bodies, _ = run("minkowski_difference() { cube(10, center=true); cube(2, center=true); }")
        assert bodies[0].body.volume() == pytest.approx(512)
        assert [round(v, 9) for v in bodies[0].body.bounding_box()] == [-4, -4, -4, 4, 4, 4]

    def test_erosion_of_one_child_is_a_no_op(self):
        bodies, _ = run("minkowski_difference() cube(5);")
        assert bodies[0].body.volume() == pytest.approx(125)

    def test_simplify_reduces_and_keeps_genus(self):
        src = ("difference() { sphere(20, $fn=64); for (a=[0:60:300]) rotate([0,0,a]) "
               "rotate([90,0,0]) cylinder(h=50, r=4, $fn=32); }")
        (dense,), _ = run(src)
        (simple,), _ = run(f"simplify() {src}")
        assert simple.body.num_tri() < dense.body.num_tri()
        assert simple.body.genus() == dense.body.genus()
        assert simple.body.volume() == pytest.approx(dense.body.volume(), rel=1e-3)

    def test_simplify_2d_and_bad_tolerances(self):
        bodies, lines = run("simplify(0.5) circle(10, $fn=128); simplify(-1) cube(3); simplify(\"x\") cube(3);")
        assert len(bodies[0].section.to_polygons()[0]) < 128
        assert [l.removesuffix(" in file <string>, line 1") for l in lines] == [
            "WARNING: simplify: tolerance must not be negative",
            "WARNING: simplify: tolerance must be a number"]


class TestSphereStyles:
    """sphere(style=), BOSL2 spheroid()'s tessellations (cpp #102).
    Triangle counts and volumes match openscad_cpp_evaluator's."""

    def test_styles(self):
        bodies, _ = run("".join(f'translate([{i*30},0,0]) sphere(10, $fn=24, style="{st}");'
                                for i, st in enumerate(["orig", "aligned", "stagger", "octa", "icosa"])))
        assert [b.body.num_tri() for b in bodies] == [572, 528, 528, 288, 500]
        # All outward: an inside-out VNF would show as a negative volume.
        assert [round(b.body.volume(), 2) for b in bodies] == [4070.7, 4070.55, 4082.12, 4024.32, 4097.25]

    def test_bad_style_warns_and_falls_back(self):
        bodies, lines = run('sphere(5, style="nope"); sphere(5, style=3);')
        assert [round(b.body.volume(), 3) for b in bodies] == [490.917, 490.917]
        assert [l.removesuffix(" in file <string>, line 1") for l in lines] == [
            'WARNING: sphere: unknown style "nope"; expected one of orig, aligned, stagger, octa, icosa',
            "WARNING: sphere: style must be a string"]


class TestFeatureDetection:
    """$_SUPPORTED_FEATURE, $_BELFRYSCAD and supported_feature() (cpp #113, #115)."""

    def test_levels(self):
        _, lines = run('echo($_SUPPORTED_FEATURE, supported_feature("separate-children"), '
                       'supported_feature("mesh-repair"), supported_feature("nope"), supported_feature(3), '
                       'supported_feature(feature="roof-op"));')
        assert lines == ["ECHO: true, 1, 0, 0, 0, 1"]

    def test_version_is_three_numbers(self):
        _, lines = run("echo(len($_BELFRYSCAD), is_num($_BELFRYSCAD[0]));")
        assert lines == ["ECHO: 3, true"]

    def test_guard_idiom(self):
        _, lines = run('echo(!is_undef($_SUPPORTED_FEATURE) && supported_feature("sphere-styles"));')
        assert lines == ["ECHO: true"]


class TestDxfDimCross:
    """dxf_dim()/dxf_cross() (cpp #100). Checked against OpenSCAD 2026.02.01
    on its own dim-all.dxf and example009.dxf; this synthetic file covers
    the same paths without shipping those."""

    DXF = "\n".join([
        "0", "SECTION", "2", "ENTITIES",
        # aligned dimension "w", from (1,1) to (4,5): length 5
        "0", "DIMENSION", "8", "dims", "1", "w", "70", "1", "13", "1", "23", "1", "14", "4", "24", "5",
        # horizontal (rotated, angle 0) dimension "h" over the same points: 3
        "0", "DIMENSION", "8", "dims", "1", "h", "70", "0", "50", "0", "13", "1", "23", "1", "14", "4", "24", "5",
        # a cross whose strokes meet at (2,3)
        "0", "LINE", "8", "cross", "10", "0", "20", "3", "11", "4", "21", "3",
        "0", "LINE", "8", "cross", "10", "2", "20", "0", "11", "2", "21", "6",
        # two lines joined end to end: an outline, not a cross
        "0", "LINE", "8", "outline", "10", "0", "20", "0", "11", "5", "21", "0",
        "0", "LINE", "8", "outline", "10", "5", "20", "0", "11", "5", "21", "5",
        "0", "ENDSEC", "0", "EOF", ""])

    def _run(self, tmp_path, body):
        from openscad_lalr_parser import getASTfromFile
        (tmp_path / "t.dxf").write_text(self.DXF)
        (tmp_path / "t.scad").write_text(body)
        lines = []
        nodes = getASTfromFile(str(tmp_path / "t.scad"))
        Evaluator(echo_fn=lines.append).evaluate(nodes, build_scopes(nodes))
        return [l.split(" in file ")[0] for l in lines]

    def test_dimensions_and_cross(self, tmp_path):
        assert self._run(tmp_path, 'echo(dxf_dim(file="t.dxf", name="w"), dxf_dim(file="t.dxf", name="h"), '
                                   'dxf_dim(file="t.dxf", name="w", scale=2), '
                                   'dxf_cross(file="t.dxf", layer="cross"));') == ["ECHO: 5, 3, 10, [2, 3]"]

    def test_failures_warn(self, tmp_path):
        assert self._run(tmp_path, 'echo(dxf_dim(file="t.dxf", name="x")); echo(dxf_dim(file="no.dxf", name="x"));'
                                   'echo(dxf_cross(file="t.dxf", layer="outline"));') == [
            "WARNING: Can't find dimension 'x' in 't.dxf', layer ''!", "ECHO: undef",
            "WARNING: Can't open DXF file 'no.dxf'!", "ECHO: undef",
            "WARNING: Can't find cross in 't.dxf', layer 'outline'!", "ECHO: undef"]


class TestListFonts:
    def test_bundled_faces_are_listed_and_their_specs_resolve(self):
        from openscad_evaluator import list_fonts
        bundled = [f for f in list_fonts() if f["path"] == "<bundled>"]
        assert [f["style"] for f in bundled] == ["Bold", "Bold Italic", "Italic", "Regular"]
        for f in bundled:
            _, lines = run(f'echo(fontmetrics(10, "{f["spec"]}").font.style);')
            assert lines == [f'ECHO: "{f["style"]}"']


class TestSvgImportFilter:
    """import(svg, id=/class=) (cpp #182)."""

    SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="cut" transform="translate(10,0)">
    <rect id="r1" class="cut outline" x="0" y="0" width="10" height="10"/>
    <rect id="r2" class="cut" x="20" y="0" width="5" height="5"/>
  </g>
  <rect id="r3" class="engrave" x="0" y="50" width="20" height="20"/>
</svg>"""

    def _areas(self, tmp_path, body):
        from openscad_lalr_parser import getASTfromFile
        (tmp_path / "d.svg").write_text(self.SVG)
        (tmp_path / "t.scad").write_text(body)
        lines = []
        nodes = getASTfromFile(str(tmp_path / "t.scad"))
        bodies, _ = Evaluator(echo_fn=lines.append).evaluate(nodes, build_scopes(nodes))
        return [round(b.section.area()) for b in bodies], [l.split(" in file ")[0] for l in lines]

    def test_id_takes_a_group_whole_with_its_transform(self, tmp_path):
        areas, _ = self._areas(tmp_path, 'import("d.svg", id="cut");')
        assert areas == [125]

    def test_class_matches_one_entry_of_the_list(self, tmp_path):
        assert self._areas(tmp_path, 'import("d.svg", class="cut");')[0] == [125]
        assert self._areas(tmp_path, 'import("d.svg", class="outline");')[0] == [100]

    def test_a_miss_warns_and_imports_nothing(self, tmp_path):
        assert self._areas(tmp_path, 'import("d.svg", id="nope");') == (
            [], ['WARNING: import() filter id = "nope" did not match anything'])


class TestSectionIdsAndGenerateFlag:
    """cpp #144, #192, #193."""

    def test_2d_shapes_pick_back_to_their_line_across_cache_hits(self):
        from openscad_evaluator.evaluator import to_renderable_bodies
        src = "square(5);\ntranslate([10,0]) circle(3);\ncolor(\"red\") translate([20,0]) square(2);"
        cache = ManifoldCache()
        for _ in range(2):
            nodes = getASTfromString(src, include_comments=False)
            ev = Evaluator(echo_fn=lambda m: None, manifold_cache=cache)
            bodies, id_to_node = ev.evaluate(nodes, build_scopes(nodes))
            slabs = to_renderable_bodies(bodies)
            ids = [int(b.body.to_mesh().run_original_id[0]) for b in slabs]
            assert [id_to_node[i].position.line for i in ids] == [1, 2, 3]
            assert ev.id_to_color[ids[2]] == (1.0, 0.0, 0.0, 1.0)

    def test_preview_height(self):
        from openscad_evaluator.evaluator import to_renderable_bodies
        bodies, _ = run("square(5);")
        assert to_renderable_bodies(bodies, height=0.1)[0].body.bounding_box()[5] == pytest.approx(0.1)

    def test_generate_false_runs_the_script_but_builds_nothing(self):
        lines = []
        nodes = getASTfromString('echo(1); cube(1); echo(1+"a");', include_comments=False)
        bodies, _ = Evaluator(echo_fn=lines.append).evaluate(nodes, build_scopes(nodes), generate=False)
        assert bodies == []
        assert lines[0] == "ECHO: 1" and lines[1].startswith("WARNING: undefined operation")


class TestCallChains:
    """id_to_call_chain (cpp #180); identical to openscad_cpp_evaluator's
    call_sites on this probe, cold and warm."""

    def test_chains(self, tmp_path):
        from openscad_lalr_parser import getASTfromFile
        (tmp_path / "lib.scad").write_text("module boxy(s) { cube(s); }\n"
                                           "module outer() { translate([0,20,0]) boxy(8); }\n")
        (tmp_path / "main.scad").write_text("use <lib.scad>\ncube(1);\ntranslate([20,0,0]) boxy(4);\n"
                                            "outer();\nmodule inner() { boxy(3); }\ninner();\ninner();\n")
        cache = ManifoldCache()
        seen = []
        for _ in range(2):
            nodes = getASTfromFile(str(tmp_path / "main.scad"))
            ev = Evaluator(echo_fn=lambda m: None, manifold_cache=cache)
            bodies, _ = ev.evaluate(nodes, build_scopes(nodes))
            seen.append([[(p.origin.rsplit("/", 1)[-1], p.line, m)
                          for p, m in ev.id_to_call_chain[int(b.body.to_mesh().run_original_id[0])]]
                         for b in bodies])
        assert seen[0] == [[], [("main.scad", 3, True)], [("lib.scad", 2, True), ("main.scad", 4, True)],
                           [("main.scad", 5, True), ("main.scad", 6, True)],
                           [("main.scad", 5, True), ("main.scad", 7, True)]]
        # A warm render attributes a whole-subtree hit to the node reusing it.
        assert seen[1] == [[], [], [("main.scad", 4, True)],
                           [("main.scad", 5, True), ("main.scad", 6, True)],
                           [("main.scad", 5, True), ("main.scad", 7, True)]]


class TestKeepMinuendColor:
    """Evaluator(keep_minuend_color=True) (cpp #173); triangle colours
    identical to openscad_cpp_evaluator's, cold and cached."""

    @staticmethod
    def _colors(src, cache=None):
        from collections import Counter
        nodes = getASTfromString(src, include_comments=False)
        ev = Evaluator(echo_fn=lambda m: None, keep_minuend_color=True, manifold_cache=cache)
        bodies, _ = ev.evaluate(nodes, build_scopes(nodes))
        return [sorted(Counter(tuple(round(float(x), 3) for x in r) for r in b.tri_colors).items())
                if b.tri_colors is not None else b.color for b in bodies]

    def test_cut_faces_take_the_minuend_colour(self):
        assert self._colors('difference() { color("red") cube(10); color("blue") translate([5,5,5]) cube(10); }') \
            == [(1.0, 0.0, 0.0, 1.0)]

    def test_each_merged_part_cuts_in_its_own_colour_through_transforms_and_the_cache(self):
        src = ('difference() { translate([0,0,1]) union() { color("red") cube(10); '
               'color("green") translate([10,0,0]) cube(10); } translate([5,5,5]) cube([20,10,10]); }')
        cache = ManifoldCache()
        expect = [[((0.0, 0.502, 0.0, 1.0), 16), ((1.0, 0.0, 0.0, 1.0), 20)]]
        assert self._colors(src, cache) == expect
        assert self._colors(src, cache) == expect


class TestTailCalls:
    """Tail calls run as a loop, as OpenSCAD's own trampoline does. Output
    identical to OpenSCAD 2026.02.01."""

    def test_deep_self_mutual_and_literal_recursion(self):
        _, lines = run("function acc(n,a=0)=n<=0?a:acc(n-1,a+1); echo(acc(20000));"
                       "function ev(n) = n == 0 ? true : od(n-1); function od(n) = n == 0 ? false : ev(n-1);"
                       "echo(ev(10001));"
                       "f = function(n, a=0) n <= 0 ? a : let(b = a + n) f(n - 1, b); echo(f(5000));")
        assert lines == ["ECHO: 20000", "ECHO: false", "ECHO: 1.25025e+7"]

    def test_echo_and_assert_in_the_chain_still_run(self):
        _, lines = run("function h(n, a=[]) = assert(n >= 0) n == 0 ? a : echo(n) h(n - 1, concat(a, [n]));"
                       "echo(h(3));")
        assert lines == ["ECHO: 3", "ECHO: 2", "ECHO: 1", "ECHO: [3, 2, 1]"]

    def test_runaway_tail_recursion_is_an_error(self):
        with pytest.raises(EvalError, match="Recursion detected calling function 'f'"):
            run("function f(n) = f(n + 1); echo(f(0));")


class TestLevelSet:
    """levelset() (cpp #124-#127, #136, #125). Volumes and areas match
    openscad_cpp_evaluator's to three decimals."""

    GRID = ("grid = [for (i=[0:20]) [for (j=[0:20]) [for (k=[0:20]) "
            "norm([-10+i, -10+j, -10+k]) - 8]]]; B = [[-10,-10,-10],[10,10,10]];")

    def _vols(self, src):
        bodies, lines = run(self.GRID + src)
        return [round(b.body.volume() if b.body else b.section.area(), 3) for b in bodies], lines

    def test_grid_function_and_bands(self):
        vols, _ = self._vols("levelset(grid, B); levelset(function(x,y,z) norm([x,y,z]) - 8, B, edge=1);"
                             "levelset(grid, B, isovalue=[-3, 0]); levelset(grid, B, isovalue=undef);"
                             "levelset(grid, B, isovalue=-100);")
        assert vols == [2122.303, 2135.033, 1612.261, 2122.303]

    def test_2d_and_the_clip_to_bounds(self):
        vols, _ = self._vols("levelset(function(x,y) norm([x,y]) - 6, [[-10,-10],[10,10]], edge=0.5);"
                             "levelset(function(x,y) x, [[-10,-10],[10,10]], edge=1);")
        assert vols == [112.931, 200.0]  # the half-plane is exact: 10 x 20

    def test_bad_arguments_warn(self):
        _, lines = self._vols("levelset(grid, [[-10,-10],[10,10,10]]); levelset(function(x,y,z) x, B);")
        assert [l.split(" in file ")[0] for l in lines] == [
            "WARNING: levelset(): bounds must be [[x0,y0],[x1,y1]] or [[x0,y0,z0],[x1,y1,z1]]",
            "WARNING: levelset(): a function field needs edge= (the sample spacing)"]

    def test_a_closure_field_is_never_cached(self):
        nodes = getASTfromString("levelset(function(x,y,z) norm([x,y,z]) - 3, [[-4,-4,-4],[4,4,4]], edge=1);",
                                 include_comments=False)
        ev = Evaluator(echo_fn=lambda m: None, manifold_cache=ManifoldCache())
        ev.evaluate(nodes, build_scopes(nodes))
        assert ev.csg_tree[0].uncacheable


class TestTextMetrics:
    """`textmetrics()`/`fontmetrics()` measure against the bundled Liberation
    Sans font (see docs/evaluator.md). Values are close to, but not bit-for-bit
    identical to, real OpenSCAD (which applies FreeType hinting we don't
    replicate) — expected strings below are this implementation's own output."""

    def test_basic_left_baseline(self):
        _, echoes = run('echo(textmetrics(text="Hello", size=10));')
        assert echoes == [
            "ECHO: { position = [1.13932, -0.135634]; size = [29.9276, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [0, 0]; advance = [31.6501, 0]; }"
        ]

    def test_size_scales_linearly(self):
        _, echoes = run('echo(textmetrics(text="Hello", size=20));')
        assert echoes == [
            "ECHO: { position = [2.27865, -0.271267]; size = [59.8551, 20.3993]; ascent = 20.128; "
            "descent = -0.271267; offset = [0, 0]; advance = [63.3002, 0]; }"
        ]

    def test_single_char_no_descender(self):
        _, echoes = run('echo(textmetrics(text="A", size=10));')
        assert echoes == [
            "ECHO: { position = [0.0271267, 0]; size = [9.20953, 9.55539]; ascent = 9.55539; "
            "descent = 0; offset = [0, 0]; advance = [9.26378, 0]; }"
        ]

    def test_empty_text_is_all_zero(self):
        _, echoes = run('echo(textmetrics(text="", size=10));')
        assert echoes == [
            "ECHO: { position = [0, 0]; size = [0, 0]; ascent = 0; descent = 0; offset = [0, 0]; "
            "advance = [0, 0]; }"
        ]

    def test_halign_center_valign_center(self):
        _, echoes = run(
            'echo(textmetrics(text="Hello", size=10, halign="center", valign="center"));'
        )
        assert echoes == [
            "ECHO: { position = [-14.6857, -5.09983]; size = [29.9276, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [-15.8251, -4.96419]; advance = [31.6501, 0]; }"
        ]

    def test_halign_right_valign_top(self):
        _, echoes = run(
            'echo(textmetrics(text="Hello", size=10, halign="right", valign="top"));'
        )
        assert echoes == [
            "ECHO: { position = [-30.5108, -10.1997]; size = [29.9276, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [-31.6501, -10.064]; advance = [31.6501, 0]; }"
        ]

    def test_halign_left_valign_bottom(self):
        _, echoes = run(
            'echo(textmetrics(text="Hello", size=10, halign="left", valign="bottom"));'
        )
        assert echoes == [
            "ECHO: { position = [1.13932, 0]; size = [29.9276, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [0, 0.135634]; advance = [31.6501, 0]; }"
        ]

    def test_spacing_scales_advance_and_size(self):
        _, echoes = run('echo(textmetrics(text="Hello", size=10, spacing=1.5));')
        assert echoes == [
            "ECHO: { position = [1.13932, -0.135634]; size = [41.8905, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [0, 0]; advance = [47.4752, 0]; }"
        ]

        _, echoes = run('echo(textmetrics(text="Hello", size=10, spacing=2));')
        assert echoes == [
            "ECHO: { position = [1.13932, -0.135634]; size = [53.8534, 10.1997]; ascent = 10.064; "
            "descent = -0.135634; offset = [0, 0]; advance = [63.3002, 0]; }"
        ]

    def test_is_object_and_member_access(self):
        _, echoes = run('echo(is_object(textmetrics(text="Hi", size=10)));')
        assert echoes == ["ECHO: true"]

        _, echoes = run('m = textmetrics(text="Hello", size=10); echo(m.size, m["ascent"]);')
        assert echoes == ["ECHO: [29.9276, 10.1997], 10.064"]

    def test_fontmetrics_structure(self):
        _, echoes = run("echo(fontmetrics(size=10));")
        assert echoes == [
            "ECHO: { nominal = { ascent = 12.5732; descent = -2.94325; }; max = { ascent = 13.6108; "
            'descent = -4.21143; }; interline = 15.9709; font = { family = "Liberation Sans"; '
            'style = "Regular"; }; }'
        ]

    def test_fontmetrics_resolves_requested_font(self):
        # Arial is metric-compatible with Liberation Sans by design (same
        # hhea-derived nominal/interline), but "max" comes from the actual
        # glyph bbox extremes in the *resolved* font's head table, so it
        # differs — proving font= actually selects a different font rather
        # than just being echoed back into the family name. Skipped where
        # Arial isn't installed (e.g. CI) — see skip_unless_font_installed.
        skip_unless_font_installed("Arial", "Arial")
        _, echoes = run('echo(fontmetrics(size=10, font="Arial"));')
        assert echoes == [
            "ECHO: { nominal = { ascent = 12.5732; descent = -2.94325; }; max = { ascent = 13.9703; "
            'descent = -4.50982; }; interline = 15.9709; font = { family = "Arial"; '
            'style = "Regular"; }; }'
        ]

    def test_fontmetrics_reports_resolved_style(self):
        skip_unless_font_installed("Times New Roman:style=Bold", "Times New Roman")
        _, echoes = run('echo(fontmetrics(size=10, font="Times New Roman:style=Bold").font);')
        assert echoes == ['ECHO: { family = "Times New Roman"; style = "Bold"; }']

    def test_textmetrics_resolves_requested_font(self):
        # Times New Roman's serif proportions measure differently from the
        # default Liberation Sans for the same text/size.
        skip_unless_font_installed("Times New Roman", "Times New Roman")
        _, echoes = run('echo(textmetrics(text="Hello", size=10, font="Times New Roman").size, '
                         'textmetrics(text="Hello", size=10, font="Times New Roman")["ascent"]);')
        assert echoes == ["ECHO: [30.1378, 9.83344], 9.64355"]


# ---------------------------------------------------------------------------
# text()
# ---------------------------------------------------------------------------

class TestText:
    """`text()` renders glyph outlines (from the same bundled Liberation Sans
    font as `textmetrics()`) as a 2D cross-section. Bbox values below come
    from `linear_extrude(height=1) text(...)` and were cross-checked against
    real OpenSCAD-dev output (see docs/evaluator.md)."""

    def test_single_char_left_baseline(self):
        bb = bbox(run('linear_extrude(height=1) text("A", size=10);')[0])
        assert bb[0] == approx(0.0271267, rel=1e-3)
        assert bb[1] == pytest.approx(0.0, abs=1e-3)
        assert bb[3] == approx(9.23665, rel=1e-3)
        assert bb[4] == approx(9.55539, rel=1e-3)

    def test_word_left_baseline(self):
        bb = bbox(run('linear_extrude(height=1) text("Hello", size=10);')[0])
        assert bb[0] == approx(1.13932, rel=1e-3)
        assert bb[1] == approx(-0.135634, rel=1e-2)
        assert bb[3] == approx(31.0669, rel=1e-3)
        assert bb[4] == approx(10.064, rel=1e-3)

    def test_halign_center_valign_center(self):
        bb = bbox(run(
            'linear_extrude(height=1) text("Hello", size=10, halign="center", valign="center");'
        )[0])
        assert bb[0] == approx(-14.6857, rel=1e-3)
        assert bb[1] == approx(-5.09983, rel=1e-3)
        assert bb[3] == approx(15.2418, rel=1e-3)
        assert bb[4] == approx(5.09983, rel=1e-3)

    def test_empty_text_produces_empty_geometry(self):
        # Previously this returned a zero-volume body whose Manifold status
        # was Error.InvalidConstruction (not a clean empty manifold) --
        # unioning that into other geometry silently zeroed out the whole
        # result. It must produce no body at all instead.
        bodies, _ = run('linear_extrude(height=1) text("");')
        assert bodies == []

    def test_empty_extrude_does_not_poison_sibling_union(self):
        bodies, _ = run(
            'union() { cube(5); linear_extrude(height=1) text(""); }'
        )
        assert len(bodies) == 1
        assert bodies[0].body.status().name == "NoError"
        assert bodies[0].body.volume() == pytest.approx(125.0)

    def test_composite_glyph_renders(self):
        bb = bbox(run('linear_extrude(height=1) text("é", size=10);')[0])
        area = (bb[3] - bb[0]) * (bb[4] - bb[1])
        assert area > 0

    def test_cff_font_renders(self):
        # CFF/OTF glyphs use cubic Bezier curves (vs. TrueType's quadratic);
        # this exercises that flattening path via a system CFF font. Skipped
        # where STIXGeneral isn't installed (e.g. CI).
        skip_unless_font_installed("STIXGeneral:style=Bold Italic", "STIXGeneral")
        bb = bbox(run(
            'linear_extrude(height=1) text("Hi", size=10, font="STIXGeneral:style=Bold Italic");'
        )[0])
        assert bb[0] == approx(-0.333333, rel=1e-3)
        assert bb[1] == pytest.approx(-0.125, abs=1e-3)
        assert bb[3] == approx(14.4444, rel=1e-3)
        assert bb[4] == approx(9.5, rel=1e-3)

    def test_spacing_increases_extent(self):
        bb1 = bbox(run('linear_extrude(height=1) text("AA", size=10, spacing=1);')[0])
        bb2 = bbox(run('linear_extrude(height=1) text("AA", size=10, spacing=2);')[0])
        assert bb2[3] > bb1[3]


# ---------------------------------------------------------------------------
# $-variable dynamic scoping into children()
# ---------------------------------------------------------------------------

class TestDollarVarChildren:
    def test_dollar_var_visible_to_children(self):
        src = """
        module m() { $x = 42; children(); }
        m() echo($x);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 42"]

    def test_dollar_var_from_for_loop_visible_to_children(self):
        src = """
        module xcopies(spacing, n=2) {
            for ($idx = [0:1:n-1]) {
                translate([($idx - n/2 + 0.5) * spacing, 0, 0])
                    children();
            }
        }
        xcopies(10, n=3) sphere(d=$idx+1);
        """
        bodies, lines = run(src)
        assert len(bodies) == 3
        widths = sorted(
            b.body.bounding_box()[3] - b.body.bounding_box()[0] for b in bodies
        )
        assert widths[0] < widths[1] < widths[2]

    def test_dollar_var_assignment_in_for_body_visible_to_children(self):
        src = """
        module m() {
            for (i = [0:2]) {
                $val = i * 10;
                children();
            }
        }
        m() echo($val);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 0", "ECHO: 10", "ECHO: 20"]

    def test_dollar_var_overrides_in_nested_module(self):
        src = """
        module outer() { $x = 1; children(); }
        module inner() { $x = 2; children(); }
        outer() inner() echo($x);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 2"]

    def test_dollar_var_from_caller_visible_without_override(self):
        src = """
        module m() { children(); }
        $x = 99;
        m() echo($x);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 99"]

    def test_children_in_for_loop_produces_multiple_bodies(self):
        src = """
        module triple() {
            for (i = [0:2]) {
                translate([i * 10, 0, 0]) children();
            }
        }
        triple() cube(1);
        """
        bodies, _ = run(src)
        assert len(bodies) == 3

    def test_children_in_let_block_preserves_dollar_vars(self):
        src = """
        module m() {
            $v = 5;
            let (x = 1) { children(); }
        }
        m() echo($v);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 5"]

    def test_three_part_range_in_for(self):
        src = """
        vals = [];
        for (i = [0:2:6]) echo(i);
        """
        _, lines = run(src)
        assert lines == ["ECHO: 0", "ECHO: 2", "ECHO: 4", "ECHO: 6"]

    def test_two_part_range_in_for(self):
        src = "for (i = [0:3]) echo(i);"
        _, lines = run(src)
        assert lines == ["ECHO: 0", "ECHO: 1", "ECHO: 2", "ECHO: 3"]

    def test_children_indexed_with_dollar_var(self):
        src = """
        module m() { $x = 10; children(0); }
        m() { cube($x); cube(1); }
        """
        bodies, _ = run(src)
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(10)

    def test_children_n_indexes_by_statement_not_body(self):
        # children(N) must index by child STATEMENT, not by output body.
        # Statement 0 is disabled (*) so it produces 0 bodies; the body list
        # is therefore just [cube6_body].  The old code returned bodies[1]
        # (OOB → nothing).  The fix evaluates statement 1 directly → cube(6).
        src = """
        module pick_second() { children(1); }
        pick_second() {
            *cube(10);  // statement 0: disabled, 0 bodies
            cube(6);    // statement 1: must be returned by children(1)
        }
        """
        bodies, _ = run(src)
        assert bodies, "children(1) must return statement 1 even when statement 0 produces 0 bodies"
        bb = bbox(bodies)
        # cube(6) → side length 6; cube(10) would be 10
        assert bb[3] - bb[0] == approx(6)

    def test_children_n_correct_stmt_when_prior_stmt_is_empty(self):
        # With three statements where statement 0 produces 0 bodies, children(1)
        # must map to the small cube and children(2) to the large cube — not shifted.
        src = """
        module pick() {
            children(1);  // must be cube(2), not cube(5)
            children(2);  // must be cube(5)
        }
        pick() {
            *cube(1);  // statement 0: disabled, 0 bodies
            cube(2);   // statement 1
            cube(5);   // statement 2
        }
        """
        bodies, _ = run(src)
        assert len(bodies) == 2
        sides = sorted(
            b.body.bounding_box()[3] - b.body.bounding_box()[0] for b in bodies
        )
        assert sides[0] == approx(2)  # cube(2)
        assert sides[1] == approx(5)  # cube(5)


# ---------------------------------------------------------------------------
# CSG tree (Evaluator.csg_tree) — Phase 1 of the evaluator refactor: an
# explicit, persistent tree built as a side effect of eager evaluation,
# purely additive (bodies/echo output are unaffected). See docs/evaluator.md
# "CSG tree" section.
# ---------------------------------------------------------------------------

class TestCSGTree:
    def test_single_primitive_one_node(self):
        _, _, ev = run_tree("cube(2);")
        assert len(ev.csg_tree) == 1
        node = ev.csg_tree[0]
        assert node.kind == "cube"
        assert node.is_builtin is True
        assert node.children == []

    def test_union_nests_children(self):
        _, _, ev = run_tree("union() { cube(1); sphere(1); }")
        assert len(ev.csg_tree) == 1
        node = ev.csg_tree[0]
        assert node.kind == "union"
        assert [c.kind for c in node.children] == ["cube", "sphere"]

    def test_highlight_wraps_one_child(self):
        _, _, ev = run_tree("#cube(2);")
        assert len(ev.csg_tree) == 1
        assert ev.csg_tree[0].kind == "highlight"
        assert len(ev.csg_tree[0].children) == 1
        assert ev.csg_tree[0].children[0].kind == "cube"

    def test_background_wraps_one_child(self):
        _, _, ev = run_tree("%sphere(2);")
        assert len(ev.csg_tree) == 1
        assert ev.csg_tree[0].kind == "background"
        assert ev.csg_tree[0].children[0].kind == "sphere"

    def test_color_wraps_one_child(self):
        _, _, ev = run_tree('color("red") cube(2);')
        assert len(ev.csg_tree) == 1
        assert ev.csg_tree[0].kind == "color"
        assert len(ev.csg_tree[0].children) == 1
        assert ev.csg_tree[0].children[0].kind == "cube"

    def test_for_produces_sibling_nodes_no_for_node(self):
        _, _, ev = run_tree("for (i=[0:2]) cube(i+1);")
        # transparent: 3 sibling cube nodes directly at root, no "for" node
        assert [n.kind for n in ev.csg_tree] == ["cube", "cube", "cube"]

    def test_if_is_transparent(self):
        _, _, ev = run_tree("if (true) cube(1); if (false) sphere(1);")
        # true branch's cube attaches directly at root; false branch never runs
        assert [n.kind for n in ev.csg_tree] == ["cube"]

    def test_intersection_for_gets_its_own_combiner_node(self):
        src = "intersection_for(i=[0:2]) rotate([0,0,i*60]) cube([10,2,10], center=true);"
        bodies, _, ev = run_tree(src)
        assert len(ev.csg_tree) == 1
        node = ev.csg_tree[0]
        assert node.kind == "intersection_for"
        assert len(node.children) == 3          # one rotate(...) per iteration
        assert node.bodies == bodies             # the combined (post-^) result, not the 3 pre-intersection bodies
        assert len(node.bodies) == 1

    def test_user_module_call_splices_body_geometry_directly(self):
        # A user-module call isn't geometry -- it's just a named wrapper
        # around whatever its body runs, so it gets no CSGNode of its own;
        # the body's own geometry (cube) lands directly in the tree.
        _, _, ev = run_tree("module foo() { cube(1); } foo();")
        assert len(ev.csg_tree) == 1
        node = ev.csg_tree[0]
        assert node.kind == "cube"

    def test_disable_produces_no_tree_node(self):
        _, _, ev = run_tree("*cube(1);")
        assert ev.csg_tree == []

    def test_nested_mix(self):
        src = """
        module box(s) { cube(s); }
        union() {
            #box(2);
            translate([5,0,0]) %sphere(1);
        }
        """
        _, _, ev = run_tree(src)
        assert len(ev.csg_tree) == 1
        union_node = ev.csg_tree[0]
        assert union_node.kind == "union"
        assert len(union_node.children) == 2
        hl, tr = union_node.children
        assert hl.kind == "highlight"
        # box(2)'s call boundary is spliced away -- its body's cube lands
        # directly under the highlight modifier.
        assert hl.children[0].kind == "cube"
        assert tr.kind == "translate"
        assert tr.children[0].kind == "background"
        assert tr.children[0].children[0].kind == "sphere"

    @pytest.mark.parametrize("src", [
        "cube(2);",
        "union() { cube(1); sphere(1); }",
        "difference() { cube([4,4,4]); cube([2,2,2]); }",
        "for (i=[0:2]) cube(i+1);",
        "module box(s) { cube(s); } box(3);",
        'color("red") translate([1,0,0]) cube(1);',
        "%cube(1); cube(2);",
    ])
    def test_flatten_matches_evaluate_result(self, src):
        # Regression proof: flattening the tree reproduces evaluate()'s own
        # result for any script with no top-level `!` (show_only).
        bodies, _, ev = run_tree(src)
        assert flatten_csg_tree(ev.csg_tree) == bodies

    def test_flatten_vs_evaluate_with_top_level_show_only(self):
        # Both statements are resolved into the tree, but only the `!`
        # subtree is generated, so flattening the tree gives the result.
        src = "cube(1); !cube(3);"
        bodies, _, ev = run_tree(src)
        assert len(ev.csg_tree) == 2                  # both top-level statements recorded
        assert flatten_csg_tree(ev.csg_tree) == bodies
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(27)

    def test_error_mid_subtree_leaves_valid_partial_tree(self):
        # An EvalError raised deep inside a subtree must not corrupt the
        # parent's accumulator — the in-progress node is simply never
        # appended, and prior completed siblings remain intact.
        src = 'cube(1); union() { sphere(1); assert(false, "boom"); }'
        nodes = getASTfromString(src, include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator(echo_fn=lambda msg: None)
        with pytest.raises(EvalError):
            ev.evaluate(nodes, root_scope)
        assert [n.kind for n in ev.csg_tree] == ["cube"]


# ---------------------------------------------------------------------------
# CSG tree Phase 2, step 1 — resolve/generate split for leaf primitives
# (cube, sphere, cylinder, polyhedron, circle, square, polygon, text).
# Generation is now fully deferred to generate_tree() (Phase 2 step 6, the
# final cutover), so these tests cover the params shape and the "mixed
# migration" scenarios exercised while the split was rolled out kind-by-kind
# — nesting a migrated leaf inside a wrapper migrated in a later step, which
# during the rollout risked a migrated node's bodies being silently dropped
# by a not-yet-migrated wrapper. Kept as regression coverage for that nesting.
# ---------------------------------------------------------------------------

class TestCSGTreeResolveGenerateSplit:
    def test_all_geometry_kinds_registered_in_dispatch(self):
        # Every geometry-producing builtin is migrated as of Phase 2 step 5.
        _, _, ev = run_tree("cube(1);")
        for kind in ("cube", "sphere", "cylinder", "polyhedron", "circle", "square", "polygon", "text",
                     "translate", "rotate", "scale", "mirror", "resize", "multmatrix", "color",
                     "hull", "minkowski", "offset", "projection",
                     "union", "difference", "intersection", "intersection_for",
                     "linear_extrude", "rotate_extrude", "roof", "surface", "import"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_cube_params_shape(self):
        _, _, ev = run_tree("cube([2,3,4], center=true);")
        params = ev.csg_tree[0].params
        assert params["size"] == [2.0, 3.0, 4.0]
        assert params["center"] is True
        assert "color" in params

    def test_sphere_params_shape(self):
        _, _, ev = run_tree("sphere(r=5, $fn=12);")
        params = ev.csg_tree[0].params
        assert "verts" in params and "tris" in params
        assert params["verts"].shape[1] == 3

    def test_cylinder_params_shape(self):
        _, _, ev = run_tree("cylinder(h=10, r1=2, r2=4);")
        params = ev.csg_tree[0].params
        assert params["h"] == 10.0
        assert params["r1"] == 2.0
        assert params["r2"] == 4.0
        assert params["segs"] >= 3

    def test_polyhedron_params_shape(self):
        src = "polyhedron(points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]], faces=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]);"
        _, _, ev = run_tree(src)
        params = ev.csg_tree[0].params
        assert "verts" in params and "tri_arr" in params

    def test_circle_square_polygon_params_shape(self):
        _, _, ev = run_tree("circle(r=3, $fn=8);")
        assert ev.csg_tree[0].params["name"] == "circle"
        assert ev.csg_tree[0].params["r"] == 3.0

        _, _, ev = run_tree("square([2,5]);")
        assert ev.csg_tree[0].params["name"] == "square"
        assert ev.csg_tree[0].params["size"] == [2.0, 5.0]

        _, _, ev = run_tree("polygon(points=[[0,0],[1,0],[0,1]]);")
        assert ev.csg_tree[0].params["name"] == "polygon"
        assert ev.csg_tree[0].params["paths"] is None

    def test_text_params_shape(self):
        _, _, ev = run_tree('text("Hi", size=10);')
        params = ev.csg_tree[0].params
        assert "font_spec" in params and "glyphs" in params and "scale" in params

    def test_migrated_leaf_inside_migrated_union(self):
        # union() was migrated in step 4; still a useful direct regression
        # check for the mixed-migration hazard this test was first written
        # for (when union was still eager: if a migrated cube/sphere
        # silently returned no bodies, union's per-statement grouping would
        # treat them as empty statements and drop them from the result).
        bb = bbox(run('union() { cube(2, center=true); translate([5,0,0]) sphere(1); }')[0])
        assert bb[3] - bb[0] == approx(7)  # spans from cube's left edge to sphere's right edge

    def test_migrated_leaf_inside_migrated_transform(self):
        # translate() was migrated in step 2 (both this and cube are now
        # resolve/generate); still a useful direct regression check.
        bb = bbox(run("translate([10,0,0]) cube(2, center=true);")[0])
        assert bb[0] == approx(9) and bb[3] == approx(11)

    def test_migrated_leaf_inside_migrated_color(self):
        # color() was migrated in step 2 (both this and sphere are now
        # resolve/generate); still a useful direct regression check.
        bodies, _ = run('color("red") sphere(2);')
        assert len(bodies) == 1
        r, g, b = bodies[0].color[:3]
        assert r == approx(1.0) and g == approx(0.0, rel=1) and b == approx(0.0, rel=1)

    def test_migrated_leaf_inside_migrated_hull_and_transform(self):
        # hull() and translate() were both migrated in later steps (3, 2);
        # still a useful direct regression check alongside cube.
        bb = bbox(run("hull() { cube(1); translate([5,0,0]) cube(1); }")[0])
        assert bb[3] - bb[0] == approx(6)

    def test_migrated_leaf_inside_for_inside_migrated_difference(self):
        # for() is transparent (Phase 1); difference() was migrated in step
        # 4 and specifically needs group_sizes bookkeeping to correctly
        # group a for loop's variable number of contributed tree children
        # into "the second statement" for its per-statement CSG grouping.
        src = "difference() { cube(4, center=true); for (i=[-1:2:1]) translate([i*3,0,0]) sphere(0.5); }"
        bb = bbox(run(src)[0])
        assert bb[3] - bb[0] == approx(4)  # still a 4x4x4 cube's extent, just with holes

    def test_polyhedron_inside_migrated_minkowski(self):
        # minkowski() was migrated in step 3; still a useful direct
        # regression check. CW-from-outside face winding (OpenSCAD
        # convention), matching TestNewBuiltins.test_polyhedron_tetrahedron.
        tet = "polyhedron(points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]], faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]])"
        bodies, _ = run(f"minkowski() {{ {tet}; cube(0.1); }}")
        assert len(bodies) == 1
        assert bodies[0].body.volume() > 0


# ---------------------------------------------------------------------------
# CSG tree Phase 2, step 2 — resolve/generate split for transforms
# (translate/rotate/scale/mirror/multmatrix/resize) and color.
# ---------------------------------------------------------------------------

class TestCSGTreeStep2Transforms:
    def test_transform_kinds_registered(self):
        _, _, ev = run_tree("cube(1);")
        for kind in ("translate", "rotate", "scale", "mirror", "resize", "multmatrix", "color"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_translate_params_shape(self):
        _, _, ev = run_tree("translate([1,2,3]) cube(1);")
        node = ev.csg_tree[0]
        assert node.kind == "translate"
        assert node.params["name"] == "translate"
        assert node.params["args"][0] == [1, 2, 3]
        assert len(node.children) == 1 and node.children[0].kind == "cube"

    def test_resize_params_shape_and_generate_uses_child_bbox(self):
        # resize's generate step needs its own (already-generated) child's
        # bounding_box() — confirmed safe since it's this node's own child,
        # not a different node's output.
        bb = bbox(run("resize([4,4,4]) sphere(1);")[0])
        assert bb[3] - bb[0] == approx(4)
        assert bb[4] - bb[1] == approx(4)

    def test_color_params_shape(self):
        _, _, ev = run_tree('color([0,1,0,1]) cube(1);')
        node = ev.csg_tree[0]
        assert node.kind == "color"
        assert node.params["rgba"] == (0.0, 1.0, 0.0, 1.0)

    def test_migrated_transform_wraps_migrated_transform(self):
        # Both translate and scale are migrated — nested migrated wrappers.
        bb = bbox(run("translate([10,0,0]) scale([2,2,2]) cube(1, center=true);")[0])
        assert bb[0] == approx(9) and bb[3] == approx(11)

    def test_migrated_color_wraps_migrated_transform_wraps_migrated_leaf(self):
        bodies, _ = run('color("blue") translate([1,0,0]) sphere(1);')
        assert len(bodies) == 1
        r, g, b = bodies[0].color[:3]
        assert r == approx(0.0, rel=1) and b == approx(1.0)

    def test_migrated_transform_wraps_migrated_offset_wraps_migrated_extrude(self):
        # offset() was migrated in step 3, linear_extrude in step 5 — all
        # three kinds in this nesting are now migrated.
        bb = bbox(run("linear_extrude(height=1) translate([5,0]) offset(r=1) square(2);")[0])
        assert bb[3] - bb[0] == approx(4)  # 2x2 square offset(r=1) -> 4x4, translate doesn't change extent
        assert bb[0] == approx(4) and bb[3] == approx(8)

    def test_migrated_color_wraps_migrated_union(self):
        # union() was migrated in step 4 — migrated color wrapping a
        # migrated union of two migrated leaves.
        bodies, _ = run('color("red") union() { cube(1); translate([3,0,0]) cube(1); }')
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[3] - bb[0] == approx(4)
        assert bodies[0].color[:3][0] == approx(1.0)

    def test_flatten_matches_evaluate_result_with_transforms_and_color(self):
        for src in [
            "translate([1,0,0]) cube(1);",
            "rotate([0,0,45]) cube(1);",
            "scale([2,1,1]) sphere(1);",
            "mirror([1,0,0]) cube(1);",
            'color("green") cylinder(h=2, r=1);',
            "resize([2,2,2]) sphere(1);",
        ]:
            bodies, _, ev = run_tree(src)
            assert flatten_csg_tree(ev.csg_tree) == bodies


# ---------------------------------------------------------------------------
# CSG tree Phase 2, step 3 — resolve/generate split for topology (hull,
# minkowski, projection, offset).
# ---------------------------------------------------------------------------

class TestCSGTreeStep3Topology:
    def test_topology_kinds_registered(self):
        _, _, ev = run_tree("cube(1);")
        for kind in ("hull", "minkowski", "offset", "projection"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_hull_minkowski_params_empty(self):
        # hull()/minkowski() take no arguments — only children matter.
        _, _, ev = run_tree("hull() { cube(1); sphere(1); }")
        assert ev.csg_tree[0].params == {}
        _, _, ev = run_tree("minkowski() { cube(1); sphere(1); }")
        assert ev.csg_tree[0].params == {}

    def test_offset_params_shape(self):
        _, _, ev = run_tree("offset(r=2) square(4);")
        params = ev.csg_tree[0].params
        assert params["r"] == 2
        assert params["delta"] is None
        assert params["segs"] is not None

        _, _, ev = run_tree("offset(delta=1, chamfer=true) square(4);")
        params = ev.csg_tree[0].params
        assert params["delta"] == 1
        assert params["chamfer"] is True
        assert params["segs"] is None

    def test_projection_params_shape(self):
        _, _, ev = run_tree("projection(cut=true) cube(2);")
        assert ev.csg_tree[0].params == {"cut": True}
        _, _, ev = run_tree("projection() cube(2);")
        assert ev.csg_tree[0].params == {"cut": False}

    def test_migrated_hull_wraps_migrated_leaves(self):
        bb = bbox(run("hull() { cube(1); translate([5,0,0]) cube(1); }")[0])
        assert bb[3] - bb[0] == approx(6)

    def test_migrated_offset_wraps_migrated_union(self):
        # union() was migrated in step 4 — migrated offset wrapping a
        # migrated union of two migrated 2D leaves.
        src = "linear_extrude(height=1) offset(r=1) union() { square(2); translate([3,0]) square(2); }"
        bb = bbox(run(src)[0])
        assert bb[0] == approx(-1) and bb[3] == approx(6)

    def test_migrated_projection_wraps_migrated_union(self):
        src = "linear_extrude(height=1) projection() union() { cube(2); translate([3,0,0]) cube(2); }"
        bodies = run(src)[0]
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[0] == approx(0) and bb[3] == approx(5)

    def test_flatten_matches_evaluate_result_with_topology(self):
        for src in [
            "hull() { cube(1); translate([3,0,0]) sphere(1); }",
            "minkowski() { cube(1); sphere(0.2); }",
            "offset(r=1) square(2);",
            "projection() cube(2);",
        ]:
            bodies, _, ev = run_tree(src)
            assert flatten_csg_tree(ev.csg_tree) == bodies


# ---------------------------------------------------------------------------
# CSG tree Phase 2, step 4 — resolve/generate split for booleans (union/
# difference/intersection) and intersection_for. The genuinely tricky step:
# these do per-statement (or per-iteration) grouping, and for/if/let are
# transparent in the tree (Phase 1), so a single top-level statement can
# contribute a variable, unmarked number of tree children — group_sizes
# bookkeeping (measuring self._tree_stack[-1] length deltas) recovers the
# grouping without needing to inspect AST structure.
# ---------------------------------------------------------------------------

class TestCSGTreeStep4Booleans:
    def test_boolean_kinds_registered(self):
        _, _, ev = run_tree("cube(1);")
        for kind in ("union", "difference", "intersection", "intersection_for"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_union_params_shape_one_group_per_statement(self):
        _, _, ev = run_tree("union() { cube(1); sphere(1); translate([3,0,0]) cube(1); }")
        node = ev.csg_tree[0]
        assert node.kind == "union"
        assert node.params["op"] == "union"
        assert node.params["group_sizes"] == [1, 1, 1]
        assert len(node.children) == 3

    def test_intersection_for_params_shape(self):
        _, _, ev = run_tree("intersection_for(i=[0:2]) rotate([0,0,i*60]) cube([10,2,10], center=true);")
        node = ev.csg_tree[0]
        assert node.kind == "intersection_for"
        assert node.params["group_sizes"] == [1, 1, 1]

    def test_for_nested_in_union_contributes_one_group_of_many(self):
        # A single top-level `for` statement inside union() must count as
        # ONE group in group_sizes, even though it contributes 3 sibling
        # tree children (for is transparent — Phase 1) — otherwise the
        # union's per-statement grouping would misinterpret the 3 spheres
        # as 3 separate top-level statements instead of 1.
        src = "union() { cube(1); for (i=[0:2]) translate([2+i*2,0,0]) sphere(0.5); }"
        bodies, _, ev = run_tree(src)
        node = ev.csg_tree[0]
        assert node.params["group_sizes"] == [1, 3]
        assert len(node.children) == 4  # cube + 3 spheres, flattened in the tree
        bb = bbox(bodies)
        assert bb[0] == approx(0) and bb[3] == approx(6.5)

    def test_if_else_nested_in_difference_contributes_one_group(self):
        src = ("difference() { cube(4, center=true); "
               "if (true) { translate([1,0,0]) sphere(0.5); } else { sphere(2); } }")
        bodies, _, ev = run_tree(src)
        node = ev.csg_tree[0]
        assert node.params["group_sizes"] == [1, 1]  # true-branch: 1 sphere (not 2 — else never ran)
        bb = bbox(bodies)
        # Carving a small sphere out of the cube doesn't change the outer extent.
        assert bb[0] == approx(-2) and bb[3] == approx(2)

    def test_let_nested_in_intersection_contributes_one_group(self):
        src = "intersection() { cube(4, center=true); let(r=1.5) sphere(r); }"
        bodies, _, ev = run_tree(src)
        node = ev.csg_tree[0]
        assert node.params["group_sizes"] == [1, 1]
        assert len(bodies) == 1
        assert bodies[0].body.volume() < 64  # strictly smaller than the 4^3 cube

    def test_intersection_for_iteration_with_multiple_statements(self):
        # Each iteration's body can itself contain a variable number of
        # geometry statements (here via if/else) — group_sizes must track
        # per-ITERATION size, not assume 1 tree child per iteration.
        src = ("intersection_for(i=[0:1]) { "
               "if (i==0) { cube(3, center=true); } else { cube(2, center=true); } }")
        bodies, _, ev = run_tree(src)
        node = ev.csg_tree[0]
        assert node.params["group_sizes"] == [1, 1]
        bb = bbox(bodies)
        # Intersection of cube(3) and cube(2), both centered -> cube(2)'s extent.
        assert bb[0] == approx(-1) and bb[3] == approx(1)

    def test_intersection_empty_operand_anywhere_discards_whole_result(self):
        # intersection(A, ∅, B) = ∅ regardless of position — an empty
        # operand nullifies the whole result, even one already established
        # from prior non-empty statements.
        bodies = run("intersection() { cube(3); for (i = [1:0]) cube(10); cube(2); }")[0]
        assert bodies == []

    def test_difference_later_empty_operand_just_skipped_not_discarded(self):
        # Only an empty FIRST (positive) operand empties a difference — a
        # later empty operand just subtracts nothing and is skipped.
        bodies = run("difference() { cube(3); *cube(10); }")[0]
        assert len(bodies) == 1
        assert bodies[0].body.volume() == approx(27)

    def test_difference_first_empty_operand_discards_whole_result(self):
        bodies = run("difference() { for (i = [1:0]) cube(3); cube(2); }")[0]
        assert bodies == []

    def test_union_skips_disabled_middle_statement(self):
        bodies = run("union() { cube(1); *cube(10); translate([3,0,0]) cube(1); }")[0]
        assert len(bodies) == 1
        assert bodies[0].body.volume() == approx(2)

    def test_migrated_union_wraps_migrated_linear_extrude(self):
        # linear_extrude() was migrated in step 5 — migrated union wrapping
        # a migrated extrude. If linear_extrude silently returned no bodies,
        # union's per-statement grouping would drop it.
        bodies = run("union() { cube(1); translate([3,0,0]) linear_extrude(height=2) square(1); }")[0]
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[0] == approx(0) and bb[3] == approx(4)

    def test_flatten_matches_evaluate_result_with_booleans(self):
        for src in [
            "union() { cube(1); sphere(1); }",
            "difference() { cube([4,4,4]); cube([2,2,2]); }",
            "intersection() { cube(3, center=true); sphere(2); }",
            "union() { cube(1); for (i=[0:2]) translate([2+i*2,0,0]) sphere(0.5); }",
            "intersection_for(i=[0:2]) rotate([0,0,i*60]) cube([10,2,10], center=true);",
        ]:
            bodies, _, ev = run_tree(src)
            assert flatten_csg_tree(ev.csg_tree) == bodies


# ---------------------------------------------------------------------------
# CSG tree — Phase 2 step 5: extrusion + surface + import
# ---------------------------------------------------------------------------

_UNIT_CUBE_OBJ = """\
v 0.0 0.0 0.0
v 0.0 0.0 1.0
v 0.0 1.0 0.0
v 0.0 1.0 1.0
v 1.0 0.0 0.0
v 1.0 0.0 1.0
v 1.0 1.0 0.0
v 1.0 1.0 1.0
f 2 1 5
f 3 5 1
f 2 4 1
f 4 2 6
f 4 3 1
f 4 8 3
f 6 5 7
f 6 2 5
f 7 5 3
f 8 7 3
f 8 4 6
f 8 6 7
"""

# Same unit-cube topology as _UNIT_CUBE_OBJ (0-indexed, already validated
# there: volume 1, bbox [0,1]^3) -- reused to build STL/OFF/3MF fixtures for
# the other import() loaders so all four formats are tested against one
# known-good mesh instead of separately hand-transcribed (and separately
# possibly-wrong) coordinates.
_UNIT_CUBE_VERTS = [
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
]
_UNIT_CUBE_TRIS = [
    (1, 0, 4), (2, 4, 0), (1, 3, 0), (3, 1, 5), (3, 2, 0), (3, 7, 2),
    (5, 4, 6), (5, 1, 4), (6, 4, 2), (7, 6, 2), (7, 3, 5), (7, 5, 6),
]


def _unit_cube_stl_ascii() -> str:
    lines = ["solid cube"]
    for a, b, c in _UNIT_CUBE_TRIS:
        lines.append("facet normal 0 0 0")
        lines.append("outer loop")
        for i in (a, b, c):
            x, y, z = _UNIT_CUBE_VERTS[i]
            lines.append(f"vertex {x} {y} {z}")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid cube")
    return "\n".join(lines) + "\n"


def _unit_cube_stl_binary() -> bytes:
    import struct

    buf = bytearray(80)  # header (no "facet normal" text -> binary detection)
    buf += struct.pack("<I", len(_UNIT_CUBE_TRIS))
    for a, b, c in _UNIT_CUBE_TRIS:
        buf += struct.pack("<3f", 0.0, 0.0, 0.0)  # normal (unused by the loader)
        for i in (a, b, c):
            buf += struct.pack("<3f", *_UNIT_CUBE_VERTS[i])
        buf += struct.pack("<H", 0)  # attribute byte count
    return bytes(buf)


def _unit_cube_off() -> str:
    lines = ["OFF", f"{len(_UNIT_CUBE_VERTS)} {len(_UNIT_CUBE_TRIS)} 0"]
    lines += [f"{x} {y} {z}" for x, y, z in _UNIT_CUBE_VERTS]
    lines += [f"3 {a} {b} {c}" for a, b, c in _UNIT_CUBE_TRIS]
    return "\n".join(lines) + "\n"


def _unit_cube_3mf_bytes() -> bytes:
    import io
    import zipfile

    NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    verts_xml = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in _UNIT_CUBE_VERTS)
    tris_xml = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in _UNIT_CUBE_TRIS)
    model = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<model xmlns="{NS}"><resources>'
        f'<object id="1" type="model"><mesh>'
        f'<vertices>{verts_xml}</vertices>'
        f'<triangles>{tris_xml}</triangles>'
        f'</mesh></object></resources>'
        f'<build><item objectid="1"/></build></model>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


class TestMultiColorCSGMerge:
    """A real boolean CSG merge (_generate_csg) used to collapse every
    child's color into just the first child's -- e.g. union()-ing an
    opaque cube with a translucent sphere dropped the sphere's color and
    alpha entirely, rendering the whole result fully opaque. Real OpenSCAD
    preserves each part's own color/alpha through the merge (verified
    against real OpenSCAD 2022.08.22's interactive 3D view for this exact
    scenario). Evaluator._attach_tri_colors recovers this via manifold3d's
    per-triangle run_original_id/run_index provenance (already used for
    id_to_node/WYSIWYG ray-cast picking) plus a parallel Evaluator.id_to_color
    map populated by _tag/_tag_generated."""

    def test_union_mixed_colors_sets_tri_colors(self):
        src = """
        union() {
            color("lightgreen") cube(10);
            color([0,1,1,0.5]) translate([5,5,10]) sphere(d=10);
        }
        """
        bodies, _, ev = run_tree(src)
        assert len(bodies) == 1
        tc = bodies[0].tri_colors
        assert tc is not None
        distinct = np.unique(tc, axis=0)
        assert len(distinct) == 2
        assert any(np.allclose(row, (0.0, 1.0, 1.0, 0.5)) for row in distinct)  # sphere
        assert any(row[3] == 1.0 and not np.allclose(row, (0.0, 1.0, 1.0, 0.5))
                   for row in distinct)  # lightgreen cube, opaque

    def test_difference_cut_face_gets_the_cut_green(self):
        # The cylinder tool has no explicit color() -- its newly-exposed cut
        # face takes OpenSCAD's cut green (#9DCB51), which its preview paints
        # and its 3MF export keeps (checked on 2026.02.01), not the default
        # geometry colour, so a cut reads as a cut (cpp #173).
        src = """
        difference() {
            union() {
                color("lightgreen") cube(10);
                color([0,1,1,0.5]) translate([5,5,10]) sphere(d=10);
            }
            translate([5,5,-0.01]) cylinder(h=10.02, d=8);
        }
        """
        bodies, _, ev = run_tree(src)
        assert len(bodies) == 1
        tc = bodies[0].tri_colors
        assert tc is not None
        distinct = np.unique(tc, axis=0)
        assert len(distinct) == 3
        assert any(np.allclose(row, (0.0, 1.0, 1.0, 0.5)) for row in distinct)
        assert any(np.allclose(row, (157 / 255, 203 / 255, 81 / 255, 1.0)) for row in distinct)

    def test_union_same_explicit_color_leaves_tri_colors_none(self):
        # Cheap-path guarantee: if every contributing color resolves to the
        # same value, tri_colors must stay None (same single-buffer,
        # live-theme-following upload path as before this feature existed).
        src = """
        union() {
            color("red") cube(10);
            color("red") translate([5,5,10]) sphere(d=10);
        }
        """
        bodies, _, ev = run_tree(src)
        assert bodies[0].tri_colors is None

    def test_union_no_explicit_color_leaves_tri_colors_none(self):
        src = "union() { cube(10); translate([5,5,10]) sphere(d=10); }"
        bodies, _, ev = run_tree(src)
        assert bodies[0].tri_colors is None

    def test_id_to_color_populated_per_primitive(self):
        src = 'color("red") cube(1); color([0,0,1,0.4]) translate([3,0,0]) sphere(1);'
        _, _, ev = run_tree(src)
        colors = list(ev.id_to_color.values())
        assert any(c is not None and c[3] == 0.4 for c in colors)

    def test_single_child_union_no_merge_leaves_tri_colors_none(self):
        # No real boolean merge (union of ONE child) -- _generate_csg never
        # even reaches the multi-child branch, so tri_colors must stay
        # unset regardless of color.
        bodies, _, ev = run_tree('union() { color([1,0,0,0.5]) cube(1); }')
        assert bodies[0].tri_colors is None


class TestCSGTreeStep5Extrusion:
    def test_extrusion_kinds_registered(self):
        _, _, ev = run_tree("cube(1);")
        for kind in ("linear_extrude", "rotate_extrude", "roof", "surface", "import"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_linear_extrude_params_shape(self):
        _, _, ev = run_tree(
            "linear_extrude(height=5, center=true, twist=90, slices=10, scale=2) square(1);")
        params = ev.csg_tree[0].params
        assert params["height"] == approx(5)
        assert params["center"] is True
        assert params["twist"] == approx(90)
        assert params["slices"] == 10
        assert params["scale_top"] == (approx(2), approx(2))
        assert "color" in params

    def test_linear_extrude_geometry_centered(self):
        bodies = run("linear_extrude(height=5, center=true) square([2,3]);")[0]
        bb = bbox(bodies)
        assert bb[2] == approx(-2.5) and bb[5] == approx(2.5)
        assert bodies[0].body.volume() == approx(30)

    def test_rotate_extrude_params_shape_caches_fn_fa_fs(self):
        _, _, ev = run_tree("rotate_extrude($fn=32, $fa=5, $fs=1) translate([5,0]) square([2,3]);")
        params = ev.csg_tree[0].params
        assert params["angle"] == approx(360)
        assert params["fn"] == approx(32)
        assert params["fa"] == approx(5)
        assert params["fs"] == approx(1)
        assert "color" in params

    def test_rotate_extrude_segment_count_depends_on_children_bounds(self):
        # segs is computed from cs.bounds() at generate time — bounds don't
        # exist until the 2D children are generated, so this can't be
        # precomputed in resolve the way e.g. offset's segs can. A regular
        # 32-gon revolve of a shape spanning x=[5,7] hits exactly +-7 on
        # both axes (vertices land on the 0/90/180/270 degree marks).
        bodies = run("rotate_extrude($fn=32) translate([5,0]) square([2,3]);")[0]
        bb = bbox(bodies)
        assert bb[0] == approx(-7) and bb[3] == approx(7)
        assert bb[1] == approx(-7) and bb[4] == approx(7)

    def test_roof_params_shape_and_bad_method_warning(self):
        _, echo, ev = run_tree('roof(method="bogus") square(10, center=true);')
        assert ev.csg_tree[0].params["method"] == "voronoi"
        assert any("Unknown roof method" in line for line in echo)

    def test_roof_straight_skeleton_peak_height(self):
        bodies = run('roof(method="straight") square([10,10], center=true);')[0]
        bb = bbox(bodies)
        assert bb[5] == approx(5, rel=1e-2)  # square roof peaks at half the min side

    def test_surface_params_caches_parsed_heights(self, tmp_path):
        dat = tmp_path / "heights.dat"
        dat.write_text("0 0 0\n0 5 0\n0 0 0\n")
        _, _, ev = run_tree(f'surface(file="{dat}");')
        params = ev.csg_tree[0].params
        assert params["heights"] == [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]
        assert params["center"] is False

    def test_surface_geometry_from_dat_file(self, tmp_path):
        dat = tmp_path / "heights.dat"
        dat.write_text("0 0 0\n0 5 0\n0 0 0\n")
        bodies = run(f'surface(file="{dat}");')[0]
        bb = bbox(bodies)
        assert bb[3] == approx(2) and bb[4] == approx(2)  # 3x3 grid -> 2x2 footprint
        assert bb[5] == approx(5)

    def test_surface_missing_file_param_raises(self):
        with pytest.raises(EvalError):
            run("surface();")

    def test_import_obj_params_shape_caches_verts_tris(self, tmp_path):
        obj = tmp_path / "cube.obj"
        obj.write_text(_UNIT_CUBE_OBJ)
        _, _, ev = run_tree(f'import("{obj}");')
        params = ev.csg_tree[0].params
        assert params["kind"] == "mesh"
        assert len(params["verts"]) == 8
        assert len(params["tris"]) == 12

    def test_import_obj_geometry_matches_unit_cube(self, tmp_path):
        obj = tmp_path / "cube.obj"
        obj.write_text(_UNIT_CUBE_OBJ)
        bodies = run(f'import("{obj}");')[0]
        assert bodies[0].body.volume() == approx(1)
        bb = bbox(bodies)
        assert bb == (approx(0), approx(0), approx(0), approx(1), approx(1), approx(1))

    def test_import_stl_params_shape_caches_verts_tris(self, tmp_path):
        stl = tmp_path / "cube.stl"
        stl.write_text(_unit_cube_stl_ascii())
        _, _, ev = run_tree(f'import("{stl}");')
        params = ev.csg_tree[0].params
        assert params["kind"] == "mesh"
        # STL has no vertex-index concept -- each triangle carries its own
        # private copy of its 3 corner positions -- so a naive load would
        # produce 36 (12 tris * 3) unwelded verts. _weld_stl_vertices merges
        # coincident positions back down to the cube's actual 8 corners;
        # without that, manifold3d rejects the mesh as NotManifold (see
        # test_import_stl_geometry_matches_unit_cube).
        assert len(params["verts"]) == 8
        assert len(params["tris"]) == 12

    def test_import_stl_geometry_matches_unit_cube(self, tmp_path):
        stl = tmp_path / "cube.stl"
        stl.write_text(_unit_cube_stl_ascii())
        bodies = run(f'import("{stl}");')[0]
        assert bodies[0].body.volume() == approx(1)
        bb = bbox(bodies)
        assert bb == (approx(0), approx(0), approx(0), approx(1), approx(1), approx(1))

    def test_import_stl_binary_geometry_matches_unit_cube(self, tmp_path):
        # Binary STL is a separate code path in _load_stl from ASCII (no
        # shared vertex indices there either -- same welding fix applies).
        stl = tmp_path / "cube_bin.stl"
        stl.write_bytes(_unit_cube_stl_binary())
        bodies = run(f'import("{stl}");')[0]
        assert bodies[0].body.volume() == approx(1)
        bb = bbox(bodies)
        assert bb == (approx(0), approx(0), approx(0), approx(1), approx(1), approx(1))

    def test_import_off_params_shape_caches_verts_tris(self, tmp_path):
        off = tmp_path / "cube.off"
        off.write_text(_unit_cube_off())
        _, _, ev = run_tree(f'import("{off}");')
        params = ev.csg_tree[0].params
        assert params["kind"] == "mesh"
        assert len(params["verts"]) == 8
        assert len(params["tris"]) == 12

    def test_import_off_geometry_matches_unit_cube(self, tmp_path):
        off = tmp_path / "cube.off"
        off.write_text(_unit_cube_off())
        bodies = run(f'import("{off}");')[0]
        assert bodies[0].body.volume() == approx(1)
        bb = bbox(bodies)
        assert bb == (approx(0), approx(0), approx(0), approx(1), approx(1), approx(1))

    def test_import_3mf_params_shape_caches_verts_tris(self, tmp_path):
        threemf = tmp_path / "cube.3mf"
        threemf.write_bytes(_unit_cube_3mf_bytes())
        _, _, ev = run_tree(f'import("{threemf}");')
        params = ev.csg_tree[0].params
        assert params["kind"] == "mesh"
        assert len(params["verts"]) == 8
        assert len(params["tris"]) == 12

    def test_import_3mf_geometry_matches_unit_cube(self, tmp_path):
        threemf = tmp_path / "cube.3mf"
        threemf.write_bytes(_unit_cube_3mf_bytes())
        bodies = run(f'import("{threemf}");')[0]
        assert bodies[0].body.volume() == approx(1)
        bb = bbox(bodies)
        assert bb == (approx(0), approx(0), approx(0), approx(1), approx(1), approx(1))

    def test_import_svg_geometry_rect(self, tmp_path):
        svg = tmp_path / "rect.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<rect x="0" y="0" width="10" height="20"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        assert bodies[0].section.area() == approx(200)

    def test_import_svg_geometry_polygon(self, tmp_path):
        svg = tmp_path / "poly.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<polygon points="0,0 10,0 10,10 0,10"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_svg_geometry_polyline(self, tmp_path):
        # Regression: _shape_contours' polygon/polyline branch used to
        # gate its return on `tag == "polygon"`, so <polyline> always
        # returned [] regardless of its own points -- a silent no-op
        # despite being documented as supported (docs/evaluator.md).
        svg = tmp_path / "polyline.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<polyline points="0,0 10,0 10,10 0,10"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_svg_geometry_circle(self, tmp_path):
        svg = tmp_path / "circle.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<circle cx="0" cy="0" r="10"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        # 32-segment polygon approximation of a r=10 circle, not pi*r^2 exactly
        import math
        assert bodies[0].section.area() == approx(math.pi * 100, rel=0.01)

    def test_import_svg_geometry_path_rect_via_line_commands(self, tmp_path):
        svg = tmp_path / "path.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<path d="M0,0 L10,0 L10,10 L0,10 Z"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_svg_geometry_path_cubic_bezier(self, tmp_path):
        # A cubic Bezier whose control points all coincide with the
        # endpoints degenerates to a straight line -- flattening a
        # "curved" rectangle edge this way should reproduce the same
        # rectangle as the plain-line-command version above.
        svg = tmp_path / "path_bezier.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<path d="M0,0 C0,0 10,0 10,0 L10,10 L0,10 Z"/></svg>')
        bodies = run(f'import("{svg}");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_svg_y_axis_flipped(self, tmp_path):
        # SVG's y-axis points down; OpenSCAD's points up -- _apply flips it.
        svg = tmp_path / "flip.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<polygon points="0,0 10,0 10,5 0,5"/></svg>')
        _, _, ev = run_tree(f'import("{svg}");')
        contour = ev.csg_tree[0].params["contours"][0]
        ys = [pt[1] for pt in contour]
        assert min(ys) == approx(-5) and max(ys) == approx(0)

    def test_import_svg_transform_translate_applied(self, tmp_path):
        svg = tmp_path / "translated.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<rect x="0" y="0" width="10" height="10" transform="translate(5,5)"/></svg>')
        _, _, ev = run_tree(f'import("{svg}");')
        contour = ev.csg_tree[0].params["contours"][0]
        xs = [pt[0] for pt in contour]
        assert min(xs) == approx(5) and max(xs) == approx(15)

    def test_import_svg_defs_contents_skipped(self, tmp_path):
        svg = tmp_path / "defs.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<defs><rect x="0" y="0" width="10" height="10"/></defs>'
                        '<circle cx="0" cy="0" r="1"/></svg>')
        _, _, ev = run_tree(f'import("{svg}");')
        # Only the circle's contour should appear -- the rect inside <defs>
        # must not be traversed into geometry.
        assert len(ev.csg_tree[0].params["contours"]) == 1

    def test_import_svg_no_shapes_raises(self, tmp_path):
        svg = tmp_path / "empty.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        with pytest.raises(EvalError):
            run(f'import("{svg}");')

    def test_import_dxf_geometry_closed_lwpolyline(self, tmp_path):
        import ezdxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True)
        dxf = tmp_path / "square.dxf"
        doc.saveas(dxf)
        bodies = run(f'import("{dxf}");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_dxf_geometry_closed_polyline_legacy_entity(self, tmp_path):
        import ezdxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_polyline2d([(0, 0), (3, 0), (3, 3), (0, 3)], close=True)
        dxf = tmp_path / "legacy.dxf"
        doc.saveas(dxf)
        bodies = run(f'import("{dxf}");')[0]
        assert bodies[0].section.area() == approx(9)

    def test_import_dxf_open_polyline_excluded(self, tmp_path):
        # Only closed polylines become fill contours -- an open one
        # can't bound an area.
        import ezdxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_lwpolyline([(0, 0), (5, 0), (5, 5)], close=False)
        dxf = tmp_path / "open.dxf"
        doc.saveas(dxf)
        with pytest.raises(EvalError):
            run(f'import("{dxf}");')

    def test_import_dxf_layer_filter(self, tmp_path):
        import ezdxf
        doc = ezdxf.new()
        doc.layers.add("SHAPES")
        msp = doc.modelspace()
        msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True,
                            dxfattribs={"layer": "SHAPES"})
        msp.add_lwpolyline([(0, 0), (2, 0), (2, 2), (0, 2)], close=True,
                            dxfattribs={"layer": "OTHER"})
        dxf = tmp_path / "layers.dxf"
        doc.saveas(dxf)
        bodies = run(f'import(file="{dxf}", layer="SHAPES");')[0]
        assert bodies[0].section.area() == approx(100)

    def test_import_dxf_missing_ezdxf_gives_clear_error(self, tmp_path, monkeypatch):
        import sys
        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ezdxf":
                raise ImportError("simulated missing ezdxf")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "ezdxf", raising=False)
        monkeypatch.setattr(builtins, "__import__", blocked_import)
        dxf = tmp_path / "whatever.dxf"
        dxf.write_text("0\nEOF\n")
        with pytest.raises(EvalError, match="ezdxf"):
            run(f'import("{dxf}");')

    def test_import_unsupported_extension_raises(self):
        with pytest.raises(EvalError):
            run('import("nonexistent.xyz");')

    def test_import_json_as_geometry_statement_raises(self, tmp_path):
        # import() as a geometry statement (ModularCall) dispatches by
        # extension like every other format -- .json is data, not
        # geometry, and errors rather than silently producing nothing.
        j = tmp_path / "data.json"
        j.write_text('{"a": 1}')
        with pytest.raises(EvalError, match="use as an expression"):
            run(f'import("{j}");')

    def test_import_json_as_expression_converts_to_osc_values(self, tmp_path):
        # import() used as an expression (PrimaryCall, not ModularCall)
        # routes through a separate code path (_import_as_value ->
        # _json_to_osc): JSON objects -> OscObject, arrays -> lists
        # (recursively converted), scalars/null pass through natively
        # (null -> Python None -> OpenSCAD undef, same sentinel used
        # everywhere else in the evaluator).
        import json
        data = {
            "name": "widget", "count": 3, "size": 2.5, "active": True,
            "note": None, "tags": ["a", "b"], "nested": {"x": 1, "y": 2},
        }
        j = tmp_path / "data.json"
        j.write_text(json.dumps(data))
        src = (
            f'o = import("{j}");'
            'echo(o.name, o.count, o.size, o.active, o.note, o.tags, o.nested.x, o.nested);'
        )
        _, echoes = run(src)
        assert echoes == ['ECHO: "widget", 3, 2.5, true, undef, ["a", "b"], 1, { x = 1; y = 2; }']

    def test_migrated_transform_wraps_migrated_rotate_extrude(self):
        bodies = run("translate([1,0,0]) rotate_extrude($fn=16) translate([5,0]) circle(1);")[0]
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb == (approx(-5), approx(-6), approx(-1), approx(7), approx(6), approx(1))

    def test_migrated_union_wraps_migrated_roof(self):
        # migrated union wrapping a migrated roof (and a migrated cube). If
        # roof silently returned no bodies, union's per-statement grouping
        # would drop it and the combined bbox would only cover the cube.
        bodies = run(
            "union() { cube([1,1,1]); translate([20,0,0]) roof() square(4, center=true); }")[0]
        assert len(bodies) == 1
        bb = bbox(bodies)
        assert bb[3] == approx(22)  # 20 + half of the 4-wide roof footprint

    def test_flatten_matches_evaluate_result_with_extrusion(self):
        for src in [
            "linear_extrude(height=3) circle(2);",
            "rotate_extrude() translate([3,0]) circle(1);",
            'roof(method="straight") square(6, center=true);',
        ]:
            bodies, _, ev = run_tree(src)
            assert flatten_csg_tree(ev.csg_tree) == bodies


# ---------------------------------------------------------------------------
# CSG tree Phase 2, step 6 — final cutover: evaluate() is now genuinely
# two-pass. Resolve (the AST walk) builds the whole tree as plain data with
# no Manifold calls at all; generate_tree() is a separate bottom-up pass
# that does all the Manifold/CrossSection work. This also required giving
# the #/%/! tag modifiers and render()/children()/breakpoint() their own
# resolve_fn (previously handled eagerly inline in _eval_statement_impl),
# and removing _resolve_csg's resolve-time short-circuit (which relied on
# already-generated child bodies that no longer exist until generate_tree()
# runs) in favor of _generate_csg deciding discard-vs-skip purely from real
# generated bodies.
# ---------------------------------------------------------------------------

class TestCSGTreeStep6FinalCutover:
    def _resolve_only(self, src: str):
        """Run just the resolve pass (build csg_tree) without calling
        generate_tree(), to inspect the tree before any Manifold work has
        happened."""
        nodes = getASTfromString(src, include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev._resolve_use_statements(nodes, root_scope)
        ev.csg_tree = []
        ev._tree_stack = [ev.csg_tree]
        ctx = EvalContext(scope=root_scope)
        ev._root_ctx = ctx
        for node in nodes:
            ev._eval_statement(node, ctx)
        return ev

    def test_resolve_alone_leaves_bodies_empty(self):
        ev = self._resolve_only("cube(1); sphere(1);")
        assert [n.bodies for n in ev.csg_tree] == [[], []]

    def test_generate_tree_populates_bodies_after_resolve(self):
        ev = self._resolve_only("cube(1); sphere(1);")
        result = ev.generate_tree(ev.csg_tree)
        assert len(result) == 2
        assert all(n.bodies for n in ev.csg_tree)

    def test_generate_tree_works_on_partial_tree(self):
        # Simulates a debugger breakpoint mid-walk (Phase 3): resolve only
        # the first two of three top-level statements, then generate_tree()
        # just the partial tree built so far.
        nodes = getASTfromString(
            "cube(1); sphere(1); translate([5,0,0]) cube(2);", include_comments=False)
        root_scope = build_scopes(nodes)
        ev = Evaluator()
        ev._resolve_use_statements(nodes, root_scope)
        ev.csg_tree = []
        ev._tree_stack = [ev.csg_tree]
        ctx = EvalContext(scope=root_scope)
        ev._root_ctx = ctx
        ev._eval_statement(nodes[0], ctx)
        ev._eval_statement(nodes[1], ctx)
        partial = ev.generate_tree(ev.csg_tree)
        assert [n.kind for n in ev.csg_tree] == ["cube", "sphere"]
        assert len(partial) == 2

    def test_modifier_kinds_registered_with_generate_fn(self):
        _, _, ev = run_tree("cube(1);")
        for kind in ("highlight", "background", "show_only"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind in ev._GENERATE_DISPATCH

    def test_render_children_breakpoint_use_default_concatenation(self):
        # These have a resolve_fn (to build the tree correctly) but no
        # generate_fn — generate_tree()'s default (concatenate children's
        # bodies) is what reproduces their old passthrough behavior.
        _, _, ev = run_tree("cube(1);")
        for kind in ("render", "children", "breakpoint"):
            assert kind in ev._RESOLVE_DISPATCH
            assert kind not in ev._GENERATE_DISPATCH

    def test_highlight_background_show_only_tree_and_roles(self):
        bodies, _, ev = run_tree("#cube(1); %sphere(1); cube(1);")
        assert [n.kind for n in ev.csg_tree] == ["highlight", "background", "cube"]
        assert [c.kind for c in ev.csg_tree[0].children] == ["cube"]
        assert [b.role for b in bodies] == ["highlight", "background", "normal"]
        assert flatten_csg_tree(ev.csg_tree) == bodies

    def test_user_module_call_tree_splices_body_and_concatenates(self):
        bodies, _, ev = run_tree(
            "module wrap() { translate([1,0,0]) cube(2); } wrap();")
        node = ev.csg_tree[0]
        assert node.kind == "translate"
        assert flatten_csg_tree(ev.csg_tree) == bodies

    def test_module_shadowing_builtin_name_still_dispatches_to_user_module(self):
        bodies = run("module render() { cube(3); } render();")[0]
        assert bodies[0].body.volume() == approx(27)

    def test_unknown_module_warns_and_produces_no_geometry(self):
        bodies, echo = run("foobar_totally_unknown(1, 2, 3);")
        assert bodies == []
        assert any("Ignoring unknown module 'foobar_totally_unknown'" in line for line in echo)

    def test_side_effect_after_would_be_short_circuited_statement_still_fires(self):
        # Final-cutover behavior change (intentional): since resolve can no
        # longer tell a statement's geometry will end up empty (that's only
        # knowable once real bodies exist, in generate_tree()), every
        # statement is always resolved — so echo() after an empty loop
        # (an empty operand) still fires, even inside an
        # intersection() whose combined geometry result is discarded to ∅
        # by the first empty operand.
        bodies, echo = run('intersection() { for (i = [1:1:0]) cube(10); echo("fired"); cube(2); }')
        assert bodies == []
        assert echo == ['ECHO: "fired"']

    def test_flatten_matches_evaluate_result_with_modifiers_and_modules(self):
        for src in [
            "#cube(1); %sphere(1); cube(1);",
            "module wrap() { translate([1,0,0]) cube(2); } wrap();",
            "render() cube(2);",
            "module m() { children(); } m() cube(1);",
        ]:
            bodies, _, ev = run_tree(src)
            assert flatten_csg_tree(ev.csg_tree) == bodies


# ---------------------------------------------------------------------------
# Silent-wrong-result bugs backported from openscad_cpp_evaluator. Every
# expected value below is what OpenSCAD 2026.02.01 produces for the same source.
# ---------------------------------------------------------------------------

def _vol(bodies):
    return sum(b.body.volume() for b in bodies if b.body is not None)


class TestBackportedSilentBugs:
    def test_escaping_closure_keeps_its_captures(self):
        _, lines = run("function mk(x) = function(y) x + y; echo(mk(10)(5));")
        assert lines == ["ECHO: 15"]

    def test_closures_in_c_style_for_capture_each_iteration(self):
        _, lines = run("fs = [for (i = 0; i < 3; i = i + 1) function() i]; echo([for (f = fs) f()]);")
        assert lines == ["ECHO: [0, 1, 2]"]

    def test_let_bound_recursive_literal_still_works(self):
        _, lines = run("echo(let(f = function(n) n <= 0 ? 0 : n + f(n - 1)) f(5));")
        assert lines == ["ECHO: 15"]

    def test_global_evaluated_once_not_per_read(self):
        _, lines = run("r = rands(0, 1, 1)[0]; function g() = r; echo(g() == g(), g() == r);")
        assert lines == ["ECHO: true, true"]

    def test_user_function_shadows_builtin(self):
        _, lines = run("function sin(x) = 42; echo(sin(0));")
        assert lines == ["ECHO: 42"]

    def test_braced_block_is_its_own_scope(self):
        _, lines = run("q = 1; if (true) { q = 5; b = 2; $fn = 7; echo(q); } "
                       "translate([0, 0, 0]) { t = 9; } "
                       "echo(q, is_undef(b), is_undef(t), $fn);")
        assert lines[0] == "ECHO: 5"
        assert lines[-1] == "ECHO: 1, true, true, 0"
        assert not any("overwritten" in l for l in lines)

    def test_let_statement_is_sequential(self):
        _, lines = run("let(a = 1, b = a + 1) echo(b);")
        assert lines == ["ECHO: 2"]

    def test_later_for_range_sees_earlier_variable(self):
        _, lines = run("echo([for (i = [0:2], j = [0:i]) [i, j]]); for (i = [0:1], j = [0:i]) echo(i, j);")
        assert lines == ["ECHO: [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2]]",
                         "ECHO: 0, 0", "ECHO: 1, 0", "ECHO: 1, 1"]

    def test_intersection_for_range_sees_earlier_variable(self):
        # i=1: j in [0:1] -> cubes at x=0 and x=0.5, intersected -> 0.5 wide
        bodies, _ = run("intersection_for (i = [1:1], j = [0:i]) translate([j / 2, 0, 0]) cube(1);")
        assert _vol(bodies) == approx(0.5)

    def test_forwarded_children_keep_callers_children_count(self):
        _, lines = run("module w() { children(); } module w0() { children(0); } "
                       "module o() { w() echo($children); w0() echo($children); } "
                       "o() { cube(1); cube(1); cube(1); }")
        assert lines == ["ECHO: 3", "ECHO: 3"]

    def test_invalid_operand_does_not_empty_union(self):
        # the open polyhedron (a missing face) is Manifold-invalid
        bodies, _ = run("union() { cube(1); translate([5, 0, 0]) polyhedron("
                        "[[0,0,0],[1,0,0],[0,1,0],[0,0,1]], [[0,1,2],[0,3,1],[0,2,3]]); }")
        assert _vol(bodies) == approx(1)

    @pytest.mark.parametrize("src, vol", [
        ("intersection() { cube(2); *cube(1); }", 8),
        ("intersection() { *cube(10); cube(2); }", 8),
        ("difference() { *cube(10); cube(2); }", 8),
        ("intersection() { cube(2); if (false) cube(1); }", 8),
        ("intersection() { cube(2); echo(\"x\"); }", 8),
        ("intersection() { cube(2); assert(true); }", 8),
        ("intersection() { cube(2); a = 1; }", 8),
        ("intersection() { cube(2); if (true) {} }", 0),
        ("intersection() { cube(2); if (false) cube(1); else {} }", 0),
        ("intersection() { cube(2); for (i = [1:0]) cube(1); }", 0),
        ("intersection() { cube(2); union() {} }", 0),
        ("module e() {} intersection() { cube(2); e(); }", 0),
    ])
    def test_empty_operand_rule(self, src, vol):
        # A statement is an operand only if it built a node: an empty loop,
        # module call or taken `if` annihilates; `*`, echo(), assert() and an
        # untaken `if` are skipped.
        bodies, _ = run(src)
        assert _vol(bodies) == approx(vol)

    def test_echo_with_child_draws_it(self):
        bodies, lines = run('echo("hi") cube(10);')
        assert lines == ['ECHO: "hi"'] and _vol(bodies) == approx(1000)

    @pytest.mark.parametrize("src, r1, r2, zmin", [
        ("cylinder(10, 5, 2);", 5, 2, 0),        # positionals are (h, r1, r2, center)
        ("cylinder(10, 5, 2, true);", 5, 2, -5),
        ("cylinder(h=10, d=4, r=9);", 2, 2, 0),  # d beats r
        ("cylinder(10);", 1, 1, 0),
        ("cylinder(h=10, r1=5);", 5, 1, 0),      # r2 defaults to 1, not to r1
        ("cylinder(30, r=5, true);", 5, 5, 0),   # a non-number in a radius slot is ignored
    ])
    def test_cylinder_arguments(self, src, r1, r2, zmin):
        _, _, ev = run_tree(src)
        p = ev.csg_tree[0].params
        assert (p["r1"], p["r2"]) == (approx(r1), approx(r2))
        assert p["center"] == (zmin != 0)

    @pytest.mark.parametrize("lit, expected", [
        (r'"a\nb"', "a\nb"),
        (r'"\x41"', "A"),
        (r'"\x80"', "x80"),          # \x is ASCII only
        (r'"☺"', "☺"),
        (r'"\U01F600"', "\U0001F600"),
        (r'"\U01F60"', "U01F60"),    # too few digits: the letter stands for itself
        (r'"\u0000"', " "),          # NUL becomes a space
        (r'"t\tx"', "t\tx"),
        (r'"q\"q"', 'q"q'),
        (r'"b\\s"', "b\\s"),
        (r'"\q"', "q"),              # an unknown escape keeps its character
        ('"x\ny"', "xy"),            # a raw line ending contributes nothing
    ])
    def test_string_escapes(self, lit, expected):
        _, _, ev = run_tree(f"s = {lit};")
        assert ev._root_ctx.let["s"] == expected

    def test_2d_union_keeps_each_childs_colour(self):
        bodies, _ = run('union() { color("red") square(5); color("blue") translate([3, 0]) square(5); }')
        colours = {b.color[:3]: b.section.area() for b in bodies}
        # later children paint over earlier ones where they overlap
        assert colours == {(1.0, 0.0, 0.0): approx(15), (0.0, 0.0, 1.0): approx(25)}

    def test_2d_union_of_one_colour_stays_one_shape(self):
        bodies, _ = run("union() { square(5); translate([3, 0]) square(5); }")
        assert len(bodies) == 1 and bodies[0].section.area() == approx(40)

    def test_colour_of_forwarded_children_survives_union(self):
        bodies, _ = run('module setcolor(c) { color(c) children(); } '
                        'union() { setcolor("red") cube(5); color("blue") translate([10, 0, 0]) cube(5); }')
        colours = {tuple(round(float(v), 2) for v in c[:3]) for c in bodies[0].tri_colors}
        assert colours == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}


_OPEN_TETRA = "polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], [[0,1,2],[0,3,1],[0,2,3]])"


class TestFlatPreview:
    """cpp #138: a 1-unit slab, as OpenSCAD's 2D preview is, at the Z a
    translate put it."""

    def test_slab_is_one_unit_and_keeps_its_z(self):
        from openscad_evaluator.evaluator import to_renderable_bodies
        bodies, _ = run("translate([0,0,-1]) square(35); circle(5);")
        boxes = [[round(v, 6) for v in b.body.bounding_box()] for b in to_renderable_bodies(bodies)]
        assert boxes == [[0, 0, -1, 35, 35, 0], [-5, -5, 0, 5, 5, 1]]


class TestLinearSolve:
    """Output identical to openscad_cpp_evaluator's (#120, #122, #125)."""

    def test_square_least_squares_min_norm(self):
        _, lines = run("echo(linear_solve([[2,1],[1,3]], [5,10]));"
                       "echo(linear_solve([[1,1],[1,2],[1,3]], [1,2,2]));"
                       "echo(linear_solve([[1,2,3],[4,5,6]], [1,2]));")
        assert lines == ["ECHO: { x = [1, 3]; det = 5; singular = false; }",
                         "ECHO: { x = [0.666667, 0.5]; det = undef; singular = false; }",
                         "ECHO: { x = [-0.0555556, 0.111111, 0.277778]; det = undef; singular = false; }"]

    def test_singular_and_relative_tolerance(self):
        _, lines = run("echo(linear_solve([[1,2],[2,4]], [1,2])); echo(linear_solve([[2,0],[0,2]]*1e-10, [1,1]).x);")
        assert lines == ["ECHO: { x = undef; det = 0; singular = true; }", "ECHO: [5e+9, 5e+9]"]

    def test_undef_b_is_absent_and_matrix_b(self):
        _, lines = run("echo(linear_solve([[4,3],[6,3]], undef)); echo(linear_solve([[1,2],[3,4]], [[1,0],[0,1]]).x);")
        assert lines == ["ECHO: { x = undef; det = -6; singular = false; }", "ECHO: [[-2, 1], [1.5, -0.5]]"]

    def test_bad_arguments_warn(self):
        _, lines = run('echo(linear_solve(5)); echo(linear_solve([1,2])); echo(linear_solve([[1,2],[3,4]], [1]));')
        assert [l.removesuffix(" in file <string>, line 1") for l in lines if l.startswith("WARNING")] == [
            "WARNING: linear_solve() parameter could not be converted: argument 0: expected vector, found number (5)",
            "WARNING: linear_solve() requires a matrix of numbers",
            "WARNING: linear_solve() right-hand side must be a vector of 2 numbers, or a matrix with that many rows"]


class TestSeparateChildren:
    """children(separate=true): one statement per forwarded child (cpp
    #110-#112). Volumes match openscad_cpp_evaluator's."""

    @staticmethod
    def _vols(src):
        bodies, lines = run(src)
        return [round(b.body.volume(), 2) for b in bodies], lines

    def test_difference_subtracts_each_child(self):
        vols, _ = self._vols("module frame() { difference() children(separate=true); }\n"
                             "frame() { cube(20, center=true); translate([10,0,0]) cube(10, center=true); "
                             "translate([-10,0,0]) cube(10, center=true); }")
        assert vols == [8000 - 2 * 500]

    def test_children_counts_and_indexes_the_spliced_statements(self):
        _, lines = self._vols("module cnt() { echo($children); }\n"
                              "module fwd() cnt() children(separate=true);\n"
                              "fwd() { cube(1); sphere(2); cube(3); }\n"
                              "cnt() { children(separate=true); }")
        assert lines == ["ECHO: 3", "ECHO: 0"]

    def test_a_loop_child_stays_one_operand(self):
        vols, _ = self._vols("module lp() difference() children(separate=true);\n"
                             "lp() { cube(40,center=true); for (i=[-1,1]) translate([i*10,0,0]) cube(8, center=true); }")
        assert vols == [64000 - 2 * 512]

    def test_does_not_leak_through_a_module_boundary(self):
        vols, _ = self._vols("module pass() { children(separate=true); }\n"
                             "difference() pass() { cube(10); translate([5,5,5]) cube(10); }")
        assert vols == [2000 - 125]

    def test_selection_and_empty_selection(self):
        vols, _ = self._vols("module sel() difference() children([0,2], separate=true);\n"
                             "sel() { cube(10); cube(9); translate([5,0,0]) cube(10); }\n"
                             "module none() difference() { cube(10); children([], separate=true); }\n"
                             "translate([50,0,0]) none() { sphere(3); }")
        assert vols == [500, 1000]


class TestOpenMeshes:
    """An open mesh -- faces that don't close a solid -- used to vanish without
    a word, since Manifold returns an empty body for it. OpenSCAD draws it; so
    do we now, as a display-only raw_mesh. Expected results checked against
    OpenSCAD 2026.02.01."""

    def test_open_polyhedron_is_drawn_and_named(self):
        bodies, lines = run(_OPEN_TETRA + ";")
        (b,) = bodies
        assert b.body is None and b.section is None
        verts, tris = b.raw_mesh
        assert len(verts) == 4 and len(tris) == 3
        assert lines == [
            "WARNING: polyhedron: mesh is not closed -- 3 boundary edge(s), first at "
            "[10, 0, 0] - [0, 10, 0]; drawing the object as an open surface rather than a "
            "solid -- nothing is patched. hull() can still use its points, but it cannot "
            "take part in union/difference/intersection in file <string>, line 1"]

    def test_closed_polyhedron_is_silent(self):
        bodies, lines = run("polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], "
                            "[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]);")
        assert lines == [] and bodies[0].body.volume() == approx(1000 / 6)

    def test_nan_vertex_says_nothing_is_drawn(self):
        bodies, lines = run("polyhedron([[0,0,0],[1,0,0],[0,1,0],[0,0,0/0]], "
                            "[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]);")
        assert bodies[0].raw_mesh is None
        assert "NonFiniteVertex" in lines[0] and "nothing is drawn" in lines[0]

    def test_transform_and_colour_carry_it(self):
        bodies, _ = run(f'color("red") translate([20, 0, 0]) mirror([1, 0, 0]) {_OPEN_TETRA};')
        verts, _ = bodies[0].raw_mesh
        assert verts.min(axis=0).tolist() == approx([10, 0, 0])
        assert verts.max(axis=0).tolist() == approx([20, 10, 10])
        assert bodies[0].color[:3] == (1.0, 0.0, 0.0)

    @pytest.mark.parametrize("tr", [
        "rotate([30, 45, 60])", "rotate([90, 180, 270])", "rotate(33, [1, 2, 3])",
        "scale([2, -1, 3])", "mirror([1, 1, 0])", "resize([5, 0, 20])",
        "multmatrix([[1, 0.5, 0, 3], [0, 1, 0, 0], [0.2, 0, 1, 1]])",
    ])
    def test_transform_matches_the_solid_face_for_face(self, tr):
        # The same transform applied to the closed tetrahedron (by Manifold)
        # and to the open one (by _transform_raw_mesh) must give the same
        # triangles, facing the same way -- mirrors included.
        closed = ("polyhedron([[0,0,0],[10,0,0],[0,10,0],[1,2,10]], "
                  "[[0,1,2],[0,3,1],[0,2,3],[1,3,2]])")
        opened = "polyhedron([[0,0,0],[10,0,0],[0,10,0],[1,2,10]], [[0,1,2],[0,3,1],[0,2,3]])"
        m = run(f"{tr} {closed};")[0][0].body.to_mesh()
        sv, st = np.asarray(m.vert_properties[:, :3], dtype=float), np.asarray(m.tri_verts)
        rv, rt = run(f"{tr} {opened};")[0][0].raw_mesh

        def faces(v, t):
            out = {}
            for tri in t:
                n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
                out[frozenset(tuple(np.round(v[i], 3)) for i in tri)] = n / np.linalg.norm(n)
            return out
        solid = faces(sv, st)
        for key, normal in faces(rv, rt).items():
            assert key in solid and solid[key] @ normal > 0.999

    def test_boolean_drops_it_like_openscad(self):
        bodies, lines = run(f"union() {{ cube(2); translate([20, 0, 0]) {_OPEN_TETRA}; }}")
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(8)
        assert "mesh is not closed" in lines[0]

    def test_hull_uses_its_points_silently(self):
        bodies, lines = run(f"hull() {{ cube(1); {_OPEN_TETRA}; }}")
        assert bodies[0].body.volume() == approx(1000 / 6) and lines == []

    def test_warning_outside_a_hull_survives_one_inside(self):
        _, lines = run(f"hull() {_OPEN_TETRA}; {_OPEN_TETRA};")
        assert len(lines) == 1 and "mesh is not closed" in lines[0]

    def test_open_stl_import_is_drawn(self, tmp_path):
        stl = tmp_path / "open.stl"
        stl.write_text("solid t\n"
                       "facet normal 0 0 -1\nouter loop\nvertex 0 0 0\nvertex 0 10 0\nvertex 10 0 0\nendloop\nendfacet\n"
                       "facet normal 0 -1 0\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 0 10\nendloop\nendfacet\n"
                       "endsolid t\n")
        bodies, lines = run(f'import("{stl.as_posix()}");')
        assert len(bodies[0].raw_mesh[1]) == 2
        assert lines[0].startswith("WARNING: import: mesh is not closed -- 4 boundary edge(s)")


class TestRenderExpression:
    """`obj = render() { ... };` measures geometry and draws nothing. Values
    match openscad_cpp_evaluator's for the same scripts, key order included."""

    def test_measures_the_union_and_draws_nothing(self):
        bodies, lines = run("obj = render() { cube(2); translate([1, 1, 1]) cube(2); }; "
                            "echo(obj.volume, obj.area, obj.genus, obj.boundingbox, obj.dim);")
        assert bodies == []
        assert lines == ["ECHO: 15, 42, 0, [[0, 0, 0], [3, 3, 3]], 3"]

    def test_3d_keys_and_vnf_shape(self):
        _, _, ev = run_tree("obj = render() { cube(1); };")
        obj = ev._root_ctx.let["obj"]
        assert list(obj) == ["vertices", "faces", "volume", "area", "genus", "boundingbox", "dim", "vnf"]
        assert len(obj.get("vertices")) == 8 and len(obj.get("faces")) == 12  # welded, triangles
        assert obj.get("vnf") == [obj.get("vertices"), obj.get("faces")]

    def test_faces_are_clockwise_from_outside(self):
        # Feeding the faces back through polyhedron() (which takes OpenSCAD's
        # clockwise winding) gives a POSITIVE volume; reversed, it would be
        # negative -- never abs() this.
        bodies, _ = run("o = render() { cube([1, 2, 3]); }; polyhedron(o.vertices, o.faces);")
        assert bodies[0].body.volume() == approx(6)

    def test_touching_shells_are_not_welded(self):
        # Welding the shared corner vertices would make edges with four faces.
        bodies, lines = run("o = render() { cube(1); translate([1, 0, 0]) cube(1); }; "
                            "echo(len(o.vertices), o.volume); polyhedron(o.vertices, o.faces);")
        assert lines == ["ECHO: 12, 2"]
        assert bodies[0].body.volume() == approx(2)

    def test_2d(self):
        _, lines = run("echo(render() { square([2, 3]); });")
        assert lines == ["ECHO: { vertices = [[0, 0], [2, 0], [2, 3], [0, 3]]; paths = [[0, 1, 2, 3]]; "
                         "area = 6; perimeter = 10; boundingbox = [[0, 0], [2, 3]]; dim = 2; }"]

    def test_empty(self):
        _, lines = run("echo(render() {});")
        assert lines == ["ECHO: { vertices = []; faces = []; volume = 0; area = 0; genus = 0; "
                         "boundingbox = undef; dim = 0; vnf = [[], []]; }"]

    def test_cavity_genus_and_function_use(self):
        _, lines = run("function v(w) = render() { cube(w); }.volume; echo(v(3)); "
                       "o = render() { difference() { cube(10, center=true); cube(4, center=true); } }; "
                       "echo(o.genus, o.volume);")
        assert lines == ["ECHO: 27", "ECHO: -1, 936"]

    def test_dollar_args_reach_the_children(self):
        _, lines = run("echo(len(render($fn=6) { cylinder(1, 1, 1); }.vertices));")
        assert lines == ["ECHO: 12"]

    def test_open_surface_gives_its_mesh_and_says_why(self):
        _, lines = run("o = render() { polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], "
                       "[[0,1,2],[0,3,1],[0,2,3]]); }; echo(len(o.faces), o.volume, o.genus);")
        assert "render(): result is not a closed solid" in lines[1]
        assert lines[-1] == "ECHO: 3, 0, undef"


class TestBackwardsRangeWarning:
    def test_warns_where_the_range_is_written(self):
        # r is never iterated; OpenSCAD still warns, at the literal
        _, lines = run("r = [5:0];")
        assert lines == ["WARNING: begin is greater than the end, but step is positive "
                         "in file <string>, line 1"]

    @pytest.mark.parametrize("src", [
        "for (i = [3:1]) echo(i);",
        "x = [for (i = [2:0]) i];",
    ])
    def test_every_use(self, src):
        _, lines = run(src)
        assert lines == ["WARNING: begin is greater than the end, but step is positive "
                         "in file <string>, line 1"]

    @pytest.mark.parametrize("src", [
        "y = [5:1:0];",   # a written step is taken as meant (OpenSCAD warns here)
        "z = [0:-1:5];",
        "w = [1:1];",
        "v = [1 + 1e-12 : 1];",  # float arithmetic a hair past the end
    ])
    def test_silent(self, src):
        _, lines = run(src)
        assert lines == []


class TestPolyhedronFromObject:
    """polyhedron(obj), polyhedron(vnf) and polygon(obj): a render() result
    goes straight back in, as in openscad_cpp_evaluator."""

    _SOLID = "o = render() { difference() { cube(10, center=true); sphere(4, $fn=16); } };"

    @pytest.mark.parametrize("call", ["polyhedron(o);", "polyhedron(o.vnf);", "polyhedron(o.vertices, o.faces);"])
    def test_all_three_forms_rebuild_the_solid(self, call):
        bodies, lines = run(self._SOLID + call + "echo(o.volume);")
        assert bodies[0].body.volume() == approx(748.651, rel=1e-5)  # positive: winding survives
        assert lines == ["ECHO: 748.651"]

    def test_any_object_with_the_keys(self):
        bodies, _ = run("polyhedron(object(points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]], "
                        "faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]));")
        assert bodies[0].body.volume() == approx(1 / 6)

    def test_object_without_faces_errors(self):
        with pytest.raises(EvalError, match="object has no 'faces' key"):
            run("polyhedron(object(vertices=[[0,0,0]]));")

    def test_two_points_are_not_taken_for_a_vnf(self):
        # a 2-point list has a POINT second, not a list of faces
        with pytest.raises(EvalError, match="'points' and 'faces' are required"):
            run("polyhedron([[0,0,0],[1,0,0]]);")

    def test_polygon_from_object(self):
        bodies, _ = run("s = render() { difference() { square(10); translate([2, 2]) square(3); } }; polygon(s);")
        assert bodies[0].section.area() == approx(91)

    def test_polygon_object_without_paths_is_one_contour(self):
        bodies, _ = run("polygon(object(vertices=[[0,0],[4,0],[0,3]]));")
        assert bodies[0].section.area() == approx(6)


def test_top_level_block_is_drawn():
    # openscad_lalr_parser < 1.2.1 left a top-level `{ ... }` in the AST as a
    # bare list, and build_scopes() crashed on it; OpenSCAD draws the contents.
    bodies, _ = run("x = 1; { cube(2); }")
    assert len(bodies) == 1 and bodies[0].body.volume() == approx(8)


class TestCSGBackports:
    """Four CSG bugs backported from openscad_cpp_evaluator; every expected
    value is what OpenSCAD 2026.02.01 renders for the same source."""

    # --- 2D shapes under a 3D transform take its in-plane part ---

    @pytest.mark.parametrize("src, bounds", [
        ("rotate([60,0,0]) square(10);", (0, 0, 10, 5)),        # was left flat: 10 x 10
        ("rotate(90, [1,0,0]) square(10);", (0, 0, 10, 0)),     # edge-on: no area left
        ("mirror([0,0,1]) square(10);", (0, 0, 10, 10)),        # was degenerate
        ("scale([1,1,3]) square(10);", (0, 0, 10, 10)),
        ("translate([1,2,5]) square(10);", (1, 2, 11, 12)),     # z is dropped
        ("resize([20,5]) square(10);", (0, 0, 20, 5)),          # was ignored in 2D
    ])
    def test_2d_takes_the_in_plane_part(self, src, bounds):
        bodies, _ = run(src)
        assert bodies[0].section.bounds() == pytest.approx(bounds, abs=1e-9)

    def test_extruded_after_an_out_of_plane_rotation(self):
        bodies, _ = run("linear_extrude(1) rotate([60,0,0]) square(10);")
        assert bodies[0].body.volume() == approx(50)

    # --- mixed 2D/3D children: the first child's dimension wins ---

    @pytest.mark.parametrize("src, dim, kept, dropped, size", [
        ("union(){cube(2); square(3);}", 3, "3D", "2D", 8),     # crashed: None + CrossSection
        ("union(){square(3); cube(2);}", 2, "2D", "3D", 9),
        ("difference(){cube(10); square(3);}", 3, "3D", "2D", 1000),
        ("difference(){square(10); cube(3);}", 2, "2D", "3D", 100),
        ("translate([1,0,0]) {cube(2); square(3);}", 3, "3D", "2D", 8),
        ('color("red") {cube(2); square(3);}', 3, "3D", "2D", 8),
    ])
    def test_mixed_children(self, src, dim, kept, dropped, size):
        bodies, lines = run(src)
        assert lines == ["WARNING: Mixing 2D and 3D objects is not supported in file <string>, line 1",
                         f"WARNING: Ignoring {dropped} child object for {kept} operation in file <string>, line 1"]
        assert len(bodies) == 1
        assert (bodies[0].body.volume() if dim == 3 else bodies[0].section.area()) == approx(size)

    def test_dropped_child_is_an_empty_operand(self):
        bodies, _ = run("intersection(){cube(10); square(3); cube(3);}")
        assert bodies == []

    def test_top_level_keeps_both(self):
        bodies, lines = run("cube(2); square(3);")
        assert lines == [] and len(bodies) == 2

    # --- ! makes its subtree the whole model ---

    def test_ancestors_of_the_root_are_not_applied(self):
        bodies, _ = run("translate([50,0,0]) !cube(5);")
        assert bbox(bodies)[0] == approx(0)  # was 50

    def test_root_under_an_extrude_stays_2d(self):
        bodies, _ = run("linear_extrude(height=10) !circle(10);")
        assert bodies[0].body is None and bodies[0].section is not None

    def test_highlighted_sibling_goes_too(self):
        bodies, _ = run("#cube(3); !cube(5);")
        assert len(bodies) == 1 and bodies[0].body.volume() == approx(125)

    def test_second_root_warns_and_the_first_wins(self):
        bodies, lines = run("cube(1);\n!cube(5);\n\n!translate([10,0,0]) cube(2);\n")
        assert lines == ["WARNING: More than one Root Modifier (!) in file <string>, line 4"]
        assert bodies[0].body.volume() == approx(125)

    def test_root_inside_the_root_is_part_of_it(self):
        bodies, lines = run("!union() { cube(5); !translate([10,0,0]) cube(1); }")
        assert "More than one Root Modifier" in lines[0]
        assert bodies[0].body.volume() == approx(126)

    # --- 2D minkowski() ---

    @pytest.mark.parametrize("src, area, bounds", [
        ("minkowski() { square([30,20]); circle(4, $fn=24); }", 1049.693178, (-4, -4, 34, 24)),
        ("minkowski() { polygon([[0,0],[30,0],[30,10],[10,10],[10,30],[0,30]]); circle(3, $fn=16); }",
         885.4412927, (-3, -3, 33, 33)),                                          # concave A
        ("minkowski() { square(10); polygon([[0,0],[6,0],[6,2],[2,2],[2,6],[0,6]]); }",
         240, (0, 0, 16, 16)),                                              # concave B
        ("minkowski() { square(10); translate([16,0]) square(2); }", 144, (16, 0, 28, 12)),  # off-origin B
        ("minkowski() { square(10); circle(2, $fn=12); square([1,3]); }", 251.0001658, (-2, -2, 13, 15)),
        ("minkowski() { difference() { square(30); translate([10,10]) square(10); } circle(2, $fn=12); }",
         1116.000166, (-2, -2, 32, 32)),                                           # a hole
    ])
    def test_2d_minkowski(self, src, area, bounds):
        bodies, _ = run(src)  # produced nothing
        assert bodies[0].section.area() == approx(area, rel=1e-5)  # OpenSCAD's, via float32 OFF
        assert bodies[0].section.bounds() == pytest.approx(bounds, abs=1e-6)



class TestHugeRanges:
    """A range of a million elements or more is refused with a warning and no
    iterations, as OpenSCAD does; `[0:1:1/0]` iterated forever. Expected
    output is OpenSCAD 2026.02.01's."""

    @staticmethod
    def _warning(n, line=1):
        return (f"WARNING: Bad range parameter in for statement: too many elements ({n}) "
                f"in file <string>, line {line}")

    def test_just_under_the_limit_iterates(self):
        _, lines = run("echo(len([for (i=[0:999998]) i]));")
        assert lines == ["ECHO: 999999"]

    @pytest.mark.parametrize("rng, n", [
        ("[0:999999]", 1000000),
        ("[0:0.5:499999.5]", 1000000),
        ("[999999:-1:0]", 1000000),
        ("[0:1:1/0]", 4294967295),       # hung
        ("[0:-1:-1/0]", 4294967295),
        ("[0:0:5]", 4294967295),
    ])
    def test_refused(self, rng, n):
        _, lines = run(f"echo(len([for (i={rng}) i]));")
        assert lines == [self._warning(n), "ECHO: 0"]

    @pytest.mark.parametrize("rng", ["[0/0:1:-1]", "[0:1:0/0]", "[0:0/0:5]", "[5:0:0]", "[0:0:0]"])
    def test_empty_and_silent(self, rng):
        _, lines = run(f"echo([for (i={rng}) i]);")
        assert lines == ["ECHO: []"]

    def test_limit_is_per_range(self):
        _, lines = run("echo(len([for (i=[0:1100], j=[0:1100]) 1]));")
        assert lines == ["ECHO: 1.2122e+6"]  # 1212201, in OpenSCAD's 6 significant digits

    def test_statement_for(self):
        bodies, lines = run("for (i=[0:2000000]) cube(1);")
        assert bodies == [] and lines == [self._warning(2000001)]

    def test_each(self):
        _, lines = run("x = [each [0:2000000]]; echo(len(x));")
        assert lines == [self._warning(2000001), "ECHO: 0"]


class TestEach:
    """`each` expands a range and splits a string, as OpenSCAD does."""

    @pytest.mark.parametrize("src, out", [
        ('[each "12"]', '["1", "2"]'),
        ("[each [0:2]]", "[0, 1, 2]"),
        ("[each 5]", "[5]"),
        ("[each undef]", "[]"),
        ("[1, each [2,3], 4]", "[1, 2, 3, 4]"),
        ('[for (s=["ab"]) each s]', '["a", "b"]'),
    ])
    def test_each(self, src, out):
        _, lines = run(f"echo({src});")
        assert lines == [f"ECHO: {out}"]



class TestNumberFormat:
    """echo()/str() write numbers as OpenSCAD does: 6 significant digits,
    fixed notation for exponents -5..5, `e+N` without a leading zero.
    Expected strings are OpenSCAD 2026.02.01's own."""

    @pytest.mark.parametrize("expr, out", [
        ("999999", "999999"), ("1000000", "1e+6"), ("123456.7", "123457"),
        ("1234567", "1.23457e+6"), ("0.0000123456", "0.0000123456"), ("1e-7", "1e-7"),
        ("1.5e20", "1.5e+20"), ("-2500000", "-2.5e+6"), ("100000", "100000"),
        ("1/3", "0.333333"), ("2/3*1e6", "666667"), ("-1e-10", "-1e-10"),
        ("0.00001", "0.00001"), ("0.0001", "0.0001"), ("12.3456789", "12.3457"),
        ("99999.95", "99999.9"),          # printed 100000: the mantissa was rounded after dividing
        ("999999.5", "1e+6"),
        ("1e-300/1e20", "9.99989e-321"),  # subnormal: printed 1.00198e-320
    ])
    def test_matches_openscad(self, expr, out):
        _, lines = run(f"echo({expr});")
        assert lines == [f"ECHO: {out}"]

    def test_integer_results_too(self):
        # len() returns an int, which used to print in full
        _, lines = run("echo(len([for (i=[0:1100], j=[0:1100]) 1]), len(\"abc\"));")
        assert lines == ["ECHO: 1.2122e+6, 3"]

    def test_smallest_subnormal_does_not_crash(self):
        from openscad_evaluator.evaluator import _format_number
        assert _format_number(5e-324) == "4.94066e-324"  # ZeroDivisionError before


class TestArgumentWarnings:
    """Arguments a callee doesn't declare were silently dropped; OpenSCAD
    warns. Expected output is OpenSCAD 2026.02.01's."""

    @staticmethod
    def _w(msg, line=1):
        return f"WARNING: {msg} in file <string>, line {line}"

    @pytest.mark.parametrize("src, warnings, echo", [
        ("function f(a) = a; echo(f(1, 2));", ["Too many unnamed arguments supplied"], "1"),
        ("function f(a) = a; echo(f(1, b=2));", ['variable "b" not specified as parameter'], "1"),
        ("function f(a, b) = a; echo(f(1, a=2));", ['argument "a" overrides positional argument'], "2"),
        ("function f(a) = a; echo(f(a=1, a=2));", ['argument "a" supplied more than once'], "2"),
        ("function f(a) = a; echo(f(a=1, $fn=3));", [], "1"),                 # $-names are fine
        ("function f(a) = a; echo(f(1, $children=2));",
         ['variable "$children" not specified as parameter'], "1"),          # ...but not $children
        ("g = function(x) x; echo(g(1, y=2));", ['variable "y" not specified as parameter'], "1"),
        ("function f(a, b) = a; echo(f(b=1, 2));", [], "2"),
    ])
    def test_user_functions(self, src, warnings, echo):
        _, lines = run(src)
        assert lines == [self._w(w) for w in warnings] + [f"ECHO: {echo}"]

    def test_user_module(self):
        _, lines = run("module m(a) { echo(a=a); } m(1, 2); m(1, b=2);")
        assert lines == [self._w("Too many unnamed arguments supplied"), "ECHO: a = 1",
                         self._w('variable "b" not specified as parameter'), "ECHO: a = 1"]

    @pytest.mark.parametrize("src, warning", [
        ("cube(1, bogus=2);", 'variable "bogus" not specified as parameter'),
        ("cube(1, true, 3);", "Too many unnamed arguments supplied"),
        ("translate([1,0,0], 5) cube(1);", "Too many unnamed arguments supplied"),
        ('echo(textmetrics("x", bogus=1).size);', 'variable "bogus" not specified as parameter'),
    ])
    def test_builtin_modules(self, src, warning):
        _, lines = run(src)
        assert lines[0] == self._w(warning)

    def test_builtin_functions_never_warn(self):
        _, lines = run("echo(sin(bogus=30));")  # read positionally, as in OpenSCAD
        assert lines == ["ECHO: 0.5"]

    def test_bosl2_style_calls_are_silent(self):
        _, lines = run("module m(size, anchor, spin=0) { cube(size); } "
                       "m(2, anchor=[0,0,1], spin=90, $fn=8);")
        assert lines == []


class TestDollarVariables:
    """Checked against OpenSCAD 2026.02.01."""

    def test_is_undef_of_a_name_is_silent(self):
        _, lines = run("echo(is_undef(zzz), is_undef($never), is_undef(zzz) ? 1 : 2);")
        assert lines == ["ECHO: true, true, 1"]

    def test_reading_an_unknown_name_still_warns(self):
        # OpenSCAD warns for an unset $var as for any name -- only is_undef() asks
        _, lines = run("echo($never_set);")
        assert lines == ['WARNING: Ignoring unknown variable "$never_set" in file <string>, line 1',
                         "ECHO: undef"]

    def test_preview_is_false_in_a_render(self):
        _, lines = run("echo($preview);")
        assert lines == ["ECHO: false"]

    def test_parent_modules_counts_the_current_module(self):
        _, lines = run("module lvl() { echo($parent_modules); } module outer() { lvl(); } "
                       "lvl(); outer(); "
                       "module w() { children(); } module o() { w() echo($parent_modules); } o() cube(1);")
        assert lines == ["ECHO: 1", "ECHO: 2", "ECHO: 2"]


class TestBuiltinArguments:
    """Expected values are OpenSCAD 2026.02.01's."""

    @pytest.mark.parametrize("src, size", [
        ("resize([10,0,0], auto=true) cube([2,4,8]);", (10, 20, 40)),
        ("resize([10,0,0], auto=[true,false,true]) cube([2,4,8]);", (10, 4, 40)),
        ("resize([10,8,0], auto=true) cube([2,4,8]);", (10, 8, 40)),   # the larger scale
        ("resize([0,2,0], auto=true) cube([2,4,8]);", (1, 2, 4)),
        ("resize([10,0,0]) cube([2,4,8]);", (10, 4, 8)),
    ])
    def test_resize_auto(self, src, size):
        bb = bbox(run(src)[0])
        assert (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]) == approx(size)

    @pytest.mark.parametrize("src, area", [
        ("offset(2) square(10);", 191.2426789),   # r positionally: did nothing
        ("offset() square(10);", 142.5201744),    # r=1 by default: did nothing
        ("offset(delta=2) square(10);", 196),     # sharp: was chamfered
        ("offset(delta=2, chamfer=true) square(10);", 193.2547694),
    ])
    def test_offset(self, src, area):
        bodies, _ = run(src)
        assert bodies[0].section.area() == approx(area, rel=1e-3)

    def test_children_by_vector_and_range(self):
        src = ("module {m}() children({i}); {m}() {{ cube(1); translate([5,0,0]) cube(2); "
               "translate([20,0,0]) cube(3); }}")
        bodies, _ = run(src.format(m="s", i="[0:1]"))  # crashed
        assert sum(b.body.volume() for b in bodies) == approx(9)
        bodies, _ = run(src.format(m="p", i="[2, 0]"))
        assert sum(b.body.volume() for b in bodies) == approx(28)

    def test_children_out_of_bounds_warns(self):
        _, lines = run("module k() children([0:3]); k() { cube(1); cube(2); }")
        assert lines == [f"WARNING: Children index ({i}) out of bounds (2 children) in file <string>, line 1\n"
                         "TRACE: call of 'k()' in file <string>, line 1\n"
                         "TRACE: called by 'k' in file <string>, line 1"
                         for i in (2, 3)]

    def test_fill(self):
        bodies, _ = run("fill() difference() { square(20); translate([3,3]) square(14); }")
        assert bodies[0].section.area() == approx(400)

    def test_version(self):
        _, lines = run("echo(version(), version_num(), version_num([2019,5,0]), version_num([2021,1]), "
                       "version_num(5));")
        assert lines == ["ECHO: [2026, 1, 1], 2.02601e+7, 2.01905e+7, 2.02101e+7, undef"]

    def test_object_delete_entry(self):
        _, lines = run('o = object(a=42, b=53, c=8); '
                       'echo(object(o, [["d", 18], ["b"]]), object(o, [["b"], ["b", 99]]), object(o, [["zz"]]));')
        assert lines == ["ECHO: { a = 42; c = 8; d = 18; }, { a = 42; c = 8; b = 99; }, { a = 42; b = 53; c = 8; }"]

    def test_search_string_in_a_list_of_strings_is_refused(self):
        _, lines = run('echo(search("x", ["x", "yx", "x"], 0));')
        assert lines == ['WARNING: Invalid entry in search vector at index 0, required number of values in '
                         'the entry: 1. Invalid entry: "x" in file <string>, line 1', "ECHO: []"]


class TestNumbersAndStrings:
    """Byte-for-byte OpenSCAD 2026.02.01 output."""

    def test_mod_and_power(self):
        _, lines = run("echo(-7%3, 7%0, 7.5%2, -7.5%2, 7%-3, 0^-1, (-8)^(1/3), 0^0, 2^-1, (-2)^3, 10^400);")
        assert lines == ["ECHO: -1, nan, 1.5, -1.5, 1, inf, nan, 1, 0.5, -8, inf"]

    def test_degree_trig_is_exact(self):
        _, lines = run("echo(sin(45)-cos(45), sin(30), cos(60), tan(45), cos(90), tan(90), tan(270), "
                       "sin(135)==cos(45), asin(0.5), acos(0.5), atan(1), atan2(1,1), sin(1e30));")
        assert lines == ["ECHO: 0, 0.5, 0.5, 1, 0, inf, -inf, true, 30, 60, 45, 45, nan"]

    def test_range_equality(self):
        _, lines = run("echo([5:1:0]==[5:1:0], [0:2]==[0:1:2], [1:3]!=[1:4], [5:1:0]==[6:1:0], [0:2]==[0,1,2]);")
        assert lines == ["ECHO: true, true, true, true, false"]

    def test_function_values_print_like_openscad(self):
        _, lines = run('echo(str(function(x, y=2) x*y+1), function(a) a ? 1 : 2, '
                       'function(v) [for (i=[0:len(v)-1]) if (v[i] > 0) v[i]*2], '
                       'function(a) let(b=a+1) -b^2, function(o) o.x[0](1e7, t="s"));')
        assert lines == ['ECHO: "function(x, y = 2) ((x * y) + 1)", function(a) (a ? 1 : 2), '
                         'function(v) [for(i = [0 : (len(v) - 1)]) (if((v[i] > 0)) ((v[i] * 2)))], '
                         'function(a) let(b = (a + 1)) -(b ^ 2), function(o) (o.x[0])(1e+7, t = "s")']

    def test_undefined_escape_warns_once_per_escape(self):
        _, lines = run('echo("\\q\\x80"); for (i=[0:2]) echo("\\z");')
        warn = "WARNING: Undefined escape sequence in file <string>, line 1"
        assert lines == [warn, warn, 'ECHO: "qx80"', warn, 'ECHO: "z"', 'ECHO: "z"', 'ECHO: "z"']


class TestPolyhedronMeshes:
    """Behaviour matches OpenSCAD 2026.02.01 and openscad_cpp_evaluator."""

    def test_concave_face_is_ear_clipped_not_fanned(self):
        # An L-prism whose caps start at a vertex that sees only part of the
        # L: a fan overlapped itself and reported 30 units of area, not 22.
        bodies, _ = run("""
            L=[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]]; n=len(L);
            pts=concat([for(p=L)[p[0],p[1],0]], [for(p=L)[p[0],p[1],1]]);
            polyhedron(pts, concat([[for(i=[0:n-1]) (i+2)%n]], [[for(i=[n-1:-1:0]) n+(i+2)%n]],
                                   [for(i=[0:n-1]) [i,n+i,n+(i+1)%n,(i+1)%n]]));""")
        (b,) = bodies
        assert b.body.volume() == pytest.approx(5.0)
        assert b.body.surface_area() == pytest.approx(22.0)

    def test_vertices_keep_double_precision(self):
        # float32's spacing at 3e7 is 2 units, which flattened this to nothing.
        bodies, _ = run("translate([-3e7,0,0]) polyhedron([for(z=[0,1],y=[0,1],x=[0,1])[3e7+x*0.5,y,z]],"
                        "[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]);")
        assert bodies[0].body.volume() == pytest.approx(0.5)

    def test_touching_shells_are_not_welded(self):
        _, lines = run("""
            o = render() { union() {
              difference(){ cube([20,20,20],center=true); cylinder(d=8,h=60,center=true,$fn=16); }
              difference(){ cylinder(d=8,h=60,center=true,$fn=16); cube([20,20,20],center=true); }
            } };
            rt = render() { polyhedron(o); };
            c = render() { cube(10); };
            echo(o.volume, o.genus, len(o.vertices));
            echo(o.volume == rt.volume, len(o.vertices) == len(rt.vertices));
            echo(len(c.vertices));""")
        assert lines == ["ECHO: 8979.67, -1, 104", "ECHO: true, true", "ECHO: 8"]


class TestManifoldCacheProvenance:
    """A cache hit must look like a fresh generate to anything reading
    originalIDs or warnings (cpp #84, #85, #186)."""

    @staticmethod
    def _render(src, cache):
        lines = []
        nodes = getASTfromString(src, include_comments=False)
        ev = Evaluator(echo_fn=lines.append, manifold_cache=cache)
        bodies, id_to_node = ev.evaluate(nodes, build_scopes(nodes))
        picked = []
        for b in bodies:
            ids = sorted({int(i) for i in b.body.to_mesh().run_original_id}) if b.body else []
            picked.append((tuple(ids), [id_to_node[i].position.line if i in id_to_node else None for i in ids]))
        return picked, lines

    def test_identical_parts_get_their_own_ids_and_lines(self):
        picked, _ = self._render("translate([0,0,0]) cylinder(5,2);\ntranslate([9,0,0]) cylinder(5,2);",
                                 ManifoldCache())
        (ids_a, lines_a), (ids_b, lines_b) = picked
        assert ids_a != ids_b
        assert (lines_a, lines_b) == ([1], [2])

    def test_module_parts_keep_their_own_line(self):
        picked, _ = self._render("module m() { cube(1); }\ntranslate([0,5,0]) m();\ntranslate([0,9,0]) m();",
                                 ManifoldCache())
        assert picked[0][0] != picked[1][0]
        assert [lines for _, lines in picked] == [[1], [1]]

    def test_rerender_stays_pickable(self):
        cache = ManifoldCache()
        src = "translate([0,0,0]) cylinder(5,2);\ntranslate([9,0,0]) cylinder(5,2);"
        self._render(src, cache)
        picked, _ = self._render(src, cache)
        assert [lines for _, lines in picked] == [[1], [2]]

    def test_hit_replays_warnings_including_a_descendants(self):
        cache = ManifoldCache()
        src = ("translate([1,0,0]) polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], "
               "[[0,1,2],[0,3,1],[0,2,3]]);")
        runs = [self._render(src, cache)[1] for _ in range(3)]
        assert len(runs[0]) == 1 and "mesh is not closed" in runs[0][0]
        assert runs[1] == runs[0] and runs[2] == runs[0]

    def test_quiet_subtree_stays_quiet(self):
        cache = ManifoldCache()
        self._render("cube(1);", cache)
        assert self._render("cube(1);", cache)[1] == []


class TestDebugStops:
    """cpp #82 and #83."""

    @staticmethod
    def _stops(src):
        seen = []

        def hook(line, depth, forced=False, expr_level=False, expr_depth=0, origin=None, get_frames=None):
            if not expr_level:
                seen.append((line, depth, ev._last_children_positions))
            return "continue", {}
        nodes = getASTfromString(src, include_comments=False)
        ev = Evaluator(echo_fn=lambda m: None, debug_hook=hook)
        ev.evaluate(nodes, build_scopes(nodes))
        return [(line, depth) for line, depth, _ in seen], seen

    def test_call_on_its_statements_line_stops_once(self):
        stops, _ = self._stops("function f(y) = y*2;\nx = f(3);\necho(x);")
        assert stops == [(2, 0), (1, 1), (3, 0)]

    def test_several_calls_on_one_line_stop_once(self):
        stops, _ = self._stops("function f(y) = y*2;\na = [f(1), f(2), f(3)];")
        assert stops == [(2, 0), (1, 1), (1, 1), (1, 1)]

    def test_call_on_its_own_line_still_stops(self):
        stops, _ = self._stops("function f(y) = y*2;\nb = [for (i=[0:0])\n  f(i)];")
        assert (3, 0) in stops

    def test_children_call_targets_the_forwarded_children(self):
        _, seen = self._stops("module framed(gap) {\n    children();\n}\nframed(20)\n    leaf();\n"
                              "module leaf() { cube(1); }")
        assert [t for line, depth, t in seen if line == 2] == [[("<string>", 5)]]


class TestFontStyles:
    """cpp #163, #164, #171. Widths from OpenSCAD 2026.02.01, which bundles
    the same Liberation Sans faces; ours differ in the last digit or so, as
    FreeType hinting isn't replicated (see TestTextMetrics)."""

    def _width(self, expr):
        _, lines = run(f"echo(textmetrics({expr}).size[0]);")
        return float(lines[0].removeprefix("ECHO: "))

    def test_bundled_styles(self):
        assert self._width('"Hi", 10') == pytest.approx(11.0413, abs=0.01)
        assert self._width('"Hi", 10, "Liberation Sans:style=Bold"') == pytest.approx(11.9821, abs=0.01)
        assert self._width('"Hi", size=10, font=":style=Bold"') == pytest.approx(11.9821, abs=0.01)
        assert self._width('"Hi", size=10, font="Liberation Sans:style=Italic"') == pytest.approx(13.0061, abs=0.01)
        assert self._width('"Hi", size=10, font="Liberation Sans:style=Bold Italic"') == pytest.approx(13.8829, abs=0.01)

    def test_fontmetrics_names_the_style(self):
        _, lines = run('echo(fontmetrics(10, "Liberation Sans:style=Bold").font);')
        assert lines == ['ECHO: { family = "Liberation Sans"; style = "Bold"; }']

    def test_text_font_is_positional(self):
        bold, _ = run('text("Hi", 10, "Liberation Sans:style=Bold");')
        named, _ = run('text("Hi", 10, font="Liberation Sans:style=Bold");')
        regular, _ = run('text("Hi", 10);')
        assert bold[0].section.area() == pytest.approx(named[0].section.area())
        assert bold[0].section.area() != pytest.approx(regular[0].section.area())

    def test_textmetrics_takes_alignment_positionally(self):
        _, lines = run('echo(textmetrics("Hi",10,undef,undef,undef,undef,"center","top",2).position ==\n'
                       '     textmetrics("Hi",size=10,halign="center",valign="top",spacing=2).position);')
        assert lines == ["ECHO: true"]


class TestWarningAttribution:
    """Warnings below the top level name the user's line that started the
    chain, as in the C++ port (3e11352). Output identical to it."""

    def test_eval_time_warning_names_entry_and_traces(self, tmp_path):
        (tmp_path / "lib.scad").write_text("module inner() { echo(1+\"a\"); }\nmodule outer() { inner(); }\n")
        (tmp_path / "main.scad").write_text("include <lib.scad>\necho(2);\nouter();\n")
        from openscad_lalr_parser import getASTfromFile
        lines = []
        nodes = getASTfromFile(str(tmp_path / "main.scad"))
        Evaluator(echo_fn=lines.append).evaluate(nodes, build_scopes(nodes))
        lib, main = tmp_path / "lib.scad", tmp_path / "main.scad"
        assert lines[1] == "\n".join([
            f"WARNING: undefined operation (number + string) in file {lib}, line 1, from {main}, line 3",
            f"TRACE: call of 'inner()' in file {lib}, line 1",
            f"TRACE: called by 'inner' in file {lib}, line 2",
            f"TRACE: call of 'outer()' in file {lib}, line 2",
            f"TRACE: called by 'outer' in file {main}, line 3"])

    def test_generate_time_warning_names_entry_only(self):
        _, lines = run("module m() {\n polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], "
                       "[[0,1,2],[0,3,1],[0,2,3]]);\n}\nm();")
        assert len(lines) == 1
        assert lines[0].endswith("in file <string>, line 2, from <string>, line 4")

    def test_top_level_warning_stays_one_line(self):
        _, lines = run('echo(1+"a");')
        assert lines[0] == "WARNING: undefined operation (number + string) in file <string>, line 1"
