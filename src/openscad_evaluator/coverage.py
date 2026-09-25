"""Coverage: which statements, branch arms and bodies a run executed. A port
of openscad_cpp_evaluator's coverage.cpp (#169, #170, #175); the result has
the same shape as its Evaluator(coverage=True).coverage_result.

Three kinds, one vocabulary shared by the recorder and by the walk that
enumerates what COULD have run: statement (every node in a statement list;
an if/else arm's first statement is flagged `arm`), branch (a ternary arm, a
comprehension if/else arm, the right operand of `&&`/`||`), body (a
function, module or function literal ever entered)."""
from __future__ import annotations

from openscad_lalr_parser.nodes import (
    Assignment, ModularCall, ModularFor, ModularIntersectionFor, ModularLet, ModularEcho, ModularAssert,
    ModularIf, ModularIfElse, ModularModifierShowOnly, ModularModifierHighlight, ModularModifierBackground,
    ModularModifierDisable, ModuleDeclaration, FunctionDeclaration, NamedArgument, PositionalArgument,
    TernaryOp, LogicalAndOp, LogicalOrOp, PrimaryCall, PrimaryIndex, PrimaryMember, LetOp, EchoOp, AssertOp,
    FunctionLiteral, RenderExpression, CommentedExpr, ListComprehension, ListCompLet, ListCompEach,
    ListCompFor, ListCompCFor, ListCompIf, ListCompIfElse, RangeLiteral,
)

_MODIFIERS = (ModularModifierShowOnly, ModularModifierHighlight, ModularModifierBackground, ModularModifierDisable)


class _Walker:
    def __init__(self):
        self.out: list[tuple] = []  # (node, kind, arm)
        self._seen: set[int] = set()

    def emit(self, n, kind, arm=False):
        if n is not None and id(n) not in self._seen:
            self._seen.add(id(n))
            self.out.append((n, kind, arm))

    def statements(self, lst, first_is_arm=False):
        for i, s in enumerate(lst or []):
            self.statement(s, first_is_arm and i == 0, True)

    def statement(self, n, arm, emit_self):
        t = type(n)
        if t is Assignment:
            if emit_self:
                self.emit(n, "statement", arm)
            self.expr(n.expr)
        elif t is ModularCall:
            if emit_self:
                self.emit(n, "statement", arm)
            self.args(n.arguments)
            self.statements(n.children)
        elif t in (ModularFor, ModularIntersectionFor):
            if emit_self:
                self.emit(n, "statement", arm)
            for a in n.assignments:
                self.expr(a.expr)
            self.statements(n.body if isinstance(n.body, list) else [n.body])
        elif t is ModularLet:
            if emit_self:
                self.emit(n, "statement", arm)
            for a in n.assignments:
                self.expr(a.expr)
            self.statements(n.children)
        elif t in (ModularEcho, ModularAssert):
            if emit_self:
                self.emit(n, "statement", arm)
            self.args(n.arguments)
            self.statements(n.children)
        elif t is ModularIf:
            if emit_self:
                self.emit(n, "statement", arm)
            self.expr(n.condition)
            self.statements(n.true_branch, True)
        elif t is ModularIfElse:
            if emit_self:
                self.emit(n, "statement", arm)
            self.expr(n.condition)
            self.statements(n.true_branch, True)
            self.statements(n.false_branch, True)
        elif t in _MODIFIERS:
            # The modifier is the statement; the call it wraps is walked for
            # what it contains, but is not a statement of its own.
            if emit_self:
                self.emit(n, "statement", arm)
            if n.child is not None:
                self.statement(n.child, False, False)
        elif t is ModuleDeclaration:
            self.emit(n, "body")
            self.params(n.parameters)
            self.statements(n.children)
        elif t is FunctionDeclaration:
            self.emit(n, "body")
            self.params(n.parameters)
            self.expr(n.expr)

    def args(self, lst):
        for a in lst or []:
            if type(a) in (NamedArgument, PositionalArgument):
                self.expr(a.expr)

    def params(self, lst):
        for p in lst or []:
            if p.default is not None:
                self.expr(p.default)

    def expr(self, e):
        if e is None:
            return
        t = type(e)
        if t is RangeLiteral:
            self.expr(e.start)
            self.expr(e.end)
            self.expr(e.step)
        elif t in (LogicalAndOp, LogicalOrOp):
            self.expr(e.left)
            self.emit(e.right, "branch")
            self.expr(e.right)
        elif t is TernaryOp:
            self.expr(e.condition)
            self.emit(e.true_expr, "branch")
            self.expr(e.true_expr)
            self.emit(e.false_expr, "branch")
            self.expr(e.false_expr)
        elif t is PrimaryCall:
            self.expr(e.left)
            self.args(e.arguments)
        elif t is PrimaryIndex:
            self.expr(e.left)
            self.expr(e.index)
        elif t is PrimaryMember:
            self.expr(e.left)
        elif t is LetOp:
            for a in e.assignments:
                self.expr(a.expr)
            self.expr(e.body)
        elif t in (EchoOp, AssertOp):
            self.args(e.arguments)
            self.expr(e.body)
        elif t is FunctionLiteral:
            self.emit(e, "body")
            self.params(e.parameters)
            self.expr(e.body)
        elif t is RenderExpression:
            self.args(e.arguments)
            self.statements(e.children)
        elif t is CommentedExpr:
            self.expr(e.expr)
        elif t is ListComprehension:
            for el in e.elements:
                self.element(el)
        elif hasattr(e, "left") and hasattr(e, "right"):
            self.expr(e.left)
            self.expr(e.right)
        elif hasattr(e, "expr") and not hasattr(e, "name"):
            self.expr(e.expr)

    def element(self, el):
        t = type(el)
        if t is ListCompLet:
            for a in el.assignments:
                self.expr(a.expr)
            self.element(el.body)
        elif t is ListCompEach:
            self.element(el.body)
        elif t is ListCompFor:
            for a in el.assignments:
                self.expr(a.expr)
            self.element(el.body)
        elif t is ListCompCFor:
            for a in el.inits:
                self.expr(a.expr)
            self.expr(el.condition)
            for a in el.incrs:
                self.expr(a.expr)
            self.element(el.body)
        elif t is ListCompIf:
            self.expr(el.condition)
            self.emit(el.true_expr, "branch")
            self.element(el.true_expr)
        elif t is ListCompIfElse:
            self.expr(el.condition)
            self.emit(el.true_expr, "branch")
            self.element(el.true_expr)
            self.emit(el.false_expr, "branch")
            self.element(el.false_expr)
        else:
            self.expr(el)


def _summary(origin: str = "") -> dict:
    return {"origin": origin, "statements": 0, "statements_hit": 0, "branches": 0, "branches_hit": 0,
            "bodies": 0, "bodies_hit": 0, "spans": 0, "spans_hit": 0}


def _finish(f: dict) -> dict:
    def pct(hit, n):
        return 100.0 * hit / n if n else 100.0
    f["percent"] = pct(f["spans_hit"], f["spans"])
    f["statement_percent"] = pct(f["statements_hit"], f["statements"])
    f["branch_percent"] = pct(f["branches_hit"], f["branches"])
    f["body_percent"] = pct(f["bodies_hit"], f["bodies"])
    return f


def summarize(spans: list[dict]) -> dict:
    """{spans, files, total}: per-file and total counts and percentages.
    "branches" counts branch spans plus if/else arm statements, which are
    also statements; "percent" is over every span."""
    files: dict[str, dict] = {}
    total = _summary()
    for s in spans:
        f = files.setdefault(s["origin"], _summary(s["origin"]))
        hit = s["hits"] > 0
        for d in (f, total):
            d["spans"] += 1
            d["spans_hit"] += hit
            if s["kind"] == "statement":
                d["statements"] += 1
                d["statements_hit"] += hit
            if s["kind"] == "branch" or s["arm"]:
                d["branches"] += 1
                d["branches_hit"] += hit
            if s["kind"] == "body":
                d["bodies"] += 1
                d["bodies_hit"] += hit
    return {"spans": spans, "files": [_finish(files[k]) for k in sorted(files)], "total": _finish(total)}


def build_result(roots: list, extra_statements: list, hits: dict[int, int]) -> dict:
    """The coverage result for every coverable node under `roots` (the run's
    top-level statements, includes and use-injected declarations among them)
    and `extra_statements` (used files' own globals, which `use` splices
    nowhere), with each node's hit count from `hits` (keyed by id(node))."""
    w = _Walker()
    for n in roots:
        w.statement(n, False, True)
    for n in extra_statements:
        w.statement(n, False, True)
    spans, seen = [], set()
    for n, kind, arm in w.out:
        p = n.position
        key = (getattr(p, "origin", ""), p.start_offset, p.end_offset, kind)
        if key in seen:
            continue  # one file reached through two parses: its first (executed) copy wins
        seen.add(key)
        spans.append({"origin": getattr(p, "origin", "") or "", "line": p.line, "column": p.column,
                      "start": p.start_offset, "end": p.end_offset, "kind": kind, "arm": arm,
                      "hits": hits.get(id(n), 0)})
    return summarize(spans)
