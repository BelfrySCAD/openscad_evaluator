# CLAUDE.md — openscad_evaluator

## Build & Test

```bash
# Install in development mode (with DXF import support)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/openscad_evaluator --cov-report=term-missing
```

## Architecture

### Data Flow

```
OpenSCAD AST (from openscad_lalr_parser)
    ↓
Evaluator.evaluate(nodes, root_scope)
    ↓ resolve (AST walk, no Manifold calls)
CSGNode tree (plain data)
    ↓ generate (bottom-up Manifold/CrossSection construction)
ColoredBody list + originalID → AST node table
```

Two-pass, not one: `resolve` walks the whole AST and builds an explicit `CSGNode` tree describing
what to build, with zero Manifold calls; `generate_tree()` then walks that tree bottom-up and does
all the actual Manifold/CrossSection work, checking `ManifoldCache` (content-hash keyed) before
each node so an unchanged subtree — e.g. one untouched by a debugger step or a partial re-render —
reuses its previous result instead of recomputing it.

### Key Files

- `src/openscad_evaluator/evaluator.py` — everything: `Evaluator`, `EvalContext`, built-ins,
  2D/3D geometry, CSG tree, profiling, `ManifoldCache`, font/DXF/3MF import, `resolve_use_scopes`
- `src/openscad_evaluator/_css_colors.py` — static CSS/SVG color-name → RGB table (generated from
  a live Qt install, not transcribed — see the table's own module docstring for how to regenerate
  it if it ever needs updating)
- `src/openscad_evaluator/resources/fonts/` — bundled Liberation Sans fallback font (used when
  `fc-match` isn't available or a `font=` spec can't be resolved)
- `src/openscad_evaluator/export.py` — headless STL/OBJ/OFF/3MF export from a `ColoredBody` list;
  no GUI dependency, no `lib3mf` dependency either (3MF is written by hand as a ZIP of XML via
  stdlib `zipfile`/`xml.etree.ElementTree`, mirroring `evaluator.py`'s own `_load_3mf` reader,
  since `lib3mf` has limited platform support — not available on aarch64/ARM64). STL/OBJ ported
  from BelfrySCAD's own exporter so the two stay in sync.
- `src/openscad_evaluator/cli.py` / `_debug_repl.py` — the `openscad-evaluator` console script
  (`[project.scripts]`) and its `--debug` gdb-style REPL, built entirely on the public
  `debug_hook`/`error_break_fn`/`return_hook` contract (nothing evaluator-internal). `DebugRepl`
  caches source lines per-origin (`_source_lines_by_origin`, lazily populated via `_lines_for()`),
  not just the main script's own file — a breakpoint/step can land inside a `use <file>`-injected
  function/module's own body, which lives in a different file than `source_path`; `list` (both the
  automatic display on a pause and the explicit command) must read *that* file's lines. Was a real
  bug until fixed (`_list_source` used to always read the main script's lines regardless of where
  the debugger was actually paused) — see `tests/test_cli.py`'s
  `test_list_shows_source_from_use_injected_file_when_paused_there`; the C++ port
  (`openscad_cpp_evaluator`) had the identical bug in its own `DebugRepl`, fixed the same way there
  too.

  Two later additions, both ported from (or discovered missing relative to) BelfrySCAD's own GUI
  debugger (`DebugSession`, the upstream source `_debug_repl.py`'s own module docstring already
  cites for its into/over/out semantics): **`child`** (step-to-child) pauses the first time
  `children()`/`children(N)` forwards control to one of the paused call's own `{ ... }` children,
  reading `Evaluator._last_children_positions` (already existed, docstring-labeled for exactly
  this, just never wired to a REPL command); **Ctrl+C** during a running `evaluate()` now pauses
  like a breakpoint instead of raising an unhandled `KeyboardInterrupt` (Python's own default
  SIGINT behavior otherwise) — `cli.py` installs a `signal.signal(SIGINT, ...)` handler that calls
  `DebugRepl.request_pause()`, a plain flag read-and-cleared in `debug_hook()` alongside
  breakpoints/steps, mirroring `DebugSession._pause_requested`/`pause()` exactly (a GUI button
  there, a signal here). Both required the exact same fix on the C++ port's side too — see that
  port's own `CLAUDE.md` for the fuller writeup, including a real bug caught in `child`'s own
  target-position matching (raw vs. realpath'd origins disagreeing across a symlink).

  **Arrow-key command history**, added right after Ctrl+C: neither port's `DebugRepl` had any
  line-editing before this (raw `input()`/`std::getline` means up/down arrow keys just insert their
  own escape bytes instead of recalling a previous command). Here the fix is a one-liner --
  `import readline` at the top of `_debug_repl.py` (wrapped in `try`/`except ImportError`, since the
  module doesn't exist on Windows) hooks stdlib `input()` into GNU readline (or libedit, what
  Python's own `readline` module actually links against on macOS) for the rest of the process, no
  other code changed. The C++ port has no stdlib equivalent and needed a real vendored dependency
  (`yhirose/cpp-linenoise`) -- see that port's own `CLAUDE.md` for the fuller writeup, including how
  both sides were verified end to end with a `pty.openpty()`-driven harness (a real pty, not a piped
  file, since arrow-key bytes only mean anything to a raw terminal) and a real pitfall that harness
  itself hit (a synthetic pty's zeroed window size sending the C++ side's vendored library down a
  cursor-position-query fallback path that then hangs forever with no real terminal to answer it --
  an artifact of the test harness, not a bug in either port).

  **`--profile FILENAME`**, added after arrow-key history: `Evaluator`'s `profile=True`/
  `profile_result` instrumentation (see `docs/evaluator.md`'s "Profiling" section) had no CLI
  surface before this -- `cli.py` never constructed `Evaluator` with profiling on, nor printed
  anything from the result. `--profile FILENAME` now sets `profile=bool(args.profile)` and, once
  `evaluate()` returns, writes a plain-text report to `FILENAME` via `_format_profile_report()` (a
  module-level helper in `cli.py`) -- a summary (`resolve_time`/`generate_time`/`total_time`/
  `unattributed_time`) followed by one row per call site, sorted by `self_time` descending with a
  deterministic tie-break (`call_origin`, then `call_line`, then `name` -- `call_sites`' own
  storage order isn't self-time order). Ported identically, same column layout and tie-break rule,
  to the C++ port's own `formatProfileReport()` in `cli_lib.cpp`, so a report generated by either
  CLI on the same script has the same shape (the timing numbers themselves naturally differ run to
  run -- real wall-clock measurement). An unwritable output path returns exit code 1 with an
  `error: ...` message, matching every other file-write failure this CLI already handles.

  **Sort/filter/CSV options, added right after**: the original `--profile` always sorted by self
  time and always emitted plain text -- no way to sort by cumulative time or call count, no way to
  cut a large (Anklet.scad's own real-world script produces 800+ call-site rows) report down to
  just the rows that matter, and no machine-readable output. `--profile-sort {self,cumulative,
  calls,name}` (default `self`) and `--profile-min-self SECONDS`/`--profile-min-calls N` (both
  default `0`, reproducing the original unfiltered behavior) feed a new
  `_select_and_sort_call_sites()` (filter, then sort by the chosen key, every non-`name` order
  keeping the original tie-break rule); `_format_profile_report()` now dispatches to
  `_render_profile_report_text()` (the original layout, unchanged) or the new
  `_render_profile_report_csv()` (`--profile-format {text,csv}`, default `text`) -- CSV's summary
  lives in `#`-prefixed comment lines ahead of the real header/data rows (skippable via
  `pandas.read_csv(..., comment="#")` or `grep -v '^#'`), built with stdlib `csv.writer` (RFC-4180
  quoting for free, no hand-rolled escaping needed). `--profile-sort`/`--profile-format` are
  `argparse` `choices=` arguments -- invalid values exit 2 via `argparse`'s own mechanism, the same
  precedent `--format`'s own `choices=` already set, not a new inconsistency. Ported identically
  (same option names/defaults/CSV columns/comment convention) to the C++ port's own
  `selectAndSortCallSites()`/`renderProfileReportText()`/`renderProfileReportCsv()` in
  `cli_lib.cpp` -- its hand-rolled arg parser has no `choices=` equivalent, so invalid values there
  are checked explicitly and exit 1, not 2 (a harmless cross-CLI exit-code asymmetry that falls out
  of each side's own pre-existing parser). Manually cross-checked against Anklet.scad (see this
  repo's own perf-tracking memory): `--profile-format csv --profile-sort cumulative
  --profile-min-calls 100` cuts an 873-line unfiltered report down to 107 lines.

  **Stop/Restart/`info variables|modules|functions`/blank-line-repeat, added right after**: a
  meaningfully bigger debug-REPL change than any prior one -- the first to touch `cli.main()`'s own
  control flow rather than just `DebugRepl` internals. Before this, the CLI ran `evaluate()` exactly
  once per process; "quit" was the only way to abort, always exiting the whole CLI (exit 1, no
  export). Confirmed with the user up front (a real design decision, not an obvious one): **`stop`**
  aborts the current evaluation but -- unlike `quit` -- returns to the *pre-run* prompt instead of
  exiting, mirroring gdb's `kill` vs `quit` distinction; **`restart`** (while paused) aborts and
  immediately re-runs from the top with no intervening prompt, and is *also* accepted at the pre-run
  prompt itself (behaving exactly like `run` there) so a user who just typed `stop` can reflexively
  type `restart` again without hitting "Undefined command". `cli.main()` is now a `while True:` loop
  around one `Evaluator`+`evaluate()` attempt per iteration -- without `--debug` this loop still runs
  exactly once (every `repl`-related branch is `if args.debug`-guarded), so it's the identical
  single-pass behavior as before. `DEBUGGING_STOPPED_MESSAGE` (a new exported constant in
  `evaluator.py`, replacing an inline `"Debugging stopped."` literal at the one `raise EvalError(...)`
  site) is how `cli.py` tells "the debugger itself asked to abort" apart from a genuine script error
  by comparing `str(e)` against it, rather than trusting `DebugRepl`'s own mutated
  `_post_run_action` alone -- `error_break()` discards its own `_interact()`'s return value entirely
  (evaluation aborts regardless, since the *real* error is what's about to re-raise), so if a user
  typed `stop` while inspecting a genuine `assert()` failure, `_post_run_action` would still get set
  to `"stopped"` as a side effect; only the exception's own message being the exact sentinel
  prevents a real error from ever being misreported as a clean "Evaluation stopped." Verified this
  exact scenario deliberately, not just assumed safe by inspection. `DebugRepl.prepare_for_run()`
  resets exactly what a fresh run needs (`_break_on_first`/`_step_cmd`/`_step_to_child_targets`/
  `_pending_mods`) -- breakpoints, print-counter, and declared-function/module names all carry over,
  matching gdb's own `run`-after-`kill` behavior.

  `info variables`/`info modules`/`info functions` extend the existing `info breakpoints` dispatch.
  `info variables` is paused-only (reuses the exact `visible_vars` dict `print` already reads --
  zero new plumbing) and reports `No variables to show before "run".` at the pre-run prompt rather
  than silently doing nothing. `info functions`/`info modules` are static (available in both
  prompts): `_collect_declarations()` (`cli.py`) scans the fully use-resolved top-level node list
  (the same `nodes` `resolve_use_scopes()` already returns) for `FunctionDeclaration`/
  `ModuleDeclaration` instances directly -- only *top-level* declarations, matching what's
  realistically ever declared in practice. `DebugRepl.set_declared_names()` receives already-sorted
  `DeclInfo(name, params, origin, line)` dataclass instances computed once by `cli.py` (which has
  direct AST access), keeping `DebugRepl` itself fully decoupled from parser types beyond this one
  dataclass -- reusing each `ParameterDeclaration`'s own `__str__` for the `name`/`name=default`
  rendering rather than re-deriving that formatting logic a second time.

  Blank-Enter-repeats-last-command (`step`/`next`/`child`/`restart`/`continue`/`finish`/`list`, plus
  `restart`/`list` specifically at the pre-run prompt too) mirrors gdb's own convention exactly.
  `_last_repeatable_cmd`/`_last_repeatable_arg` are set at each relevant dispatch branch (not looked
  up via a separate command-name set, so alias handling falls out of each branch's own existing
  `cmd in (...)` check rather than a second, easy-to-drift-out-of-sync enumeration) and persist
  across the whole debug session (matching gdb's own single persistent "last command" register) -- a
  blank line before any repeatable command has ever been issued is an unchanged no-op.

  Manually verified every one of these end to end (not just via the test suite), same exact script
  and command sequence run against both this reference's own CLI and the C++ port's side by side --
  outputs matched. Ported identically (same command names/aliases, same `PostRunAction`-equivalent
  string values, same shared `DEBUGGING_STOPPED_MESSAGE` constant, same declared-lines scan) to the
  C++ port -- see that port's own `CLAUDE.md` for its side of the writeup, including why a
  `Scope`-enumeration API in the `openscad_cpp_parser` submodule was considered and rejected in favor
  of scanning the node list the CLI already owns.

  **`list <name>`, added right after**: the user's own follow-up question ("Can you do a
  `list funcname`?") surfaced a real gap -- `_list_source()` only ever accepted a line number,
  silently falling back to the current/start line for anything else (including a function name --
  not an error, just quietly wrong). Fixed with a `[line|name]` argument: an unparsable-as-int `arg`
  is now looked up by name in `_declared_functions`/`_declared_modules` (the same `DeclInfo` data
  `info functions`/`info modules` already has) and jumps to *that declaration's own* file:line --
  which may be a completely different file than wherever the debugger happens to be paused right now
  (e.g. a `use <file>`-injected declaration), so `list_origin` is set from `decl.origin`, not the
  caller's own `origin` parameter. The one real design question, flagged by the user up front:
  OpenSCAD's function and module namespaces are genuinely separate (`function foo(x)=x;` and
  `module foo(){}` can coexist), so an unqualified `list foo` must handle real ambiguity, not just
  pick one arbitrarily. Solved with `function:name`/`module:name` qualifiers -- reusing this REPL's
  own existing `[file:]line` colon convention (`break`/`delete` already parse a `prefix:rest` this
  exact way) rather than inventing a new syntax -- and an explicit "Both a function and a module are
  named ..." error (listing nothing) when `list foo` is ambiguous and unqualified. No `"->"` marker
  is drawn for a name-based jump (`current_line` is cleared) since it's not the paused line. An
  unrecognized, non-numeric `arg` (name matches neither namespace) now prints
  `No symbol "..." in current context.` instead of the old silent current/start-line fallback -- a
  small, deliberate behavior change on an edge case with no prior test depending on its output.
  Manually verified all five cases (unambiguous name, ambiguous name error, `function:`-qualified,
  `module:`-qualified, unknown name) plus that this still works identically while paused. Ported
  identically (same qualifier syntax, same error messages) to the C++ port -- see that port's own
  `CLAUDE.md` for its side of the writeup.

### Design Patterns

- **Callback injection, not GUI coupling**: `Evaluator.__init__` takes `echo_fn`/`debug_hook`/
  `error_break_fn`/`return_hook`, all optional plain callables. No `QObject`, no signals — a
  caller (like BelfrySCAD's debugger) wires these to its own event system; the evaluator itself
  has zero GUI-toolkit dependencies. See `examples/minimal_debugger.py` for a minimal, runnable
  `debug_hook` integration (tracing, breakpoints, variable overrides via `mods`).
- **`EvalContext`**: `__slots__`-based, threaded through recursive evaluation, carries lexical
  scope + `$`-variable dynamic scope + `let` bindings + color/children state. `child_ctx()`/
  `call_ctx()`/`let_child_ctx()` derive new contexts for different scoping situations.
- **Content-hash geometry cache**: `ManifoldCache` (opt-in, `None` by default) keys on each
  `CSGNode`'s resolved content, letting `generate_tree()` skip recomputing unchanged Manifold work
  across renders/debugger pauses. See `examples/manifold_cache_reuse.py` for a minimal, runnable
  demonstration of the speedup on an unchanged subtree.
- **Profiling**: opt-in (`Evaluator(profile=True)`), self-time + cumulative-time per call site
  (not per declaration), zero overhead when off.

See `docs/evaluator.md` for the full reference: scope processing, assignment order, the complete
built-ins table, 2D/3D geometry handling, error format, `$variables` scoping, `include`/`use`, and
the Manifold provenance / AST-to-geometry-ID mapping API used by BelfrySCAD's WYSIWYG picking.

### Test Organization

- `tests/test_evaluator.py` — the whole test suite: built-ins, scoping, CSG tree, profiling,
  `ManifoldCache`, error handling, real-script regression cases
- `tests/test_examples.py` — runs every script under `examples/` as `__main__`, so their own
  `assert`s double as regression coverage against the examples drifting out of sync with the API
- `tests/test_cli.py` — the `openscad-evaluator` CLI end to end: all four export formats, error
  exit codes, and the `--debug` REPL (breakpoints, step/next/finish/child, print, set, quit,
  Ctrl+C-during-a-running-eval via `request_pause()`) driven by monkeypatching `input()` with
  canned responses
- `tests/test_export.py` — `export.py` at the unit level: format dispatch, empty-geometry errors,
  and the pure-Python 3MF writer's structure/colors/round-trip through `_load_3mf` (and a hard
  assertion that it never imports `lib3mf`)
