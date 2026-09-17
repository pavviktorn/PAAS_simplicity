#!/usr/bin/env python3
"""Scope-aware undefined-name checker (a focused stand-in for pyflakes, which isn't installed and
which I will not add to the project's deliberately isolated venv).

Written because a real bug shipped past `py_compile`: a patch introduced `none_r` in sweep.py where
the defined variable was `base_row`. Python only raises NameError when the line EXECUTES, and that
line sits after every imbalance experiment and before sweep.json is written -- so a ~20-minute sweep
would run to completion and then discard all of its work. Compilation cannot catch this; a scope walk
can.

Reports a name only when it is read in a scope where nothing ever binds it, and it is not a builtin
and not bound at module level or in an enclosing function. Conservative by design: it prefers missing
a dubious case to crying wolf.
"""
import ast, builtins, sys

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__package__"}


def bound_names(node):
    """Every name this scope binds, anywhere in it (Python has no block scope)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def walk_scopes(node, enclosing, path, findings):
    """enclosing = set of names visible from outer scopes."""
    here = bound_names(node) | enclosing
    # names read directly in THIS scope (not inside nested function bodies)
    nested = [n for n in ast.iter_child_nodes(node)]
    def reads(n, stop_at_funcs=True):
        for c in ast.walk(n):
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                yield c
    inner_funcs = [n for n in ast.walk(node)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node]
    inner_ids = set()
    for f in inner_funcs:
        for c in reads(f):
            inner_ids.add(id(c))
    for c in reads(node):
        if id(c) in inner_ids:
            continue
        if c.id not in here and c.id not in BUILTINS:
            findings.append((path, c.lineno, c.id,
                             getattr(node, "name", "<module>")))
    for f in inner_funcs:
        walk_scopes(f, here, path, findings)


def check(path):
    tree = ast.parse(open(path).read(), path)
    findings = []
    walk_scopes(tree, set(), path, findings)
    return findings


def check_use_before_assign(path):
    """Module-level use-before-assignment. check() cannot see this: Python has no block scope, so a
    name assigned ANYWHERE in the module counts as bound and a load before that line still parses and
    still passes a scope walk. It only fails at RUN TIME -- which for a report generator means after
    the whole pipeline has finished. Real instance caught this way: a paragraph on page 2 referenced
    `m = test["metrics"]`, assigned on page 3, and the render died at the last step of a 4-hour run.
    Linear approximation: for module-level statements only, flag a Load whose line precedes every
    module-level binding of that name. Function bodies are skipped (they run later, by definition)."""
    tree = ast.parse(open(path).read(), path)
    first_bind = {}

    # MINIMUM line per name, not the first one ast.walk happens to reach. ast.walk is BREADTH-first, so
    # traversal order is depth order, not source order. Concretely: `for k, pg in enumerate(xs)` puts
    # `pg` inside a Tuple (depth 2 below the For) while a later `for pg in ys` has `pg` as a direct
    # child (depth 1) -- BFS therefore saw the LATER line first, setdefault locked it in, and every
    # read between the two lines was reported as use-before-assignment. That false positive fired on
    # real code (build_exp25.py:612) and the cost of noise here is that the tool stops being trusted.
    def bind(name, lineno):
        if name not in first_bind or lineno < first_bind[name]:
            first_bind[name] = lineno

    for node in tree.body:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bind(n.id, n.lineno)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(n.name, n.lineno)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    bind((al.asname or al.name).split(".")[0], n.lineno)
            elif isinstance(n, ast.arg):
                bind(n.arg, 0)
    # Skip function/class bodies (they run later) AND comprehension/lambda scopes. In a comprehension
    # the element expression is written BEFORE the `for` clause -- `f(v) for v in xs` -- so a
    # line-ordering heuristic flags the target as used-before-assigned when it is nothing of the kind.
    # Verified against real hits: infer.py `[... for p in paths]` and build_exp24 `{k: v for k, v in ...}`.
    skip = set()
    for node in tree.body:
        for c in ast.walk(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                              ast.Lambda)):
                for d in ast.walk(c):
                    skip.add(id(d))
    out = []
    for node in tree.body:
        for n in ast.walk(node):
            if id(n) in skip:
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                fb = first_bind.get(n.id)
                if fb is not None and fb > n.lineno and n.id not in BUILTINS:
                    out.append((path, n.lineno, n.id, fb))
    return out


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        for path, line, name, scope in check(p):
            print(f"{path}:{line}: undefined name '{name}' in {scope}()")
            bad += 1
    for p in sys.argv[1:]:
        for path, line, name, fb in check_use_before_assign(p):
            print(f"{path}:{line}: '{name}' used at module level before its assignment on line {fb}")
            bad += 1
    print(f"[check_names] {bad} finding(s) (undefined + module-level use-before-assignment) "
          f"across {len(sys.argv)-1} file(s)")
    sys.exit(1 if bad else 0)
