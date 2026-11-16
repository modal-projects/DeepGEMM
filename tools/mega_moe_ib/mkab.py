#!/usr/bin/env python3
"""Build ONE header tree that holds several kernels, so a single dev-server request can time them back to back.

    mkab.py OUT  NAME=TREE[:DEFINE] ...

Run it inside a checkout of the DeepGEMM repository (or set DG_REPO to one): commit-ish arguments are resolved
there and OUT becomes one of its worktrees.

  OUT     directory to (re)create; it becomes a detached git worktree of the FIRST tree's commit, so that
          `devclient.py ab --repo OUT --force` can diff it against the server's base commit.
  TREE    a commit-ish of the DeepGEMM repository, or a directory (a worktree: its files as they are on disk,
          uncommitted edits included).
  DEFINE  the arm define that selects this tree (tested as `defined(D) and D`). The LAST tree needs none: it is
          the fallback. Selection is tested in the order given, so put the most specific define first.

Typical uses:
    # experiment vs base:        arm EXP = DG_MEGA_MOE_EXP=1     arm BASE = (empty)
    mkab.py OUT exp=<experiment checkout>:DG_MEGA_MOE_EXP base=<base tag>
    # experiment vs base vs a reference kernel in one request (on a server whose host serves all three: the base
    # kernel's arm then carries DG_MEGA_MOE_IB_COMPAT_ABI=1, see docs/mega_moe_ib/REPRODUCTION.md):
    mkab.py OUT exp=<experiment checkout>:DG_MEGA_MOE_EXP base=<base tag>:DG_MEGA_MOE_IB_COMPAT_ABI old=<reference branch>

The selecting define must reach the kernel source. A host that prints only the plan into the JIT source (the
in-repo host) does not forward arm variables as defines: there, compare two trees in two requests instead.

Every header under deep_gemm/include/deep_gemm that is byte-identical in all trees that have it is copied once.
A header that differs becomes a dispatcher (#if DEFINE ... #include <deep_gemm/_ab/NAME/path>), the variants live in
deep_gemm/include/deep_gemm/_ab/NAME/. A tree that lacks a dispatched header gets `#error` in its branch.
The kernel code of an arm is the same as from its own tree (same preprocessed text up to #line), so its code hash
(`code [ARM]: size=hash` in the result) must equal the hash of the same arm built from the tree alone: check it.
"""
import os
import pathlib
import subprocess
import sys

GIT = os.path.expanduser(os.environ.get('DG_REPO') or subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip() or '.')
INC = 'deep_gemm/include/deep_gemm'


def sh(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, **k).stdout


def read_tree(spec):
    """{relative header path: text}, description"""
    p = pathlib.Path(os.path.expanduser(spec))
    if p.is_dir():
        root = p / INC
        files = {str(f.relative_to(root)): f.read_text() for f in root.rglob('*') if f.is_file() and '/_ab/' not in str(f)}
        head = sh('git', '-C', str(p), 'rev-parse', '--short', 'HEAD').strip()
        dirty = subprocess.run(['git', '-C', str(p), 'diff', '--quiet', 'HEAD', '--', INC]).returncode != 0
        untracked = sh('git', '-C', str(p), 'ls-files', '--others', '--exclude-standard', INC).strip()
        return files, head + ('+dirty' if dirty or untracked else ''), sh('git', '-C', str(p), 'rev-parse', 'HEAD').strip()
    commit = sh('git', '-C', GIT, 'rev-parse', spec).strip()
    names = sh('git', '-C', GIT, 'ls-tree', '-r', '--name-only', commit, INC).split()
    files = {n[len(INC) + 1:]: sh('git', '-C', GIT, 'show', f'{commit}:{n}') for n in names}
    return files, commit[:7], commit


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = pathlib.Path(os.path.expanduser(sys.argv[1])).resolve()
    trees = []
    for a in sys.argv[2:]:
        name, _, rest = a.partition('=')
        if rest.count(':') and not pathlib.Path(os.path.expanduser(rest)).exists() and rest.rsplit(':', 1)[1].startswith('DG_'):
            spec, define = rest.rsplit(':', 1)
        else:
            spec, define = rest, None
        files, desc, commit = read_tree(spec)
        trees.append(dict(name=name, define=define, files=files, desc=desc, commit=commit))
    if any(t['define'] is None for t in trees[:-1]):
        sys.exit('every tree but the last needs a :DEFINE')
    if out.exists():
        subprocess.run(['git', '-C', GIT, 'worktree', 'remove', '--force', str(out)], check=True)
    sh('git', '-C', GIT, 'worktree', 'add', '-q', '--detach', str(out), trees[0]['commit'])
    root = out / INC
    # start from an empty include tree: every file is written below
    for f in sorted(root.rglob('*'), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    paths = sorted(set().union(*(t['files'] for t in trees)))
    dispatched = []
    for rel in paths:
        have = [t for t in trees if rel in t['files']]
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if len({t['files'][rel] for t in have}) == 1:
            dst.write_text(have[0]['files'][rel])
            continue
        lines = []
        for i, t in enumerate(trees):
            if t['define'] is not None:
                lines.append(('#if' if i == 0 else '#elif') + f' defined({t["define"]}) and {t["define"]}')
            else:
                lines.append('#else')
            if rel in t['files']:
                v = root / '_ab' / t['name'] / rel
                v.parent.mkdir(parents=True, exist_ok=True)
                v.write_text(t['files'][rel])
                lines.append(f'#include <deep_gemm/_ab/{t["name"]}/{rel}>')
            else:
                lines.append(f'#error "{rel} does not exist in tree {t["name"]}"')
        lines.append('#endif')
        dst.write_text('\n'.join(lines) + '\n')
        dispatched.append(rel)
    print(f'{out}: ' + ', '.join(f'{t["name"]}={t["desc"]}' + (f' [{t["define"]}]' if t['define'] else ' [fallback]') for t in trees))
    print(f'  {len(paths)} headers, {len(dispatched)} dispatched: ' + ' '.join(dispatched))


if __name__ == '__main__':
    main()
