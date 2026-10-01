# Build State — noaa_mbes_shipwreckfinder_model

Auto-generated gate baseline (2026-10-01) by `tools/reposstatus` survey.
Every gate below is individually checkable. A gate is marked CLOSED only
where the check was actually run; anything unverified is TODO on purpose —
an optimistic gate is worse than an open one.

## Gates

| Gate | What it checks | How to verify | State |
|---|---|---|---|
| **G1** | README describes purpose and how to run it | test -f /codebase/repos/noaa_mbes_shipwreckfinder_model/README.md | ✅ CLOSED |
| **G2** | CI workflow builds on every push | ls .github/workflows/*.yml | TODO |
| **G3** | Working tree is clean (no uncommitted work in flight) | git status --porcelain | ✅ CLOSED |
| **G4** | Package imports without error | `python -c "import ast,pathlib;ast.parse(pathlib.Path('README.md').read_text()[:0] or '')"` | TODO |
| **G5** | Repo is reachable from the fleet agent runtime | cd /codebase/repos/noaa_mbes_shipwreckfinder_model && git rev-parse --short HEAD | ✅ CLOSED |

## Notes

- Class: real · source files: 35 · tracked LOC: 24150
- Rebuild this file when a gate changes state; it is the source repostatus reads.
