# AGENTS.md — Repository Instructions for AI Agents

This repository uses the **GRAVITAS** engineering reliability protocol to ensure discipline, scoped execution, and proof-backed verification.

## The 6 Non-Negotiable Invariants

1. **Ground before mutating**: Read the target file, its tests, and its callers before editing. Never act on assumed file state.
2. **Stay in declared scope**: Only modify files within the declared task scope.
3. **Completion needs evidence**: A task or bug fix is complete only when a fresh validator (`pytest`) execution passes after the last edit. Stale green runs or prose assertions do not count.
4. **Never exact-retry a failure**: If a test or command fails, diagnose the root cause and change the strategy before retrying.
5. **Satisfy the contract**: Acceptance criteria and required tests gate completion.
6. **Say what you cannot verify**: Mark uncertain or unverifiable claims explicitly.

## Verification

Before claiming any task is complete, run the verification suite and verify clean output:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
or via the Gravitas verification runner:
```powershell
.\.venv\Scripts\python.exe .\.agents\skills\gravitas\scripts\verify.py
```

## Report Format

Always structure completion responses with:
- **What Changed**: Specific files and behavior changes.
- **Evidence**: Verifiable command output and test suite results.
- **Why**: Rationale and underlying architectural decisions.
