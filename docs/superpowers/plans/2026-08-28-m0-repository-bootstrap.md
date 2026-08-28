# M0 Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the approved monolithic roadmap into five independently executable milestone plans while preserving every original task and establishing an auditable public-repository baseline.

**Architecture:** Keep the approved design as the binding authority and the existing 21-task plan as the complete roadmap. Add a milestone index plus five self-contained execution plans that copy, rather than paraphrase, their assigned task bodies so task briefs remain decision-complete.

**Tech Stack:** Markdown, Git, PowerShell validation, Superpowers plan conventions.

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

## Global Constraints

- Preserve all 21 original tasks, their interfaces, commands, test expectations, and commit messages.
- M1 contains Tasks 1-8; M2 contains Tasks 9-12; M3 contains Tasks 13-14; M4 contains Tasks 15-17; M5 contains Tasks 18-21.
- Each milestone plan must repeat the required plan header, global constraints, relevant planned file structure, full task bodies, milestone acceptance gate, Git tag, and push gate.
- The existing complete plan remains the traceability roadmap and links to all five milestone plans.
- No source code, downloaded model, engine binary, screenshot, database, API key, or build output is added in M0.

---

### Task 1: Split the roadmap into five executable milestone plans

**Files:**
- Create: `docs/superpowers/plans/README.md`
- Create: `docs/superpowers/plans/2026-08-28-m1-foundation.md`
- Create: `docs/superpowers/plans/2026-08-28-m2-recognition-sync.md`
- Create: `docs/superpowers/plans/2026-08-28-m3-pikafish-analysis.md`
- Create: `docs/superpowers/plans/2026-08-28-m4-deepseek-coach.md`
- Create: `docs/superpowers/plans/2026-08-28-m5-release.md`
- Modify: `docs/superpowers/plans/2026-08-28-xiangqi-learning-agent.md`
- Modify: `docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md`
- Modify: `docs/status/m0-bootstrap.md`

**Interfaces:**
- Consumes: the approved design and the existing complete 21-task implementation plan.
- Produces: five task-complete milestone plans and one roadmap index.

- [ ] **Step 1: Create the milestone index and five self-contained plans**

Copy the assigned task sections verbatim from the complete plan. Add milestone-specific goals, acceptance gates, tag names `v0.1.0-m1` through `v0.1.0-m5`, and a rule that pushing occurs only after tests, review, privacy scan, and a clean worktree.

- [ ] **Step 2: Update roadmap and design milestone descriptions**

Link the five plans from the complete roadmap, change the design's implementation boundary from the old milestone count to the approved five milestone groups, and remove the obsolete statement that Git has not been initialized.

- [ ] **Step 3: Run structural validation**

Verify that all five files exist; all global Task numbers 1 through 21 appear exactly once across the five milestone plans; every plan contains `Goal`, `Architecture`, `Tech Stack`, `Spec`, and `Global Constraints`; all code fences are balanced; and no `TBD`, `TODO`, `FIXME`, or placeholder text remains.

- [ ] **Step 4: Update M0 status and commit**

Record the exact validation result in `docs/status/m0-bootstrap.md`, run `git diff --check`, and commit all M0 documentation changes with:

~~~powershell
git add docs
git commit -m "docs: split roadmap into milestone execution plans"
~~~
