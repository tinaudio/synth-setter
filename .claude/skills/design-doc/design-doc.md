---
name: design-doc
description: Guide structured creation of design docs for systems, features, or architectural decisions. Synthesizes best practices from leading engineering organizations and ML-specific templates.
---

# Design Doc Skill

Use this skill when the user wants to write a design doc, technical spec, RFC, architecture document, or system design document.

## Trigger Conditions

- User mentions: "design doc", "write a design doc", "technical spec", "system design", "RFC", "architecture doc"
- User is about to build something non-trivial and needs to document the approach
- User asks to document an existing system's architecture

## Workflow

This skill provides the **template and structure** for design documents.

### Step 1: Context Questions

Before starting, ask:
1. **What type of document?** (new system design, migration/refactor, feature addition, decision record)
2. **Who is the primary audience?** (internal team, open source contributors, recruiting/portfolio, future self + AI)
3. **What's the desired impact?** (understand & contribute, evaluate decisions, both)
4. **Where should it live?** (docs/ in repo, project root, wiki, etc.)

### Step 2: Select Sections

Not every design doc needs every section. Use this decision guide:

| Section | When to Include | When to Skip |
|---------|----------------|--------------|
| Context & Motivation | Always | Never |
| Goals & Non-Goals | Always | Never |
| Design Principles | System has >3 governing principles | Simple feature addition |
| Success Metrics | Measurable outcomes matter | Pure refactoring |
| System Overview | Always | Never |
| Data Flow Diagrams | System has >2 components | Single-module change |
| Detailed Design | Always for new systems | Light for incremental changes |
| Alternatives Considered | Always (even if brief) | Never — reviewers always ask "why not X?" |
| Operations & Infrastructure | System has operational concerns | Pure library/algorithm work |
| Implementation Roadmap | Multi-stage implementation | Single PR |
| Open Questions & Risks | Always | Never |
| Known Limitations | System has intentional trade-offs worth documenting | Trivial scope |
| Appendices (Tech Stack, Glossary) | External audience, open source | Internal team that knows the stack |

### Step 3: Draft Using Template

Use the template below.

---

## Template

```markdown
# Design Doc: [Title]

> **Status**: Draft | In Review | Approved | Implemented
> **Author**: [name]
> **Last Updated**: [date]

---

## 1. Context & Motivation

[Why does this system/change exist? What problem does it solve? What prompted it now?]

Guidelines:
- 2-4 paragraphs max
- Lead with the problem, not the solution
- Include enough context that a new team member understands the motivation
- If this is a fork/extension of existing work, explain the relationship

## 2. Goals & Non-Goals

### Goals
- [Frame as product outcomes — what the user experiences, not what you implement]
- [Good: "No dangerous buttons — every command is safe to run at any time"]
- [Bad: "Deterministic outputs + last-writer-wins storage"]

### Design Principles
- [Governing principles that motivate architecture decisions]
- [State near the top, reference section numbers where each is realized in detail]

### Success Metrics
| Metric | Target | How to Measure |
|--------|--------|---------------|
| ... | ... | ... |

### Non-Goals
- [Explicit things this system does NOT attempt — prevents scope creep]

### Do Not Build
- [Explicit list of tempting features/approaches that must NOT be built]
- [Makes anti-goals as visible as goals and prevents scope creep]

## 3. System Overview

[3-5 sentences describing the high-level architecture. A reader should understand the system's shape after reading this section alone.]

## 4. Data Flow & Architecture Diagrams

[ASCII or mermaid diagrams. Include:
- Component interaction diagram
- Data flow diagram
- File/storage structure diagram (if applicable)]

## 5. Detailed Design

### 5.1 [Core Design Decision 1]
### 5.2 [Core Design Decision 2]
### ...

[For each subsection:
- State the decision
- Explain WHY this approach was chosen
- Describe failure modes and how they're handled
- Include code snippets only when they clarify the design (not for API specifications)]

## 6. [Domain-Specific Section]

[For ML systems: Methodology, Data, Validation]
[For distributed systems: Coordination, Consistency, Fault Tolerance]
[For APIs: Interface Design, Versioning, Migration]

## 7. Alternatives Considered

[Imagine a critic reading this doc and asking "why didn't you just use X?" Preempt those questions here.]

### 7.1 [Alternative A]
**[Rejected | Considered, deferred | Interesting, worth revisiting].**
[What it is, why it seemed reasonable, why it was dropped]

## 8. Operations & Infrastructure

### Security & Credentials
### Monitoring & Observability
### Cost Model

## 9. Implementation Roadmap

[Summary table linking to detailed implementation plans]

## 10. Open Questions, Risks & Limitations

### Open Questions
1. [Numbered list of unresolved decisions with options]

### Known Limitations
[Intentional trade-offs and architectural constraints — not bugs. For each: what the limitation is, why it's acceptable at current scale, and what the mitigation path is if it becomes a problem. Distinct from risks (probabilistic) and non-goals (deliberate exclusions).]

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|

---

## Appendix A: Tech Stack
## Appendix B: References
## Appendix C: Glossary
```

---

## Quality Gates

Before considering the doc complete, apply these tests:

### Vacation Test
> "Could a teammate who has never seen this system implement it from this doc alone?"

If no: the doc is missing critical design details or context. Add them.

### Skeptic Test
> "Are the most likely reviewer objections addressed preemptively?"

Common objections:
- "Why not use [existing tool/framework]?" → Alternatives Considered section
- "How does this handle [failure mode]?" → Detailed Design failure modes
- "What happens when [scale/edge case]?" → Design should address or explicitly scope out
- "Is this over-engineered?" → Non-Goals section, thin abstractions principle

### Length Check
- 5-15 pages for most systems (Google guideline: 10-20 pages for large systems)
- If shorter: probably missing detail
- If longer: probably including implementation details that belong in code

---

## Anti-Patterns

Avoid these in design docs:

| Anti-Pattern | Why It's Bad | What to Do Instead |
|-------------|-------------|-------------------|
| Raw API specifications | Becomes stale, belongs in code | Describe the interface concept, link to generated docs |
| Irrelevant code snippets | Noise, hard to maintain | Only include code that clarifies a design decision |
| Data dumps | Unreadable, no insight | Summarize with a table, link to raw data |
| Excessive bullet points | Feels comprehensive but lacks substance | Write prose for complex reasoning, bullets for lists |
| Fill-in-the-blank thinking | Template sections filled mechanically | Skip sections that don't apply, add custom ones that do |
| Writing for the author | Author already knows the context | Write for the reader who doesn't — use the Vacation Test |
| Burying the lede | Problem/motivation buried after technical details | Context & Motivation comes first |

---

## Writing Guidelines

These patterns produce stronger design docs:

**Language & Style:**
- Use plain language for section headers — "Operations & Infrastructure" not "Cross-Cutting Concerns"
- Don't frame changes as v1 → v2 migrations when readers won't know v1 exists. Describe the system as-is.
- Prefer "fully parallel" over "embarrassingly parallel" — skip jargon that adds no precision

**Goals:**
- Frame goals as product outcomes the user experiences, not implementation details
- Include explicit "Do not build" lists to prevent scope creep
- Validation/data integrity concerns are important enough to be goals, not buried in subsections
- Design principles should be stated near the top (Goals section) with detailed realization mapped later

**Concreteness:**
- Show concrete examples: actual CLI output, directory trees, JSON schemas, Pydantic models
- When describing concurrency/safety, work through specific scenarios rather than abstract guarantees
- Use comparison tables to separate concerns: input spec vs output record, logs vs reports

**Structure:**
- Trim reference lists to 1-2 most important sources in the doc. Full lists go in supporting materials.
- Aspirational features that probably won't be implemented go in appendices, not main sections

---

## Audience Adaptation

Adjust tone and depth based on audience:

| Audience | Emphasis | Tone | Extra Sections |
|----------|----------|------|---------------|
| **Internal engineering team** | Design decisions, tradeoffs, failure modes | Direct, assumes domain knowledge | Skip glossary, light on context |
| **Open source contributors** | Setup, architecture overview, contribution paths | Welcoming, explains domain terms | Glossary, tech stack, setup guide |
| **Recruiting / portfolio** | Quality of thinking, clear communication | Engineering-first but accessible | Clean diagrams, polished prose |
| **Future self + AI assistants** | Decision rationale, what was tried and rejected | Reference-style, searchable | Alternatives considered, open questions |

---

## Lessons Learned

Patterns and improvements discovered through real design doc authoring:

**Document Structure:**
- **Navigable index at the top.** For docs longer than ~5 sections, add a table of contents with anchor links. Readers need to jump to sections, not scroll.
- **Separate implementation details from design decisions.** The order should be: requirements/goals → proposed solution & interfaces → algorithms/logic/important details → implementation details (libraries, config, fine details). Don't mix what logging framework you use into the section about your coordination model.
- **Future work in a walled-off section.** All future/aspirational work belongs in a clearly labeled "Out of Scope" or "Future Work" section. It must NOT be referenced from the main design sections — otherwise readers can't tell what's built vs. what's planned.
- **Typical workflow near the top.** Show the end-to-end CLI example before the detailed design. Readers understand the system shape from a concrete example faster than from architecture diagrams.
- **Artifact taxonomy in one place.** When a system produces multiple files (configs, specs, reports, logs, output records), list them all in a single table with: artifact name, format, who produces it, who consumes it, and what it's for. A flow diagram helps too.

**Design Rigor:**
- **Add a failure modes section.** Go beyond the happy path. Enumerate concrete failure scenarios (corrupt overwrites, partial uploads, non-atomic cross-file writes, stale markers) with consequence and mitigation for each. Readers (and your future self) need this.
- **Document storage consistency assumptions.** If your system depends on storage semantics (strong consistency, conditional writes, atomic operations), document exactly what you're relying on and link to the provider's documentation to prove the assumptions hold.
- **Cross-reference sections.** When one section references another (e.g., "see §7.6 for concurrency"), use hyperlinks, not just section numbers. Readers in markdown viewers can click through.
- **Link external technologies.** First mention of any external service or library should link to its documentation. Glossary entries should also have links.

**Naming:**
- **Names should describe semantics, not mechanism.** `dataset.complete` (completion marker) over `dataset.lock` (implies mutex). `debug log` (what you'd use it for) over `event log` (mechanism). `status` (what the command shows) over `poll` (implies provider-specific behavior).
- **Alternatives comparison chart.** When comparing multiple alternatives, a table with ticks/crosses on key criteria (cost, ops burden, resumability, etc.) followed by 2-3 sentences per meaningful alternative works better than paragraph-per-alternative for 5+ options.
- **Split alternatives into tiers.** Keep "Alternatives Considered" for alternatives that need real explanation (why they were tempting, why rejected). Move trivial rejections to a "Discarded Choices" table with one-line reasons.

**State Models & Lifecycle:**
- **Frame lifecycle markers as commit points.** `.valid` is "staged shard committed" not "validation passed." A three-marker state machine (started → staged → finalized) is clean and easy to reason about. Name markers by what they commit, not what triggered them.
- **Config vs CLI flag: the "what it is" test.** If something changes what the artifact *is* (not just how it's produced), it belongs in the frozen spec — not as a CLI flag. Example: `output_format` is part of the dataset definition because HDF5 and WDS produce different artifacts under the same `run_id`.
- **Lexicographic over timestamps for deterministic selection.** When a design needs a deterministic tiebreaker (e.g., picking one shard from multiple attempts), prefer lexicographic ordering over timestamps. Timestamps depend on clock accuracy and storage behavior; lexicographic ordering is purely deterministic.

**Validation & Trust Boundaries:**
- **Tiered validation.** Expensive validation once at the source (worker), cheap checks frequently (reconciliation), moderate checks at gates (finalize). Don't re-validate what's already been validated — document the trust chain that justifies skipping re-validation.
- **Pydantic at trust boundaries, dataclasses internally.** Pydantic is for where data enters the system from external sources (user config, JSON from R2). Dataclasses are for internal contracts where data is already validated and you just need typed containers. Implementation guidance about which tool to use where belongs in implementation details, not architecture sections.

**Process:**
- **"Who watches the watchmen" section.** For systems with self-monitoring (reconciliation checking its own work), explicitly address: what if the monitoring itself has a bug? Document defense-in-depth (multiple validation layers, end-to-end checks, manual spot-checking).
- **Data plane vs control plane.** When both data and coordination flow through the same infrastructure, call it out explicitly. It's a simplification (one system to manage) with a trade-off (single point of failure for both planes).
- **Evaluate review feedback: accept, refine, or push back.** When incorporating external feedback, don't blindly implement. For each item: accept as-is, refine (accept the concern but propose a better solution), or push back with clear rationale. State the case, propose a counter, defer to the reviewer for the final call.

---

## References

These templates and articles informed this skill:

**Template structure:**
- [Industrial Empathy — Design Doc: A Design Doc](https://www.industrialempathy.com/posts/design-doc-a-design-doc/)
- [Industrial Empathy — Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/)
- [Eugene Yan — ML Design Doc Template](https://github.com/eugeneyan/ml-design-docs)
- [HashiCorp PRD Template](https://www.hashicorp.com/en/how-hashicorp-works/articles/prd-template)
- [HashiCorp RFC Template](https://www.hashicorp.com/en/how-hashicorp-works/articles/rfc-template)
- [Chromium Design Documents](https://www.chromium.org/developers/design-documents/)
- [freeCodeCamp — How to Write a Good Software Design Document](https://www.freecodecamp.org/news/how-to-write-a-good-software-design-document-66fcf019569c/)

**Writing style & process:**
- [Eugene Yan — Writing Docs: Why, What, How](https://eugeneyan.com/writing/writing-docs-why-what-how/)
- [Eugene Yan — ML Design Docs](https://eugeneyan.com/writing/ml-design-docs/)
- [Eugene Yan — Design Patterns for ML Systems](https://eugeneyan.com/writing/design-patterns/)
- [Basecamp — Shape Up: Write the Pitch](https://basecamp.com/shapeup/1.5-chapter-06#examples)
- [Caitie McCaffrey — Design Docs, Markdown, and Git](https://www.caitiem.com/2020/03/29/design-docs-markdown-and-git/)
- [The Anatomy of an Amazon 6-Pager](https://writingcooperative.com/the-anatomy-of-an-amazon-6-pager-fc79f31a41c9)
- [Vicki Boykis — Writing for Distributed Teams](https://vickiboykis.com/2021/07/17/writing-for-distributed-teams/)
- [Stitch Fix — Remote Decision Making](https://multithreaded.stitchfix.com/blog/2020/12/07/remote-decision-making/)
- [Machine Words — Writing Technical Design Docs](https://medium.com/machine-words/writing-technical-design-docs-71f446e42f2e)
- [Machine Words — Writing Technical Design Docs Revisited](https://medium.com/machine-words/writing-technical-design-docs-revisited-850d36570ec)

**Key insight from Google's practice:** "The benefits in organizational consensus around design, documentation, senior review must justify the creation overhead." Not everything needs a design doc — use the 1 engineer-month rule as a threshold.
