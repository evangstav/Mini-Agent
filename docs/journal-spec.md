# Journal: Daily Work Extraction + Knowledge Learning

> Spec v0.1 — April 2026

## 1. Problem Statement

A day's work in Gas Town produces valuable signal scattered across beads,
git commits, mail, audit logs, and session transcripts. This knowledge
evaporates between sessions. We need an automated pipeline that:

1. Sweeps all daily activity into a structured journal
2. Extracts durable knowledge from journals
3. Feeds knowledge back into future agent sessions

## 2. Data Sources

### 2.1 Beads (issue tracking)
```bash
bd list --status=closed --since=today    # Issues completed today
bd list --status=open                     # Still in progress
bd list --status=in_progress              # Active work
```
Fields: id, title, description, status, assignee, closed reason, created/closed timestamps.

### 2.2 Git History
```bash
git log --since="midnight" --all --oneline   # Per-rig commits
```
Fields: hash, message, author, timestamp, files changed, insertions/deletions.

### 2.3 Audit Logs (Mini-Agent)
```
~/.mini-agent/sessions/<id>/transcript.jsonl
```
Fields: timestamp, tool name, arguments, success, result, duration_ms, token counts.

### 2.4 Session Transcripts (Claude Code)
```
~/.claude/projects/<slug>/transcripts/
```
JSONL files with full conversation history per session.

### 2.5 Mail & Escalations
```bash
gt mail inbox --all    # All messages (including archived)
```

### 2.6 Memory Files
```
~/.claude/projects/<slug>/memory/    # Existing Dream output
.claude/memory/                      # Project-level memories
```

## 3. Journal Format

### 3.1 File Location
```
~/.gastown/journals/YYYY/MM/YYYY-MM-DD.md
```

### 3.2 Template

```markdown
# Daily Journal — YYYY-MM-DD

## Summary
_One paragraph: what was the main thrust of today's work?_

## Work Completed
- [issue-id] Title — outcome (rig: rigname)
- [issue-id] Title — outcome (rig: rigname)

## Commits
| Rig | Commits | Files Changed | +/- |
|-----|---------|---------------|-----|
| conceptlang | 12 | 45 | +3823/-81108 |
| miniagent | 28 | 89 | +5200/-1200 |

## Key Decisions
- Decision 1 — why it was made, what alternatives were considered
- Decision 2 — context and rationale

## What Went Wrong
- Problem 1 — what happened, how it was resolved
- Problem 2 — root cause, mitigation

## Patterns Observed
- Pattern 1 — recurring theme across work
- Pattern 2 — what worked well / what didn't

## Token & Cost
- Total tokens: NNK across N sessions
- Estimated cost: $X.XX
- Most expensive operation: (what, tokens)

## Agent Performance
- Polecats spawned: N
- Completed successfully: N
- Got stuck / needed nudge: N
- Produced empty results: N

## Open Threads
- Issue still in progress
- Question unanswered
- Decision deferred

## Tomorrow Suggestions
- What should be prioritized next based on today's work
```

## 4. Knowledge Extraction

### 4.1 Memory Types (from existing Dream schema)

| Type | What to Extract | Example |
|------|----------------|---------|
| `feedback` | Process improvements, workflow corrections | "Always run integration pass after parallel polecats" |
| `project` | Technical decisions, architecture state | "Mini-Agent compaction uses structured 5-section template" |
| `user` | User preferences, working style | "User prefers architecture reviews before big merges" |
| `reference` | Links to external resources, tools | "Compaction research: gist.github.com/badlogic/..." |

### 4.2 Extraction Rules

Extract knowledge that is:
- **Durable** — true beyond today (not "deployed feature X" but "feature X uses pattern Y")
- **Actionable** — changes future behavior (not observations, but instructions)
- **Non-obvious** — not derivable from reading the code

Do NOT extract:
- What was done (that's the journal itself)
- Code patterns (read the code)
- Git history (use git log)

### 4.3 Deduplication

Before writing a new memory:
1. Read existing MEMORY.md index
2. Check if a memory on this topic already exists
3. If yes: update the existing memory file
4. If no: create new memory file + update index

## 5. Commands

### 5.1 `gt journal`

Generate today's journal by sweeping all data sources.

```bash
gt journal                        # Generate for today
gt journal --date=2026-04-01      # Specific date
gt journal --rig=miniagent        # Single rig only
gt journal --dry-run              # Preview without writing
```

**Implementation:**
1. Collect data from all sources (beads, git, audit, mail)
2. Build structured data payload
3. Call LLM with journal template + data
4. Write to `~/.gastown/journals/YYYY/MM/YYYY-MM-DD.md`
5. Print summary to terminal

### 5.2 `gt learn`

Extract durable knowledge from recent journals.

```bash
gt learn                          # Extract from last 3 days
gt learn --days=7                 # Last week
gt learn --journal=path           # Specific journal file
gt learn --dry-run                # Preview without writing
```

**Implementation:**
1. Read recent journal files
2. Read existing memory index (MEMORY.md)
3. Call LLM with extraction prompt + journals + existing memories
4. LLM returns: updates (modify existing), creates (new memories), deletes (stale)
5. Apply changes to memory files
6. Rebuild MEMORY.md index

### 5.3 `gt reflect`

Interactive review — human reads journal, adds annotations, corrects errors.

```bash
gt reflect                        # Open today's journal in $EDITOR
gt reflect --date=2026-04-01      # Specific date
```

After editing, `gt learn` can re-extract from the annotated journal.

## 6. Integration with Existing Systems

### 6.1 Dream Consolidation
Dream runs per-session (agent level). Journal runs per-day (human level).
They write to the same memory directory but at different granularity.
Dream captures agent-level patterns. Journal captures human-level decisions.

### 6.2 SystemPromptBuilder
Already injects `.claude/memory/` files into system prompts.
No changes needed — extracted knowledge automatically appears in future sessions.

### 6.3 Gas Town Hooks
Journal generation can be triggered by:
- Manual command (`gt journal`)
- Cron job (end of day)
- GT hook (on `gt rig dock` — when you're done for the day)

## 7. LLM Prompts

### 7.1 Journal Generation Prompt

```
You are generating a daily engineering journal from structured activity data.

DATA:
{beads_json}
{git_log}
{audit_summary}
{mail_summary}

TEMPLATE:
{journal_template}

Rules:
1. Be specific — use issue IDs, file names, concrete numbers
2. "Key Decisions" should capture WHY, not just WHAT
3. "What Went Wrong" should include root cause, not just symptoms
4. "Patterns Observed" should be generalizable insights
5. "Tomorrow Suggestions" should be actionable and prioritized
6. Keep it concise — someone should read this in 2 minutes
```

### 7.2 Knowledge Extraction Prompt

```
You are extracting durable knowledge from engineering journals.

JOURNALS:
{journal_content}

EXISTING MEMORIES:
{memory_index}

Extract knowledge that will improve future agent sessions. Return JSON:
{
  "updates": [{"file": "existing_file.md", "new_content": "..."}],
  "creates": [{"name": "topic", "type": "feedback|project|user|reference",
               "description": "one line", "content": "..."}],
  "deletes": ["stale_file.md"]
}

Rules:
1. Only extract DURABLE knowledge (true beyond today)
2. Only extract ACTIONABLE knowledge (changes behavior)
3. Check existing memories — UPDATE don't duplicate
4. Each memory should have Why + How to Apply lines
5. Keep memories concise (< 500 chars each)
```

## 8. Implementation Plan

| Component | File | LOC | Notes |
|-----------|------|-----|-------|
| Data collectors | `gt_journal/collectors.py` | ~150 | Sweep beads, git, audit, mail |
| Journal generator | `gt_journal/journal.py` | ~100 | LLM call with template |
| Knowledge extractor | `gt_journal/learn.py` | ~120 | LLM call with extraction prompt |
| CLI commands | `gt_journal/cli.py` | ~80 | gt journal, gt learn, gt reflect |
| Memory writer | `gt_journal/memory.py` | ~60 | Write/update memory files + index |
| **Total** | | **~510** | |

This could live in the GT codebase (Go) or as a standalone Python tool
that calls `bd` and `git` commands. Python is easier since it can reuse
the Mini-Agent LLM client for the LLM calls.

## 9. Open Questions

1. **Where should this live?** GT (Go) or Mini-Agent (Python) or standalone?
2. **Which LLM for journal/extraction?** Cheap model (Haiku) or smart model (Sonnet)?
3. **How often to run `gt learn`?** Daily? Weekly? On demand?
4. **Should journals be committed to git?** They contain useful project history.
5. **Privacy** — journals contain full work context. Should they be gitignored?
