# Token Efficiency Strategy: Complete Overview

**Date**: 2026-06-09  
**Context**: GitHub token pricing increased. We must optimize.  
**Goal**: Complete 4-week Sprint with <50 Copilot tool calls.

---

## What Changed

### Before (Inefficient)
- ❌ Exploratory searches (semantic_search, file_search)
- ❌ Running tests via tools (run_in_terminal)
- ❌ Reading entire directories
- ❌ Multiple read_file calls per task
- ❌ Asking "what should I do next?"
- **Result**: 200+ tool calls per sprint

### After (Token-Optimized)
- ✅ You read files locally (VS Code, Ctrl+F)
- ✅ You run tests locally (terminal, <1 second)
- ✅ You make decisions (read dev plan, choose file)
- ✅ Copilot creates/edits 1 file at a time
- ✅ Clear scope: "Day 1.1, implement X with Y methods"
- **Result**: <50 tool calls per sprint (4x reduction)

---

## Three Documents You Now Have

### 1. `.copilot-instructions.md` (Read This First!)
**What**: Rules for me (Copilot) + you (workflow)  
**When**: Before each session  
**Contains**:
- Tool usage rules (what I can/can't do)
- Efficient request patterns (examples)
- Testing strategy (no tool calls)
- Daily workflow (15-min setup)

**Action**: Open in VS Code, **bookmark it**. Reference daily.

---

### 2. `V7P3R_vX_DEVELOPMENT PLAN.md` (Your Roadmap)
**What**: 4 Sprints → 4 weeks → Phase 1 training  
**When**: Pick one day per session  
**Structure**:
- Sprint 1: Data serialization (Days 1-3)
- Sprint 2: HalfKA features (Days 1-4)
- Sprint 3: Training loop (Days 1-6)
- Sprint 4: Production (Days 1-5)
- Token efficiency rules (mandatory reading)

**Action**: Each morning, pick a DAY (e.g., "Day 1.1"), read task, tell me exactly what to implement.

---

### 3. `ENVIRONMENT_SETUP.md` (One-Time Setup)
**What**: VS Code configuration + terminal workflow  
**When**: Do once, reference when things break  
**Contains**:
- Terminal commands (activate venv, run tests)
- VS Code extensions (Pylance, Pytest)
- Settings (auto-format, error highlighting)
- Git workflow (optional checkpoints)
- Troubleshooting (common issues)

**Action**: Follow the 5-minute quick setup. Keep terminal open.

---

## The New Workflow (Simple!)

### Each Day (Repeat)

```
1. Read dev plan: "What's today's task?"
   Example: "Day 1.1: Implement binary_format_converter.py"

2. Open VS Code (same folder every day)
   - v10.0/
   - Terminal already open (Ctrl+`)

3. You handle this (NO Copilot calls):
   - Read src/existing_module.py (5 min)
   - Understand patterns (copy style)
   - Open new file: src/binary_format_converter.py
   - Write code (1-2 hours)
   - Save (Ctrl+S, auto-formats)
   - See errors (Problems panel)
   - Fix locally (you try first)

4. Test locally (NO Copilot calls):
   - Terminal: python -m pytest tests/test_binary_converter.py -v
   - Result: Green ✅ or Red ❌
   - If Red: See error, fix code, retry

5. Stuck after 15 min? Ask Copilot (1 specific question):
   "I'm implementing pgn_to_binary() in src/binary_format_converter.py.
   The test fails because [error message].
   Here's my code: [paste].
   How do I fix this?"

6. Copilot responds (1 tool call):
   replace_string_in_file: Fixes your code.
   You test again (terminal).
   Problem solved.

7. Move to next day (repeat).
```

### Weekly (Progress Check)

```
Friday 5pm:
- Run full test suite: pytest tests/ -v
- Check all tests pass
- Read dev plan for next week
- Commit to git (if using): git add .; git commit -m "Week X complete"
```

---

## Token Cost Breakdown

### Old Approach (Inefficient)

```
Per Day:
- semantic_search (explore codebase): 10 tokens
- file_search (find files): 5 tokens
- read_file (read multiple): 20 tokens
- get_errors (run tools): 5 tokens
- runSubagent (ask for help): 30 tokens
- run_in_terminal (test execution): 10 tokens
TOTAL: ~80 tokens/day

Per Sprint (5 work days):
80 × 5 = 400 tokens/sprint

For 4 sprints:
400 × 4 = 1,600 tokens (EXPENSIVE!)
```

### New Approach (Token-Optimized)

```
Per Day:
- You read files (VS Code): 0 tokens
- You run tests (terminal): 0 tokens
- You fix errors (IDE): 0 tokens
- 1 Copilot question (if needed): 5 tokens
- 1 create_file or replace_string: 10 tokens
TOTAL: ~15 tokens/day (optional)

Per Sprint (5 work days):
15 × 5 = 75 tokens/sprint

For 4 sprints:
75 × 4 = 300 tokens (4x SAVINGS!)
```

**Savings**: 1,300 tokens = money + fast iteration

---

## When to Use Copilot (Efficient Scenarios)

### Scenario 1: Implement a File (1 Tool Call)

```
You: "Implement src/halfdka_features.py with:
- Function get_halfdka_index(piece, square, king_square) -> int
  (converts board position to HalfKA feature index)
- Function get_active_features(board) -> List[int]
  (returns all active features for a position)
- King bucket mapping (32 zones)

Use this pattern from src/my_existing_module.py
[paste relevant code]

Dependencies: chess library. Save to src/halfdka_features.py"

Copilot: Creates the file (1 create_file tool call)

You: Test locally, verify it works
```

**Cost**: 1 tool call, 5 minutes to validate

---

### Scenario 2: Fix a Specific Bug (1 Tool Call)

```
You: "Fix the bug in src/train.py line 45:

Current code:
[paste 5 lines with problem]

Error when running:
[paste exact error message]

What's wrong?"

Copilot: Explains + suggests fix (1 replace_string call)

You: Apply fix, retest locally
```

**Cost**: 1 tool call, 2 minutes to verify

---

### Scenario 3: Code Review (1 Tool Call)

```
You: "Review src/accumulator_architecture.py for correctness.
I tested it locally (all tests pass).
Just want your feedback on:
- Perspective symmetry implementation
- Memory efficiency
- Gradient flow

Here's the code: [paste full file]"

Copilot: Reviews + comments (1 read + 1 comment, <10 tokens)

You: Integrate feedback, retest
```

**Cost**: 1 tool call, helps you learn

---

## What NOT to Do (Save Tokens)

| Bad Request | Cost | Better Approach | Cost |
|-------------|------|-----------------|------|
| "Review my entire codebase" | 50+ tokens | "Review src/accumulator.py" | 2 tokens |
| "Implement Sprint 2 for me" | 100+ tokens | "Create src/halfdka_features.py" | 10 tokens |
| "Find bugs in my code" | 30+ tokens | "This test fails: [error]. Fix it." | 5 tokens |
| "What should I do next?" | 20+ tokens | "Read dev plan, pick file, tell me" | 0 tokens |
| "Run these tests" | 15+ tokens | "You run tests, show me results" | 0 tokens |

---

## Your Checklist (Before Each Session)

Print this or bookmark it:

### Start of Day
- [ ] Opened `.copilot-instructions.md`
- [ ] Read today's task from dev plan (which Day?)
- [ ] Opened VS Code to v10.0/
- [ ] Terminal open and venv activated
- [ ] Have you read existing code patterns? (5 min)

### During Work
- [ ] Writing code (no Copilot)
- [ ] Saving frequently (Ctrl+S)
- [ ] Errors visible in Problems panel
- [ ] Testing locally (terminal)
- [ ] Trying to fix errors yourself first (10+ min)

### Before Asking Copilot
- [ ] Specific file + specific problem
- [ ] Exact error message included
- [ ] Code snippet provided (not whole file)
- [ ] What you've tried already
- [ ] Clear question ("fix this" not "make it better")

### After Copilot Response
- [ ] Understood the change (read it)
- [ ] Tested locally (terminal)
- [ ] All tests pass (✅)
- [ ] Can explain what was fixed
- [ ] Moved to next task

---

## Success Looks Like

### Week 1 (Sprint 1)
- ✅ binary_format_converter.py working
- ✅ position_filters.py working
- ✅ pytorch_dataset.py working
- ✅ Benchmarks meet targets
- ✅ <15 Copilot questions asked
- ✅ You understand every line

### Week 2-3 (Sprint 2-3)
- ✅ HalfKA features implemented
- ✅ Training loop started
- ✅ Multi-signal loss working
- ✅ ELO measurements starting
- ✅ <15 Copilot questions asked
- ✅ Monitoring active

### Week 4 (Sprint 4)
- ✅ Syzygy integration done
- ✅ Quantization working
- ✅ ONNX export ready
- ✅ v2 model checkpoint saved
- ✅ <10 Copilot questions asked
- ✅ Ready for Phase 1

### Total
- ✅ 4 weeks complete
- ✅ <50 Copilot tool calls (4x savings)
- ✅ 1,300 tokens saved
- ✅ You understand the codebase
- ✅ Ready for Phase 2

---

## Frequently Asked Questions

### Q: What if I'm completely stuck?

**A**: Follow the 3-step process:
1. Try for 15 minutes locally (copy patterns, test)
2. Search VS Code docs (Ctrl+F)
3. Ask Copilot with: [exact error] + [code] + [what you tried]

**Never**: "I don't know how to do this, help me figure it out"

---

### Q: Should I use Copilot Chat for quick answers?

**A**: No. Use it only for:
- File creation (create_file tool)
- File editing (replace_string tool)
- Code review (read + answer)

**Not for**:
- Exploration ("what's in this file?")
- Brainstorming ("what should I build?")
- Discussions ("what do you think?")

---

### Q: What if I run out of tokens?

**A**: You won't if you follow this plan. But if you do:
- Stop asking Copilot
- Work locally (read existing code, copy patterns)
- Test everything yourself
- Come back when tokens reset

---

### Q: Can I ask follow-up questions?

**A**: Yes, but **one follow-up per request**.

**Good**:
```
Request: "Fix this bug in src/train.py"
Follow-up: "Can you explain why that was the issue?"
```

**Bad**:
```
Request: "Fix this bug"
Follow-up: "Can you also..."
Follow-up: "And can you..."
Follow-up: "One more thing..."
```

---

### Q: What if the dev plan is wrong?

**A**: Adjust as needed, but **document it**:
- Update V7P3R_vX_DEVELOPMENT PLAN.md
- Explain why (performance, logic, etc.)
- Move forward

**Don't**: Rewrite entire plans constantly. Stick with decisions.

---

## One-Page Summary

```
GOAL:        Complete 4 Sprints in 4 weeks
APPROACH:    File-by-file, locally-tested, minimal Copilot
TOOL USE:    <50 calls total (was 200+)
TOKEN SAVE:  1,300 tokens

DAILY:
1. Read dev plan (which day?)
2. Open existing code (copy patterns)
3. Write + test locally (no Copilot)
4. If stuck 15+ min: Ask Copilot (1 question)
5. Validate + move next day

RULES:
✅ Specific file
✅ Specific task
✅ Local validation first
✅ Exact error messages
❌ Exploratory searches
❌ Multiple files per session
❌ Running tests via tools
❌ Vague questions

FILES TO KEEP OPEN:
1. .copilot-instructions.md (rules + patterns)
2. V7P3R_vX_DEVELOPMENT PLAN.md (daily tasks)
3. ENVIRONMENT_SETUP.md (when things break)

START:
1. Read .copilot-instructions.md (today, 10 min)
2. Follow ENVIRONMENT_SETUP.md (today, 5 min)
3. Pick Day 1.1 from dev plan (tomorrow, start)
4. Implement binary_format_converter.py (no Copilot needed)
5. Test locally (terminal, pass tests)
```

---

## Final Message

You have:
- ✅ A complete development plan (4 sprints, 4 weeks)
- ✅ Clear rules for Copilot (when + how to use)
- ✅ Environment setup (one-time, 5 minutes)
- ✅ Daily workflow (simple, repeatable)
- ✅ Token savings (4x reduction)

**What's missing**: You starting.

**Your move**: Pick Day 1.1, open a file, implement code.

**Copilot's role**: Create files, fix bugs, answer specific questions. Nothing more.

**Result**: 4 weeks later, Phase 1 training starts with master-level architecture + 1,300 tokens saved.

---

**Status**: 🟢 Ready to build  
**Next Step**: Read `.copilot-instructions.md`  
**Timeline**: Start tomorrow, Day 1.1  
**Goal**: Master-level chess engine in 24 weeks  

Let's go. 🚀
