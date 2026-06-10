# VS Code Environment Setup for Token Efficiency

**Goal**: Eliminate Copilot tool use where possible. Work fast locally.

---

## 5-Minute Quick Setup

### 1. Terminal Setup (One-Time)

```powershell
# Open PowerShell in VS Code (Ctrl+`)
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python (should show correct path)
python --version
pip list | grep torch  # Should show pytorch installed
```

**Keep this terminal open** throughout your work.

---

### 2. VS Code Extensions (Recommended)

Install these to **avoid needing Copilot for syntax/errors**:

```
Ctrl+Shift+X (Extensions)
```

Search and install:
- ✅ **Pylance** (Python language server) - shows errors in real-time
- ✅ **Python** (Microsoft) - debugging, running scripts
- ✅ **Pytest** - run tests with one click
- ✅ **Thunder Client** or **REST Client** - test HTTP endpoints
- ❌ **Remove**: Copilot Labs (not needed, wastes attention)

---

### 3. VS Code Settings (Optimize for Coding, Not Suggestions)

Press `Ctrl+,` to open Settings, then search:

```json
// Search: "Python Linting"
"python.linting.enabled": true,
"python.linting.pylintEnabled": true,

// Search: "Format on Save"
"editor.formatOnSave": true,
"python.formatting.provider": "black",

// Search: "Copilot Inline"
"github.copilot.enable": {
  "plaintext": false,
  "markdown": false,
  "yaml": false,
  "json": false
},

// Search: "Auto Save"
"files.autoSave": "onFocusChange",

// Search: "Word Wrap"
"editor.wordWrap": "on",

// Search: "Minimap"
"editor.minimap.enabled": true
```

This makes **errors visible immediately** without asking Copilot.

---

### 4. File Tree Shortcut

Left side: Click Explorer icon (Ctrl+Shift+E)

```
v7p3r-chess-ai/
└── v10.0/
    ├── .copilot-instructions.md  ← READ THIS BEFORE EACH SESSION
    ├── V7P3R_vX_DEVELOPMENT PLAN.md  ← Your roadmap
    ├── src/  ← Expand this for your modules
    ├── tests/  ← Your unit tests
    ├── data/  ← Your datasets (gitignore'd)
    └── models/  ← Your checkpoints (gitignore'd)
```

**Pro tip**: Right-click `src/` → "New File" to create modules.

---

### 5. Running Code Without Copilot

#### Option A: Quick Test (Fastest)

```powershell
# Terminal (Ctrl+`)
python -c "from src.my_module import MyClass; print(MyClass().method())"
```

#### Option B: Full Script

```powershell
# Terminal
python src/my_module.py
```

#### Option C: Debugging

```powershell
# Terminal (runs with debugger)
python -m pdb src/my_module.py
```

#### Option D: Unit Tests (Recommended)

```powershell
# Terminal (runs all tests in a file)
python -m pytest tests/test_my_module.py -v
```

**Result**: Green checkmarks = no errors. Red X = fix needed.

---

## Daily Workflow (15 Minutes Setup)

### Morning (When You Start)

1. **Open VS Code** to v10.0/ folder
2. **Open terminal** (Ctrl+`)
3. **Activate venv** (if not already):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. **Read today's task** from V7P3R_vX_DEVELOPMENT PLAN.md
5. **Open one file** to understand patterns (Ctrl+P)

### During Work

1. **Write code** in VS Code (no Copilot needed)
2. **Save file** (Ctrl+S) - auto-formatted
3. **See errors** in Problems panel (Ctrl+Shift+M)
4. **Fix locally** (try to understand the error yourself)
5. **Test** (run pytest or python script)

### When Stuck (Ask Copilot)

Only when:
- You've tried locally 15+ minutes
- Error message is confusing
- Need specific API documentation

Example:
```
"I'm getting TypeError: 'NoneType' object is not subscriptable 
at line 45 in src/halfdka_features.py

Here's the code:
[paste 10 lines]

Why is this happening?"
```

---

## Common Commands (Memorize These)

| Task | Command | Shortcut |
|------|---------|----------|
| Open file | `Ctrl+P`, type name | Fast! |
| Search in file | `Ctrl+F` | Within one file |
| Replace in file | `Ctrl+H` | Edit quickly |
| Open terminal | `Ctrl+`` | Run code here |
| Show errors | `Ctrl+Shift+M` | Problems panel |
| Run tests | `python -m pytest tests/test_*.py -v` | In terminal |
| Format code | `Shift+Alt+F` | Auto-fixes style |
| Debug hover | Hover on variable | Pylance shows type |

---

## Git Setup (Optional, Recommended)

Initialize git to save checkpoints:

```powershell
# Terminal (one time)
cd v7p3r-chess-ai
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git add -A
git commit -m "Initial commit: v10.0 setup"
```

After each Sprint:
```powershell
git add src/new_module.py tests/test_*.py
git commit -m "Sprint 1: Implement binary_format_converter.py"
git log --oneline  # See history
```

**Benefit**: Can revert if something breaks.

---

## Troubleshooting

### Problem: Terminal shows "command not found: python"

**Fix**:
```powershell
# Get full path
Get-Command python.exe
# Activate venv again
.\.venv\Scripts\Activate.ps1
```

### Problem: Pylance shows "cannot import module"

**Fix**:
```powershell
# Add to terminal
python -m pip install -e .
```

### Problem: Pytest says "ModuleNotFoundError"

**Fix**: Run from v10.0/ directory:
```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v10.0"
python -m pytest tests/ -v
```

### Problem: Copilot doesn't show suggestions

**That's good!** Use terminal + Pylance errors instead.

---

## Token-Saving Checklist

Before asking Copilot, check:

- [ ] Opened the file locally? (Ctrl+P)
- [ ] Read 30-50 lines? (understand patterns)
- [ ] Tried running it? (terminal)
- [ ] Saw the error? (Problems panel or terminal output)
- [ ] Searched VS Code docs? (Ctrl+F in this instructions file)
- [ ] Tried for 10+ minutes? (persistence matters)

If all yes → Ask Copilot with your exact error.  
If no → Do the checklist first.

---

## Success Metrics

By end of Week 1, you should:

✅ Have binary_format_converter.py working  
✅ Ran benchmarks locally (>50 MB/sec)  
✅ Created tests that pass  
✅ Asked Copilot <5 questions total  
✅ Saved 50+ tokens vs exploratory approach  

---

**Ready? Start here**: Open `.copilot-instructions.md` again, pick Day 1.1, implement binary_format_converter.py. 

**No Copilot needed for first implementation—just code.** 🚀
