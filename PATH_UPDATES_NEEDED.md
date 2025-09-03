# Path Reference Updates Needed

After reorganization, the following files may need path reference updates:

## V1.0 Files
- Update import paths in Python files
- Update model loading paths
- Update data file references

## V2.0 Files  
- Update import paths in src/ modules
- Update model saving/loading paths
- Update configuration file references
- Update tournament package paths

## V3.0 Files
- Will need fresh configuration once development begins

## Recommended Approach
1. Test each version independently
2. Update imports as needed
3. Verify model loading/saving works
4. Update any hardcoded paths

## Common Path Updates
- `models/` → `v2.0/models/` (for v2.0)
- `src/` → `v2.0/src/` (for v2.0 imports)
- `data/` → `v{version}/data/`

## Quick Fixes Needed
- Update incremental_trainer.py to use v2.0/models/
- Update enhanced training scripts to use v2.0/ paths
- Test tournament packages still work
