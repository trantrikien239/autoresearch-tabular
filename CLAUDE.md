# CLAUDE.md

Read `program.md` for your full instructions.

## Quick reference

```bash
# Run experiment (ALWAYS redirect output)
python3 run_experiment.py "description" > run.log 2>&1
grep "^mean_score:\|^std_score:\|^n_features:" run.log

# Run analysis
python3 run_analysis.py > analysis.log 2>&1

# If run crashed
tail -n 50 run.log
```

## Edit only
- `features.py` — feature engineering + training data curation
- `model.py` — model choice + hyperparameters (N_ESTIMATORS set in prepare.py)
- `analysis.py` — analysis code and primitives
- `LEARNING.md` — your research journal

## Never edit
- `prepare.py`, `run_experiment.py`, `run_analysis.py`, `program.md`, `CLAUDE.md`

## Accept / reject
- Improved: keep commit
- Not improved: `git reset --hard HEAD~1`
- Crashed: `tail -n 50 run.log`, fix or revert
