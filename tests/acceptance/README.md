# Acceptance Tests

This directory contains acceptance test scripts that verify the system meets all acceptance criteria defined in the project specification.

## Purpose

Acceptance tests are end-to-end tests that validate:
- The system meets all functional requirements
- All acceptance criteria defined in DEV_SPEC.md are satisfied
- Integration between different components works correctly

## Structure

```
tests/acceptance/
├── README.md                          # This file
├── verify_c4_acceptance.py           # C4: Splitter Integration acceptance test
└── verify_<task_id>_acceptance.py   # Other acceptance tests (to be added)
```

## Running Acceptance Tests

### Run a specific acceptance test

```bash
# C4: Splitter Integration
python tests/acceptance/verify_c4_acceptance.py
```

### Run all acceptance tests

```bash
# From project root
python tests/acceptance/verify_c4_acceptance.py
# (Add more scripts as they are created)
```

## Acceptance Test Reports

After running acceptance tests, detailed reports are generated in:
- `docs/test/reports/` - Acceptance test reports with detailed results

## Difference from Unit/Integration Tests

| Test Type | Purpose | Scope | Location |
|-----------|---------|-------|----------|
| **Unit Tests** | Test individual functions/classes in isolation | Small, focused | `tests/unit/` |
| **Integration Tests** | Test interaction between multiple components | Medium, component-level | `tests/integration/` |
| **Acceptance Tests** | Verify system meets all requirements | Large, end-to-end | `tests/acceptance/` |

## Writing New Acceptance Tests

When creating a new acceptance test:

1. **Name it `verify_<task_id>_acceptance.py`**
   - Example: `verify_c4_acceptance.py`, `verify_c5_acceptance.py`

2. **Follow the structure:**
   ```python
   class TaskXAcceptanceVerifier:
       def verify(self, test_file):
           # Verify all acceptance criteria
           pass

   def main():
       verifier = TaskXAcceptanceVerifier()
       results = verifier.verify(test_file)
       verifier.print_summary(results)
   ```

3. **Test against real data:**
   - Use real PDF files from `docs/test/`
   - Test with edge cases
   - Validate against all acceptance criteria in DEV_SPEC.md

4. **Generate reports:**
   - Print detailed test results
   - Generate markdown reports in `docs/test/reports/`

## References

- **Project Specification:** `DEV_SPEC.md` / `.claude/skills/auto-coder/references/`
- **Task Completion Summaries:** `docs/task-completion-summaries/`
- **Acceptance Test Reports:** `docs/test/reports/`
