# Reproducing the Project

## 1. Tested Environment

The final release baseline was validated on:

- **Python**: 3.11.9
- **pytest**: 9.1.1
- **mcp**: 2.1.1
- **opentelemetry-api**: 1.44.0
- **opentelemetry-sdk**: 1.44.0
- **Platform**: Windows (PowerShell). The project is also
  compatible with any platform that supports
  `asyncio.create_subprocess_exec`.

These are the tested versions, not the only supported versions. The
current dependency set (mcp>=2,<3) requires Python 3.10 or newer.
The code also uses `from __future__ import annotations` and PEP 585
generic syntax (`dict[str, ...]`).

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

## 3. Install Development Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` pulls in runtime dependencies from
`requirements.txt` plus test-only dependencies (pytest).

## 4. Run the Full Test Suite

```bash
python -m pytest -q
```

## 5. Run Warnings-as-Errors Validation

Single-line (cross-platform):

```bash
python -m pytest -q -W error::DeprecationWarning -W error::ResourceWarning
```

## 6. Run Benchmark-Specific Tests

### Controlled suite (Phase 7 Steps 1–5)

```bash
python -m pytest tests/test_phase7_step4_smoke.py -q
```

### Filesystem suite (Phase 7 Steps 6–8)

```bash
python -m pytest tests/test_phase7_step6_smoke.py -q
```

### Integrated multi-fault (Phase 7 Step 9)

```bash
python -m pytest tests/test_phase7_step9_integrated.py -q
```

### Reporting / final audit (Phase 7 Step 10)

```bash
python -m pytest tests/test_phase7_step10_reporting.py tests/test_phase7_step10_final_audit.py -q
```

## 7. Expected Baseline

```
1521 passed
0 DeprecationWarning
0 ResourceWarning
```

## 8. Determinism Notes

- All benchmark scenarios are deterministic and offline (no network,
  no LLM, no external services).
- Functional results are deterministic across repeated runs:
  completion, failure reason, counts, fault placement, final
  repository snapshot, oracle result.
- NOT deterministic (and not compared across runs): UUIDs,
  timestamps, wall-clock duration, temp paths, PIDs, raw pytest
  timing.

## 9. Platform Notes

- Filesystem benchmark tests spawn real `sys.executable -m pytest`
  subprocesses. On a loaded machine, the timeout scenario may take
  3–5 seconds per trial.
- The integrated multi-fault test takes approximately 3 minutes due
  to multiple real pytest subprocess invocations.
- No tests require network access.
- No tests require a real LLM API key.
