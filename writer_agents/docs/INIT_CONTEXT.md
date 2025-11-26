# Writer Agents - Init Context

## Focus
- AutoGen logging + Google Docs hygiene (2025-11-13)
- Smoke test: `python writer_agents/scripts/generate_optimized_motion.py --case-summary "Test seal motion" --refinement-mode auto`

## Troubleshooting
### AutoGen exploration fails without OpenAI creds
- Symptom: `AutoGen exploration failed: The api_key client option must be set...`
- Impact: workflow aborts before iterative refinement or Google Docs updates
- Fix: export `OPENAI_API_KEY` or configure local-only drafting before rerunning

> **Do this next**
> ```bash
> export OPENAI_API_KEY="sk-your-key"
> python writer_agents/scripts/generate_optimized_motion.py --case-summary "Test seal motion" --refinement-mode auto
> ```

### Local-only dry run to bypass AutoGen API calls
- Set `MATRIX_USE_LOCAL=true` + `MATRIX_LOCAL_MODEL=phi3:mini` to keep exploration local
- Pair with `--refinement-mode off` when you only need SK + CatBoost validation
- Expect more conservative drafts because the AutoGen writers are skipped

> **Do this next**
> ```bash
> export MATRIX_USE_LOCAL=true
> export MATRIX_LOCAL_MODEL="phi3:mini"
> python writer_agents/scripts/generate_optimized_motion.py --case-summary "Test seal motion" --refinement-mode off
> ```

## Run Snapshot
- Timestamp: 2025-11-13T14:01Z
- Command: `python writer_agents/scripts/generate_optimized_motion.py --case-summary "Test seal motion" --refinement-mode auto`
- Result: failed with AutoGen API key error; see troubleshooting steps above
