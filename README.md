# downgrade

> **Model routers claim large cost savings. Nobody measures whether quality silently regressed. This measures it.**

[![CI](https://github.com/thebharathkumar/downgrade/actions/workflows/ci.yml/badge.svg)](https://github.com/thebharathkumar/downgrade/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

Model routers (Fireworks FireRouter/Nexus, OpenRouter auto, LiteLLM) send "easy" requests
to cheaper models and report the savings. What they do not report is whether the answer got
worse on the way. `downgrade` runs a fixed task suite against a router at every routing
preference, captures the full trajectory of every run, and classifies what changed.

The distinction that matters:

- **LOUD** regression: the final answer is wrong, the run errored, or it never terminated.
  An outcome-only judge catches these.
- **SILENT** regression: the final answer is indistinguishable from baseline, but the path
  that produced it degraded. A tool call vanished. A constraint got dropped. A number in the
  answer came from nowhere.

Silent regressions are the interesting ones, because the thing most teams deploy as a quality
gate (an LLM judging the final answer) is close to blind to them by construction.

## Status

Week 1 of 3. The harness, the classifier, OTel emission and the CLI are in scope. The
statistics layer (sequential testing, Benjamini-Hochberg, Kupiec and Christoffersen
backtests) and the published report land in weeks 2 and 3 and are **not** built yet.

No sweep results are published here yet. When they are, if the data shows no silent
regression at any routing preference, that is what will be reported. The classifier is not
tuned to produce findings.

## The six silent sub-classes

Detected in priority order. Each finding carries the evidence that produced it: which step,
what the baseline arm did, what this run did instead.

| # | Sub-class | Detector | How |
|--:|:----------|:---------|:----|
| 1 | `unsupported_claim` | judge | A claim in the final message that no step supports |
| 2 | `missing_tool_call` | structural | A call reliable in baseline, absent here |
| 3 | `dropped_constraint` | structural | A task-declared constraint predicate that now fails |
| 4 | `redundant_retry` | structural | Repeated calls baseline did not need |
| 5 | `fabricated_value` | structural | A value in the answer that no tool call produced |
| 6 | `unsupported_citation` | judge | A citation that no longer supports its claim |

Four of six are structural. That is the point: they are decided by set difference, argument
diffing and value provenance over the trajectory, not by an LLM's opinion. The two judge
detectors live in a separate, clearly-labelled code path (`downgrade/classify/judge.py`) so
their precision can be measured independently of the structural ones. `classify(mode=...)`
runs either half alone.

`dropped_constraint` is structural because every task in the suite declares its constraints
as machine-checkable predicates in its YAML. A constraint that cannot be expressed as a
predicate means the task is badly authored, and it gets rewritten rather than handed to a
judge.

## Design notes that are easy to get wrong

**Fresh conversation per run.** FireRouter caches its routing decision within a conversation,
so a preference change takes several turns to take effect. A runner that reuses a
conversation across tasks or arms is measuring the previous arm's cached route. `AgentLoop`
therefore has no parameter through which prior message state can enter, mints a new
`conversation_id` per run, and a test asserts the isolation directly.

**Comparison is set-vs-set, never run-vs-run.** At any temperature above zero, one
downgraded run differing from one baseline run is indistinguishable from sampling noise.
Each arm runs R replicates, and findings are reported as rates over those replicates.

**Sampling settings are pinned and recorded.** Temperature and seed are fixed across every
arm, stored on every individual request, and folded into a `config_fingerprint` carried by
every trajectory. Comparing two arms with different fingerprints is refused rather than
silently reported, because that comparison measures the drift, not the routing.

**Token usage arrives in the final streaming chunk.** A partially-consumed stream records
`complete=False` rather than a plausible-looking zero.

## OpenTelemetry

Spans follow the GenAI semantic conventions from
[`open-telemetry/semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai).
Everything in that repo is `Development`; nothing is Stable.

Standard attributes used: `gen_ai.operation.name`, `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`,
`gen_ai.conversation.id`.

**Non-standard attributes.** The GenAI conventions define no attribute for which model a
router selected. `gen_ai.response.model` already carries the served model, so it is not
duplicated. These are emitted under a clearly-namespaced prefix and are **not** part of any
standard:

| Attribute | Meaning |
|:----------|:--------|
| `downgrade.route.preference` | The `x-routing-preference` value sent (1-5) |
| `downgrade.route.arm` | Which experimental arm this call belongs to |
| `downgrade.route.primary_model` | The declared primary of the router pair |
| `downgrade.route.downgraded` | Whether the served model differs from the primary |
| `downgrade.route.turn_index` | Position in the conversation, since routing is cached |
| `downgrade.route.source` | How the route was attributed (`response_model`/`header`) |

`downgrade.route.source` records the provenance of the attribution itself, because the only
per-request attribution the API reliably offers is the response `model` field.

## Prior work reused

This project reuses machinery from two earlier repos by the same author:

- [**dungeon-traces**](https://github.com/thebharathkumar/dungeon-traces) supplies the
  shape of the classifier: a documented priority ladder that returns the first matching
  category, and per-fact divergence records that carry `believed` / `actual` / provenance
  rather than a bare label. `downgrade`'s detectors are the same idea with baseline
  trajectories in place of ground-truth world state.
- [**agent-triage**](https://github.com/thebharathkumar/agent-triage) supplies the trace
  adapter protocol, the pydantic event schema, the before/after batch comparison structure,
  and the OTLP receiving path.

## License

MIT. See [LICENSE](./LICENSE).
