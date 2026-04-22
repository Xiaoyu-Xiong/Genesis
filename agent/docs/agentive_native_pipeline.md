# Agentive-Native Pipeline Upgrade Plan

This document describes the planned migration of the current `agent/` optimization stack toward a more agentive-native
architecture with the explicit goal of reducing OpenAI token spend without weakening the current Genesis workflow.

It is scoped to three concrete upgrades:

1. Strong prompt-cache-friendly generator / critic structure
2. Two-stage critic
3. Retrieval-based critic agent

It also defines the logging and validation work needed to measure whether the migration actually reduces cost.

## Current State

The current implementation already uses several agentive-native OpenAI features:

- `Responses API`
- `previous_response_id` in generator refinement loops
- `prompt_cache_key`
- hosted prompt references
- tool calling in the IR generator

Relevant code:

- [agent/llm_generator/client/openai_client.py](../llm_generator/client/openai_client.py)
- [agent/llm_generator/agents/ir_agent.py](../llm_generator/agents/ir_agent.py)
- [agent/llm_critic/critic.py](../llm_critic/critic.py)
- [agent/opt/pipeline.py](../opt/pipeline.py)

However, the current pipeline is still expensive because:

- optimization rounds recreate large prompts repeatedly
- critic evaluation still preloads large evidence bundles
- there is no component-level usage accounting
- prompt caching is not treated as a first-class design constraint

## OpenAI Cost Model Constraints

Three OpenAI behaviors shape the migration:

1. `previous_response_id` simplifies state management but does **not** make earlier tokens free. Chained history is
   still billed as input tokens.
2. prompt caching can materially reduce input token cost, but only for exact prefix matches
3. hosted prompts help keep the static prefix stable, which improves cache hit rates

That means the migration should focus on:

- reducing how much context is sent per call
- maximizing stable prompt prefixes
- only escalating to expensive analysis when needed

## Migration Order

The implementation order is:

1. cache-friendly prompt structure and usage accounting
2. two-stage critic
3. retrieval-based critic agent

This order is deliberate:

- step 1 is low-risk and makes later gains measurable
- step 2 reduces the number of expensive critic calls
- step 3 reduces the size of the expensive critic calls that remain

## Step 1: Strong Prompt-Cache-Friendly Structure

### Goal

Keep the longest possible prefix of generator and critic requests stable across rounds and cases so that prompt caching
has a meaningful chance to reduce billed input tokens.

### Generator Changes

Target files:

- [agent/llm_generator/agents/ir_agent.py](../llm_generator/agents/ir_agent.py)
- [agent/llm_generator/agents/xml_agent.py](../llm_generator/agents/xml_agent.py)
- [agent/llm_generator/client/openai_client.py](../llm_generator/client/openai_client.py)

Implementation:

1. split prompt construction into explicit segments:
   - static system / process requirement prefix
   - task block
   - revision block
   - dynamic feedback tail
2. keep the static segments first in the message list
3. enable `prompt_cache_retention="24h"` by default for generator requests
4. retain hosted prompt support, but do not require it for the optimization to work

### Critic Changes

Target files:

- [agent/llm_critic/critic.py](../llm_critic/critic.py)
- [agent/llm_critic/prompting.py](../llm_critic/prompting.py)
- [agent/llm_generator/client/openai_client.py](../llm_generator/client/openai_client.py)

Implementation:

1. split critic input into:
   - stable rubric / schema prefix
   - compact digest block
   - dynamic evidence block
2. keep rubric/schema in a fixed prefix
3. enable `prompt_cache_retention="24h"` by default for critic requests

### Why This Saves Tokens

This step does not reduce raw token counts by itself. It reduces **billed input cost** by increasing prompt-cache hits
on repeated static prefixes.

### Success Metric

For repeated rounds, response usage should show non-zero `cached_tokens` on both generator and critic calls.

## Step 2: Two-Stage Critic

### Goal

Avoid paying full-critic cost on rounds where a cheaper compact pass is already decisive.

### Target Files

- [agent/llm_critic/critic.py](../llm_critic/critic.py)
- [agent/llm_critic/prompting.py](../llm_critic/prompting.py)
- [agent/configs.py](../configs.py)
- [agent/opt/pipeline.py](../opt/pipeline.py)

### Implementation

Add a critic pipeline with two stages:

1. Stage A: compact critic
   - fewer frames
   - smaller frame width
   - compact digest only
2. Stage B: expensive critic
   - only runs when Stage A is uncertain or low-confidence

Stage A should produce:

- verdict
- confidence
- needs_escalation
- summary
- priority fix sketch

Stage B should produce the full structured critique currently expected by the optimization loop.

### Why This Saves Tokens

This step reduces the **number of expensive critic invocations**.

### Success Metric

Across a suite, only a subset of rounds should escalate to Stage B, while the rest finish at Stage A.

## Step 3: Retrieval-Based Critic Agent

### Goal

When Stage B is needed, avoid stuffing the entire IR, event pack, XML, and full evidence block into one prompt.

### Target Files

- [agent/llm_critic/critic.py](../llm_critic/critic.py)
- new retrieval tool files under `agent/llm_critic/`
- [agent/llm_generator/client/openai_client.py](../llm_generator/client/openai_client.py)

### Implementation

Add a small critic tool library with read-only tools, for example:

- `get_critic_bootstrap`
- `get_ir_scene`
- `get_ir_body`
- `get_ir_actions`
- `get_event_execution`
- `get_event_timeline_slice`
- `get_event_crash`
- `get_xml_body`

Stage B becomes a tool-using Responses loop:

1. seed with compact digest and stable rubric prefix
2. let the model request only the slices it needs
3. return the final structured critique JSON

Video frames may still be supplied directly as initial multimodal evidence in the first implementation; retrieval tools
focus first on structured text artifacts.

### Why This Saves Tokens

This reduces the **size of each expensive critic call**, because the model only requests the evidence it needs instead
of receiving the entire evidence bundle unconditionally.

### Success Metric

For escalated rounds, total critic input tokens should drop relative to the current full-critic baseline while verdict
quality remains comparable.

## Usage Accounting

### Goal

Measure whether the migration actually saves tokens.

### Target Files

- [agent/llm_generator/client/openai_client.py](../llm_generator/client/openai_client.py)
- generator and critic call sites
- [agent/opt/pipeline.py](../opt/pipeline.py)
- new reporting script under `agent/scripts/`

### Implementation

1. preserve `usage` from Responses API calls
2. attach usage metadata to:
   - generator round logs
   - XML generation logs
   - critic stage logs
3. write per-round usage files under each `round_*` directory
4. add a suite-level aggregation script that summarizes:
   - prompt tokens
   - cached tokens
   - output tokens
   - reasoning tokens when present
   - component split: generator / xml / critic_stage1 / critic_stage2

### Why This Saves Tokens

This step does not save tokens directly. It makes token-saving changes measurable and prevents blind optimization.

## Validation Plan

### Local Code-Level Validation

After each implementation step:

1. run small local smoke tests for the modified component
2. keep output schemas backward-compatible where possible
3. confirm round artifacts still write successfully

### Suite Validation

Primary suite:

- [agent/scripts/run_opt_deformable_texture_suite.sh](../scripts/run_opt_deformable_texture_suite.sh)

Validation target:

- one H100-80 GPU
- full suite run
- per-case, per-round token accounting for all OpenAI-facing components

### Expected Metrics

Success is defined as:

1. suite finishes without systemic `BrokenProcessPool` regressions
2. token usage logs are present for every round that reaches the corresponding component
3. cached token counts become visible in repeated generator / critic calls
4. Stage B critic call count is meaningfully lower than total critic opportunities

## Risks

### Prompt Caching Risk

If dynamic content leaks into the front of the prompt, cache hit rates will remain poor even after adding retention.

### Critic Regression Risk

If Stage A is too weak, it may pass cases that require deeper analysis or escalate too often and erase savings.

### Retrieval-Agent Risk

If critic tools are too coarse, the model will still request oversized payloads and savings will be limited.

### Logging Risk

If usage accounting is only stored in final summaries and not per-round artifacts, debugging cost regressions will be hard.

## Final Deliverables

After implementation, the repository should contain:

1. cache-friendly generator and critic prompting
2. two-stage critic
3. retrieval-based Stage B critic
4. component-level usage logging
5. suite-level usage summary artifacts
