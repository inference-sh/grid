---
name: add-llm-api-provider
description: "Add a new LLM API provider to inference.sh grid with native SDK, streaming, shared helper, deploy, test, and pricing. Use when creating a batch of LLM model apps from a provider like Anthropic, OpenAI, Google, xAI."
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebFetch
---

## Add LLM API Provider

Use when adding a new LLM provider (Anthropic, OpenAI, Google, xAI, etc.) to the inference.sh grid as a set of chat apps with a shared helper module.

### Process

1. **Research the provider API** — Fetch official documentation. For each model, extract:
   - Model ID (exact API string, e.g. `claude-sonnet-4-6`)
   - Context window and max output tokens
   - Pricing per million tokens (input/output)
   - Capabilities: vision, tool use, reasoning/thinking, streaming format
   - Auth method and header format
   - Collect into `README.md` and `pricing.md` at the provider root

2. **Create provider directory and shared helper** — 
   ```bash
   mkdir -p tmp/grid/api/{provider}
   ```
   Write `{provider}_helper.py` with:
   - **Message conversion**: Provider-specific format from `LLMInput` (context messages, images, tool calls, tool results). Handle consecutive same-role message merging if required.
   - **Tool conversion**: OpenAI format → provider format
   - **Streaming function**: `stream_completion(client, input_data, model)` → yields output dicts with `response`, `reasoning`, `tool_calls`, `output_meta`
   - **Non-streaming function**: `complete(client, input_data, model)` → returns output dict
   - **Token tracking**: Extract `input_tokens`/`output_tokens` from API response, emit as `OutputMeta(inputs=[TextMeta(tokens=N)], outputs=[TextMeta(tokens=N)])`
   - **Error handling**: `handle_api_error()` that extracts provider error messages
   - **Reasoning/thinking config**: Map `ReasoningEffortEnum` to provider's thinking parameter format

3. **Scaffold each model app** — Use `belt app init`, never create files manually:
   ```bash
   cd tmp/grid/api/{provider}
   belt app init {model-name} --lang python   # repeat for each model
   ```

4. **Symlink the shared helper** into each app:
   ```bash
   for app in model-a model-b model-c; do
     ln -sf ../{provider}_helper.py "$app/{provider}_helper.py"
   done
   ```

5. **Implement each app's inference.py** — Thin wrapper (~50 lines):
   - `AppInput` extends `LLMInput` + capability mixins (`ReasoningCapabilityMixin`, `ToolsCapabilityMixin`, `ImageCapabilityMixin`)
   - `AppOutput` extends `ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput`
   - `App.setup()` — no parameters, init the SDK client from `os.getenv("{PROVIDER}_KEY")`
   - `App.run()` — delegates to `stream_completion()` or `complete()` based on `input_data.stream`
   - Set `DEFAULT_MODEL`, `context_size`, and `max_tokens` per model

6. **Update inf.yml for each app** — Set:
   - `namespace: {provider}`, `name: {model-name}`
   - `category: chat`
   - `metadata.capabilities: [reasoning, image]` (as appropriate)
   - `secrets: [{key: {PROVIDER}_KEY}]`
   - `resources: gpu: {count: 0, type: none}, ram: 0` (API-only)

7. **Update requirements.txt** for each app:
   ```
   pydantic >= 2.0.0
   inferencesh >= 0.6.30
   {provider-sdk} >= {version}
   ```

8. **Deploy all apps**:
   ```bash
   for app in model-a model-b model-c; do
     cd tmp/grid/api/{provider}/$app && belt app deploy
   done
   ```

9. **Cloud-test every app** — Not just one, ALL of them:
   ```bash
   belt app run {provider}/{model-name} --json --input '{"text": "simple test prompt"}'
   ```
   Verify: `status_text: "completed"`, non-empty `response`, `output_meta` with token counts.

10. **Run pricing agents for all apps** — Get app/version IDs, then run:
    ```bash
    belt app get {provider}/{model-name} --json  # extract app.id and app.version_id
    belt agent run infsh/pricing-agent "<pricing instructions>" \
      --context version_id=<vid> --context app_id=<aid>
    ```
    Group by pricing tier (e.g. all $5/$25 models together). After agents draft and evaluate, approve publish:
    ```bash
    belt chat pending                    # list pending approvals
    belt chat approve <invocation-id>    # approve each publish
    ```

### Rules

- **NEVER create inf.yml, inference.py, __init__.py, or app directories manually.** Always use `belt app init`. The scaffolded `__init__.py` and package structure are required for the validator. Reason: manual creation misses required fields, uses wrong field names, and breaks the deployment validator.

- **setup() takes no parameters.** The `metadata` parameter is deprecated. Use `self.context.metadata` if needed. `self.logger` is NOT available in `setup()` — use `print()` for setup logging. Reason: both caused runtime errors during testing.

- **LLMInput field is `text`, not `prompt`.** The input schema inherits from `LLMInput` which uses `text` as the primary field. Reason: `prompt` causes a Pydantic validation error at runtime.

- **AppOutput MUST extend BaseAppOutput** (not just BaseModel) when using `output_meta`. Without it, `output_meta` is silently dropped from the response.

- **Read files before overwriting with Write tool.** The Write tool requires a prior Read of the file in the same conversation. After `belt app init`, read the scaffolded file first.

- **Use relative imports for the symlinked helper.** `from .{provider}_helper import func` — not absolute imports. The app runs as a package inside the deployment container.

- **Deploy and cloud-test EVERY app, not just one.** Different model IDs can have different API behaviors. A working sonnet doesn't guarantee a working haiku. Reason: we shipped with only partial testing initially.

- **Run pricing agents AFTER deployment**, not before. The pricing agent needs the app submitted to the store, which happens during deploy. If the agent says "not submitted to store", tell it to submit first.

- **belt secrets are masked locally.** `belt secrets get` doesn't return the raw value — you can't use it for local testing env vars. Deploy and cloud-test instead.

- **Pricing CEL expressions need double() casts.** `text_tokens()` returns int; dividing by a double causes a CEL type error. Always use `double(text_tokens(inputs)) / 1000000.0 * double(prices.var)`. Reason: multiple pricing agents failed on this before self-correcting.

- **Group models by pricing tier** for pricing agent efficiency. Don't run 6 identical pricing agents — group the 3 Opus models, 2 Sonnet models, etc. by their $/MTok rate.
