# Consensus — Multi-AI Provider Plugin for Claude Code

Get unstuck by querying multiple AI providers (GPT-6 Astra, Gemini 4 Argon, DeepSeek V4 Pro) for diverse perspectives, right from within Claude Code.

The plugin lives in its own repository, [mark-kimura/consensus-plugin](https://github.com/mark-kimura/consensus-plugin), which is included here in the `consensus-plugin/` folder. Current version: 1.7.5.

## How It Works

When you're stuck on a problem, Consensus queries multiple AI providers concurrently — GPT-6 Astra, Gemini 4 Argon, and DeepSeek V4 Pro via OpenRouter by default — then synthesizes their responses to surface common themes, unique insights, and actionable takeaways.

## Installation

### 1. Install `uv` (Python package runner)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set API Keys

Only one key is needed for the default setup:

```bash
export OPENROUTER_API_KEY="sk-or-..."   # Routes all three default providers
```

Optional, for direct API access (bypassing OpenRouter):

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI direct API
export GEMINI_API_KEY="AI..."           # Google Gemini direct API
```

Add them to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist across sessions.

### 3. Install the Plugin

**Option A — Third-Party Marketplace**:

```
/plugin marketplace add mark-kimura/kahidreamers-marketplace
/plugin install consensus@kahidreamers-marketplace
```

**Option B — Official Plugin Directory** (not yet available — submitted, under review by Anthropic):

```
/plugin install consensus@claude-plugin-directory
```

### 4. Enable Auto-Update (Recommended)

For the official directory, auto-update is enabled by default. For third-party marketplaces, enable it manually:

1. Run `/plugin` in Claude Code
2. Go to **Marketplaces**
3. Select **kahidreamers-marketplace**
4. Choose **Enable auto-update**

Without this, you'll need to run `/plugin update consensus@kahidreamers-marketplace` manually after each release.

## Usage

### `/consensus:ask`

```
/consensus:ask help me optimize the SSE connection handling
```

Or without arguments — Claude infers from conversation context:

```
/consensus:ask
```

This also auto-triggers when you say things like:
- "What do other AIs think about this approach?"
- "Can I get a consensus on this?"
- "Cross-check this with other models"

### `/consensus:config`

```
/consensus:config              # Show current providers (same as check)
/consensus:config update       # Update all providers to latest models
/consensus:config add llama 4  # Add a new provider via OpenRouter
/consensus:config remove deepseek  # Remove a provider
/consensus:config openai       # Interactively configure OpenAI (model + routing)
/consensus:config --reset      # Reset all providers to plugin defaults
```

### What Happens

1. Claude analyzes your request and conversation context
2. Crafts a comprehensive prompt with technical details
3. Queries all available providers concurrently
4. Synthesizes findings: agreements, unique perspectives, and takeaways

All intermediate files go to a temp directory — nothing is written to your project.

## Configuration

The plugin works out of the box with just `OPENROUTER_API_KEY`. Use `/consensus:config` to manage providers interactively, or customize `consensus_config.json` directly:

- **Provider routing**: Route through OpenRouter or call APIs directly (`/consensus:config openai`)
- **Models**: Add, remove, or update providers (`/consensus:config add llama 4`)
- **Search modes**: Set default to `"web"` (includes web search) or `"none"` (faster, no web)
- **Token limits**: Adjust `max_tokens` per provider

### Default Providers

| Provider | `none` mode model | `web` mode model |
|----------|-------------------|------------------|
| OpenAI | `openai/gpt-6-astra` | `openai/gpt-6-astra:online` |
| Gemini | `google/gemini-4-argon` | `google/gemini-4-argon:online` |
| DeepSeek | `deepseek/deepseek-v4-pro-0813` | `deepseek/deepseek-v4-pro-0813:online` |

All three route through OpenRouter by default.

### Config Resolution Order

1. `./consensus_config.json` (project root — for project-specific overrides; API keys and endpoints in this file are ignored, because the file may come from any repository you open)
2. `~/.claude/consensus_config.json` (user config — written by `/consensus:config`, survives plugin updates)
3. Plugin's built-in `consensus_config.json`
4. Hardcoded defaults

In every config file, endpoints must be `https://` URLs on `api.openai.com`, `generativelanguage.googleapis.com` or `openrouter.ai`. Any other endpoint is replaced by the default, so API keys are only ever sent to those three hosts.

## Search Modes

By default, providers include current web information in their responses. If you'd rather get purely model-based perspectives without web search, just say so naturally:

- "Get a consensus on this, no web search"
- "What do other models think — don't use the web"
- "/consensus:ask without web search, compare approaches to caching"

Claude will detect your intent and run in the appropriate mode:

- **`web`** (default): Providers include current web information. Good for questions about recent technologies or current best practices.
- **`none`**: Pure model knowledge, no web search. Faster responses. Good for coding questions, architecture decisions, or when you want unbiased model perspectives.

## Repository Structure

```
consensus/
├── consensus-plugin/              # The plugin (separate repo: mark-kimura/consensus-plugin)
│   ├── .claude-plugin/
│   │   └── plugin.json            # Plugin manifest
│   ├── commands/
│   │   └── config.md              # /consensus:config command
│   ├── skills/
│   │   └── ask/
│   │       └── SKILL.md           # /consensus:ask skill (also auto-triggers)
│   ├── hooks/
│   │   └── hooks.json             # SessionStart: checks API keys + uv
│   ├── scripts/
│   │   ├── consensus.py           # Core engine (uv run auto-resolves deps)
│   │   └── check-setup.sh         # Setup validator
│   └── consensus_config.json      # Default config template
├── consensus.py                   # Older standalone script (see below)
├── consensus_config.json          # Config for the older standalone script
├── requirements.txt               # Dependencies for the older standalone script
├── .claude/commands/consensus.md  # /consensus command for this folder (runs the plugin script)
├── LICENSE
└── README.md
```

## Older Standalone Script

The `consensus.py` file at the top level of this repository is the version that existed before the plugin. It is not updated with the plugin. It queries GPT-5 (direct OpenAI API), Gemini 2.5 Pro and Perplexity Sonar (through OpenRouter), and saves results to `consensus_docs/`.

```bash
pip install -r requirements.txt
python consensus.py your_prompt.md [--search-mode web|none] [--output-dir DIR] [--config CONFIG]
```

For the current models and features, use the plugin.

## Requirements

- [uv](https://docs.astral.sh/uv/) — auto-installs Python dependencies on first run
- At least one API key (`OPENROUTER_API_KEY` recommended)
- Claude Code

## License

MIT License — free for any use, including commercial use. See the LICENSE file for details.
