# Consensus AI Query Tool

Query multiple AI providers concurrently to get diverse perspectives and expert advice on complex topics. This tool combines responses from GPT-5, Gemini 2.5 Pro, and Perplexity to provide comprehensive analysis and consensus findings.

## Features

- **Multi-Provider Querying**: Concurrent queries to OpenAI GPT-5, Google Gemini 2.5 Pro, and Perplexity Sonar
- **Flexible Configuration**: Support for both direct APIs and OpenRouter proxy
- **Environment Variable Support**: Secure API key management via environment variables
- **Search Modes**: Web search integration or fast no-web responses
- **Consolidated Analysis**: Automatic response consolidation and consensus analysis
- **Claude Code Integration**: Built-in `/consensus` command for Claude Code users

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. API Key Setup

Set your API keys as environment variables (recommended):

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

Alternatively, configure them in `consensus_config.json` (not recommended for security).

### 3. Basic Usage

```bash
# Query with web search (default)
python consensus.py your_prompt.md

# Query without web search for faster responses
python consensus.py your_prompt.md --search-mode none

# Custom output directory
python consensus.py your_prompt.md --output-dir custom_results
```

### 4. Create a Prompt File

Create a markdown file with your question or topic:

```markdown
# Should AI Development Be Regulated?

Analyze the current state of AI regulation proposals, considering:
- Technical feasibility of different regulatory approaches
- Economic impacts on innovation and industry
- International coordination challenges
- Timeline considerations for implementation

Please provide specific policy recommendations.
```

## Claude Code Integration

If you're using [Claude Code](https://claude.ai/code), this tool includes a built-in `/consensus` command:

### Usage Examples

```bash
# Let Claude Code analyze context and create the prompt
/consensus

# Provide specific focus area
/consensus "help me optimize database query performance"

# Ask about complex technical decisions
/consensus "should I use microservices or monolith for this project?"
```

### Workflow

The Claude Code `/consensus` command will:

1. **Analyze your request**: Determine focus based on your input or current context
2. **Craft comprehensive prompt**: Create detailed prompt with:
   - Problem statement and technical context
   - Previous attempts and outcomes
   - Supporting evidence and constraints
   - Specific questions for AI providers
3. **Save prompt**: Write to `consensus_docs/prompt-{timestamp}.md`
4. **Execute queries**: Run concurrent queries to all providers
5. **Analyze results**: Provide consensus analysis with:
   - Common themes and agreements
   - Unique perspectives from each provider
   - Conflict resolution and synthesis
   - Actionable recommendations

### Expected Output

- Input prompt: `consensus_docs/prompt-{timestamp}.md`
- Consolidated responses: `consensus_docs/consolidated-{timestamp}.md`
- Automatic consensus analysis by Claude Code

## Configuration

### Environment Variables (Recommended)

```bash
OPENAI_API_KEY     # OpenAI API key
GEMINI_API_KEY     # Google Gemini API key  
OPENROUTER_API_KEY # OpenRouter API key (for Perplexity)
```

### Configuration File

The `consensus_config.json` file allows detailed customization:

```json
{
  "providers": {
    "openai": {
      "enabled": true,
      "use_openrouter": false,
      "model": "gpt-5"
    },
    "gemini": {
      "enabled": true,
      "use_openrouter": true,
      "openrouter_model": {
        "none": "google/gemini-2.5-pro",
        "web": "google/gemini-2.5-pro:online"
      }
    },
    "perplexity": {
      "enabled": true,
      "use_openrouter": true,
      "openrouter_model": {
        "none": "perplexity/sonar",
        "web": "perplexity/sonar-reasoning"
      }
    }
  },
  "settings": {
    "default_search_mode": "web",
    "max_tokens": 4000,
    "temperature": 0.7
  }
}
```

## Search Modes

- **`web`**: Include current web information (default)
  - Best for recent technologies, current events, latest research
  - Uses models with web search capabilities
- **`none`**: Fast responses without web search
  - Best for coding questions, established concepts, faster responses
  - Uses base models without web integration

## AI Providers

### OpenAI GPT-5
- Latest GPT model with Responses API
- Advanced reasoning capabilities
- Direct API or OpenRouter support

### Google Gemini 2.5 Pro
- Large context window (up to 32K tokens)
- Thinking capabilities
- Supports both direct API and OpenRouter

### Perplexity Sonar
- Built-in web search and research
- Real-time information access
- Multiple model variants (sonar, sonar-reasoning, sonar-deep-research)

## Output Format

Results are saved as timestamped markdown files in `consensus_docs/`:

```
consensus_docs/
├── prompt-20250819_105745.md          # Your input prompt
└── consolidated-20250819_105948.md    # All provider responses
```

### Consolidated Response Structure

```markdown
# Consolidated AI Response - 2025-08-19 10:59:48

**Query:** Your question summary...
**Providers Used:** OpenAI, Gemini, Perplexity

## OpenAI Response
[OpenAI's response...]

## Gemini Response  
[Gemini's response...]

## Perplexity Response
[Perplexity's response...]
```

## Advanced Usage

### Custom Configuration

```bash
python consensus.py prompt.md --config custom_config.json
```

### Provider Selection

Enable/disable providers in the config file:

```json
{
  "providers": {
    "openai": {"enabled": true},
    "gemini": {"enabled": false},
    "perplexity": {"enabled": true}
  }
}
```

### Model Selection

Configure specific models for different search modes:

```json
{
  "providers": {
    "gemini": {
      "openrouter_model": {
        "none": "google/gemini-2.5-pro",
        "web": "google/gemini-2.5-pro:online"
      }
    }
  }
}
```

## Best Practices

### Prompt Writing
- Be specific about your problem or question
- Include relevant context and constraints
- Specify what type of response you need
- Mention decision criteria and trade-offs

### When to Use Web Mode
- ✅ Recent technologies, current events, latest research
- ✅ Industry trends and market analysis
- ✅ Regulatory and policy questions
- ❌ Established programming concepts
- ❌ Mathematical or theoretical questions

### Security
- Use environment variables for API keys
- Don't commit API keys to version control
- Regularly rotate your API keys
- Review generated prompts before sharing

## Troubleshooting

### No API Keys Found
```
Error: No API keys found. Please set environment variables:
- OPENAI_API_KEY for OpenAI
- GEMINI_API_KEY for Gemini  
- OPENROUTER_API_KEY for Perplexity (via OpenRouter)
```

**Solution**: Set the required environment variables or configure `consensus_config.json`

### Provider Failures
If a provider fails, the tool continues with available providers and notes the failure in output.

### Rate Limiting
- OpenAI: Respect tier limits
- Gemini: Built-in retry logic with exponential backoff
- OpenRouter: Handles multiple provider rate limits

## Requirements

- Python 3.7+
- aiohttp>=3.8.0
- Valid API keys for desired providers

## License

This project is available for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with multiple providers
5. Submit a pull request

## Support

For issues and questions:
1. Check this README for common solutions
2. Review the configuration file format
3. Verify API key setup
4. Test with a simple prompt first