---
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - LS
  - BashOutput
  - KillBash
description: Query multiple AI providers concurrently for expert advice with comprehensive prompts
argument-hint: "[optional: specific focus area or leave empty for context-based]"
---

# Consensus - Query Multiple AIs

Claude Code will analyze your request and current context to craft a comprehensive prompt, then query multiple AI providers concurrently (GPT-5, Gemini 2.5 Pro, Perplexity via openrouter.ai) and consolidate their responses.

Arguments provided: $ARGUMENTS

## Workflow:

This command will:
1. **Analyze your request**: Claude Code will determine what problem to focus on based on:
   - Your explicit instructions (if provided in $ARGUMENTS)
   - Recent conversation context and issues you've been working on
   - Current codebase state and recent errors or challenges

2. **Craft a comprehensive prompt**: Claude Code will create a detailed prompt including:
   - Problem statement with full technical context
   - What has been tried and the outcomes
   - Relevant code snippets, configurations, and error messages
   - Specific questions for the AI providers to address

3. **Save prompt to file**: Write the crafted prompt to a markdown file in the `consensus_docs` folder

4. **Execute concurrent queries**: Run the consensus.py script to query multiple providers simultaneously

5. **Analyze responses and present consensus findings**: All responses will be saved as `consensus_docs/consolidated-{timestamp}.md`, and Claude Code will analyze the results to identify common themes, unique perspectives, and overall consensus

## Your Input Options:

- **With specific focus**: `/consensus "help me optimize the SSE connection handling"`
- **Without input**: `/consensus` (Claude Code will infer from context)

## Expected Input:
- Optional: Your instructions for what the prompt should focus on
- If no input provided, Claude Code will analyze the current context

## How Claude Code Will Structure the Multi-AI Prompt:

Claude Code will craft a comprehensive prompt with these essential components:

1. **Problem Statement**: Start with a clear, concise description of the core issue. Be specific about what's not working, what error you're encountering, or what goal you're trying to achieve.

2. **Previous Attempts and Their Outcomes**: Document each approach you've tried, why you tried it, and what happened. Include both successful partial solutions and complete failures - this helps AI providers avoid suggesting redundant solutions.

3. **Environmental Context**: Provide complete technical details including:
   - Programming languages, frameworks, and major library versions
   - Infrastructure setup (cloud providers, deployment methods, CI/CD)
   - System constraints (performance requirements, security policies, budget)
   - Team constraints (skill levels, available time, organizational policies)
   - Any relevant compliance or regulatory requirements

4. **Supporting Evidence**: Include concrete data such as:
   - Specific error messages with stack traces
   - Performance metrics or benchmarks
   - Relevant configuration files or code snippets
   - Links to documentation you've already consulted
   - Similar issues from forums or GitHub that didn't solve your problem

5. **Desired Outcome**: Clearly articulate what success looks like. Specify if you need:
   - A theoretical explanation of concepts
   - Practical implementation guidance
   - Code review or architecture validation
   - Alternative approaches ranked by trade-offs
   - Industry best practices and standards

6. **Decision Criteria**: Help AI providers prioritize solutions by specifying what matters most:
   - Performance optimization vs. code simplicity
   - Short-term fixes vs. long-term maintainability
   - Cost considerations vs. ideal solutions
   - Time-to-market vs. technical debt

7. **Response Requirements**: Include explicit instructions that:
   - AI providers must provide direct answers using their best judgment and available knowledge
   - No follow-up questions or requests for additional information are allowed
   - If information is incomplete or uncertain, providers should make reasonable assumptions and clearly state them
   - Providers should work with the information provided and give their best analysis/recommendations

## Command Execution:

After crafting the prompt, Claude Code will:
1. Check if consensus_docs directory exists (create only if needed)
2. Save the prompt to a file in consensus_docs: `consensus_docs/prompt-{timestamp}.md`
3. Execute the consensus.py script and wait for completion
4. Analyze the consolidated responses and present consensus findings

**Execution Process:**
```bash
# Default: web search mode (uses config default)
python consensus.py consensus_docs/prompt-{timestamp}.md

# Explicit no-web mode only when requested
python consensus.py consensus_docs/prompt-{timestamp}.md --search-mode none
```

**Simple Waiting Approach:**
- Claude Code executes the script and waits for it to complete
- No complex monitoring or real-time progress updates
- Results are presented once the process finishes successfully

## Search Modes:

The script supports two search modes:
- **web**: Include current web information (default mode, good for recent technologies)
- **none**: Fast responses without web search (good for coding questions)

## AI Providers:

The script queries multiple providers concurrently:
- **OpenAI GPT-5**: Using the latest Responses API with reasoning capabilities
- **Google Gemini 2.5 Pro**: With thinking capabilities and large context
- **Perplexity Sonar**: With built-in web search and research capabilities

Each provider offers different strengths and perspectives on your question.

## Expected Output:
- A timestamped markdown file: `consensus_docs/consolidated-YYYYMMDD_HHMMSS.md`
- Contains responses from all available AI providers
- Shows which providers were used
- All input prompts and output responses are organized in the `consensus_docs` folder
- Claude Code will automatically read the consolidated results and provide consensus analysis including:
  - **Common themes**: Areas where multiple AI providers agree
  - **Unique perspectives**: Insights or approaches mentioned by only one provider
  - **Consensus summary**: Overall agreement or majority opinion
  - **Minority viewpoints**: Dissenting opinions or alternative approaches
  - **Key takeaways**: Actionable insights synthesized from all responses

## Requirements:
- Python dependencies installed (see requirements.txt)
- Valid API keys configured in `consensus_config.json` or environment variables
- The `consensus_docs` folder in the project root (checked automatically)

## Context Gathering:

When no specific focus is provided, Claude Code will:
1. Review recent error messages and debugging attempts
2. Analyze current git changes and recent commits
3. Check for failing tests or build issues
4. Consider the conversation history for recurring problems
5. Examine CLAUDE.md for project-specific challenges

This ensures all AI providers receive the most relevant context for providing actionable advice with diverse perspectives.

## Consensus Analysis Approach:

Claude Code will analyze the multiple AI responses to provide:

1. **Agreement Identification**: Find common recommendations, approaches, or conclusions across providers
2. **Perspective Mapping**: Identify which provider contributed which unique insights
3. **Confidence Assessment**: Evaluate how strongly the consensus is supported
4. **Conflict Resolution**: Address any contradictory advice between providers
5. **Synthesis**: Combine the best elements from all responses into actionable guidance

The goal is to leverage the collective intelligence of multiple AI systems while highlighting where they differ and why those differences might matter.