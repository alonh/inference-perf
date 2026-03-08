# Advanced OTEL Test Traces

This directory contains complex OpenTelemetry trace files demonstrating advanced agentic workflows and dependency patterns.

## Files

### multi_agent_research.json
Complex multi-agent research workflow with parallel execution and synthesis.
- **Nodes**: 12
- **Pattern**: Coordinator → Parallel Agents → Synthesis → Follow-up
- **Use case**: Multi-agent research system
- **Complexity**: ⭐⭐⭐⭐ Advanced

**Structure**:
1. Coordinator dispatches research tasks
2. Three agents work in parallel (Market, Environmental, Technology)
3. Coordinator synthesizes results
4. User follow-up triggers second round of parallel research
5. Final synthesis

**Key Features**:
- Parallel fan-out (1 → N)
- Fan-in aggregation (N → 1)
- Multi-level dependencies
- Real-world agentic pattern

### code_review_workflow.json
Comprehensive code review workflow with multiple reviewers.
- **Nodes**: 15
- **Pattern**: Coordinator → Parallel Reviews → Synthesis → Iterations
- **Use case**: Automated code review system
- **Complexity**: ⭐⭐⭐⭐⭐ Expert

**Key Features**:
- Multiple parallel review streams
- Iterative refinement
- Complex dependency chains
- Realistic enterprise workflow

### parallel_requests.json
Basic parallel execution pattern.
- **Nodes**: 5
- **Pattern**: Root → 2 Parallel → Aggregation
- **Use case**: Parallel task execution
- **Complexity**: ⭐⭐ Intermediate

### fan_in_dependencies.json
Fan-in pattern where multiple nodes converge.
- **Nodes**: 4
- **Pattern**: 2 Parallel → 1 Aggregator
- **Use case**: Parallel data gathering with aggregation
- **Complexity**: ⭐⭐ Intermediate

### prefix_sharing_example.json
Demonstrates KV cache prefix sharing optimization.
- **Nodes**: 7
- **Pattern**: Shared prefix with branching
- **Use case**: Efficient context reuse
- **Complexity**: ⭐⭐⭐ Advanced

## Documentation

- **DEPENDENCY_INFERENCE_EXPLAINED.md**: Deep dive into how dependencies are inferred
- **COMPLEX_SCENARIOS.md**: Detailed analysis of complex patterns
- **PREFIX_SHARING_EXAMPLE.md**: KV cache optimization examples

## Usage

### Session-Based Replay (Recommended)
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/advanced/session-based.yml
```

### Multi-Session Concurrent Replay
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/advanced/multi-session.yml
```

### With Mock Client
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/advanced/mock-advanced.yml
```

## Key Concepts

### Dependency Types
1. **Causal**: Output of node A appears in input of node B
2. **Temporal**: Node B starts after node A completes (timing-based)
3. **Prefix Sharing**: Nodes share common message prefix (KV cache optimization)

### Async Dependency Handling
The system uses async waiting to prevent deadlocks:
- Workers don't block synchronously
- Dependencies resolved before HTTP dispatch
- Output substitution happens at runtime

### Output Substitution
Recorded assistant messages are replaced with actual LLM outputs during replay:
- Ensures realistic growing-context patterns
- Maintains causal relationships
- Enables proper KV cache behavior

## Testing Patterns

1. **Linear Chains**: Start with simple_chain.json
2. **Parallel Execution**: Try parallel_requests.json
3. **Fan-in/Fan-out**: Explore fan_in_dependencies.json
4. **Multi-Agent**: Test with multi_agent_research.json
5. **Complex Workflows**: Challenge with code_review_workflow.json

## Performance Considerations

- **Concurrency**: Adjust `num_workers` based on dependency depth
- **Rate Limiting**: Use `session_rate` to control load
- **Timeouts**: Dependencies have 5-minute timeout by default
- **Memory**: Large traces may require increased worker memory

## Troubleshooting

### Deadlock Issues
If nodes wait indefinitely:
- Check predecessor_node_ids are correct
- Verify all nodes register outputs
- Increase timeout in config

### Missing Outputs
If output substitution fails:
- Check NodeOutputRegistry is shared (mp_manager)
- Verify process_response registers outputs
- Look for errors in predecessor nodes

## Creating Custom Traces

See the existing traces as templates. Key requirements:
- Valid JSON structure
- Proper span timing
- Complete message history
- Realistic output text