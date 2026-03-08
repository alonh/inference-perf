# Complex Multi-Agent Test Scenarios

This document describes the complex multi-agent test traces and their expected dependency patterns.

## Test Case 4: Multi-Agent Research (`multi_agent_research.json`)

### Scenario Description
A comprehensive research workflow where a coordinator dispatches multiple specialized agents in parallel, synthesizes their results, interacts with the user, and repeats the pattern for follow-up questions.

### Workflow Structure

**Phase 1: Initial Research (0-11s)**
```
span_001_coordinator (0-1.5s)
    ↓ (dispatches 3 agents in parallel)
├─ span_002_market_agent (2-5s)
├─ span_003_env_agent (2.5-6s)  
└─ span_004_tech_agent (3-7s)
    ↓ (all feed into synthesis)
span_005_synthesis (8-11s)
```

**Phase 2: User Interaction (12-13.5s)**
```
span_006_user_response (12-13.5s)
    ← depends on span_005_synthesis
```

**Phase 3: Deep Dive Research (14-22s)**
```
span_006_user_response
    ↓ (triggers new research)
├─ span_007_infra_market (14-17s)
├─ span_008_infra_tech (14.5-18s)
└─ span_009_infra_policy (15-18.5s)
    ↓ (all feed into synthesis)
span_010_infra_synthesis (19-22s)
```

**Phase 4: Follow-up (23-28s)**
```
span_011_user_followup (23-24.5s)
    ← depends on span_010_infra_synthesis
    ↓
span_012_cost_analysis (25-28s)
```

### Expected Dependencies

**Detected by Containment Heuristic:**
- `span_006_user_response` → `span_005_synthesis` ✓ (synthesis output in user response input)
- `span_011_user_followup` → `span_010_infra_synthesis` ✓ (synthesis output in followup input)

**Not Detected (Synthesis Input is Summary, Not Verbatim):**
- `span_005_synthesis` should depend on `span_002`, `span_003`, `span_004`
  - Reason: Synthesis input contains *summaries* of agent outputs, not verbatim text
  - The coordinator rewrites/summarizes the outputs before passing to synthesis
- `span_010_infra_synthesis` should depend on `span_007`, `span_008`, `span_009`
  - Same reason as above

### Key Insights

1. **Parallel Agent Execution**: Agents 002, 003, 004 run concurrently (overlapping times)
   - No forced ordering between them ✓
   - All are independent roots ✓

2. **Fan-in Pattern**: Multiple agents → single synthesis
   - Detected when synthesis input contains verbatim agent outputs
   - Not detected when coordinator summarizes/rewrites outputs

3. **User Interaction Breaks**: Clear dependency on synthesis before user response
   - Detected correctly ✓

4. **Multi-Phase Workflow**: Pattern repeats for follow-up questions
   - Each phase has its own parallel agents → synthesis → user interaction cycle

### Statistics
- Total events: 12
- Detected dependencies: 2 (user interactions depending on synthesis)
- Parallel execution phases: 2 (agents in phase 1 and phase 3)
- Synthesis points: 2 (span_005, span_010)

---

## Test Case 5: Code Review Workflow (`code_review_workflow.json`)

### Scenario Description
An iterative code review workflow where multiple specialized reviewers analyze code in parallel, provide feedback, generate refactored versions, and create comprehensive test suites.

### Workflow Structure

**Phase 1: Initial Review (0-9s)**
```
span_001_initial (0-1s)
    ↓ (dispatches 3 reviewers in parallel)
├─ span_002_security (1.5-4s)
├─ span_003_performance (2-4.5s)
└─ span_004_quality (2.5-5s)
    ↓ (all feed into synthesis)
span_005_synthesis (6-9s)
```

**Phase 2: User Requests Refactoring (10-20s)**
```
span_006_user_request_fix (10-11s)
    ← depends on span_005_synthesis
    ↓ (dispatches 3 refactoring agents in parallel)
├─ span_007_security_fix (11.5-15s)
├─ span_008_performance_fix (12-15.5s)
└─ span_009_quality_fix (12.5-16s)
    ↓ (all feed into synthesis)
span_010_refactor_synthesis (17-20s)
```

**Phase 3: User Requests Testing Strategy (21-30s)**
```
span_011_user_question (21-22s)
    ← depends on span_010_refactor_synthesis
    ↓ (dispatches 3 test agents in parallel)
├─ span_012_test_unit (22.5-25s)
├─ span_013_test_integration (23-25.5s)
└─ span_014_test_security (23.5-26s)
    ↓ (all feed into synthesis)
span_015_test_synthesis (27-30s)
```

### Expected Dependencies

**Detected by Containment Heuristic:**
- `span_006_user_request_fix` → `span_005_synthesis` (review summary in user request)
- `span_011_user_question` → `span_010_refactor_synthesis` (refactor summary in question)

**Not Detected (Same Reason as Multi-Agent Research):**
- Synthesis spans should depend on their respective agent outputs
- But coordinator summarizes/rewrites outputs before synthesis

### Key Insights

1. **Three-Phase Iterative Workflow**:
   - Phase 1: Review → Synthesis → User feedback
   - Phase 2: Refactor → Synthesis → User feedback
   - Phase 3: Testing → Synthesis

2. **Consistent Pattern**: Each phase follows the same structure
   - Parallel agent execution (3 agents per phase)
   - Synthesis combining results
   - User interaction triggering next phase

3. **Progressive Refinement**: Each phase builds on previous
   - Review identifies issues
   - Refactor fixes issues
   - Testing validates fixes

4. **Parallel Specialization**: Different agent types per phase
   - Phase 1: Security, Performance, Quality reviewers
   - Phase 2: Security, Performance, Quality fixers
   - Phase 3: Unit, Integration, Security testers

### Statistics
- Total events: 15
- Detected dependencies: 2 (user interactions)
- Parallel execution phases: 3 (3 agents per phase)
- Synthesis points: 3 (spans 005, 010, 015)
- Phases: 3 (review, refactor, test)

---

## Dependency Inference Limitations

### What Works Well ✓
1. **Sequential chains**: When output text appears verbatim in next input
2. **User interactions**: Conversation history includes previous responses
3. **Parallel detection**: Overlapping time ranges prevent forced ordering
4. **Fan-in from user**: Multiple agent outputs quoted in user message

### What Doesn't Work ❌
1. **Coordinator summarization**: When coordinator rewrites agent outputs
2. **Implicit dependencies**: Logical dependencies not reflected in text
3. **Structured data**: When outputs are JSON/structured, not prose
4. **Paraphrasing**: When input references output concepts but not exact text

### Recommendations for Better Dependency Detection

**Option 1: Explicit Dependency Metadata**
Add `depends_on_span_ids` to span attributes:
```json
{
  "attributes": {
    "exgentic.depends_on": "span_002;span_003;span_004"
  }
}
```

**Option 2: Include Verbatim Outputs in Synthesis Input**
Instead of summarizing, include full outputs:
```json
{
  "role": "user",
  "content": "Synthesize: [AGENT_002_OUTPUT] ... [AGENT_003_OUTPUT] ..."
}
```

**Option 3: Use Parent Span IDs**
Set `parent_span_id` to indicate dependencies:
```json
{
  "span_id": "span_005_synthesis",
  "parent_span_id": "span_001_coordinator"
}
```

**Option 4: Semantic Similarity (Advanced)**
Use embeddings to detect semantic dependencies:
- Compute embeddings for outputs and inputs
- High cosine similarity → likely dependency
- More expensive but handles paraphrasing

---

## Replay Behavior

### With Current Dependencies
The replay will:
1. ✓ Dispatch parallel agents correctly (no forced ordering)
2. ✓ Wait for synthesis before user interaction
3. ✓ Preserve wall-clock timing between phases
4. ✗ Not wait for agents before synthesis (missing dependencies)

### Impact on Prefix Caching
- **Positive**: Parallel agents may share system prompts (cache hit)
- **Positive**: User interactions include conversation history (cache hit)
- **Neutral**: Synthesis may start before agents complete (but will fail in real system)

### Workaround for Testing
If you need accurate dependencies for testing:
1. Add explicit `depends_on_span_ids` to trace
2. Or use `parent_span_id` to indicate coordination
3. Or modify synthesis inputs to include verbatim outputs

---

## Conclusion

These complex test cases demonstrate:
- ✓ Parallel multi-agent execution patterns
- ✓ Multi-phase iterative workflows
- ✓ User interaction dependency detection
- ✗ Coordinator-mediated fan-in patterns (limitation)

The dependency inference works well for direct text containment but struggles with coordinator-mediated workflows where outputs are summarized or restructured. For production use, consider adding explicit dependency metadata to traces.