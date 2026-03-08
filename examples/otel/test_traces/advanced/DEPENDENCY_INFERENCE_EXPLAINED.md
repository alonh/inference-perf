# Understanding `infer_dependencies` Return Values

The `infer_dependencies` function returns two data structures: `deps` and `metadata`. Here's what each contains:

## Return Value Structure

```python
def infer_dependencies(...) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    return deps, metadata
```

## 1. `deps` - The Dependency Graph

**Type**: `Dict[str, List[str]]`

**Purpose**: Maps each event to its list of predecessor events (the events it depends on)

**Structure**:
```python
deps = {
    "event_id_1": [],                           # No dependencies (root event)
    "event_id_2": ["event_id_1"],              # Depends on event_id_1
    "event_id_3": ["event_id_2", "event_id_1"] # Depends on event_id_2 and event_id_1
}
```

**Real Example** (from prefix_sharing_example):
```python
deps = {
    "span_independent_1": [],
    "span_independent_2": ["span_independent_1"],
    "span_conv_turn1": ["span_independent_2", "span_independent_1"],
    "span_conv_turn2": ["span_conv_turn1", "span_independent_2", "span_independent_1"],
    "span_conv_turn3": ["span_conv_turn2", "span_conv_turn1", "span_independent_2"],
    # ... more events
}
```

### Key Points about `deps`:
- **Empty list** `[]` means the event has no dependencies (it's a root/independent event)
- **Order matters**: Dependencies are sorted by score (highest first), so the first dependency is the most important
- **Max dependencies**: Limited by `max_deps` parameter (default: 3)
- **Used for**: Scheduling events in replay - an event can only start after its dependencies complete

---

## 2. `metadata` - Detailed Dependency Information

**Type**: `Dict[str, Dict[str, Any]]`

**Purpose**: Provides detailed information about WHY and HOW events are related

**Structure**:
```python
metadata = {
    "event_id": {
        "dependency_types": List[str],           # Type of each dependency
        "prefix_sharing_candidates": List[str],  # Events with prefix overlap
        "prefix_scores": Dict[str, float]        # Prefix overlap scores (0.0-1.0)
    }
}
```

### 2.1 `dependency_types`

**Type**: `List[str]`

**Purpose**: Explains the TYPE of relationship for each dependency (parallel to `deps` list)

**Possible Values**:
- `"sequential"` - Event B's input contains Event A's output (causal dependency)
- `"prefix_sharing"` - Event B's input extends Event A's input (KV cache opportunity)
- `"sequential+prefix_sharing"` - Both sequential AND prefix sharing (strongest relationship)
- `"timestamp"` - Only temporal ordering (fallback when no content relationship found)
- `""` (empty) - Dependency exists but type couldn't be determined

**Real Example**:
```python
# For span_conv_turn2:
deps["span_conv_turn2"] = ["span_conv_turn1", "span_independent_2", "span_independent_1"]
metadata["span_conv_turn2"]["dependency_types"] = [
    "sequential+prefix_sharing",  # Corresponds to span_conv_turn1
    "timestamp",                   # Corresponds to span_independent_2
    "timestamp"                    # Corresponds to span_independent_1
]
```

### 2.2 `prefix_sharing_candidates`

**Type**: `List[str]`

**Purpose**: Lists ALL events that share a significant prefix with this event's input (not limited by `max_deps`)

**Criteria**: Prefix overlap score >= 0.5 (50% of shorter input is shared prefix)

**Real Example**:
```python
metadata["span_conv_turn3"]["prefix_sharing_candidates"] = [
    "span_conv_turn2",  # Turn 3 extends Turn 2's input
    "span_conv_turn1"   # Turn 3 also extends Turn 1's input
]
```

**Key Difference from `deps`**:
- `deps` is limited to `max_deps` (default 3) best dependencies
- `prefix_sharing_candidates` includes ALL events with significant prefix overlap
- Used for KV cache analysis and optimization opportunities

### 2.3 `prefix_scores`

**Type**: `Dict[str, float]`

**Purpose**: Quantifies HOW MUCH prefix overlap exists with each candidate

**Score Range**: 0.0 to 1.0
- `1.0` = Perfect prefix match (100% of shorter input is shared)
- `0.5` = Minimum threshold (50% overlap)
- Higher scores = Better KV cache reuse opportunity

**Real Example**:
```python
metadata["span_conv_turn3"]["prefix_scores"] = {
    "span_conv_turn2": 1.0,  # Perfect prefix match
    "span_conv_turn1": 1.0   # Also perfect prefix match
}
```

---

## Complete Real-World Example

Let's trace through `span_conv_turn2` from our example:

### Input Data:
```
Turn 1: "System: You are helpful... User: What is a hash table?"
Turn 2: "System: You are helpful... User: What is a hash table? Assistant: A hash table is... User: What is the time complexity?"
```

### Output:

#### `deps`:
```python
deps["span_conv_turn2"] = [
    "span_conv_turn1",      # Most important dependency
    "span_independent_2",   # Less important
    "span_independent_1"    # Least important
]
```

#### `metadata`:
```python
metadata["span_conv_turn2"] = {
    "dependency_types": [
        "sequential+prefix_sharing",  # Turn 1 → Turn 2: output appears in input AND prefix shared
        "timestamp",                   # Independent 2 → Turn 2: only temporal ordering
        "timestamp"                    # Independent 1 → Turn 2: only temporal ordering
    ],
    "prefix_sharing_candidates": [
        "span_conv_turn1"  # Only Turn 1 shares prefix (Independent calls don't)
    ],
    "prefix_scores": {
        "span_conv_turn1": 1.0  # 100% prefix match (Turn 2 extends Turn 1 perfectly)
    }
}
```

---

## How They're Used

### 1. In Replay Scheduling (`deps`):
```python
# Event can only start after all its dependencies complete
for event in events:
    wait_for_completion(deps[event.event_id])
    start_event(event)
```

### 2. In KV Cache Analysis (`metadata`):
```python
# Calculate potential KV cache savings
for event in events:
    candidates = metadata[event.event_id]["prefix_sharing_candidates"]
    for candidate in candidates:
        score = metadata[event.event_id]["prefix_scores"][candidate]
        if score >= 0.8:  # High reuse opportunity
            print(f"Event {event.event_id} can reuse {score*100}% of {candidate}'s KV cache")
```

### 3. In Dependency Visualization:
```python
# Show why events are related
for event_id, dep_list in deps.items():
    types = metadata[event_id]["dependency_types"]
    for dep_id, dep_type in zip(dep_list, types):
        print(f"{dep_id} → {event_id} ({dep_type})")
```

---

## Summary Table

| Field | Location | Purpose | Example Value |
|-------|----------|---------|---------------|
| `deps[event_id]` | Top-level dict | List of predecessor event IDs | `["event_1", "event_2"]` |
| `dependency_types` | metadata | Why each dependency exists | `["sequential", "timestamp"]` |
| `prefix_sharing_candidates` | metadata | All events with prefix overlap | `["event_1", "event_3"]` |
| `prefix_scores` | metadata | Quantify prefix overlap | `{"event_1": 1.0, "event_3": 0.75}` |

---

## Key Insights

1. **`deps` is for execution**: Determines when events can run
2. **`metadata` is for analysis**: Explains relationships and optimization opportunities
3. **Prefix sharing ≠ Dependency**: An event can share a prefix without being a dependency (if it's beyond `max_deps`)
4. **Dependency types matter**: `sequential+prefix_sharing` is the strongest relationship, indicating both causal and KV cache benefits
5. **Scores guide optimization**: Higher prefix scores = better KV cache reuse opportunities