# Special Tokens Reference

This document describes the special tokens available in Splintr's `cl100k_base` and `o200k_base` tokenizers, including the extended agent token vocabulary.

## Table of Contents

- [Overview](#overview)
- [Design Rationale](#design-rationale)
- [Token ID Allocation](#token-id-allocation)
- [OpenAI Standard Tokens](#openai-standard-tokens)
- [Agent Token Categories](#agent-token-categories)
  - [1. Conversation Structure](#1-conversation-structure)
  - [2. Reasoning / Chain-of-Thought](#2-reasoning--chain-of-thought)
  - [3. ReAct Agent Loop](#3-react-agent-loop)
  - [4. Tool / Function Calling](#4-tool--function-calling)
  - [5. Code Execution](#5-code-execution)
  - [6. RAG / Citations](#6-rag--citations)
  - [7. Memory / State](#7-memory--state)
  - [8. Control Tokens](#8-control-tokens)
  - [9. Multimodal](#9-multimodal)
  - [10. Document Structure](#10-document-structure)
- [Usage Examples](#usage-examples)
- [Python API Reference](#python-api-reference)
- [Rust API Reference](#rust-api-reference)

---

## Overview

Splintr extends the standard OpenAI tokenizer vocabularies with **54 additional special tokens** designed for building modern AI agent systems. These tokens provide semantic structure for:

- Multi-turn chat conversations (ChatML format)
- Chain-of-Thought reasoning (System 2 thinking)
- ReAct-style agent loops (Reason + Act)
- Tool/function calling with error handling
- Code execution environments
- Retrieval-Augmented Generation (RAG) with citations
- Long-term memory and state persistence
- Multimodal content placeholders
- Structured document parsing

---

## Design Rationale

### Why Special Tokens?

Special tokens serve as **semantic markers** that help models understand the structure and intent of different parts of the input. Unlike regular text that gets split into subword tokens, special tokens are:

1. **Atomic**: Always encoded as a single token ID, never split
2. **Unambiguous**: Cannot be confused with regular text
3. **Efficient**: Single token vs multiple tokens for delimiters
4. **Trainable**: Models can learn specific behaviors associated with each token

### Why Extend the Vocabulary?

OpenAI's standard tokenizers include only basic special tokens (`<|endoftext|>`, `<|fim_*|>`, etc.). Modern agent architectures require richer semantic markers to:

- **Separate concerns**: Distinguish thinking from output, actions from observations
- **Enable parsing**: Reliably extract structured data from model outputs
- **Support training**: Provide clear signals for fine-tuning agent behaviors
- **Maintain compatibility**: Work alongside existing tokenizer infrastructure

### Token Naming Convention

All tokens follow the `<|name|>` / `<|/name|>` pattern:

- Opening tags: `<|name|>` - marks the start of a semantic block
- Closing tags: `<|/name|>` - marks the end of a semantic block
- Standalone tokens: `<|name|>` - single markers (e.g., `<|pad|>`, `<|stop|>`)

This convention mirrors XML/HTML for familiarity while using `<|...|>` to avoid conflicts with actual markup in training data.

---

## Token ID Allocation

### Avoiding Conflicts

Token IDs are carefully allocated to avoid conflicts with OpenAI's reserved ranges:

| Model         | Regular Tokens | OpenAI Reserved | Agent Tokens    | Total   |
| ------------- | -------------- | --------------- | --------------- | ------- |
| `cl100k_base` | 0-100,255      | 100,257-100,276 | 100,277-100,330 | 100,331 |
| `o200k_base`  | 0-199,997      | 199,999-200,018 | 200,019-200,072 | 200,073 |

### Why These Ranges?

- **OpenAI compatibility**: Agent tokens start after OpenAI's last known special token
- **Future-proofing**: Gap between OpenAI tokens and agent tokens allows for OpenAI additions
- **Consistency**: Same token semantics map to different IDs per vocabulary, but maintain relative ordering

---

## OpenAI Standard Tokens

These tokens are part of the original OpenAI tokenizer specification:

### cl100k_base (GPT-4, GPT-3.5-turbo)

| Token               | ID     | Purpose                    |
| ------------------- | ------ | -------------------------- |
| `<\|endoftext\|>`   | 100257 | End of document marker     |
| `<\|fim_prefix\|>`  | 100258 | Fill-in-the-middle: prefix |
| `<\|fim_middle\|>`  | 100259 | Fill-in-the-middle: middle |
| `<\|fim_suffix\|>`  | 100260 | Fill-in-the-middle: suffix |
| `<\|endofprompt\|>` | 100276 | End of prompt marker       |

### o200k_base (GPT-4o)

| Token               | ID     | Purpose                |
| ------------------- | ------ | ---------------------- |
| `<\|endoftext\|>`   | 199999 | End of document marker |
| `<\|endofprompt\|>` | 200018 | End of prompt marker   |

---

## Agent Token Categories

### 1. Conversation Structure

**Purpose**: Standard ChatML-style tokens for multi-turn conversations.

| Token             | cl100k ID | o200k ID | Description                                     |
| ----------------- | --------- | -------- | ----------------------------------------------- |
| `<\|system\|>`    | 100277    | 200019   | System instructions defining assistant behavior |
| `<\|user\|>`      | 100278    | 200020   | User input/queries                              |
| `<\|assistant\|>` | 100279    | 200021   | Assistant responses                             |
| `<\|im_start\|>`  | 100280    | 200022   | Generic message start (ChatML)                  |
| `<\|im_end\|>`    | 100281    | 200023   | Generic message end (ChatML)                    |

**Rationale**: These tokens implement the [ChatML format](https://github.com/openai/openai-python/blob/main/chatml.md) used by OpenAI and adopted widely for chat model training. The `im_start`/`im_end` tokens provide a generic wrapper, while role-specific tokens (`system`, `user`, `assistant`) enable direct role marking.

**Example**:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

---

### 2. Reasoning / Chain-of-Thought

**Purpose**: Enable System 2 (slow, deliberate) reasoning similar to DeepSeek-R1 or OpenAI o1.

| Token          | cl100k ID | o200k ID | Description              |
| -------------- | --------- | -------- | ------------------------ |
| `<\|think\|>`  | 100282    | 200024   | Start of reasoning block |
| `<\|/think\|>` | 100283    | 200025   | End of reasoning block   |

**Rationale**: Chain-of-Thought (CoT) prompting significantly improves model performance on complex tasks. Dedicated thinking tokens allow:

- **Training**: Models learn to "think before answering"
- **Inference**: Thinking can be hidden from users in production
- **Analysis**: Reasoning traces can be extracted for debugging/evaluation

**Example**:

```
<|think|>
The user is asking about the capital of France.
I know that Paris is the capital and largest city of France.
It has been the capital since the 10th century.
<|/think|>
The capital of France is Paris.
```

---

### 3. ReAct Agent Loop

**Purpose**: Implement the ReAct (Reason + Act) paradigm for autonomous agents.

| Token            | cl100k ID | o200k ID | Description                     |
| ---------------- | --------- | -------- | ------------------------------- |
| `<\|plan\|>`     | 100284    | 200026   | High-level strategy formulation |
| `<\|/plan\|>`    | 100285    | 200027   | End of plan                     |
| `<\|step\|>`     | 100286    | 200028   | Individual step within plan     |
| `<\|/step\|>`    | 100287    | 200029   | End of step                     |
| `<\|act\|>`      | 100288    | 200030   | Action intent declaration       |
| `<\|/act\|>`     | 100289    | 200031   | End of action                   |
| `<\|observe\|>`  | 100290    | 200032   | Environment feedback            |
| `<\|/observe\|>` | 100291    | 200033   | End of observation              |

**Rationale**: The [ReAct paper](https://arxiv.org/abs/2210.03629) demonstrated that interleaving reasoning and acting improves agent performance. These tokens create a structured loop:

1. **Plan**: Agent decides overall strategy
2. **Step**: Break plan into discrete actions
3. **Act**: Declare intent to perform action
4. **Observe**: Receive and process environment feedback
5. Repeat until task complete

**Example**:

```
<|plan|>
To answer this question, I need to:
1. Search for current weather data
2. Extract the temperature
3. Format the response
<|/plan|>
<|step|>Searching for weather data<|/step|>
<|act|>search("London weather today")<|/act|>
<|observe|>Temperature: 18°C, Condition: Partly cloudy<|/observe|>
<|step|>Formatting response<|/step|>
The current temperature in London is 18°C with partly cloudy skies.
```

---

### 4. Tool / Function Calling

**Purpose**: Structured tool use with explicit success/error handling.

| Token             | cl100k ID | o200k ID | Description                 |
| ----------------- | --------- | -------- | --------------------------- |
| `<\|function\|>`  | 100292    | 200034   | Function call specification |
| `<\|/function\|>` | 100293    | 200035   | End of function call        |
| `<\|result\|>`    | 100294    | 200036   | Successful return value     |
| `<\|/result\|>`   | 100295    | 200037   | End of result               |
| `<\|error\|>`     | 100296    | 200038   | Execution error             |
| `<\|/error\|>`    | 100297    | 200039   | End of error                |

**Rationale**: Function calling is fundamental to agent capabilities. Separating `<|act|>` (intent) from `<|function|>` (technical payload) allows:

- **Intent**: "I want to check the weather" (`<|act|>`)
- **Implementation**: `{"name": "get_weather", "args": {...}}` (`<|function|>`)

The `<|error|>` token is critical for robust agents—it signals that the previous action failed, enabling retry logic without confusing errors with valid outputs.

**Example**:

```
<|function|>{"name": "get_weather", "args": {"city": "London", "units": "celsius"}}<|/function|>
<|result|>{"temperature": 18, "condition": "partly_cloudy", "humidity": 65}<|/result|>
```

**Error handling**:

```
<|function|>{"name": "get_stock_price", "args": {"symbol": "INVALID"}}<|/function|>
<|error|>{"code": "SYMBOL_NOT_FOUND", "message": "Stock symbol 'INVALID' not found"}<|/error|>
```

---

### 5. Code Execution

**Purpose**: Jupyter notebook-style code interpreter flow.

| Token           | cl100k ID | o200k ID | Description           |
| --------------- | --------- | -------- | --------------------- |
| `<\|code\|>`    | 100298    | 200040   | Code block to execute |
| `<\|/code\|>`   | 100299    | 200041   | End of code block     |
| `<\|output\|>`  | 100300    | 200042   | Execution output      |
| `<\|/output\|>` | 100301    | 200043   | End of output         |
| `<\|lang\|>`    | 100302    | 200044   | Language identifier   |
| `<\|/lang\|>`   | 100303    | 200045   | End of language tag   |

**Rationale**: Code execution is a powerful agent capability. These tokens model the notebook paradigm:

- Code cells with explicit language tags
- Captured stdout/return values
- Clear separation between code and output

**Example**:

```
<|code|><|lang|>python<|/lang|>
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

area = calculate_circle_area(5)
print(f"Area: {area:.2f}")
<|/code|>
<|output|>Area: 78.54<|/output|>
```

---

### 6. RAG / Citations

**Purpose**: Retrieval-Augmented Generation with source attribution.

| Token            | cl100k ID | o200k ID | Description             |
| ---------------- | --------- | -------- | ----------------------- |
| `<\|context\|>`  | 100304    | 200046   | Retrieved context block |
| `<\|/context\|>` | 100305    | 200047   | End of context          |
| `<\|quote\|>`    | 100306    | 200048   | Direct quotation        |
| `<\|/quote\|>`   | 100307    | 200049   | End of quote            |
| `<\|cite\|>`     | 100308    | 200050   | Citation reference      |
| `<\|/cite\|>`    | 100309    | 200051   | End of citation         |
| `<\|source\|>`   | 100310    | 200052   | Source metadata         |
| `<\|/source\|>`  | 100311    | 200053   | End of source           |

**Rationale**: RAG systems retrieve relevant documents to ground model responses. These tokens enable:

- **Grounded generation**: Model sees retrieved context explicitly
- **Citation training**: Model learns to cite sources
- **Verification**: Outputs can be traced back to sources
- **Hallucination reduction**: Clear separation of retrieved vs generated content

**Example**:

```
<|context|>
<|source|>wikipedia:Paris<|/source|>
Paris is the capital and most populous city of France. With an official
estimated population of 2,102,650 residents in January 2023 in an area of
more than 105 km², Paris is the fourth-most populated city in the European Union.
<|/context|>

Based on the retrieved information, Paris is the capital of France with a
population of approximately <|quote|>2,102,650 residents<|/quote|>
<|cite|>wikipedia:Paris<|/cite|>.
```

---

### 7. Memory / State

**Purpose**: Long-term memory and state persistence across sessions.

| Token           | cl100k ID | o200k ID | Description         |
| --------------- | --------- | -------- | ------------------- |
| `<\|memory\|>`  | 100312    | 200054   | Store information   |
| `<\|/memory\|>` | 100313    | 200055   | End of memory block |
| `<\|recall\|>`  | 100314    | 200056   | Retrieved memory    |
| `<\|/recall\|>` | 100315    | 200057   | End of recall       |

**Rationale**: Persistent memory enables agents to:

- Remember user preferences across conversations
- Build up knowledge over time
- Maintain continuity in long-running tasks

The separation of `memory` (write) and `recall` (read) mirrors database semantics.

**Example**:

```
<|memory|>User prefers concise responses. User's name is Alice.<|/memory|>

... later in conversation ...

<|recall|>User prefers concise responses. User's name is Alice.<|/recall|>
Hello Alice! Here's a brief answer: The capital of France is Paris.
```

---

### 8. Control Tokens

**Purpose**: Sequence control and formatting.

| Token        | cl100k ID | o200k ID | Description                 |
| ------------ | --------- | -------- | --------------------------- |
| `<\|pad\|>`  | 100316    | 200058   | Padding for batch alignment |
| `<\|stop\|>` | 100317    | 200059   | Generation stop signal      |
| `<\|sep\|>`  | 100318    | 200060   | Segment separator           |

**Rationale**: These are utility tokens for training and inference:

- **pad**: Aligns sequences in batches (has no semantic meaning)
- **stop**: Alternative to `<|endoftext|>` for stopping generation
- **sep**: Separates segments without implying document boundaries

---

### 9. Multimodal

**Purpose**: Placeholders for non-text content.

| Token          | cl100k ID | o200k ID | Description   |
| -------------- | --------- | -------- | ------------- |
| `<\|image\|>`  | 100319    | 200061   | Image content |
| `<\|/image\|>` | 100320    | 200062   | End of image  |
| `<\|audio\|>`  | 100321    | 200063   | Audio content |
| `<\|/audio\|>` | 100322    | 200064   | End of audio  |
| `<\|video\|>`  | 100323    | 200065   | Video content |
| `<\|/video\|>` | 100324    | 200066   | End of video  |

**Rationale**: Multimodal models need to mark where non-text embeddings are inserted. These tokens serve as:

- **Placeholders**: Mark positions for embedding injection
- **Delimiters**: Wrap base64-encoded or referenced content
- **Training signals**: Help models learn cross-modal attention

**Example**:

```
Describe what you see in this image:
<|image|>base64_encoded_image_data_here<|/image|>

The image shows a sunset over the ocean with vibrant orange and purple colors.
```

---

### 10. Document Structure

**Purpose**: Semantic layout for parsing structured documents.

| Token            | cl100k ID | o200k ID | Description            |
| ---------------- | --------- | -------- | ---------------------- |
| `<\|title\|>`    | 100325    | 200067   | Document/section title |
| `<\|/title\|>`   | 100326    | 200068   | End of title           |
| `<\|section\|>`  | 100327    | 200069   | Semantic section       |
| `<\|/section\|>` | 100328    | 200070   | End of section         |
| `<\|summary\|>`  | 100329    | 200071   | Content summary        |
| `<\|/summary\|>` | 100330    | 200072   | End of summary         |

**Rationale**: When processing structured documents (papers, reports, documentation), these tokens help:

- **Preserve structure**: Maintain document hierarchy in tokenized form
- **Enable extraction**: Reliably parse titles, sections, summaries
- **Support generation**: Train models to produce well-structured output

**Example**:

```
<|title|>Climate Change Impact Assessment<|/title|>

<|summary|>
This report examines the effects of climate change on coastal ecosystems,
finding significant impacts on biodiversity and recommending adaptive strategies.
<|/summary|>

<|section|>
<|title|>Introduction<|/title|>
Climate change represents one of the most significant challenges...
<|/section|>

<|section|>
<|title|>Methodology<|/title|>
We analyzed data from 50 coastal monitoring stations...
<|/section|>
```

---

## Usage Examples

### Python

```python
from Splintr import Tokenizer, CL100K_AGENT_TOKENS

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode text with special tokens
text = "<|think|>Let me reason about this...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)

# Check for specific tokens
if CL100K_AGENT_TOKENS.THINK in tokens:
    print("Contains thinking block")

# Decode back to text
decoded = tokenizer.decode(tokens)
assert decoded == text

# Access token IDs programmatically
print(f"THINK token ID: {CL100K_AGENT_TOKENS.THINK}")        # 100282
print(f"FUNCTION token ID: {CL100K_AGENT_TOKENS.FUNCTION}")  # 100292
```

### Rust

```rust
use Splintr::{Tokenizer, cl100k_agent_tokens, CL100K_BASE_PATTERN};

// Access token constants
let think_id = cl100k_agent_tokens::THINK;           // 100282
let function_id = cl100k_agent_tokens::FUNCTION;     // 100292

// Use in your agent implementation
fn extract_thinking(tokens: &[u32]) -> Option<(usize, usize)> {
    let start = tokens.iter().position(|&t| t == cl100k_agent_tokens::THINK)?;
    let end = tokens.iter().position(|&t| t == cl100k_agent_tokens::THINK_END)?;
    Some((start, end))
}
```

---

## Python API Reference

### CL100K_AGENT_TOKENS

```python
from Splintr import CL100K_AGENT_TOKENS

# Conversation
CL100K_AGENT_TOKENS.SYSTEM          # 100277
CL100K_AGENT_TOKENS.USER            # 100278
CL100K_AGENT_TOKENS.ASSISTANT       # 100279
CL100K_AGENT_TOKENS.IM_START        # 100280
CL100K_AGENT_TOKENS.IM_END          # 100281

# Thinking
CL100K_AGENT_TOKENS.THINK           # 100282
CL100K_AGENT_TOKENS.THINK_END       # 100283

# ReAct
CL100K_AGENT_TOKENS.PLAN            # 100284
CL100K_AGENT_TOKENS.PLAN_END        # 100285
CL100K_AGENT_TOKENS.STEP            # 100286
CL100K_AGENT_TOKENS.STEP_END        # 100287
CL100K_AGENT_TOKENS.ACT             # 100288
CL100K_AGENT_TOKENS.ACT_END         # 100289
CL100K_AGENT_TOKENS.OBSERVE         # 100290
CL100K_AGENT_TOKENS.OBSERVE_END     # 100291

# Tool/Function
CL100K_AGENT_TOKENS.FUNCTION        # 100292
CL100K_AGENT_TOKENS.FUNCTION_END    # 100293
CL100K_AGENT_TOKENS.RESULT          # 100294
CL100K_AGENT_TOKENS.RESULT_END      # 100295
CL100K_AGENT_TOKENS.ERROR           # 100296
CL100K_AGENT_TOKENS.ERROR_END       # 100297

# Code
CL100K_AGENT_TOKENS.CODE            # 100298
CL100K_AGENT_TOKENS.CODE_END        # 100299
CL100K_AGENT_TOKENS.OUTPUT          # 100300
CL100K_AGENT_TOKENS.OUTPUT_END      # 100301
CL100K_AGENT_TOKENS.LANG            # 100302
CL100K_AGENT_TOKENS.LANG_END        # 100303

# RAG
CL100K_AGENT_TOKENS.CONTEXT         # 100304
CL100K_AGENT_TOKENS.CONTEXT_END     # 100305
CL100K_AGENT_TOKENS.QUOTE           # 100306
CL100K_AGENT_TOKENS.QUOTE_END       # 100307
CL100K_AGENT_TOKENS.CITE            # 100308
CL100K_AGENT_TOKENS.CITE_END        # 100309
CL100K_AGENT_TOKENS.SOURCE          # 100310
CL100K_AGENT_TOKENS.SOURCE_END      # 100311

# Memory
CL100K_AGENT_TOKENS.MEMORY          # 100312
CL100K_AGENT_TOKENS.MEMORY_END      # 100313
CL100K_AGENT_TOKENS.RECALL          # 100314
CL100K_AGENT_TOKENS.RECALL_END      # 100315

# Control
CL100K_AGENT_TOKENS.PAD             # 100316
CL100K_AGENT_TOKENS.STOP            # 100317
CL100K_AGENT_TOKENS.SEP             # 100318

# Multimodal
CL100K_AGENT_TOKENS.IMAGE           # 100319
CL100K_AGENT_TOKENS.IMAGE_END       # 100320
CL100K_AGENT_TOKENS.AUDIO           # 100321
CL100K_AGENT_TOKENS.AUDIO_END       # 100322
CL100K_AGENT_TOKENS.VIDEO           # 100323
CL100K_AGENT_TOKENS.VIDEO_END       # 100324

# Document
CL100K_AGENT_TOKENS.TITLE           # 100325
CL100K_AGENT_TOKENS.TITLE_END       # 100326
CL100K_AGENT_TOKENS.SECTION         # 100327
CL100K_AGENT_TOKENS.SECTION_END     # 100328
CL100K_AGENT_TOKENS.SUMMARY         # 100329
CL100K_AGENT_TOKENS.SUMMARY_END     # 100330
```

### O200K_AGENT_TOKENS

Same structure as above, with IDs starting at 200019.

---

## Rust API Reference

### cl100k_agent_tokens module

```rust
use Splintr::cl100k_agent_tokens;

// All constants follow the same naming as Python
cl100k_agent_tokens::SYSTEM          // 100277
cl100k_agent_tokens::THINK           // 100282
cl100k_agent_tokens::FUNCTION        // 100292
// ... etc
```

### o200k_agent_tokens module

```rust
use Splintr::o200k_agent_tokens;

o200k_agent_tokens::SYSTEM           // 200019
o200k_agent_tokens::THINK            // 200024
// ... etc
```

---

## See Also

- [README.md](../README.md) - Project overview and quick start
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - ReAct: Synergizing Reasoning and Acting in Language Models
- [ChatML Specification](https://github.com/openai/openai-python/blob/main/chatml.md) - Chat Markup Language
