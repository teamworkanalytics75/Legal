# Legal Liaison Agent - Natural Language Chat Interface

## Overview

The Legal Liaison Agent provides a conversational interface that acts as a liaison between you and the entire legal analysis system, similar to ChatGPT but connected to your Conductor, Research, ML, and BN systems.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LegalLiaisonAgent (Main Interface)         │
│  - Manages conversation context                         │
│  - Routes questions to appropriate components           │
│  - Decides quick answer vs full workflow                │
└─────────────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Question  │ │   BN      │ │ Context   │
│Classifier │ │ QueryMapper│ │ Manager   │
└───────────┘ └───────────┘ └───────────┘
        │           │           │
        └───────────┼───────────┘
                    ↓
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
┌───────────┐ ┌───────────┐ ┌───────────┐
│  Quick    │ │   Full    │ │   Chat    │
│  Answer   │ │ Workflow  │ │ Interface │
│  Engine   │ │Orchestrator│ │           │
└───────────┘ └───────────┘ └───────────┘
```

## Components

### 1. QuestionClassifier (`writer_agents/code/QuestionClassifier.py`)

Classifies user questions to determine:
- **Question type**: PROBABILITY, RESEARCH, ANALYSIS, WRITING, HYBRID
- **Required components**: RESEARCH, ML, BN, WRITING
- **Complexity level**: QUICK, MODERATE, COMPLEX

**Example:**
```python
classifier = QuestionClassifier()
classification = classifier.classify("What's the probability my case would constitute national security?")
# Returns: QuestionType.PROBABILITY, [BN, RESEARCH], ComplexityLevel.QUICK
```

### 2. BNQueryMapper (`writer_agents/code/BNQueryMapper.py`)

Maps natural language questions to Bayesian Network queries:
- Extracts keywords/entities from questions
- Maps to BN nodes using entity mappings
- Determines target nodes to query
- Constructs evidence dict for BN inference

**Example:**
```python
mapper = BNQueryMapper()
query = mapper.map_question_to_bn_query("What's the probability of national security risk?")
# Returns: BNQuery with evidence={"Statement_1": "Present"}, target_nodes=["National_Security_Risk"]
```

### 3. ContextManager (`writer_agents/code/ContextManager.py`)

Manages conversation history with:
- **Sliding window**: Last N messages (default: 10)
- **Fact extraction**: Verified facts from system outputs
- **Hallucination prevention**: Doesn't carry forward uncertain info
- **Context summarization**: Condenses old context

### 4. QuickAnswerEngine (`writer_agents/code/QuickAnswerEngine.py`)

Fast-path answers (<10 seconds) via:
- Direct database queries
- Simple BN queries
- CatBoost feature lookup
- Research result summaries

### 5. FullWorkflowOrchestrator (`writer_agents/code/FullWorkflowOrchestrator.py`)

Triggers full Conductor workflows:
- Converts questions to CaseInsights
- Runs full research → ML → writing pipeline
- Generates comprehensive reports

### 6. LegalLiaisonAgent (`writer_agents/code/LegalLiaisonAgent.py`)

Main conversational interface that:
- Coordinates all components
- Manages conversation flow
- Formats responses (conversational + structured)
- Integrates with Conductor, Research, ML, BN systems

### 7. Chat Interface (`writer_agents/scripts/chat_interface.py`)

Interactive CLI interface with:
- Persistent chat session
- History management (`/history`)
- Verified facts (`/facts`)
- Export conversation (`/export`)
- Force full workflow (`/full`)

## Usage

### Basic Usage

```python
from writer_agents.code.LegalLiaisonAgent import LegalLiaisonAgent
from writer_agents.code.WorkflowOrchestrator import Conductor
from writer_agents.code.case_law_researcher import CaseLawResearcher

# Initialize components
conductor = Conductor()
case_law_researcher = CaseLawResearcher()

# Create liaison agent
liaison = LegalLiaisonAgent(
    conductor=conductor,
    case_law_researcher=case_law_researcher,
    memory_store=conductor.memory_store
)

# Ask questions
response = await liaison.ask("What's the probability my case would constitute national security?")
print(response.answer)
```

### Interactive Chat

```bash
cd writer_agents/scripts
python chat_interface.py
```

Commands:
- `/help` - Show help
- `/history` - Show conversation history
- `/facts` - Show verified facts
- `/clear` - Clear conversation history
- `/export` - Export conversation to JSON
- `/full` - Force full workflow for next question
- `/quit` - Exit chat

## Question Types

### Probability Questions
- "What's the probability of X?"
- "Percent chance that Y?"
- "How likely is Z?"

**Handled by**: BNQueryMapper → QuickAnswerEngine (BN queries)

### Research Questions
- "Find cases about X"
- "What does case law say about Y?"
- "Search for precedents on Z"

**Handled by**: QuickAnswerEngine (CaseLawResearcher)

### Analysis Questions
- "Analyze my case"
- "Evaluate my motion"
- "What about X?"

**Handled by**: FullWorkflowOrchestrator (full workflow)

### Writing Questions
- "Write a motion"
- "Draft a section"
- "Create a brief"

**Handled by**: FullWorkflowOrchestrator (full workflow)

## Configuration

### Question Classification Rules
`writer_agents/config/question_classification_rules.json`

Defines:
- Keywords for each question type
- Required components
- Complexity indicators

### BN Node Mappings
`writer_agents/config/bn_node_mappings.json`

Maps entities/keywords to BN nodes:
- Organizations: "Harvard" → "Harvard_Involvement"
- Concepts: "national security" → "National_Security_Risk"
- Evidence: "statement 1" → "Statement_1"

## Example Conversation

```
You: What's the percent probability my case allegations if factually plausible would constitute a matter of national security justifying sealing?

Assistant: Based on Bayesian Network analysis:

  - National_Security_Risk: 72.3% probability of High
  - Sealing_Justification: 68.1% probability of Justified

Found 5 relevant cases. Top matches:

  - Case A v. B (85% relevance)
  - Case C v. D (78% relevance)

Sources: BN, Research

Would you like a full comprehensive analysis?
```

## Integration Points

- **Conductor**: For full workflows
- **CaseLawResearcher**: For research queries
- **RefinementLoop**: For ML analysis
- **BnAdapter**: For probability queries
- **EpisodicMemoryBank**: For conversation history

## Future Enhancements

- Web interface (Streamlit/Gradio)
- Multi-turn conversation refinement
- Visual probability displays
- Export to PDF/Word reports
- Integration with Google Docs

