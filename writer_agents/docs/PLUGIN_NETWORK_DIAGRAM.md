# 📊 Plugin Network Architecture Diagram

**Created:** 2025-11-06
**Purpose:** Visual representation of the plugin system hierarchy and relationships

---

## 🎯 Quick Reference

- **Plugin Registry:** [plugin_registry.json](../config/plugin_registry.json)
- **Interactive HTML:** [plugin_network.html](plugin_network.html)
- **Master Checklist:** [LEGAL_CHECKLISTS_SCAFFOLD.md](LEGAL_CHECKLISTS_SCAFFOLD.md)

---

## 📐 Mermaid Diagram

```mermaid
graph TB
    subgraph "WorkflowOrchestrator (Conductor)"
        WO[WorkflowOrchestrator<br/>Main Orchestrator]
    end

    subgraph "Quality Systems"
        RL[RefinementLoop<br/>Sub-Coordinator]
        QGP[QualityGatePipeline<br/>Validation System]
    end

    subgraph "Document Level Plugins"
        SS[Section Structure Plugin]
        CR[Citation Retrieval Plugin]
        WC[Word Count Plugin]
    end

    subgraph "Section Level Plugins"
        SR1[Statutory Requirement 1<br/>Person Found/Resides]
        SR2[Statutory Requirement 2<br/>Foreign Proceeding]
        SR3[Statutory Requirement 3<br/>Interested Person]

        IF1[Intel Factor 1<br/>Participant Status]
        IF2[Intel Factor 2<br/>Receptivity]
        IF3[Intel Factor 3<br/>Circumvention]
        IF4[Intel Factor 4<br/>Undue Burden]

        CE[Case Enforcement Plugins<br/>Intel, O'Keeffe, Brandi-Dohrn, etc.]

        FR45[FRCP Rule 45 Plugin]
        FR26[FRCP Rule 26 Plugin]
    end

    subgraph "Paragraph Level Plugins"
        PS[Paragraph Structure Plugin]
        PP[Per-Paragraph Plugin]
        PM[Paragraph Monitor Plugin]
    end

    subgraph "Sentence Level Plugins"
        SS2[Sentence Structure Plugin]
        SC[Sentence Count Plugin]
        SL[Sentence Length Plugin]
    end

    subgraph "Word Level Plugins"
        WC2[Word Count Plugin]
        WCH[Word Choice Plugin]
        WF[Word Frequency Plugin]
    end

    subgraph "Character Level Plugins"
        CC[Character Count Plugin]
        FM[Formatting Plugin]
        CF[Citation Format Plugin]
    end

    subgraph "CatBoost Features"
        ED[Enumeration Density Plugin<br/>SHAP: 0.312]
        R45M[Rule 45 Mentions Plugin<br/>SHAP: 0.144]
        CC2[Custodian Count Plugin<br/>SHAP: -0.309]
        SB[Scope Breadth Plugin]
    end

    subgraph "Content Features"
        PR[Privacy Plugin]
        HA[Harassment Plugin]
        SA[Safety Plugin]
        RE[Retaliation Plugin]
        PI[Public Interest Plugin]
        TR[Transparency Plugin]
        PHC[Privacy Harm Count Plugin]
    end

    subgraph "Specialized Plugins"
        NS[National Security Plugins]
        HG[Foreign Government Plugin]
        HL[Harvard Lawsuit Plugin]
        BA[Balancing Test Plugins]
        TM[Timing Argument Plugin]
    end

    WO -->|VALIDATE/REFINE phases| RL
    WO -->|Quality Gates| QGP

    RL -->|Coordinates| SS
    RL -->|Coordinates| CR
    RL -->|Coordinates| SR1
    RL -->|Coordinates| SR2
    RL -->|Coordinates| SR3
    RL -->|Coordinates| IF1
    RL -->|Coordinates| IF2
    RL -->|Coordinates| IF3
    RL -->|Coordinates| IF4
    RL -->|Coordinates| CE
    RL -->|Prioritizes by SHAP| ED
    RL -->|Prioritizes by SHAP| R45M
    RL -->|Prioritizes by SHAP| CC2

    QGP -->|Validates| SS
    QGP -->|Validates| CR
    QGP -->|Validates| FR45
    QGP -->|Validates| FR26

    SS -->|Requires| SR1
    SR1 -->|Requires| IF1
    IF1 -->|Requires| PS
    PS -->|Requires| SS2
    SS2 -->|Requires| WC2
    WC2 -->|Requires| CC

    style WO fill:#e1f5ff
    style RL fill:#fff4e1
    style QGP fill:#ffe1f5
    style ED fill:#e1ffe1
    style R45M fill:#e1ffe1
    style CC2 fill:#ffe1e1
```

---

## 🔄 Integration Flow

```mermaid
sequenceDiagram
    participant WO as WorkflowOrchestrator
    participant RL as RefinementLoop
    participant CB as CatBoost Model
    participant SK as SK Plugins
    participant QGP as QualityGatePipeline

    WO->>RL: analyze_draft(draft)
    RL->>CB: extract_features(draft)
    CB-->>RL: feature_scores, shap_values
    RL->>RL: identify_weak_features()
    RL->>SK: strengthen_features(weak_features)
    SK-->>RL: edit_requests
    RL->>RL: apply_improvements()
    RL-->>WO: improved_draft, analysis

    WO->>QGP: run_quality_gates(draft)
    QGP->>SK: validate_citations()
    QGP->>SK: validate_structure()
    QGP->>SK: validate_evidence()
    SK-->>QGP: validation_results
    QGP-->>WO: gate_results
```

---

## 📈 Plugin Hierarchy

```mermaid
graph TD
    A[Document Level] --> B[Section Level]
    B --> C[Paragraph Level]
    C --> D[Sentence Level]
    D --> E[Word Level]
    E --> F[Character Level]

    A --> A1[Section Structure]
    A --> A2[Citation Retrieval]
    A --> A3[Word Count]

    B --> B1[Statutory Requirements]
    B --> B2[Intel Factors]
    B --> B3[Case Enforcement]
    B --> B4[Federal Rules]

    C --> C1[Paragraph Structure]
    C --> C2[Per-Paragraph]
    C --> C3[Paragraph Monitor]

    D --> D1[Sentence Structure]
    D --> D2[Sentence Count]
    D --> D3[Sentence Length]

    E --> E1[Word Count]
    E --> E2[Word Choice]
    E --> E3[Word Frequency]

    F --> F1[Character Count]
    F --> F2[Formatting]
    F --> F3[Citation Format]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#ffe1e1
    style F fill:#f5e1ff
```

---

## 🎯 Key Relationships

### Dependency Chain
1. **Document Level** → Foundation for all other levels
2. **Section Level** → Depends on document structure
3. **Paragraph Level** → Depends on section structure
4. **Sentence Level** → Depends on paragraph structure
5. **Word Level** → Depends on sentence structure
6. **Character Level** → Depends on word structure

### Integration Points
- **RefinementLoop** coordinates all plugins during VALIDATE/REFINE phases
- **QualityGatePipeline** validates outputs using selected plugins
- **CatBoost** provides SHAP importance for prioritization
- **WorkflowOrchestrator** manages overall workflow and calls subsystems

---

**Last Updated:** 2025-11-06
**Status:** ✅ Complete
