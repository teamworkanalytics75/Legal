# Legal Checklists Plugin Network - Complete Scaffold

**Created:** 2025-01-11
**Purpose:** Comprehensive visual scaffold of all legal checklists, gates, and enforcement plugins from document level down to character level.

---

## Quick Reference

### Core Documents
- [Plugin Registry](config/plugin_registry.json) - JSON mapping of all checklists to plugins
- [Plugin Network Diagram](PLUGIN_NETWORK_DIAGRAM.md) - Mermaid diagram of plugin hierarchy
- [Interactive Visualization](plugin_network.html) - HTML interactive tree view

### Key Checklists
- [Section 1782 Statutory Requirements](#section-1782-statutory-requirements-3-elements)
- [Intel Factors](#intel-factors-4-factors)
- [Section 1782 Case Law](#section-1782-case-law-checklists)
- [Master Procedural Checklist](rules_registry/output/MASTERCHECKLISTEXPANDED.md) - 471 rules

---

## Document Level Gates

### 1. Document Structure Requirements
**Plugin:** `section_structure_plugin.py` (exists)
**Rules:** `rules/section_structure.json`

**Checklist:**
- [ ] All required sections present
- [ ] Section order correct
- [ ] Table of contents (if required)
- [ ] Caption/title page
- [ ] Signature block

### 2. Document-Level Citations
**Plugin:** `citation_retrieval_plugin.py` (exists)
**Rules:** `rules/citation_requirements.json`

**Checklist:**
- [ ] All required cases cited
- [ ] Citation format correct
- [ ] Pin cites included
- [ ] Parallel citations included

### 3. Document-Level Word Count
**Plugin:** `word_count_plugin.py` (needs creation)
**Rules:** `rules/word_count_rules.json` (needs creation)

**Checklist:**
- [ ] Total word count within limits
- [ ] Minimum word count met
- [ ] Maximum word count not exceeded

---

## Section Level Gates

### 1. Section 1782 Statutory Requirements (3 Elements)

#### Element 1: Person Found/Resides in District
**Plugin:** `statutory_requirement_1_plugin.py` (needs creation)
**Rules:** `rules/statutory_requirement_1_rules.json` (needs creation)
**Priority:** CRITICAL (Required)

**Checklist:**
- [ ] Discovery target identified
- [ ] District where target is found/resides specified
- [ ] Evidence of residency/location provided
- [ ] Jurisdictional basis established

**Legal Standard:**
> "(1) the person from whom discovery is sought resides (or is found) in the district of the district court to which the application is made" - Brandi-Dohrn v. IKB Deutsche Industriebank AG, 673 F.3d 76, 80 (2d Cir. 2012)

#### Element 2: Foreign Proceeding Exists
**Plugin:** `statutory_requirement_2_plugin.py` (needs creation)
**Rules:** `rules/statutory_requirement_2_rules.json` (needs creation)
**Priority:** CRITICAL (Required)

**Checklist:**
- [ ] Foreign proceeding identified
- [ ] Foreign tribunal specified
- [ ] Proceeding status (pending or reasonably contemplated)
- [ ] Evidence of proceeding provided (filing documents, court orders)

**Legal Standard:**
> "(2) the discovery is for use in a proceeding before a foreign tribunal" - Brandi-Dohrn, 673 F.3d at 80

**Key Cases:**
- Intel Corp. v. Advanced Micro Devices, Inc., 542 U.S. 241 (2004) - "within reasonable contemplation"
- Mees v. Buiter, 793 F.3d 291 (2d Cir. 2015) - "reasonably contemplated"

#### Element 3: Interested Person/Applicant
**Plugin:** `statutory_requirement_3_plugin.py` (needs creation)
**Rules:** `rules/statutory_requirement_3_rules.json` (needs creation)
**Priority:** CRITICAL (Required)

**Checklist:**
- [ ] Applicant identified
- [ ] Applicant's interest in foreign proceeding established
- [ ] "Reasonable interest" demonstrated
- [ ] Applicant is party to foreign proceeding OR has right to submit information

**Legal Standard:**
> "(3) the application is made by a foreign or international tribunal or 'any interested person'" - Brandi-Dohrn, 673 F.3d at 80

**Key Cases:**
- Intel, 542 U.S. at 256 - "interested person" includes complainants with "reasonable interest"

---

### 2. Intel Factors (4 Factors)

#### Intel Factor 1: Participant Status
**Plugin:** `intel_factor_1_participant_plugin.py` (needs creation)
**Rules:** `rules/intel_factor_1_rules.json` (needs creation)
**Priority:** HIGH (Discretionary but critical)

**Checklist:**
- [ ] Discovery target identified as non-participant
- [ ] Evidence that target is NOT a party to foreign proceeding
- [ ] If target IS a party, justification for Section 1782 aid
- [ ] Separate legal entities distinguished (if applicable)

**Legal Standard:**
> "Whether 'the person from whom discovery is sought is a participant in the foreign proceeding,' in which case 'the need for § 1782(a) aid generally is not as apparent as it ordinarily is when evidence is sought from a nonparticipant'" - Intel, 542 U.S. at 264

**Key Cases:**
- Gorsoan Ltd. v. Bullock, 652 F. App'x 44 (2d Cir. 2016) - Related entity being party doesn't bar discovery from non-party
- Pishevar v. Fusion GPS, 2017 WL 1184305 (D.D.C. 2017) - Non-participant status weighs in favor

**Enforcement Points:**
- Present tense status ("IS a participant") not future intent
- Separate entities are separate
- Failed attempts to add as party don't change status

#### Intel Factor 2: Receptivity of Foreign Tribunal
**Plugin:** `intel_factor_2_receptivity_plugin.py` (needs creation)
**Rules:** `rules/intel_factor_2_rules.json` (needs creation)
**Priority:** HIGH (Discretionary but critical)

**Checklist:**
- [ ] Foreign tribunal identified
- [ ] Evidence of receptivity (Hague Evidence Convention, prior cases, etc.)
- [ ] No authoritative proof of non-receptivity
- [ ] Hong Kong cases: Cite Hague Evidence Convention signatory status

**Legal Standard:**
> "The nature of the foreign tribunal, the character of the proceedings underway abroad, and the receptivity of the foreign government or the court or agency abroad to U.S. federal-court judicial assistance" - Intel, 542 U.S. at 264

**Key Cases:**
- O'Keeffe v. Adelson, 646 F. App'x 263 (3d Cir. 2016) - Hong Kong is "receptive to U.S. judicial assistance" (Hague Evidence Convention)
- Pishevar v. Fusion GPS - UK courts receptive to U.S. assistance

**Enforcement Points:**
- Presumption of receptivity (courts require "authoritative proof" of non-receptivity)
- Hague Evidence Convention signatory = evidence of receptivity
- Hong Kong = explicitly found receptive in O'Keeffe

#### Intel Factor 3: Circumvention Concerns
**Plugin:** `intel_factor_3_circumvention_plugin.py` (needs creation)
**Rules:** `rules/intel_factor_3_rules.json` (needs creation)
**Priority:** HIGH (Discretionary but critical)

**Checklist:**
- [ ] No evidence of bad faith
- [ ] Not attempting to avoid foreign court orders
- [ ] Not using Section 1782 to evade foreign procedural safeguards
- [ ] Valid foreign proceeding exists
- [ ] Discovery target is separate entity (not sham proceeding)

**Legal Standard:**
> "Whether the § 1782(a) request conceals an attempt to circumvent foreign proof-gathering restrictions or other policies of a foreign country or the United States" - Intel, 542 U.S. at 264

**Key Cases:**
- Brandi-Dohrn v. IKB Deutsche Industriebank AG, 673 F.3d 76, 81-83 (2d Cir. 2012) - "Mere fact that discovery might not be obtainable under foreign law does not, by itself, suggest circumvention"
- O'Keeffe v. Adelson - No requirement to seek letter rogatory first

**Enforcement Points:**
- Circumvention requires clear evidence of bad faith
- Valid proceeding exists = no circumvention
- Separate entities = no circumvention
- No requirement to exhaust foreign procedural options

#### Intel Factor 4: Undue Burden/Intrusiveness
**Plugin:** `intel_factor_4_burden_plugin.py` (needs creation)
**Rules:** `rules/intel_factor_4_rules.json` (needs creation)
**Priority:** MEDIUM (Discretionary)

**Checklist:**
- [ ] Request is narrowly tailored
- [ ] Not overly broad
- [ ] Proportional to foreign proceeding
- [ ] Specific custodian(s) identified
- [ ] Time period limited
- [ ] Document categories specific

**Legal Standard:**
> "Whether the request is 'unduly intrusive or burdensome'" - Intel, 542 U.S. at 264-65

**Key Cases:**
- Pishevar v. Fusion GPS - Court limited overbroad requests but granted narrow ones
- O'Keeffe v. Adelson - Narrow requests approved

**Enforcement Points:**
- Narrow tailoring = positive factor
- Specific custodians = positive factor
- Limited time periods = positive factor
- CatBoost signals: enumeration_density = positive, custodian_count = negative if too high

---

### 3. Section 1782 Case Law Checklists

#### Intel Corp. v. Advanced Micro Devices, Inc. (542 U.S. 241)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** CRITICAL

**Checklist:**
- [ ] Case cited with full citation
- [ ] Intel factors framework explained
- [ ] "Interested person" definition referenced
- [ ] "Foreign proceeding" definition referenced
- [ ] Non-participant status emphasized

#### O'Keeffe v. Adelson (Hong Kong Defamation)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** HIGH (Direct precedent)

**Checklist:**
- [ ] Case cited (multiple circuits available)
- [ ] Hong Kong receptivity holding cited
- [ ] Hague Evidence Convention mentioned
- [ ] Deposition request precedent cited
- [ ] "No letter rogatory required" holding cited

#### Brandi-Dohrn v. IKB Deutsche Industriebank AG (673 F.3d 76)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** HIGH

**Checklist:**
- [ ] Case cited
- [ ] "No requirement for foreign discoverability" holding cited
- [ ] Circumvention standard cited
- [ ] Statutory requirements framework cited

#### Gorsoan Ltd. v. Bullock (652 F. App'x 44)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** HIGH (Addresses "add as party" argument)

**Checklist:**
- [ ] Case cited (if Harvard raises "add as party" argument)
- [ ] "Related entity being party doesn't bar discovery from non-party" holding
- [ ] "Participation doesn't automatically foreclose Section 1782 aid" holding

#### Pishevar v. Fusion GPS (2017 WL 1184305)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** HIGH (Similar fact pattern)

**Checklist:**
- [ ] Case cited (if applicable)
- [ ] Non-participant status holding
- [ ] Deposition request granted precedent
- [ ] Narrow tailoring approved precedent

#### ZF Automotive US, Inc. v. Luxshare, Ltd. (596 U.S. 450)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** MEDIUM

**Checklist:**
- [ ] Case cited (if arbitration tribunal issue)
- [ ] "Foreign tribunal" definition clarified

#### In re del Valle Ruiz (939 F.3d 520)
**Plugin:** Generated via `case_enforcement_plugin_generator.py`
**Rules:** Auto-generated from `master_case_citations.json`
**Priority:** MEDIUM

**Checklist:**
- [ ] Case cited (if "resides or is found" issue)
- [ ] Extraterritorial reach limits

---

### 4. Federal Rules Checklists

#### FRCP Rule 45 (Subpoenas)
**Priority:** HIGH (Section 1782 uses Rule 45)
**Checklist:**
- [ ] Subpoena format correct
- [ ] Service requirements met
- [ ] Geographic limits respected
- [ ] Notice to parties (if required)

#### FRCP Rule 26 (Discovery Scope)
**Priority:** MEDIUM
**Checklist:**
- [ ] Relevance requirement met
- [ ] Proportionality requirement met
- [ ] Privilege issues addressed

#### FRAP Requirements
**Priority:** LOW (If appealing)
**Checklist:**
- [ ] Appeal deadlines met
- [ ] Notice of appeal filed
- [ ] Record on appeal complete

---

## Paragraph Level Gates

### 1. Paragraph Structure
**Plugin:** `paragraph_structure_plugin.py` (exists)
**Rules:** `rules/paragraph_structure_rules.json` (exists)

**Checklist:**
- [ ] Paragraph word count within limits
- [ ] Paragraph sentence count appropriate
- [ ] Paragraph coherence
- [ ] Topic sentence present

### 2. Paragraph Content
**Plugin:** `per_paragraph_plugin.py` (exists)
**Rules:** `rules/per_paragraph_rules.json` (exists)

**Checklist:**
- [ ] Each paragraph has clear purpose
- [ ] Paragraphs support section thesis
- [ ] Transition between paragraphs

### 3. Paragraph Monitoring
**Plugin:** `paragraph_monitor_plugin.py` (exists)
**Rules:** `rules/paragraph_monitor_rules.json` (exists)

**Checklist:**
- [ ] Paragraph count per section appropriate
- [ ] Paragraph length variation
- [ ] Paragraph structure consistency

---

## Sentence Level Gates

### 1. Sentence Structure
**Plugin:** `sentence_structure_plugin.py` (exists)
**Rules:** `rules/sentence_structure_rules.json` (exists)

**Checklist:**
- [ ] Sentence word count within limits
- [ ] Sentence complexity appropriate
- [ ] Sentence variety (not all simple or all complex)
- [ ] Run-on sentences avoided

### 2. Sentence Count
**Plugin:** `sentence_count_plugin.py` (needs creation)
**Rules:** `rules/sentence_count_rules.json` (needs creation)

**Checklist:**
- [ ] Sentences per paragraph: 3-8 (typical)
- [ ] Maximum sentence count per paragraph
- [ ] Minimum sentence count per paragraph

### 3. Sentence Length
**Plugin:** `sentence_length_plugin.py` (needs creation)
**Rules:** `rules/sentence_length_rules.json` (needs creation)

**Checklist:**
- [ ] Average sentence length: 15-25 words (legal writing)
- [ ] Maximum sentence length: 40 words
- [ ] Minimum sentence length: 5 words
- [ ] Long sentence ratio < 20%

---

## Word Level Gates

### 1. Word Count
**Plugin:** `word_count_plugin.py` (needs creation)
**Rules:** `rules/word_count_rules.json` (needs creation)

**Checklist:**
- [ ] Document word count within limits
- [ ] Section word count appropriate
- [ ] Paragraph word count appropriate
- [ ] Sentence word count appropriate

### 2. Word Choice
**Plugin:** `word_choice_plugin.py` (needs creation)
**Rules:** `rules/word_choice_rules.json` (needs creation)

**Checklist:**
- [ ] Legal terminology used correctly
- [ ] Avoid informal language
- [ ] Avoid contractions
- [ ] Technical terms defined

### 3. Word Frequency
**Plugin:** `word_frequency_plugin.py` (needs creation)
**Rules:** `rules/word_frequency_rules.json` (needs creation)

**Checklist:**
- [ ] Key terms used consistently
- [ ] Avoid excessive repetition
- [ ] Keyword density appropriate
- [ ] Legal terms frequency tracked

### 4. CatBoost Feature Keywords
**Plugin:** Various (enumeration_density exists, others need creation)

**Checklist:**
- [ ] Enumeration density: High (positive signal)
- [ ] Custodian count: Low (1-3 ideal, avoid 20+)
- [ ] Rule 45 mentions: Present but not excessive
- [ ] Citation count: Appropriate for section
- [ ] Privacy/harassment/safety/retaliation mentions (as applicable)

---

## Character Level Gates

### 1. Character Count
**Plugin:** `character_count_plugin.py` (needs creation)
**Rules:** `rules/character_count_rules.json` (needs creation)

**Checklist:**
- [ ] Document character count within limits
- [ ] Section character count appropriate
- [ ] Paragraph character count appropriate
- [ ] Line length considerations

### 2. Formatting
**Plugin:** `formatting_plugin.py` (needs creation)
**Rules:** `rules/formatting_rules.json` (needs creation)

**Checklist:**
- [ ] Font consistent
- [ ] Font size appropriate
- [ ] Margins correct
- [ ] Line spacing correct
- [ ] Headers/footers formatted

### 3. Citation Format (Character Level)
**Plugin:** `citation_format_plugin.py` (needs creation)
**Rules:** `rules/citation_format_rules.json` (needs creation)

**Checklist:**
- [ ] Citation format: Bluebook compliant
- [ ] Case names italicized
- [ ] Pin cites correct format
- [ ] Parallel citations correct
- [ ] Period placement correct
- [ ] Comma placement correct

---

## CatBoost Feature Checklists

### High-Impact Features (Positive Signals)

#### 1. Enumeration Density
**Plugin:** `enumeration_density_plugin.py` (exists)
**Rules:** `rules/enumeration_density_rules.json` (exists)
**SHAP Value:** 0.312 (positive)

**Checklist:**
- [ ] Use numbered lists (1., 2., 3.)
- [ ] Use bullet points
- [ ] Use lettered lists (a., b., c.)
- [ ] Use headers/subheaders
- [ ] Break up dense text with enumeration
- [ ] Target: High enumeration density (34+ = very high)

#### 2. Rule 45 Mentions
**Plugin:** `rule_45_mentions_plugin.py` (needs creation)
**Rules:** `rules/rule_45_mentions_rules.json` (needs creation)
**SHAP Value:** 0.144 (positive)

**Checklist:**
- [ ] Rule 45 cited appropriately
- [ ] "Undue burden" language used
- [ ] "Quash" or "quashing" mentioned
- [ ] Burden minimization arguments
- [ ] Target: 10-20 mentions (not excessive)

### High-Impact Features (Negative Signals - Minimize)

#### 3. Custodian List Count
**Plugin:** `custodian_count_plugin.py` (needs creation)
**Rules:** `rules/custodian_count_rules.json` (needs creation)
**SHAP Value:** -0.309 (negative)

**Checklist:**
- [ ] Minimize number of custodians
- [ ] Target: 1-3 custodians (narrow request)
- [ ] Avoid: 20+ custodians (overly broad)
- [ ] Justify each custodian individually

#### 4. Request Scope Breadth
**Plugin:** `scope_breadth_plugin.py` (needs creation)
**Rules:** `rules/scope_breadth_rules.json` (needs creation)

**Checklist:**
- [ ] Narrow time period
- [ ] Specific document categories
- [ ] Specific search terms
- [ ] Avoid: "All documents" language
- [ ] Avoid: Unlimited time periods

---

## Master Procedural Checklist Integration

**Source:** `rules_registry/output/MASTERCHECKLISTEXPANDED.md`
**Total Rules:** 471
**Rule Categories:** 24

### Case Law Categories
- Attorney-Client Privilege (Upjohn)
- Discovery - Privilege (Hickman)
- E-Discovery (Zubulake)
- Expert Testimony (Daubert)
- Pleading Standards (Iqbal, Twombly)
- Section 1782 (Intel, ZF Automotive)
- Summary Judgment (Anderson, Celotex)

### Federal Rules Categories
- FRCP (Rules 1-87+)
- FRE (Rules 101-1103)
- FRAP (Rules 1-48+)

### Statutory Categories
- 28 U.S.C. § 1782 (Section 1782)
- 28 U.S.C. § 1331 (Federal question jurisdiction)
- 42 U.S.C. § 1983 (Civil rights)
- Spoliation statutes (18 U.S.C. § 1519)

### Sanctions Categories
- Rule 11 Sanctions
- Discovery Sanctions
- Disclosure Sanctions
- ESI Preservation

**Note:** Each item in the master checklist should have a corresponding plugin or be grouped into a category plugin.

---

## Plugin Status Summary

### Existing Plugins (✅)
- `enumeration_density_plugin.py`
- `paragraph_structure_plugin.py`
- `sentence_structure_plugin.py`
- `citation_retrieval_plugin.py`
- `required_case_citation_plugin.py`
- `privacy_harm_count_plugin.py`
- `public_interest_plugin.py`
- `transparency_argument_plugin.py`
- `mentions_privacy_plugin.py`
- `mentions_harassment_plugin.py`
- `mentions_safety_plugin.py`
- `mentions_retaliation_plugin.py`
- `per_paragraph_plugin.py`
- `paragraph_monitor_plugin.py`
- `case_enforcement_plugin_generator.py` (generates case-specific plugins)

### Plugins Needing Creation (❌)
- `statutory_requirement_1_plugin.py` (Person found/resides)
- `statutory_requirement_2_plugin.py` (Foreign proceeding)
- `statutory_requirement_3_plugin.py` (Interested person)
- `intel_factor_1_participant_plugin.py`
- `intel_factor_2_receptivity_plugin.py`
- `intel_factor_3_circumvention_plugin.py`
- `intel_factor_4_burden_plugin.py`
- `sentence_count_plugin.py`
- `sentence_length_plugin.py`
- `word_count_plugin.py`
- `word_choice_plugin.py`
- `word_frequency_plugin.py`
- `character_count_plugin.py`
- `formatting_plugin.py`
- `citation_format_plugin.py`
- `rule_45_mentions_plugin.py`
- `custodian_count_plugin.py`
- `scope_breadth_plugin.py`

---

## Plugin Dependency Graph

```
Document Level
├── Section Structure Plugin
├── Citation Retrieval Plugin
└── Word Count Plugin
    │
    ├── Section Level
    │   ├── Statutory Requirement Plugins (3)
    │   ├── Intel Factor Plugins (4)
    │   ├── Case Enforcement Plugins (14+)
    │   └── FRCP Rule Plugins (as needed)
    │       │
    │       ├── Paragraph Level
    │       │   ├── Paragraph Structure Plugin
    │       │   ├── Per Paragraph Plugin
    │       │   └── Paragraph Monitor Plugin
    │       │       │
    │       │       ├── Sentence Level
    │       │       │   ├── Sentence Structure Plugin
    │       │       │   ├── Sentence Count Plugin
    │       │       │   └── Sentence Length Plugin
    │       │       │       │
    │       │       │       ├── Word Level
    │       │       │       │   ├── Word Count Plugin
    │       │       │       │   ├── Word Choice Plugin
    │       │       │       │   └── Word Frequency Plugin
    │       │       │       │       │
    │       │       │       │       └── Character Level
    │       │       │       │           ├── Character Count Plugin
    │       │       │       │           ├── Formatting Plugin
    │       │       │       │           └── Citation Format Plugin
```

---

## Integration Points

### RefinementLoop Integration
- File: `writer_agents/code/sk_plugins/FeaturePlugin/feature_orchestrator.py`
- All plugins integrated into enhancement pipeline
- Priority/ordering system
- Dependency resolver

### QualityGatePipeline Integration
- File: `writer_agents/code/WorkflowOrchestrator.py`
- All gates added to quality gate system
- Thresholds from CatBoost analysis
- Required vs optional gates

---

## Next Steps

1. ✅ Identify all checklists (THIS DOCUMENT)
2. ✅ Create missing plugins (Phase 2) - 61 plugins now exist
3. ✅ Create plugin registry JSON (Phase 4) - Updated 2025-11-06
4. ✅ Create Mermaid diagram (Phase 4) - [PLUGIN_NETWORK_DIAGRAM.md](PLUGIN_NETWORK_DIAGRAM.md)
5. ✅ Create HTML visualization (Phase 4) - [plugin_network.html](plugin_network.html)
6. ✅ Update master checklist (Phase 1) - Updated 2025-11-06
7. ✅ Integrate with RefinementLoop (Phase 5) - Fully integrated in WorkflowOrchestrator
8. ✅ Integrate with QualityGatePipeline (Phase 5) - Fully integrated in WorkflowOrchestrator

**Status:** ✅ All tasks completed as of 2025-11-06

