# 🔬 Plugin Categorization Architecture - Complete Design

**Date**: 2025-01-XX
**Design Philosophy**: Each SK plugin is its own enforcer, categorized by enforcement type

---

## 🎯 Design Principles

### **1. Each Plugin is an Enforcer**
- **Every SK plugin** enforces something specific
- **Single duty** - each plugin does ONE thing
- **Categorized by enforcement type**:
  - Formatting enforcement
  - Legal enforcement (court rules, evidence rules)
  - Citation enforcement
  - Rule enforcement (FRCP, FRE, etc.)

### **2. Master Exclusion Plugin**
- **ONE dedicated plugin** tracks all irrelevant rules
- **Acts as memory** of what's not important and why
- Documents why each irrelevant rule doesn't apply
- Ensures system knows there's no hidden relevant rules

### **3. Categorization Structure**
- **Formatting Enforcement**: Margins, word count, page limits, font, spacing
- **Legal Enforcement**: Court rules, evidence rules, FRCP, FRE, FRAP
- **Citation Enforcement**: Bluebook format, citation requirements
- **Rule Enforcement**: Specific rule compliance (Rule 5.2, etc.)

---

## 📊 Plugin Categories

### **Category 1: Formatting Enforcement Plugins**

**Purpose**: Each plugin enforces a specific formatting requirement

**Philosophy**: Each SK plugin is its own enforcer - one plugin per formatting rule

**Examples**:
- `margin_requirements_enforcement_plugin.py` - Enforce margin rules (1 inch)
- `word_count_limit_enforcement_plugin.py` - Enforce word count maximums
- `page_limit_enforcement_plugin.py` - Enforce page maximums
- `font_requirements_enforcement_plugin.py` - Enforce font rules (Times New Roman, 12pt)
- `line_spacing_enforcement_plugin.py` - Enforce line spacing (double spacing)
- `paragraph_spacing_enforcement_plugin.py` - Enforce paragraph spacing
- `header_footer_enforcement_plugin.py` - Enforce header/footer requirements
- `page_numbering_enforcement_plugin.py` - Enforce page numbering
- `caption_requirements_enforcement_plugin.py` - Enforce caption format
- `signature_block_enforcement_plugin.py` - Enforce signature block format

**Count**: ~20-30 formatting enforcement plugins (one per formatting rule)

---

### **Category 2: Legal Enforcement Plugins**

**Purpose**: Each plugin enforces a specific legal rule (court rules, evidence rules, FRCP, FRE, FRAP)

**Philosophy**: Each SK plugin is its own enforcer - one plugin per legal rule

**Note**: Evidence rules (FRE) are included in this category as part of court rules/legal enforcement

**Subcategories**:

#### **2a. Federal Rules of Civil Procedure (FRCP) - Court Rules**
- `rule_5_2_privacy_protection_enforcement_plugin.py` - Rule 5.2 (Privacy Protection)
- `rule_7_pleadings_enforcement_plugin.py` - Rule 7 (Pleadings)
- `rule_11_signatures_enforcement_plugin.py` - Rule 11 (Signatures)
- `rule_26_discovery_enforcement_plugin.py` - Rule 26 (Discovery)
- ... (other FRCP rules)

#### **2b. Federal Rules of Evidence (FRE) - Evidence Rules (Included in Court Rules)**
- `rule_401_relevance_enforcement_plugin.py` - Rule 401 (Relevance)
- `rule_402_admissibility_enforcement_plugin.py` - Rule 402 (Admissibility)
- `rule_403_exclusion_enforcement_plugin.py` - Rule 403 (Exclusion)
- ... (other FRE rules - evidence rules are part of court rules)

#### **2c. Federal Rules of Appellate Procedure (FRAP) - Court Rules**
- (if applicable to district court motions)

#### **2d. Case Law Rules - Legal Rules**
- `upjohn_attorney_client_privilege_enforcement_plugin.py` - Upjohn v. United States
- `intel_factor_1_enforcement_plugin.py` - Intel Factor 1
- `section_1782_requirement_1_enforcement_plugin.py` - Section 1782 Requirement 1
- ... (other case law rules)

**Count**: ~50 legal enforcement plugins (from 50 relevant rules)

---

### **Category 3: Citation Enforcement Plugins**

**Purpose**: Each plugin enforces a specific citation format requirement

**Philosophy**: Each SK plugin is its own enforcer - one plugin per citation rule

**Examples**:
- `bluebook_citation_format_enforcement_plugin.py` - Enforce Bluebook format
- `case_citation_requirements_enforcement_plugin.py` - Enforce case citation format
- `statute_citation_requirements_enforcement_plugin.py` - Enforce statute citation format
- `pin_citation_requirements_enforcement_plugin.py` - Enforce pin citation requirements
- `string_citation_requirements_enforcement_plugin.py` - Enforce string citation requirements
- `parallel_citation_requirements_enforcement_plugin.py` - Enforce parallel citations

**Count**: ~10-15 citation enforcement plugins (one per citation rule)

---

### **Category 4: Master Exclusion Plugin** (1 plugin)

**Purpose**: Understand which rules of the entire set are NOT applicable and why

**Name**: `irrelevant_rules_exclusion_plugin.py`

**Philosophy**:
- **ONE dedicated plugin** to understand which rules are not applicable and why
- Acts as **memory** of what's not important and why
- Ensures the whole system knows there's no database of relevant rules it doesn't know about

**Duty**:
- Acts as **memory** of what's not important and why
- Tracks all 421 irrelevant rules from the entire 471-rule set
- Documents exclusion reasons for each irrelevant rule
- Ensures system knows there's no hidden relevant rules
- Provides clean data for validation
- **Understands which rules don't apply and why** (like a memory)

**Key Features**:
- Loads all 421 irrelevant rules from RulesExpanded.json (out of 471 total)
- Maintains exclusion reasons for each rule
- Verifies no relevant rules are missing
- Provides system-wide exclusion tracking
- Acts as **memory/knowledge base** of excluded rules
- **Ensures system knows complete rule set** - no hidden relevant rules

**Count**: 1 plugin (master exclusion tracker)

---

## 📋 Complete Plugin Inventory

### **Formatting Enforcement Plugins** (~20-30 plugins)
- Margin requirements
- Word count limits
- Page maximums
- Font requirements
- Line spacing
- Paragraph spacing
- Header/footer
- Page numbering
- Caption format
- Signature block
- ... (other formatting rules)

### **Legal Enforcement Plugins** (~50 plugins)

**From 50 relevant rules**:
- **FRCP Rules**: Rule 5.2, Rule 7, Rule 11, Rule 26, etc.
- **FRE Rules**: Rule 401, Rule 402, Rule 403, etc.
- **Case Law**: Upjohn, Intel factors, Section 1782, etc.
- **Court Rules**: Local rules, district court rules, etc.

### **Citation Enforcement Plugins** (~10-15 plugins)
- Bluebook format
- Case citations
- Statute citations
- Pin citations
- String citations
- Parallel citations
- ... (other citation rules)

### **Master Exclusion Plugin** (1 plugin)
- Tracks all 421 irrelevant rules
- Documents exclusion reasons
- Acts as memory of excluded rules

---

## 🔧 Plugin Categorization Logic

### **How to Categorize Rules**

```python
def categorize_rule(rule: Dict) -> str:
    """Categorize rule by enforcement type."""
    rule_type = rule.get('rule_type', '').lower()
    citation = rule.get('citation_user', '').lower()
    text_excerpt = rule.get('text_excerpt', '').lower()

    # Formatting rules
    if any(keyword in rule_type or keyword in text_excerpt for keyword in [
        'margin', 'word count', 'page limit', 'font', 'spacing',
        'line spacing', 'paragraph', 'header', 'footer', 'page number',
        'caption', 'signature', 'formatting', 'technical'
    ]):
        return 'formatting'

    # Citation rules
    elif any(keyword in rule_type or keyword in text_excerpt for keyword in [
        'citation', 'bluebook', 'pin cite', 'string cite', 'parallel cite',
        'case citation', 'statute citation'
    ]):
        return 'citation'

    # Legal rules (FRCP, FRE, FRAP, case law)
    elif any(keyword in rule_type or keyword in citation for keyword in [
        'federal rule', 'frcp', 'fre', 'frap', 'case law', 'court rule',
        'evidence', 'procedure', 'privilege', 'discovery', 'motion'
    ]):
        return 'legal'

    # Default to legal
    else:
        return 'legal'
```

---

## 📊 Final Plugin Count by Category

### **Total: ~81-95 Enforcement Plugins + 1 Master Exclusion**

**Breakdown**:
- **Formatting Enforcement**: ~20-30 plugins (each enforces one formatting rule)
- **Legal Enforcement**: ~50 plugins (court rules, evidence rules - each enforces one legal rule)
- **Citation Enforcement**: ~10-15 plugins (each enforces one citation rule)
- **Master Exclusion**: 1 plugin (tracks all 421 irrelevant rules - acts as memory)

**Plus Existing Plugins**:
- **Existing FeaturePlugins**: ~70 plugins (CatBoost features, etc.)
- **Case Enforcement**: 21 plugins (dynamic)

**Total Active**: ~170-185 plugins

### **Key Point**:
- **Each SK plugin is its own enforcer** - one plugin per rule
- **One master exclusion plugin** - understands which rules don't apply and why (acts as memory)

---

## 🎯 Benefits of Categorization

### **1. Clear Organization**
- ✅ Plugins organized by enforcement type
- ✅ Easy to find and maintain
- ✅ Clear responsibilities

### **2. Efficient Enforcement**
- ✅ Each plugin enforces ONE specific thing
- ✅ Categorized by type for easy management
- ✅ No overlap or confusion

### **3. Complete Coverage**
- ✅ Formatting: All formatting rules enforced
- ✅ Legal: All relevant legal rules enforced
- ✅ Citations: All citation rules enforced
- ✅ Exclusion: All irrelevant rules tracked

### **4. Master Exclusion Memory**
- ✅ ONE plugin tracks all irrelevant rules
- ✅ Acts as memory of what's not important
- ✅ Documents why each doesn't apply
- ✅ Ensures no hidden relevant rules

---

## ✅ Summary

**Architecture**:
- ✅ **Formatting Enforcement**: ~20-30 plugins (margins, word count, font, etc.)
- ✅ **Legal Enforcement**: ~50 plugins (FRCP, FRE, case law, court rules)
- ✅ **Citation Enforcement**: ~10-15 plugins (Bluebook, citations)
- ✅ **Master Exclusion**: 1 plugin (tracks all 421 irrelevant rules - acts as memory)

**Key Points**:
- ✅ Each SK plugin is its own enforcer
- ✅ Plugins categorized by enforcement type
- ✅ Master exclusion plugin acts as memory of excluded rules
- ✅ Complete coverage of all rules

**This is the optimal categorized architecture!** 🎯

