# 🔬 Atomic Plugin Architecture - Complete 471-Rule Coverage

**Date**: 2025-01-XX
**Vision**: Maximum modularity, perfect recall, single duty enforcement for ALL 471 rules

---

## 🎯 Design Philosophy

### **1. Atomic Modularity**
- **One Plugin = One Rule = One Duty**
- Each plugin enforces EXACTLY ONE rule
- No shared responsibilities
- Complete independence

### **2. Perfect Recall**
- **ALL 471 rules** get plugins
- **No exceptions** - formatting, legal, procedural, evidentiary
- **Complete coverage** - margins, word count, page limits, legal arguments
- **Every rule** matters

### **3. Exclusion Plugins (Inverse Logic)**
- **Irrelevant rules** get "exclusion plugins"
- **Document WHY** they don't apply (clean data)
- **Prevent false positives** - system knows what NOT to check
- **Redundancy for validation** - explicit exclusion documentation

### **4. Complete Rule Coverage**
- **Federal Rules of Civil Procedure** (FRCP)
- **Federal Rules of Evidence** (FRE)
- **Federal Rules of Appellate Procedure** (FRAP)
- **Case Law Rules**
- **Local Rules**
- **Formatting Rules** (margins, word count, page limits)
- **Technical Rules** (font, spacing, citation format)

---

## 📊 Complete Plugin Architecture

### **Total Plugins: 471** (one per rule)

**Breakdown**:
- **Enforcement Plugins**: ~50 (relevant rules - enforce compliance)
- **Exclusion Plugins**: ~421 (irrelevant rules - document exclusion)
- **Formatting Plugins**: ~20-30 (technical rules - enforce formatting)
- **Legal Plugins**: ~400+ (legal rules - enforce legal arguments)

---

## 🔬 Plugin Types

### **Type 1: Enforcement Plugins** (Relevant Rules)

**Purpose**: Enforce rule compliance

**Example**: `rule_5_2_privacy_protection_enforcement_plugin.py`
```python
class Rule5_2PrivacyProtectionEnforcementPlugin(BaseFeaturePlugin):
    """
    Single Duty: Enforce Federal Rule of Civil Procedure 5.2
    Privacy Protection For Filings Made with the Court

    Rule: Rule 5.2(a)(3) - Pseudonym usage
    Rule: Rule 5.2(a)(1) - Redaction requirements

    Duty:
    - Check pseudonym requirements
    - Check redaction requirements (SSN, DOB, account numbers)
    - Check sealing standards
    - Check balancing test (privacy vs. public interest)
    - Verify compliance with Rule 5.2
    """
    def __init__(self, kernel, chroma_store, rules_dir, **kwargs):
        super().__init__(kernel, "rule_5_2_privacy_protection_enforcement",
                        chroma_store, rules_dir, **kwargs)
        self.rule_id = "5.2"
        self.rule_citation = "Federal 5.2"
        self.rule_type = "Federal Rules Civil Procedure"
        self.duty = "Enforce Rule 5.2 compliance"
        self.is_enforcement = True

    async def enforce_rule(self, draft_text: str) -> FunctionResult:
        """
        Single duty: Enforce Rule 5.2 compliance.

        Returns:
            FunctionResult with compliance status, violations, recommendations
        """
        violations = []
        compliance_checks = []

        # Check pseudonym requirements
        if not self._check_pseudonym_usage(draft_text):
            violations.append("Pseudonym not used when required")

        # Check redaction requirements
        redaction_issues = self._check_redaction_requirements(draft_text)
        violations.extend(redaction_issues)

        # Check balancing test
        if not self._check_balancing_test(draft_text):
            violations.append("Balancing test (privacy vs. public interest) not addressed")

        return FunctionResult(
            success=len(violations) == 0,
            value={
                "rule_id": self.rule_id,
                "rule_citation": self.rule_citation,
                "compliant": len(violations) == 0,
                "violations": violations,
                "compliance_checks": compliance_checks,
                "duty": self.duty
            }
        )
```

### **Type 2: Exclusion Plugins** (Irrelevant Rules)

**Purpose**: Document why rule doesn't apply (inverse logic)

**Example**: `rule_56_summary_judgment_exclusion_plugin.py`
```python
class Rule56SummaryJudgmentExclusionPlugin(BaseFeaturePlugin):
    """
    Single Duty: Document exclusion of Federal Rule of Civil Procedure 56
    Summary Judgment

    Rule: Rule 56 - Summary Judgment

    Exclusion Reason: Motion type is seal/pseudonym, not summary judgment motion

    Duty:
    - Check if motion is NOT a summary judgment motion
    - Document exclusion reason (motion type mismatch)
    - Prevent false positive enforcement
    - Provide clean data for validation
    """
    def __init__(self, kernel, chroma_store, rules_dir, **kwargs):
        super().__init__(kernel, "rule_56_summary_judgment_exclusion",
                        chroma_store, rules_dir, **kwargs)
        self.rule_id = "56"
        self.rule_citation = "Federal 56"
        self.rule_type = "Federal Rules Civil Procedure"
        self.duty = "Document exclusion of Rule 56"
        self.is_enforcement = False
        self.is_exclusion = True
        self.exclusion_reason = "Motion type is seal/pseudonym, not summary judgment motion"

    async def check_exclusion(self, draft_text: str) -> FunctionResult:
        """
        Single duty: Document why Rule 56 does NOT apply.

        Returns:
            FunctionResult with exclusion status and reason
        """
        # Check motion type
        motion_type = self._detect_motion_type(draft_text)

        # Rule 56 applies to summary judgment motions
        # Seal/pseudonym motions are NOT summary judgment motions
        is_excluded = motion_type != "summary_judgment"

        return FunctionResult(
            success=True,
            value={
                "rule_id": self.rule_id,
                "rule_citation": self.rule_citation,
                "excluded": is_excluded,
                "exclusion_reason": self.exclusion_reason,
                "motion_type": motion_type,
                "verification": "Rule 56 (Summary Judgment) does not apply to seal/pseudonym motions",
                "duty": self.duty
            }
        )
```

### **Type 3: Formatting Enforcement Plugins** (Technical Rules)

**Purpose**: Enforce formatting requirements

**Examples**:
- `margin_requirements_enforcement_plugin.py` - Enforce margin rules (1 inch, etc.)
- `word_count_limit_enforcement_plugin.py` - Enforce word count maximums
- `page_limit_enforcement_plugin.py` - Enforce page maximums
- `font_requirements_enforcement_plugin.py` - Enforce font rules (Times New Roman, 12pt)
- `line_spacing_enforcement_plugin.py` - Enforce line spacing (double spacing)
- `citation_format_enforcement_plugin.py` - Enforce Bluebook citation format

**Example**: `margin_requirements_enforcement_plugin.py`
```python
class MarginRequirementsEnforcementPlugin(BaseFeaturePlugin):
    """
    Single Duty: Enforce margin requirements

    Rule: Local Rules - Margin requirements (typically 1 inch)

    Duty:
    - Check top margin (1 inch)
    - Check bottom margin (1 inch)
    - Check left margin (1 inch)
    - Check right margin (1 inch)
    - Verify compliance
    """
    async def enforce_margin_requirements(self, draft_text: str) -> FunctionResult:
        """
        Single duty: Enforce margin requirements.
        """
        # Check margins (if document structure available)
        # Return violations or compliance
        pass
```

### **Type 4: Legal Argument Plugins** (Legal Rules)

**Purpose**: Enforce legal argument requirements

**Examples**:
- `section_1782_requirement_1_enforcement_plugin.py` - Enforce statutory requirement 1
- `intel_factor_1_enforcement_plugin.py` - Enforce Intel Factor 1
- `balancing_test_requirement_enforcement_plugin.py` - Enforce balancing test
- `privilege_requirements_enforcement_plugin.py` - Enforce privilege rules
- `discovery_scope_enforcement_plugin.py` - Enforce discovery scope rules

---

## 📋 Complete Plugin Generation Plan

### **Phase 1: Generate All 471 Plugins**

**Approach**: Automated generation from RulesExpanded.json

**Plugin Types**:
1. **Enforcement Plugins** (50 relevant rules)
2. **Exclusion Plugins** (421 irrelevant rules)
3. **Formatting Plugins** (technical rules)
4. **Legal Plugins** (legal argument rules)

### **Generation Strategy**:

```python
class AtomicPluginGenerator:
    """
    Generate atomic plugins for all 471 rules.
    """

    @staticmethod
    def generate_all_plugins(rules_data: List[Dict],
                            relevance_report: Dict) -> Dict[str, BaseFeaturePlugin]:
        """
        Generate all 471 plugins (enforcement + exclusion).

        Args:
            rules_data: All 471 rules from RulesExpanded.json
            relevance_report: Rule relevance report (50 relevant, 421 irrelevant)

        Returns:
            Dictionary of all 471 plugins
        """
        plugins = {}
        relevant_rule_ids = {r['rule_id'] for r in relevance_report['relevant_rules']}

        for rule in rules_data:
            rule_id = rule.get('rule_id')
            is_relevant = rule_id in relevant_rule_ids

            if is_relevant:
                # Create enforcement plugin
                plugin = AtomicPluginGenerator.create_enforcement_plugin(rule)
            else:
                # Create exclusion plugin
                plugin = AtomicPluginGenerator.create_exclusion_plugin(rule)

            plugins[plugin.name] = plugin

        return plugins

    @staticmethod
    def create_enforcement_plugin(rule: Dict) -> BaseFeaturePlugin:
        """Create enforcement plugin for relevant rule."""
        # Generate plugin code
        # Single duty: Enforce this rule
        pass

    @staticmethod
    def create_exclusion_plugin(rule: Dict) -> BaseFeaturePlugin:
        """Create exclusion plugin for irrelevant rule."""
        # Generate plugin code
        # Single duty: Document exclusion
        pass
```

---

## 🔧 Plugin Structure Details

### **Enforcement Plugin Structure**

```python
class Rule{ID}_{Name}EnforcementPlugin(BaseFeaturePlugin):
    """
    Single Duty: Enforce {Rule Citation}

    Rule: {Rule Description}
    Source: {Source File}
    Rule Type: {Rule Type}
    Relevance: APPLIES (from relevance report)

    Duty:
    - {Specific duty 1}
    - {Specific duty 2}
    - {Specific duty 3}
    """

    def __init__(self, kernel, chroma_store, rules_dir, **kwargs):
        super().__init__(kernel, f"rule_{id}_{name}_enforcement",
                        chroma_store, rules_dir, **kwargs)
        self.rule_id = "{ID}"
        self.rule_citation = "{Citation}"
        self.rule_type = "{Rule Type}"
        self.duty = "Enforce {Rule Description}"
        self.is_enforcement = True
        self.is_exclusion = False
        self.relevance_status = "APPLIES"

    async def enforce_rule(self, draft_text: str) -> FunctionResult:
        """
        Single duty: Enforce this specific rule.

        Returns:
            FunctionResult with compliance status
        """
        # Implementation
        pass
```

### **Exclusion Plugin Structure**

```python
class Rule{ID}_{Name}ExclusionPlugin(BaseFeaturePlugin):
    """
    Single Duty: Document exclusion of {Rule Citation}

    Rule: {Rule Description}
    Exclusion Reason: {Why it doesn't apply}
    Source: {Source File}
    Rule Type: {Rule Type}
    Relevance: DOES NOT APPLY (from relevance report)

    Duty:
    - Check if rule applies (should return False)
    - Document exclusion reason
    - Provide clean data for validation
    - Prevent false positive enforcement
    """

    def __init__(self, kernel, chroma_store, rules_dir, **kwargs):
        super().__init__(kernel, f"rule_{id}_{name}_exclusion",
                        chroma_store, rules_dir, **kwargs)
        self.rule_id = "{ID}"
        self.rule_citation = "{Citation}"
        self.rule_type = "{Rule Type}"
        self.duty = "Document exclusion of {Rule Description}"
        self.is_enforcement = False
        self.is_exclusion = True
        self.relevance_status = "DOES NOT APPLY"
        self.exclusion_reason = "{Exclusion Reason}"

    async def check_exclusion(self, draft_text: str) -> FunctionResult:
        """
        Single duty: Document why this rule does NOT apply.

        Returns:
            FunctionResult with exclusion status and reason
        """
        # Implementation
        pass
```

---

## 📊 Complete Plugin Inventory (471 Plugins)

### **Enforcement Plugins** (50 plugins)

**Relevant Rules** (from relevance report):
1. ✅ Rule 5.2 - Privacy Protection (ENFORCE) - Rank #1
2. ✅ Rule 53 - Masters (ENFORCE)
3. ✅ Rule 50 - Judgment (ENFORCE)
4. ✅ Upjohn v. United States - Attorney-Client Privilege (ENFORCE)
5. ✅ Rule 67 - Deposit into Court (ENFORCE)
6. ... (45 more relevant rules)

### **Exclusion Plugins** (421 plugins)

**Irrelevant Rules** (from relevance report):
1. ✅ Rule 56 - Summary Judgment (EXCLUDE - motion type mismatch)
2. ✅ Rule 12 - Motion to Dismiss (EXCLUDE - not applicable)
3. ✅ Rule 23 - Class Actions (EXCLUDE - not applicable)
4. ✅ Rule 23.1 - Derivative Actions (EXCLUDE - not applicable)
5. ... (417 more exclusion rules)

**Exclusion Reasons**:
- Motion type mismatch (seal/pseudonym vs. summary judgment, etc.)
- Rule type mismatch (Federal Rules Appellate Procedure vs. Civil Procedure)
- Not applicable to motion type
- Wrong jurisdiction
- Wrong procedural stage

### **Formatting Plugins** (~20-30 plugins)

**Technical Rules**:
- ✅ Margin requirements (1 inch)
- ✅ Word count limits
- ✅ Page maximums
- ✅ Font requirements (Times New Roman, 12pt)
- ✅ Line spacing (double spacing)
- ✅ Citation format (Bluebook)
- ✅ Header/footer requirements
- ✅ Page numbering
- ✅ Caption requirements
- ✅ Signature block requirements

### **Legal Plugins** (~400+ plugins)

**Legal Rules**:
- ✅ Section 1782 requirements
- ✅ Intel factors
- ✅ Discovery rules
- ✅ Privilege rules
- ✅ Evidence rules
- ✅ Case law rules
- ✅ Local rules

---

## 🚀 Implementation Strategy

### **Step 1: Create Plugin Generator**

**File**: `atomic_plugin_generator.py`

**Function**: Generate all 471 plugins from RulesExpanded.json

**Output**:
- Enforcement plugins (50)
- Exclusion plugins (421)
- All plugins as static files or dynamic factory

### **Step 2: Generate Static Plugins**

**Option A: All Static** (471 files)
- Generate all 471 plugin files
- Maximum modularity
- Easy to navigate

**Option B: Hybrid** (Recommended)
- High-value enforcement plugins → Static files (~50)
- Exclusion plugins → Dynamic factory (~421)
- Formatting plugins → Static files (~20-30)

### **Step 3: Plugin Registration**

**File**: `atomic_plugin_registry.py`

**Function**: Register all 471 plugins

**Integration**: Integrate with RefinementLoop

---

## 🎯 Benefits of Complete Atomic Architecture

### **1. Perfect Modularity**
- ✅ One plugin per rule
- ✅ Single duty enforcement
- ✅ No shared responsibilities
- ✅ Complete independence

### **2. Perfect Recall**
- ✅ Every rule covered (471/471)
- ✅ No gaps
- ✅ Complete validation

### **3. Clean Data**
- ✅ Exclusion plugins document WHY rules don't apply
- ✅ Prevents false positives
- ✅ Redundancy for validation
- ✅ Clear audit trail

### **4. Maximum Scalability**
- ✅ Easy to add new rules
- ✅ Easy to modify individual rules
- ✅ No cascading changes
- ✅ Easy to test

### **5. Complete Coverage**
- ✅ Formatting rules (margins, word count, page limits)
- ✅ Legal rules (legal arguments)
- ✅ Procedural rules (FRCP)
- ✅ Evidentiary rules (FRE)
- ✅ Case law rules
- ✅ Local rules

---

## 📋 Plugin Generation Script

### **Generator Implementation**

```python
#!/usr/bin/env python3
"""
Atomic Plugin Generator - Generate all 471 plugins (one per rule).

Creates:
- Enforcement plugins for relevant rules
- Exclusion plugins for irrelevant rules
- Formatting plugins for technical rules
- Legal plugins for legal argument rules
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AtomicPluginGenerator:
    """Generate atomic plugins for all 471 rules."""

    def __init__(self, rules_path: Path, relevance_report_path: Path, output_dir: Path):
        self.rules_path = rules_path
        self.relevance_report_path = relevance_report_path
        self.output_dir = output_dir
        self.rules_data = self._load_rules()
        self.relevance_report = self._load_relevance_report()

    def generate_all_plugins(self):
        """Generate all 471 plugins."""
        logger.info("Generating 471 atomic plugins...")

        relevant_rule_ids = {
            r['rule_id'] for r in self.relevance_report.get('relevant_rules', [])
        }

        enforcement_count = 0
        exclusion_count = 0

        for rule in self.rules_data:
            rule_id = rule.get('rule_id')
            is_relevant = rule_id in relevant_rule_ids

            if is_relevant:
                self._generate_enforcement_plugin(rule)
                enforcement_count += 1
            else:
                self._generate_exclusion_plugin(rule)
                exclusion_count += 1

        logger.info(f"Generated {enforcement_count} enforcement plugins")
        logger.info(f"Generated {exclusion_count} exclusion plugins")
        logger.info(f"Total: {enforcement_count + exclusion_count} plugins")

    def _generate_enforcement_plugin(self, rule: Dict):
        """Generate enforcement plugin for relevant rule."""
        # Generate plugin code
        pass

    def _generate_exclusion_plugin(self, rule: Dict):
        """Generate exclusion plugin for irrelevant rule."""
        # Generate plugin code
        pass
```

---

## ✅ Summary

**Vision**: **Complete atomic modularity with 471 plugins**

**Design**:
- ✅ **One plugin per rule** (471 plugins)
- ✅ **Single duty enforcement** (each plugin does ONE thing)
- ✅ **Perfect recall** (every rule covered)
- ✅ **Exclusion plugins** (document why irrelevant rules don't apply)
- ✅ **Clean data** (redundancy for validation)
- ✅ **No limits** (maximum modularity)

**Coverage**:
- ✅ **50 enforcement plugins** (relevant rules)
- ✅ **421 exclusion plugins** (irrelevant rules)
- ✅ **Formatting plugins** (margins, word count, page limits)
- ✅ **Legal plugins** (legal arguments)
- ✅ **Complete coverage** (471/471 rules)

**Next Steps**:
1. Create plugin generator script
2. Generate all 471 plugins
3. Register all plugins
4. Integrate with RefinementLoop

**Maximum modularity, perfect recall, complete coverage!** 🎯

