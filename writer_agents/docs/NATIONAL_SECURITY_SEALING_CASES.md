# National Security/Sealing Supporting Cases - Added to System

## Overview

Added 4 high-priority Section 1782 cases that support arguments for national security-based sealing in Section 1782 discovery applications.

## Cases Added to Database & Plugin System

### 1. Friends for All Children, Inc. v. Lockheed Aircraft Corp. (D.D.C. 1976)
- **Cluster ID:** 1752777
- **Priority:** HIGH
- **Plugin:** `case_enforcement_friends_for_all_children_inc_v_lockheed_aircraft_c`
- **Why it supports you:**
  - Direct precedent for filing materials "under seal" in discovery proceedings
  - Shows courts approve protective orders for confidential documents
  - Demonstrates sealing for work-product privilege materials
- **Key Points:**
  - Sealing
  - Protective orders
  - Confidential documents
  - Section 1782

### 2. In Re Application of Chevron Corp. (S.D.N.Y.)
- **Cluster IDs:** 1587460, 2541883
- **Priority:** HIGH
- **Plugin:** `case_enforcement_in_re_application_of_chevron_corp`
- **Why it supports you:**
  - Section 1782 granted for sensitive international matters
  - Shows courts grant discovery even for sensitive materials
  - Establishes "highly likely to be directly relevant" standard for foreign proceedings
- **Key Points:**
  - Section 1782
  - Foreign proceedings
  - Sensitive materials
  - International arbitration

### 3. In Re: Application of International Mineral Resources B.V. (D.D.C. 2014)
- **Cluster ID:** 2821079
- **Priority:** HIGH
- **Plugin:** `case_enforcement_in_re_application_of_international_mineral_resourc`
- **Why it supports you:**
  - References "In re Sealed Case" precedent (754 F.2d 395, 399)
  - Shows sealing is established practice in discovery contexts
  - Section 1782 granted with sealing considerations
- **Key Points:**
  - Section 1782
  - Sealing
  - Sealed Case precedent

## Database Updates

- All 8 cases verified in database
- National security tags applied where appropriate
- Cases tagged with `section_1782_discovery` corpus_subset
- Database updates complete

## Plugin System Integration

### Plugin Registry
- **Location:** `writer_agents/config/plugin_registry.json`
- **Status:** 4 new plugins added
- **Category:** case_enforcement
- **Priority:** HIGH
- **Status:** active, enabled

### Master Checklist
- **Location:** `rules_registry/output/MASTERCHECKLISTEXPANDED.md`
- **Status:** Updated with 3 new case entries
- **Section:** Case Law - Section 1782

### Plugin Generator
- **Location:** `writer_agents/code/sk_plugins/FeaturePlugin/case_enforcement_plugin_generator.py`
- **Status:** Plugins will be generated dynamically at runtime
- **Enforcement:** Each plugin enforces citation of its respective case with proper checklists

## Plugin Enforcement Checklist

Each plugin will enforce:

### Friends for All Children v. Lockheed
- [ ] Case cited
- [ ] "Filed under seal" precedent mentioned
- [ ] Protective orders for confidential documents referenced
- [ ] Sealing for work-product privilege materials mentioned

### Chevron Corp.
- [ ] Case cited
- [ ] Section 1782 granted for sensitive international matters cited
- [ ] "Highly likely to be directly relevant" standard referenced
- [ ] International arbitration discovery precedent mentioned

### International Mineral Resources
- [ ] Case cited
- [ ] "In re Sealed Case" precedent (754 F.2d 395, 399) referenced
- [ ] Sealing as established practice mentioned
- [ ] Section 1782 with sealing considerations noted

## Usage

These plugins will automatically:
1. Check if cases are cited in your drafts
2. Verify proper citation format
3. Ensure key holdings are referenced
4. Generate edit requests if citations are missing
5. Suggest improvements based on successful case patterns

## Next Steps

1. Plugins are automatically loaded by the plugin generator
2. Test with a sample Section 1782 motion
3. Verify citations are enforced correctly
4. Monitor plugin performance in validation phase

## Related Cases (Medium Priority)

The following cases were also identified but assigned medium priority:
- STATE EX REL. JOINT COMMITTEE v. Bonar (national security in subpoena contexts)
- McConnell v. Federal Election Commission (national security considerations)
- Texas v. United States ("national security risk" as valid basis)
- United States v. Philip Morris USA, Inc. (confidential materials and protective orders)

These can be added as plugins later if needed.

