# The 12 Critical Seal Facts

## Overview

These are the **12 mandatory facts** that must be included in every motion to seal. The system validates that all 12 facts are present using regex pattern matching defined in `personal_facts_verifier.py`.

## The 12 Facts

### 1. **hk_statement** - Hong Kong Statement of Claim
- **Patterns**: `hong\s+kong\s+statement\s+of\s+claim`, `\bhk\s+statement\b`
- **Description**: References the Hong Kong Statement of Claim (Action No. 771)
- **Fallback Text**: "The Hong Kong Statement of Claim (Action No. 771) documents Harvard-linked retaliation and is part of the sealed record."

### 2. **ogc_emails** - Harvard OGC Emails
- **Patterns**: `\bogc\b`, `office\s+of\s+general\s+counsel`
- **Description**: References the Harvard OGC emails / Office of General Counsel
- **Fallback Text**: "Harvard's Office of the General Counsel was notified via the April 2025 email chain but failed to respond."

### 3. **date_april_7_2025** - April 7, 2025 OGC Notice
- **Patterns**: `april\s+7,\s*2025`, `7\s+april\s+2025`
- **Description**: Mentions April 7, 2025 OGC notice
- **Fallback Text**: "On April 7, 2025, Plaintiff warned Harvard OGC about the Hong Kong litigation and retaliation findings."

### 4. **date_april_18_2025** - April 18, 2025 OGC Follow-up
- **Patterns**: `april\s+18,\s*2025`, `18\s+april\s+2025`
- **Description**: Mentions April 18, 2025 OGC follow-up
- **Fallback Text**: "By April 18, 2025 Harvard still had not responded, escalating the retaliation risk documented in the HK court record."

### 5. **date_june_2_2025** - June 2, 2025 HK Filing
- **Patterns**: `june\s+2,\s*2025`, `2\s+june\s+2025`
- **Description**: Mentions June 2, 2025 HK filing
- **Fallback Text**: "Hong Kong authorities intensified arrests on June 2, 2025, underscoring the safety risks of public disclosure."

### 6. **date_june_4_2025** - June 4, 2025 Arrests/Threats
- **Patterns**: `june\s+4,\s*2025`, `4\s+june\s+2025`
- **Description**: Mentions June 4, 2025 arrests / threats
- **Fallback Text**: "On June 4, 2025, authorities tied the disclosures to the Tiananmen commemoration crackdown, amplifying the danger."

### 7. **allegation_defamation** - Defamation Allegation
- **Patterns**: `\bdefamation\b`, `\bdefamatory\b`
- **Description**: Explains defamation allegation

### 8. **allegation_privacy_breach** - Privacy Breach
- **Patterns**: `privacy\s+breach`, `\bprivacy\b`
- **Description**: Describes privacy breach
- **Fallback Text**: "Plaintiff's private educational records were exposed without consent, creating a direct privacy breach."

### 9. **allegation_retaliation** - Retaliation
- **Patterns**: `\bretaliation\b`, `\bretaliatory\b`
- **Description**: Describes retaliation

### 10. **allegation_harassment** - Harassment
- **Patterns**: `\bharassment\b`, `\bharassing\b`
- **Description**: Describes harassment
- **Fallback Text**: "Plaintiff has endured a sustained campaign of harassment and retaliation tied to Harvard-affiliated actors."

### 11. **timeline_april_ogc_emails** - April 2025 OGC Email Timeline
- **Patterns**: 
  - `april\s+2025[^.]{0,120}ogc`
  - `ogc[^.]{0,120}april\s+2025`
  - `april\s+2025[^.]{0,120}office\s+of\s+general\s+counsel`
- **Description**: Timeline reference to April 2025 OGC emails
- **Fallback Text**: "In April 2025, Harvard's Office of the General Counsel was warned via email yet refused to engage, escalating the dispute."

### 12. **timeline_june_2025_arrests** - June 2025 Arrests Timeline
- **Patterns**: 
  - `june\s+2025[^.]{0,120}arrest`
  - `arrest[^.]{0,120}june\s+2025`
  - `june\s+2025[^.]{0,120}detention`
- **Description**: Timeline reference to June 2025 arrests or threats
- **Fallback Text**: "By June 2025, Hong Kong authorities intensified arrests linked to the disclosures, heightening the safety risks."

## Source Code

The fact rules are defined in:
- **Verifier**: `personal_facts_verifier.py` (lines 73-142) - `DEFAULT_FACT_RULES`
- **Enforcement**: `fact_payload_utils.py` (lines 266-352) - `SEAL_CRITICAL_FACT_RULES`

## Validation Process

1. **Pattern Matching**: Uses strict regex patterns to verify facts are present
2. **Fact Enforcement**: System programmatically injects missing facts if LLM omits them
3. **Coverage Tracking**: Reports percentage of facts covered (target: 100%)
4. **Retry Logic**: If facts are missing, system retries with explicit fact checklists

