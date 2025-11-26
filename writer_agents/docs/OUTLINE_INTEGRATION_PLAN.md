# 📋 Outline Integration Plan

## 🎯 Goal

Integrate the **perfect outline structure** (reverse-engineered from CatBoost analysis) into the writing system so that:
1. **Plugins know which section they belong to**
2. **Plugins recalibrate their targets** based on outline requirements
3. **Section order is enforced** (Legal Standard → Factual Background is critical!)
4. **Enumeration requirements are met** (11+ instances, bullet points throughout)

---

## 📁 Files Created

### 1. **`outline_manager.py`** ✅
- Manages perfect outline structure
- Defines 9 sections with positions, importance, plugins
- Enforces critical transitions (Legal Standard → Factual Background)
- Provides enumeration requirements

### 2. **`plugin_calibrator.py`** ✅
- Recalibrates plugins based on outline section requirements
- Updates plugin targets based on section needs
- Organizes plugins by section and priority
- Validates plugin configuration

---

## 🔧 Integration Points

### ✅ Completed:

1. **RefinementLoop.__init__()** - OutlineManager and PluginCalibrator initialization
2. **Plugin target recalibration** - `_recalibrate_plugin_targets_from_outline()`
3. **Outline structure loaded** - Perfect outline with 9 sections and critical transitions

### ⚠️ Still Needed:

1. **Conductor._initialize_feature_orchestrator()** - Pass outline_manager to RefinementLoop
2. **Drafting phase** - Use outline to organize section generation
3. **Section validation** - Validate draft sections match perfect outline
4. **Plugin organization** - Organize plugins by section during drafting

---

## 📊 Perfect Outline Structure

```
1. Introduction
2. Legal Standard ← CRITICAL (importance: 64.80)
3. Factual Background ← CRITICAL (must immediately follow Legal Standard)
4. Privacy Harm / Good Cause
5. Danger / Safety Arguments ← Place early
6. Public Interest Analysis
7. Balancing Test ← Place prominently
8. Proposed Protective Measures
9. Conclusion
```

### Critical Requirements:

- **Transition**: Legal Standard → Factual Background (MUST be consecutive)
- **Enumeration**: 11+ instances throughout motion
- **Bullet Points**: Use extensively (importance: 31.56)
- **Section Order**: Matters for persuasion

---

## 🚀 Next Steps

1. ✅ Create OutlineManager and PluginCalibrator (DONE)
2. ⏳ Integrate into RefinementLoop initialization (PARTIAL)
3. ⏳ Update Conductor to pass outline_manager
4. ⏳ Use outline in drafting phase to organize sections
5. ⏳ Validate draft sections against outline structure
6. ⏳ Organize plugins by section during analysis

---

## 💡 How It Works

### Plugin Calibration Flow:

```
Perfect Outline → OutlineManager → PluginCalibrator → Plugin Targets Updated
```

1. **OutlineManager** loads perfect outline structure
2. **PluginCalibrator** generates calibrations for each plugin
3. **RefinementLoop** uses calibrations to update plugin targets
4. **Plugins** know their section, priority, and target values

### Example:

```python
# Plugin: mentions_privacy
# Section: privacy_harm
# Calibration:
{
    "section": "privacy_harm",
    "min_privacy_mentions": 5,  # From CatBoost analysis
    "enumeration_required": True,
    "priority": "required"
}
```

---

## ✅ Status

**Created**: OutlineManager and PluginCalibrator ✅
**Integrated**: Partial (RefinementLoop initialization) ⏳
**Next**: Complete integration in Conductor and drafting phase

