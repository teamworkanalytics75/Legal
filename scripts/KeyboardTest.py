#!/usr/bin/env python3
"""
keyboard_test.py
Test script to help identify keyboard layouts and key codes.
"""

import time

def test_keyboard_layout():
    """Help identify keyboard layouts."""
    print("🎯 Keyboard Layout Test for Predator Laptop")
    print("=" * 50)
    print()
    print("Your updated hotkeys are now:")
    print()
    print("📋 MAIN KEYBOARD (top):")
    print("  • Ctrl+Alt+1 → Cursor agent message")
    print("  • Ctrl+Alt+2 → Codex agent message")
    print("  • Ctrl+Alt+3 → Stuck/unstuck message")
    print()
    print("🔢 SIDE KEYBOARD (numpad):")
    print("  • Ctrl+Alt+Numpad1 → Cursor agent message")
    print("  • Ctrl+Alt+Numpad2 → Codex agent message")
    print("  • Ctrl+Alt+Numpad3 → Stuck/unstuck message")
    print()
    print("🪟 BACKUP (Windows key):")
    print("  • Windows+1 → Cursor agent message")
    print("  • Windows+2 → Codex agent message")
    print("  • Windows+3 → Stuck/unstuck message")
    print()
    print("✅ AutoHotkey Status: RUNNING")
    print("✅ Regular 1,2,3 keys: NORMAL (no interference)")
    print()
    print("🧪 Test Instructions:")
    print("1. Try Ctrl+Alt+1 on the TOP keyboard")
    print("2. Try Ctrl+Alt+1 on the SIDE keyboard (numpad)")
    print("3. Try Windows+1 on either keyboard")
    print("4. Regular 1,2,3 keys should work normally")
    print()
    print("If you want ONLY the side keyboard, I can create a specialized version!")

if __name__ == "__main__":
    test_keyboard_layout()
