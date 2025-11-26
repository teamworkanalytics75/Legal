from cursor_tracker import CursorTracker

t = CursorTracker()

print("=== SAMPLE DATA (Created for Demo) ===")
print("Session 1:", t.data['sessions'][0]['project_context'], "-", t.data['sessions'][0]['start_time'][:10])
print("Session 2:", t.data['sessions'][1]['project_context'], "-", t.data['sessions'][1]['start_time'][:10])

print("\n=== YOUR REAL DATA (Just Created) ===")
print("Session 3:", t.data['sessions'][2]['project_context'], "-", t.data['sessions'][2]['start_time'][:10])

print("\n=== SAMPLE QUESTIONS ===")
for i, q in enumerate(t.data['questions'][:3]):
    print(f"{i+1}. {q['question']} ({q['category']})")

print("\n=== YOUR REAL QUESTION ===")
print(f"4. {t.data['questions'][-1]['question']} ({t.data['questions'][-1]['category']})")

print("\n=== SUMMARY ===")
print(f"Total sessions: {len(t.data['sessions'])}")
print(f"Sample sessions: 2")
print(f"Your real sessions: 1")
print(f"Total questions: {len(t.data['questions'])}")
print(f"Sample questions: 3")
print(f"Your real questions: 1")
