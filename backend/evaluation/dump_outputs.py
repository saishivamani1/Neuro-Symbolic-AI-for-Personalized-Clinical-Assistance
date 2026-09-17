import sys, os, json
sys.path.insert(0, r'c:\Users\saish\OneDrive\Documents\minor\backend')
os.chdir(r'c:\Users\saish\OneDrive\Documents\minor\backend')
from app.reasoning.engine import SymbolicRuleEngine
engine = SymbolicRuleEngine()

with open(r'c:\Users\saish\OneDrive\Documents\minor\backend\evaluation\humanized_cases.json') as f:
    cases = json.load(f)

print('Actual engine outputs per humanized case:')
print('=' * 70)
for case in cases:
    result = engine.evaluate(case['patient_facts'])
    fired = sorted([r.rule_id for r in result.triggered_rules])
    facts = sorted([f.name for f in result.derived_facts])
    alerts = sorted([a.name for a in result.critical_alerts])
    print("Case: %s  domain=%s" % (case['case_id'], case['domain']))
    print("  Fired Rules : %s" % fired)
    print("  Derived     : %s" % facts)
    print("  Alerts      : %s" % alerts)
    print()
