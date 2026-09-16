"""Check a necessary condition for monotone aggregation on the four fixed pairs."""
import hashlib,json
from pathlib import Path
p=Path("results/P029_clause_diagnostic.json");d=json.loads(p.read_text())
cases=[]
for c in d["cases"]:
    clauses=[r for r in c["claims"] if r["claim_index"]>0]
    delta=[r["gold_minus_error"] for r in clauses]
    cases.append({"id":c["id"],"n_clauses":len(delta),
                  "all_clauses_strictly_favor_error":all(x<0 for x in delta),
                  "largest_gold_minus_error":max(delta)})
d["dominance_audit"]={
    "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases":cases,
    "n_error_dominated_cases":sum(c["all_clauses_strictly_favor_error"] for c in cases),
    "argument":"For the fixed clause score vectors, s(gold)<=s(error) coordinatewise implies F(s(gold))<=F(s(error)) for every coordinatewise nondecreasing aggregation F. It cannot strictly rank gold above error.",
    "scope":"Necessary-order property only for these four post-hoc pairs, this exact decomposition and frozen pointwise scores. This is a standard Pareto-order property, not a new theorem.",
    "does_not_rule_out":["Improving clause estimation","Adding/changing visual evidence","Better or alternative faithful decompositions","Joint dependency/context models","Other examples"]}
p.write_text(json.dumps(d,indent=2)+"\n")
print("error-dominated cases",d["dominance_audit"]["n_error_dominated_cases"],"/",len(cases))
