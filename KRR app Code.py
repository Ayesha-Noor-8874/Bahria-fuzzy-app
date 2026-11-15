from __future__ import annotations

import json, math, os, sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================  1. ENGINE


class InputError(ValueError):
    """Raised when user input is missing, non-numeric or out of range."""

    def __init__(self, messages):
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("; ".join(self.messages))


# ----------------------------------------------------------------- MFs
def trapmf(x, a, b, c, d):
    """Trapezoid (a,b,c,d). a==b or c==d gives a vertical (shoulder) edge."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if d > c:
        m = (x > c) & (x < d)
        y[m] = (d - x[m]) / (d - c)
    y[(x >= b) & (x <= c)] = 1.0
    return y


def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)


@dataclass
class Variable:
    name: str
    lo: float
    hi: float
    terms: Dict[str, Tuple[float, ...]]
    unit: str = ""
    n: int = 1001

    @property
    def universe(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.n)

    def mf(self, term: str, x) -> np.ndarray:
        p = self.terms[term]
        return trimf(x, *p) if len(p) == 3 else trapmf(x, *p)

    def fuzzify(self, x: float) -> Dict[str, float]:
        xa = np.array([float(x)])
        return {t: float(self.mf(t, xa)[0]) for t in self.terms}

    def label(self, x: float) -> str:
        """Linguistic term with the highest membership at x."""
        mu = self.fuzzify(x)
        return max(mu, key=mu.get)


# ----------------------------------------------------------------- rules
AND = lambda *v: min(v)
OR = lambda *v: max(v)
NOT = lambda v: 1.0 - v


@dataclass
class Rule:
    id: str
    if_text: str
    then_text: str
    source: str                                   # handbook clause or "design"
    strength: Callable[[dict], float]
    then: List[Tuple[str, Optional[str]]]         # (output var, term) or (flag, None)
    note: str = ""


@dataclass
class Inference:
    fired: List[dict]
    all_rules: List[dict]
    crisp: Dict[str, Optional[float]]
    agg: Dict[str, Tuple[np.ndarray, np.ndarray]]
    flags: Dict[str, float]
    fuzzified: Dict[str, Dict[str, float]] = field(default_factory=dict)


def infer(rules: List[Rule], ctx: dict, outputs: Dict[str, Variable]) -> Inference:
    agg = {k: (v.universe, np.zeros(v.n)) for k, v in outputs.items()}
    flags: Dict[str, float] = {}
    rows = []
    for r in rules:
        w = float(max(0.0, min(1.0, r.strength(ctx))))
        rows.append({"id": r.id, "if": r.if_text, "then": r.then_text,
                     "source": r.source, "strength": round(w, 4), "note": r.note})
        if w <= 0:
            continue
        for var, term in r.then:
            if var in outputs:
                x, mu = agg[var]
                np.maximum(mu, np.minimum(w, outputs[var].mf(term, x)), out=mu)
            else:
                flags[var] = max(flags.get(var, 0.0), w)
    crisp = {}
    for k, (x, mu) in agg.items():
        area = mu.sum()
        crisp[k] = float((x * mu).sum() / area) if area > 0 else None   # centroid
    fired = sorted((r for r in rows if r["strength"] > 0), key=lambda r: -r["strength"])
    return Inference(fired, rows, crisp, agg, flags)


# ----------------------------------------------------------------- validation
def num(value, name, lo, hi, errors, integer=False):
    """Validate a numeric input; append to errors and return None if bad."""
    if value is None or (isinstance(value, str) and not value.strip()):
        errors.append(f"{name} is required.")
        return None
    if isinstance(value, bool):
        errors.append(f"{name} must be a number.")
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number (got {value!r}).")
        return None
    if math.isnan(v) or math.isinf(v):
        errors.append(f"{name} must be a finite number.")
        return None
    if not lo <= v <= hi:
        errors.append(f"{name} must be between {lo:g} and {hi:g} (got {v:g}).")
        return None
    if integer and v != int(v):
        errors.append(f"{name} must be a whole number (got {v:g}).")
        return None
    return v


def choice(value, name, options, errors):
    if value not in options:
        errors.append(f"{name} must be one of {list(options)} (got {value!r}).")
        return None
    return value


# =====================================================================  2. MODULE 1: ATTENDANCE (s.1.27)



THRESHOLD = 75.0   # Handbook 1.27.1

# ---------------------------------------------------------------- variables
ATT = Variable("Attendance Percentage", 0, 100, {
    "Very Low":     (0, 0, 35, 50),
    "Low":          (35, 50, 65),
    "Borderline":   (60, 75, 85),
    "Satisfactory": (75, 75, 90, 95),     # vertical edge AT 75 : "at least 75%"
    "High":         (90, 95, 100, 100),
}, "%")
UNF = Variable("Unforeseen Situation", 0, 10, {
    "None": (0, 0, 1, 2), "Low": (1, 3, 5), "Moderate": (3, 5, 7), "Severe": (6, 8, 10, 10),
}, "0-10")
TUT = Variable("Tutorial Completion", 0, 100, {
    "Incomplete": (0, 0, 20, 40), "Partial": (25, 50, 75), "Complete": (70, 90, 100, 100),
}, "%")
RISK = Variable("Attendance Risk", 0, 100, {
    "Low": (0, 0, 20, 35), "Medium": (25, 50, 75), "High": (65, 80, 100, 100),
})
A_ELIG = Variable("Examination Eligibility", 0, 100, {
    "Not Eligible": (0, 0, 25, 45), "Eligible": (55, 75, 100, 100),
})
INPUT_VARS = {"Attendance": ATT, "Unforeseen": UNF, "Tutorial": TUT}
A_OUTPUTS = {"Attendance Risk": RISK, "Examination Eligibility": A_ELIG}

# ---------------------------------------------------------------- rules
H = "Handbook "
A_RULES = [
    Rule("F-ATT-01", "Attendance is Very Low", "Risk is High AND Eligibility is Not Eligible", H + "1.27.1",
         lambda c: c["att"]["Very Low"], [("Attendance Risk", "High"), ("Examination Eligibility", "Not Eligible")]),
    Rule("F-ATT-02", "Attendance is Low", "Risk is High AND Eligibility is Not Eligible", H + "1.27.1",
         lambda c: c["att"]["Low"], [("Attendance Risk", "High"), ("Examination Eligibility", "Not Eligible")]),
    Rule("F-ATT-03", "Attendance is Borderline", "Risk is Medium", H + "1.27.1",
         lambda c: c["att"]["Borderline"], [("Attendance Risk", "Medium")],
         "Risk/review flag only - does NOT grant permission to sit the exam below 75 %."),
    Rule("F-ATT-04", "Attendance is Satisfactory OR High", "Risk is Low AND Eligibility is Eligible", H + "1.27.1",
         lambda c: OR(c["att"]["Satisfactory"], c["att"]["High"]),
         [("Attendance Risk", "Low"), ("Examination Eligibility", "Eligible")]),
    Rule("F-ATT-05", "Unforeseen Situation is Severe AND Attendance is Borderline",
         "Risk is High; ADVISORY: unforeseen-case review", H + "1.27.1 (25 % relaxation)",
         lambda c: AND(c["unf"]["Severe"], c["att"]["Borderline"]),
         [("Attendance Risk", "High"), ("flag:review", None)],
         "Design assumption: a severe situation is likely to use up the remaining 25 % slack. "
         "Advisory only - 1.27.1 says no shortfall is condoned."),
    Rule("F-ATT-06", "BUHO Approval is Approved AND a sports/extracurricular conflict exists",
         "ADVISORY: tutorial arrangement is required", H + "1.27.2",
         lambda c: AND(c["approved"], c["clash"]), [("flag:tutorial_required", None)]),
    Rule("F-ATT-07", "Tutorial Completion is Complete AND BUHO Approval is Approved",
         "ADVISORY: missed-class attendance credit is permitted", H + "1.27.2",
         lambda c: AND(c["tut"]["Complete"], c["approved"], c["clash"]), [("flag:credit_ok", None)],
         "Credit is applied by the university; the attendance entered here is not changed automatically."),
    Rule("F-ATT-08", "Tutorial Completion is Incomplete OR (conflict exists AND approval is missing)",
         "ADVISORY: missed-class attendance credit is not permitted", H + "1.27.2",
         lambda c: AND(c["clash"], OR(c["tut"]["Incomplete"], NOT(c["approved"]))), [("flag:credit_no", None)],
         "Extension of the doc rule: 1.27.2 requires PRIOR approval, so no approval = no credit."),
    Rule("F-ATT-09", "Student is Postgraduate AND requested relaxation is above 25 %",
         "ADVISORY: attendance relaxation is not permitted", H + "1.27.3",
         lambda c: AND(c["pg"], c["relax_req"]), [("flag:pg_no_relax", None)]),
    Rule("F-ATT-10", "Attendance is below 75 %", "Shortfall is not condoned; Risk is High", H + "1.27.1",
         lambda c: c["below"], [("flag:not_condoned", None), ("Attendance Risk", "High")],
         "Crisp rule (membership 0/1): the handbook threshold is binding. "
         "Added 'Risk is High' because Borderline (60-75) alone would give only 'Medium' to a barred student."),
    Rule("F-ATT-11", "Attendance is below 75 %", "Eligibility is Not Eligible", H + "1.27.1",
         lambda c: c["below"], [("Examination Eligibility", "Not Eligible")], "Crisp rule."),
    Rule("F-ATT-12", "Attendance is 75 % or above", "Eligibility is Eligible", H + "1.27.1",
         lambda c: NOT(c["below"]), [("Examination Eligibility", "Eligible")], "Crisp rule; 75 is INCLUDED."),
]

A_FLAG_TEXT = {
    "review": "Severe unforeseen situation while borderline: speak to your HoD early. "
              "This is advice only - the handbook does not condone any shortfall below 75 %.",
    "tutorial_required": "BUHO has approved the activity: tutorials must be arranged for the missed classes.",
    "credit_ok": "Tutorials are complete and BUHO approved: the missed classes may be credited to your record.",
    "credit_no": "Missed classes cannot be credited yet (tutorials incomplete or no prior BUHO approval).",
    "pg_no_relax": "Postgraduate scholars get no relaxation beyond the permissible 25 %.",
    "not_condoned": "Attendance is below 75 %: the shortfall cannot be condoned and attendance once marked cannot be changed.",
}

LEVELS = ("Undergraduate", "Postgraduate")


def hours_to_percent(attended, total):
    """Helper for the GUI: attended / total contact hours -> percentage."""
    e = []
    t = num(total, "Total contact hours", 0.0001, 10_000, e)
    a = num(attended, "Attended hours", 0, 10_000, e)
    if not e and a > t:
        e.append("Attended hours cannot exceed total contact hours.")
    if e:
        raise InputError(e)
    return 100.0 * a / t


def assess_attendance(attendance, unforeseen=0, tutorial=0, buho_approved=False,
                      clash=False, level="Undergraduate", relaxation_request=False):
    errs = []
    att = num(attendance, "Attendance %", 0, 100, errs)
    unf = num(unforeseen, "Unforeseen situation severity", 0, 10, errs)
    tut = num(tutorial, "Tutorial completion %", 0, 100, errs)
    choice(level, "Level", LEVELS, errs)
    if errs:
        raise InputError(errs)

    ctx = {
        "att": ATT.fuzzify(att), "unf": UNF.fuzzify(unf), "tut": TUT.fuzzify(tut),
        "approved": 1.0 if buho_approved else 0.0, "clash": 1.0 if clash else 0.0,
        "pg": 1.0 if level == "Postgraduate" else 0.0,
        "relax_req": 1.0 if relaxation_request else 0.0,
        "below": 1.0 if att < THRESHOLD else 0.0,          # exact comparison, no rounding
    }
    res = infer(A_RULES, ctx, A_OUTPUTS)
    risk, score = res.crisp["Attendance Risk"], res.crisp["Examination Eligibility"]

    official_eligible = att >= THRESHOLD                    # binding policy gate
    fuzzy_eligible = score is not None and score >= 50
    return {
        "module": "attendance",
        "inputs": {"attendance": att, "unforeseen": unf, "tutorial": tut, "level": level},
        "eligible": official_eligible,
        "decision": "ELIGIBLE to sit the final exam" if official_eligible else "NOT ELIGIBLE to sit the final exam",
        "risk_score": risk, "risk_label": RISK.label(risk) if risk is not None else None,
        "elig_score": score, "elig_label": A_ELIG.label(score) if score is not None else None,
        "policy_consistent": fuzzy_eligible == official_eligible,
        "margin": round(att - THRESHOLD, 4),
        "fuzzified": {"Attendance": ctx["att"], "Unforeseen": ctx["unf"], "Tutorial": ctx["tut"]},
        "advice": [A_FLAG_TEXT[k] for k in A_FLAG_TEXT if res.flags.get("flag:" + k, 0) > 0],
        "flags": {k[5:]: v for k, v in res.flags.items()},
        "fired": res.fired, "all_rules": res.all_rules, "agg": res.agg,
    }


# =====================================================================  3. MODULE 2: MERIT SCHOLARSHIP (s.4.4)



T_NON_ENG, T_ENG = 3.75, 3.60                      # 4.4.3.3
AWARDS = (0, 30, 40, 70)                           # 4.4.1 / 4.4.2
UG_AWARD = {1: 70, 2: 40, 3: 30}
PG_AWARD = {1: 70, 2: 40}

# GPA margin D = GPA - T.  "Meets Requirement" covers EVERY D >= 0 (so a student well
# above the cut-off is never weakened by the OR of two sloping sets); "High" is nested in it.
GPA = Variable("GPA Margin (GPA - T)", -4.0, 0.4, {
    "Below Requirement": (-4.0, -4.0, -0.20, 0.0),
    "Borderline":        (-0.10, 0.0, 0.10),
    "Meets Requirement": (0.0, 0.0, 0.40, 0.40),
    "High":              (0.15, 0.30, 0.40, 0.40),
}, "GPA points", n=2001)
S_ELIG = Variable("Scholarship Eligibility", 0, 100, {
    "Not Eligible": (0, 0, 25, 45), "Eligible": (55, 75, 100, 100)})
PCT = Variable("Scholarship Percentage", 0, 100, {
    "No Scholarship": (0, 0, 5), "30% Scholarship": (25, 30, 35),
    "40% Scholarship": (35, 40, 45), "70% Scholarship": (65, 70, 75)}, "%")
S_OUTPUTS = {"Scholarship Eligibility": S_ELIG, "Scholarship Percentage": PCT}
H = "Handbook "

NO_DISQ = lambda c: AND(NOT(c["staff_conc"]), NOT(c["excl_sem"]), c["regular"], NOT(c["tie_lost"]),
                        NOT(c["multi_blocked"]))
GPA_OK = lambda c: c["gpa"]["Meets Requirement"]
E_ALL = lambda c: AND(GPA_OK(c), c["in_range"], NO_DISQ(c))      # overall eligibility degree

S_RULES = [
    Rule("F-MS-01", "Undergraduate AND Position is First AND eligible", "Scholarship is 70 %", H + "4.4.1",
         lambda c: AND(c["ug"], c["pos"] == 1, E_ALL(c)), [("Scholarship Percentage", "70% Scholarship")]),
    Rule("F-MS-02", "Undergraduate AND Position is Second AND eligible", "Scholarship is 40 %", H + "4.4.1",
         lambda c: AND(c["ug"], c["pos"] == 2, E_ALL(c)), [("Scholarship Percentage", "40% Scholarship")]),
    Rule("F-MS-03", "Undergraduate AND Position is Third AND eligible", "Scholarship is 30 %", H + "4.4.1",
         lambda c: AND(c["ug"], c["pos"] == 3, E_ALL(c)), [("Scholarship Percentage", "30% Scholarship")]),
    Rule("F-MS-04", "Postgraduate AND Position is First AND eligible", "Scholarship is 70 %", H + "4.4.2",
         lambda c: AND(c["pg"], c["pos"] == 1, E_ALL(c)), [("Scholarship Percentage", "70% Scholarship")]),
    Rule("F-MS-05", "Postgraduate AND Position is Second AND eligible", "Scholarship is 40 %", H + "4.4.2",
         lambda c: AND(c["pg"], c["pos"] == 2, E_ALL(c)), [("Scholarship Percentage", "40% Scholarship")]),
    Rule("F-MS-06", "Non-Engineering AND GPA margin is Below Requirement", "Eligibility is Not Eligible",
         H + "4.4.3.3", lambda c: AND(c["noneng"], c["gpa"]["Below Requirement"]),
         [("Scholarship Eligibility", "Not Eligible")]),
    Rule("F-MS-07", "Non-Engineering AND GPA Meets Requirement AND in award range AND no disqualifier",
         "Eligibility is Eligible", H + "4.4.3.3 / 4.4.1-4.4.2",
         lambda c: AND(c["noneng"], E_ALL(c)), [("Scholarship Eligibility", "Eligible")]),
    Rule("F-MS-08", "Engineering AND GPA margin is Below Requirement", "Eligibility is Not Eligible",
         H + "4.4.3.3", lambda c: AND(c["eng"], c["gpa"]["Below Requirement"]),
         [("Scholarship Eligibility", "Not Eligible")]),
    Rule("F-MS-09", "Engineering AND GPA Meets Requirement AND in award range AND no disqualifier",
         "Eligibility is Eligible", H + "4.4.3.3 / 4.4.1-4.4.2",
         lambda c: AND(c["eng"], E_ALL(c)), [("Scholarship Eligibility", "Eligible")]),
    Rule("F-MS-10", "Tied on the same position AND the other student has the higher percentage",
         "Eligibility is Not Eligible (for this position)", H + "4.4.3.4",
         lambda c: c["tie_lost"], [("Scholarship Eligibility", "Not Eligible")],
         "Doc rule says 'equal GPA'; the handbook only says 'higher percentage' (see report)."),
    Rule("F-MS-10b", "Tied AND my percentage is higher, or percentages are also equal",
         "ADVISORY: award retained (a double tie awards all tied students)", H + "4.4.3.4",
         lambda c: AND(c["tie_kept"], E_ALL(c)), [("flag:tie_kept", None)]),
    Rule("F-MS-11", "Naval Fee Concession is Yes AND scholarship is awarded",
         "ADVISORY: only one benefit may be availed at a time", H + "4.4.3.6",
         lambda c: AND(c["naval"], E_ALL(c)), [("flag:one_benefit", None)]),
    Rule("F-MS-12", "University staff AND already has a fee concession", "Eligibility is Not Eligible",
         H + "4.4.3.7", lambda c: c["staff_conc"], [("Scholarship Eligibility", "Not Eligible")]),
    Rule("F-MS-13", "Semester is Summer OR Final (Graduating)", "Eligibility is Not Eligible",
         H + "4.4.3.8", lambda c: c["excl_sem"], [("Scholarship Eligibility", "Not Eligible")]),
    Rule("F-MS-14", "Two-year MS/MPhil AND semester is Third", "ADVISORY: coursework GPA is the GPA considered",
         H + "4.4.3.10", lambda c: AND(c["msm"], c["sem3"]), [("flag:coursework_gpa", None)]),
    Rule("F-MS-15", "Two-year MS/MPhil AND semester is not Third",
         "ADVISORY: handbook does not define the GPA basis for this semester", H + "4.4.3.10",
         lambda c: AND(c["msm"], NOT(c["sem3"])), [("flag:coursework_na", None)],
         "Doc rule is ambiguous; the system applies the standard GPA test and warns."),
    Rule("F-MS-16", "Position is outside the award range (UG: >3, PG: >2)", "Eligibility is Not Eligible",
         H + "4.4.1 / 4.4.2", lambda c: NOT(c["in_range"]), [("Scholarship Eligibility", "Not Eligible")],
         "Added: completes the rule base (doc's 'Other' position)."),
    Rule("F-MS-17", "Student is not following the roadmap / regular course load", "Eligibility is Not Eligible",
         H + "4.4.3.5", lambda c: NOT(c["regular"]), [("Scholarship Eligibility", "Not Eligible")],
         "Added: clause 4.4.3.5 was missing from the doc."),
    Rule("F-MS-18", "Student already receives another scholarship/waiver and has not chosen the merit award",
         "Eligibility is Not Eligible for the merit award (only one benefit)", H + "4.4.3.9",
         lambda c: c["multi_blocked"], [("Scholarship Eligibility", "Not Eligible")],
         "Added: clause 4.4.3.9 was missing from the doc."),
    Rule("F-MS-19", "GPA margin is Borderline", "ADVISORY: you are within 0.10 of the cut-off - verify your GPA",
         "Design (fuzzy boundary)", lambda c: c["gpa"]["Borderline"], [("flag:borderline", None)],
         "Advisory only; the official test is the exact comparison GPA >= T."),
    Rule("F-MS-00", "NOT eligible", "Scholarship is 0 %", H + "4.4",
         lambda c: NOT(E_ALL(c)), [("Scholarship Percentage", "No Scholarship")]),
]

S_FLAG_TEXT = {
    "tie_kept": "Tie-break: the award is kept (higher percentage, or a double tie where all tied students receive it).",
    "one_benefit": "Naval fee concession: you may avail only ONE benefit at a time.",
    "coursework_gpa": "Two-year MS/MPhil, 3rd semester: only the COURSEWORK GPA is considered.",
    "coursework_na": "MS/MPhil outside the 3rd semester: the handbook does not say which GPA applies; confirm with the Examination Office.",
    "borderline": "Your GPA is within 0.10 of the minimum - double-check the exact GPA (3.75 non-eng. / 3.60 eng.).",
}
LEVELS = ("Undergraduate", "Postgraduate")
PROGRAMS = ("Non-Engineering", "Engineering")
SEMESTERS = ("Regular", "Summer", "Final (Graduating)")
TIES = ("No tie", "Tied - my percentage is higher", "Tied - percentages are also equal", "Tied - my percentage is lower")


def assess_scholarship(gpa, position, level="Undergraduate", program="Non-Engineering",
                       semester_type="Regular", staff=False, existing_concession=False,
                       naval=False, regular_load=True, tie="No tie", other_scholarship=False,
                       keep_merit=True, ms_mphil_2yr=False, ms_semester=1):
    errs = []
    g = num(gpa, "GPA", 0, 4, errs)
    pos = num(position, "Position", 1, 100000, errs, integer=True)
    choice(level, "Level", LEVELS, errs); choice(program, "Programme", PROGRAMS, errs)
    choice(semester_type, "Semester type", SEMESTERS, errs); choice(tie, "Tie status", TIES, errs)
    if ms_mphil_2yr:
        if level != "Postgraduate":
            errs.append("MS/MPhil option is only valid for Postgraduate students.")
        num(ms_semester, "MS/MPhil semester", 1, 4, errs, integer=True)
    if errs:
        raise InputError(errs)

    T = T_ENG if program == "Engineering" else T_NON_ENG
    D = round(g - T, 6)
    ug = level == "Undergraduate"
    in_range = pos <= (3 if ug else 2)
    ctx = {
        "gpa": GPA.fuzzify(D), "pos": int(pos),  # compared with == in rules
        "ug": float(ug), "pg": float(not ug),
        "eng": float(program == "Engineering"), "noneng": float(program != "Engineering"),
        "in_range": float(in_range),
        "staff_conc": float(bool(staff and existing_concession)),
        "excl_sem": float(semester_type != "Regular"),
        "regular": float(bool(regular_load)),
        "tie_lost": float(tie == TIES[3]), "tie_kept": float(tie in TIES[1:3]),
        "naval": float(bool(naval)),
        "multi_blocked": float(bool(other_scholarship) and not keep_merit),
        "msm": float(bool(ms_mphil_2yr)), "sem3": float(int(ms_semester or 0) == 3),
    }
    res = infer(S_RULES, ctx, S_OUTPUTS)
    elig, pct = res.crisp["Scholarship Eligibility"], res.crisp["Scholarship Percentage"]
    snapped = min(AWARDS, key=lambda a: abs(a - pct)) if pct is not None else 0

    official = bool(g >= T and in_range and not ctx["staff_conc"] and not ctx["excl_sem"]
                    and regular_load and not ctx["tie_lost"] and not ctx["multi_blocked"])
    expected_amount = (UG_AWARD if ug else PG_AWARD).get(int(pos), 0) if official else 0
    return {
        "module": "scholarship",
        "inputs": {"gpa": g, "position": int(pos), "level": level, "program": program, "T": T, "D": D},
        "eligible": official, "percentage": snapped, "raw_percentage": pct,
        "elig_score": elig, "elig_label": S_ELIG.label(elig) if elig is not None else None,
        "policy_consistent": (snapped == expected_amount) and ((elig is not None and elig >= 50) == official),
        "decision": (f"ELIGIBLE - {snapped}% merit scholarship" if snapped else "NOT ELIGIBLE for a merit scholarship"),
        "gpa_margin": D, "gpa_label": GPA.label(D),
        "fuzzified": {"GPA margin": ctx["gpa"]},
        "advice": [S_FLAG_TEXT[k] for k in S_FLAG_TEXT if res.flags.get("flag:" + k, 0) > 0],
        "flags": {k[5:]: v for k, v in res.flags.items()},
        "reasons": _reasons(g, T, in_range, ug, ctx, tie, semester_type, regular_load),
        "fired": res.fired, "all_rules": res.all_rules, "agg": res.agg,
    }


def _reasons(g, T, in_range, ug, c, tie, sem, regular):
    r = []
    if g < T:
        r.append(f"GPA {g:.2f} is below the minimum {T:.2f} (4.4.3.3).")
    if not in_range:
        r.append(f"Position is outside the award range ({'top 3' if ug else 'top 2'}) (4.4.1/4.4.2).")
    if c["staff_conc"]:
        r.append("Staff member already availing a fee concession (4.4.3.7).")
    if c["excl_sem"]:
        r.append(f"No merit scholarship for the {sem} semester (4.4.3.8).")
    if not regular:
        r.append("Not following the regular course load / roadmap (4.4.3.5).")
    if c["tie_lost"]:
        r.append("Lost the tie-break on percentage (4.4.3.4).")
    if c["multi_blocked"]:
        r.append("Another scholarship/waiver is being availed instead (4.4.3.9).")
    return r


# =====================================================================  4. PLOTS

PALETTE = ["#2a6f97", "#e07a1f", "#3a9d5d", "#b5446e", "#7b6bb0"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("membership")
    ax.grid(alpha=.25)


def fig_variable(var, marker=None, vline=None, vline_label=None, title=None):
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    x = var.universe
    for i, t in enumerate(var.terms):
        ax.plot(x, var.mf(t, x), lw=2, color=PALETTE[i % 5], label=t)
    if marker is not None:
        ax.axvline(marker, color="k", ls="--", lw=1.2)
        ax.text(marker, 1.03, f" input = {marker:g}", fontsize=8, va="bottom")
    if vline is not None:
        ax.axvline(vline, color="#c0392b", lw=1.2)
        ax.text(vline, 0.5, f" {vline_label or vline}", color="#c0392b", fontsize=8, rotation=90, va="center")
    ax.set_xlabel(f"{var.name} {('(' + var.unit + ')') if var.unit else ''}")
    ax.set_title(title or var.name, fontsize=10, loc="left")
    _style(ax); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.28), frameon=False)
    fig.tight_layout()
    return fig


def fig_aggregate(var, agg, crisp, label=None):
    x, mu = agg
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for i, t in enumerate(var.terms):
        ax.plot(x, var.mf(t, x), lw=1, color=PALETTE[i % 5], alpha=.35)
    ax.fill_between(x, mu, color="#2a6f97", alpha=.35, label="aggregated output (MAX of clipped rule outputs)")
    ax.plot(x, mu, color="#2a6f97", lw=1.6)
    if crisp is not None:
        ax.axvline(crisp, color="#c0392b", lw=1.6)
        ax.text(crisp, 1.03, f" centroid = {crisp:.1f}", color="#c0392b", fontsize=8, va="bottom")
    ax.set_xlabel(var.name); ax.set_title(label or f"Aggregation & centroid - {var.name}", fontsize=10, loc="left")
    _style(ax); fig.tight_layout()
    return fig


def fig_surface(xs, ys, z, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.contourf(xs, ys, z, levels=np.linspace(0, 100, 21), cmap="RdYlGn_r")
    fig.colorbar(im, ax=ax, label="crisp output")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


# =====================================================================  5. EXAMPLES, TRACEABILITY

EXAMPLES = [
 ("A1", "attendance", "Comfortable attendance (88%)", dict(attendance="88", level="Undergraduate")),
 ("A2", "attendance", "Exactly on the 75% line", dict(attendance="75")),
 ("A3", "attendance", "Just short: 74%", dict(attendance="74")),
 ("A4", "attendance", "70% but a severe family bereavement", dict(attendance="70", unforeseen="9")),
 ("A5", "attendance", "Sports trial: BUHO approved, tutorials done, 78%", dict(attendance="78", clash=True, buho_approved=True, tutorial="95")),
 ("A6", "attendance", "Postgraduate at 72% asking for more relaxation", dict(attendance="72", level="Postgraduate", relaxation_request=True)),
 ("S1", "scholarship", "UG topper, 3.92 GPA, 1st position", dict(gpa="3.92", position="1")),
 ("S2", "scholarship", "Non-engineering student at exactly 3.75", dict(gpa="3.75", position="2")),
 ("S3", "scholarship", "Engineering student at 3.62 (passes only because T = 3.60)", dict(gpa="3.62", position="3", program="Engineering")),
 ("S4", "scholarship", "Non-engineering student at 3.74, 1st position", dict(gpa="3.74", position="1")),
 ("S5", "scholarship", "Summer-semester topper", dict(gpa="3.95", position="1", semester_type="Summer")),
 ("S6", "scholarship", "Staff member who already has a fee concession", dict(gpa="3.90", position="1", staff=True, existing_concession=True)),
 ("S7", "scholarship", "PG 3rd position (only top two awarded)", dict(gpa="3.90", position="3", level="Postgraduate")),
 ("S8", "scholarship", "Lost a tie on percentage", dict(gpa="3.90", position="1", tie="Tied - my percentage is lower")),
]

TRACEABILITY = [
 {
  "no": 1,
  "area": "Class Attendance",
  "text": "It shall be mandatory for students to attend at least 75% of the Total Contact Hours in a Subject/Course of Study failing which they will not be allowed to sit in the final examination.",
  "section": "1.27.1",
  "page_in_doc": "22",
  "colour": "#FF9494",
  "rules": [
   "F-ATT-01",
   "F-ATT-02",
   "F-ATT-03",
   "F-ATT-04",
   "F-ATT-11",
   "F-ATT-12"
  ]
 },
 {
  "no": 2,
  "area": "Class Attendance",
  "text": "The 25% relaxation in attendance is to cater for unforeseen situations like sickness, bereavement in the family, law and order situation, untoward incident etc.",
  "section": "1.27.1",
  "page_in_doc": "22",
  "colour": "#80FFC8",
  "rules": [
   "F-ATT-03",
   "F-ATT-05"
  ]
 },
 {
  "no": 3,
  "area": "Class Attendance",
  "text": "On no account, any shortfall in attendance shall be condoned. Attendance once marked shall not be changed.",
  "section": "1.27.1",
  "page_in_doc": "22",
  "colour": "#FF94F0",
  "rules": [
   "F-ATT-10"
  ]
 },
 {
  "no": 4,
  "area": "Class Attendance",
  "text": "Where class attendance clashes with a sports event or any other extra-curricular activity, prior approval of the BUHO shall be sought for participation in the sports event or the extra-curricular activity.",
  "section": "1.27.2",
  "page_in_doc": "22",
  "colour": "#A4FF80",
  "rules": [
   "F-ATT-06"
  ]
 },
 {
  "no": 5,
  "area": "Class Attendance",
  "text": "If BUHO approves such participation, tutorials shall be arranged to make up for the loss of academic activity.",
  "section": "1.27.2",
  "page_in_doc": "22",
  "colour": "#B294FF",
  "rules": [
   "F-ATT-06"
  ]
 },
 {
  "no": 6,
  "area": "Class Attendance",
  "text": "Only after the tutorials for the missed classes have been held that attendance for the missed classes shall be marked and credited to the student’s attendance record.",
  "section": "1.27.2",
  "page_in_doc": "22",
  "colour": "#FFED80",
  "rules": [
   "F-ATT-07",
   "F-ATT-08"
  ]
 },
 {
  "no": 7,
  "area": "Class Attendance",
  "text": "In the Post graduate programs, the scholars shall not be permitted any relaxation in attendance in the coursework classes, other than the permissible 25%.",
  "section": "1.27.3",
  "page_in_doc": "22",
  "colour": "#94D1FF",
  "rules": [
   "F-ATT-09"
  ]
 },
 {
  "no": 8,
  "area": "Merit Scholarship – Undergraduate",
  "text": "Top scorer students from each batch, achieving first three (03) position in their Semester Examination shall be awarded Merit Scholarship as per following:",
  "section": "4.4.1",
  "page_in_doc": "52",
  "colour": "#FFAD80",
  "rules": [
   "F-MS-01",
   "F-MS-02",
   "F-MS-03"
  ],
  "note": "INCOMPLETE in the Word doc: also add the full 70/40/30 (UG) and 70/40 (PG) lists from the handbook."
 },
 {
  "no": 9,
  "area": "Merit Scholarship – Postgraduate",
  "text": "Top scorer students from each batch, achieving first two (02) position in their Semester Examination shall be awarded Merit Scholarship as per following:",
  "section": "4.4.2",
  "page_in_doc": "52",
  "colour": "#94FFD5",
  "rules": [
   "F-MS-04",
   "F-MS-05"
  ],
  "note": "INCOMPLETE in the Word doc: also add the full 70/40/30 (UG) and 70/40 (PG) lists from the handbook."
 },
 {
  "no": 10,
  "area": "Merit Scholarship – GPA Eligibility",
  "text": "The students must achieve minimum GPA ≥ 3.75 in order to qualify for the award of Merit Scholarship for non-engineering programs & in case of engineering programs, student has to achieve minimum GPA ≥ 3.60 as per detail mentioned above (4.4.1 & 4.4.2).",
  "section": "4.4.3.3",
  "page_in_doc": "52",
  "colour": "#ED80FF",
  "rules": [
   "F-MS-06",
   "F-MS-07",
   "F-MS-08",
   "F-MS-09",
   "F-MS-19"
  ]
 },
 {
  "no": 11,
  "area": "Merit Scholarship – Tie",
  "text": "In case of a tie, the student with the higher percentage will get the Scholarship..",
  "section": "4.4.3.4",
  "page_in_doc": "52",
  "colour": "#F3FF94",
  "rules": [
   "F-MS-10",
   "F-MS-10b"
  ]
 },
 {
  "no": 12,
  "area": "Merit Scholarship – Naval Fee Concession",
  "text": "Student availing Naval fee concession will also be eligible for Merit scholarship as per policy, however they can only avail one benefit at a time",
  "section": "4.4.3.6",
  "page_in_doc": "53",
  "colour": "#80D1FF",
  "rules": [
   "F-MS-11"
  ]
 },
 {
  "no": 13,
  "area": "Merit Scholarship – Staff Fee Concession",
  "text": "Bahria University Faculty and administrative staff qualify for Merit scholarship shall not be considered for award of scholarship if availing already any fee concession.",
  "section": "4.4.3.7",
  "page_in_doc": "52",
  "colour": "#FF94BE",
  "rules": [
   "F-MS-12"
  ]
 },
 {
  "no": 14,
  "area": "Merit Scholarship – Semester Exclusions",
  "text": "Merit Scholarship shall not be provided for the Summer Semester, final semester (Graduating Batch).",
  "section": "4.4.3.8",
  "page_in_doc": "53",
  "colour": "#91FF80",
  "rules": [
   "F-MS-13"
  ]
 },
 {
  "no": 15,
  "area": "Merit Scholarship – MS/MPhil",
  "text": "For MS/ MPhil only for 02 years of program, for the purpose of award of Merit Scholarship, the GPA of course work shall only be considered in the 3rd semester.",
  "section": "4.4.3.10",
  "page_in_doc": "53",
  "colour": "#A094FF",
  "rules": [
   "F-MS-14",
   "F-MS-15"
  ]
 }
]


# =====================================================================  6. TEST SET (expected answers written by hand from the handbook)


TIE = ["No tie", "Tied - my percentage is higher", "Tied - percentages are also equal", "Tied - my percentage is lower"]
TE_ = {"error": True}
ok = lambda pct=None: {"eligible": True, **({"pct": pct} if pct is not None else {})}
no = lambda: {"eligible": False}

TEST_ATT = [  # id, kind, description, kwargs, expected
 ("A01", "normal", "85% attendance", dict(attendance=85), ok()),
 ("A02", "normal", "95% attendance", dict(attendance=95), ok()),
 ("A03", "normal", "100% attendance", dict(attendance=100), ok()),
 ("A04", "normal", "60% attendance", dict(attendance=60), no()),
 ("A05", "normal", "40% attendance", dict(attendance=40), no()),
 ("A06", "normal", "0% attendance", dict(attendance=0), no()),
 ("A07", "boundary", "exactly 75% (at least 75% is mandatory)", dict(attendance=75), ok()),
 ("A08", "boundary", "74.99%", dict(attendance=74.99), no()),
 ("A09", "boundary", "74%", dict(attendance=74), no()),
 ("A10", "boundary", "75.01%", dict(attendance=75.01), ok()),
 ("A11", "boundary", "76%", dict(attendance=76), ok()),
 ("A12", "boundary", "74.9999%", dict(attendance=74.9999), no()),
 ("A13", "normal", "65% (borderline-low)", dict(attendance=65), no()),
 ("A14", "normal", "70%, severe unforeseen situation: shortfall never condoned", dict(attendance=70, unforeseen=8), no()),
 ("A15", "normal", "80%, severe unforeseen situation: still eligible", dict(attendance=80, unforeseen=8), ok()),
 ("A16", "normal", "70%, BUHO-approved sports clash, tutorials complete: entered record stays 70%", dict(attendance=70, clash=True, buho_approved=True, tutorial=100), no()),
 ("A17", "normal", "Postgraduate 72% asking extra relaxation", dict(attendance=72, level="Postgraduate", relaxation_request=True), no()),
 ("A18", "boundary", "30 of 40 hours = 75%", dict(attendance=lambda: hours_to_percent(30, 40)), ok()),
 ("A19", "boundary", "29 of 40 hours = 72.5%", dict(attendance=lambda: hours_to_percent(29, 40)), no()),
 ("A20", "invalid", "attendance -5", dict(attendance=-5), TE_),
 ("A21", "invalid", "attendance 100.5", dict(attendance=100.5), TE_),
 ("A22", "invalid", "attendance 'abc'", dict(attendance="abc"), TE_),
 ("A23", "invalid", "attendance missing (None)", dict(attendance=None), TE_),
 ("A24", "invalid", "attendance NaN", dict(attendance=float("nan")), TE_),
 ("A25", "invalid", "unforeseen severity 11 (scale is 0-10)", dict(attendance=80, unforeseen=11), TE_),
 ("A26", "invalid", "attended hours 50 of 40", dict(attendance=lambda: hours_to_percent(50, 40)), TE_),
]
TEST_SCH = [
 ("S01", "normal", "UG non-eng GPA 3.90, 1st", dict(gpa=3.90, position=1), ok(70)),
 ("S02", "normal", "UG non-eng GPA 3.85, 2nd", dict(gpa=3.85, position=2), ok(40)),
 ("S03", "normal", "UG non-eng GPA 3.80, 3rd", dict(gpa=3.80, position=3), ok(30)),
 ("S04", "normal", "UG 4th position", dict(gpa=3.95, position=4), no()),
 ("S05", "normal", "PG GPA 3.90, 1st", dict(gpa=3.90, position=1, level="Postgraduate"), ok(70)),
 ("S06", "normal", "PG GPA 3.88, 2nd", dict(gpa=3.88, position=2, level="Postgraduate"), ok(40)),
 ("S07", "normal", "PG 3rd position (only top two)", dict(gpa=3.90, position=3, level="Postgraduate"), no()),
 ("S08", "boundary", "non-eng GPA exactly 3.75, 1st", dict(gpa=3.75, position=1), ok(70)),
 ("S09", "boundary", "non-eng GPA 3.74, 1st", dict(gpa=3.74, position=1), no()),
 ("S10", "boundary", "non-eng GPA 3.76, 2nd", dict(gpa=3.76, position=2), ok(40)),
 ("S11", "boundary", "engineering GPA exactly 3.60, 1st", dict(gpa=3.60, position=1, program="Engineering"), ok(70)),
 ("S12", "boundary", "engineering GPA 3.59, 1st", dict(gpa=3.59, position=1, program="Engineering"), no()),
 ("S13", "boundary", "engineering GPA 3.70, 2nd", dict(gpa=3.70, position=2, program="Engineering"), ok(40)),
 ("S14", "boundary", "non-eng GPA 3.70, 2nd (same GPA fails for non-eng)", dict(gpa=3.70, position=2), no()),
 ("S15", "boundary", "non-eng GPA 3.7499 (strict reading of >= 3.75)", dict(gpa=3.7499, position=1), no()),
 ("S16", "normal", "Summer semester", dict(gpa=3.9, position=1, semester_type="Summer"), no()),
 ("S17", "normal", "Final (graduating) semester", dict(gpa=3.9, position=1, semester_type="Final (Graduating)"), no()),
 ("S18", "normal", "Staff with existing fee concession", dict(gpa=3.9, position=1, staff=True, existing_concession=True), no()),
 ("S19", "normal", "Staff WITHOUT fee concession", dict(gpa=3.9, position=1, staff=True), ok(70)),
 ("S20", "normal", "Naval fee concession holder, 1st (one benefit only)", dict(gpa=3.9, position=1, naval=True), ok(70)),
 ("S21", "normal", "Tie, my percentage lower", dict(gpa=3.9, position=1, tie=TIE[3]), no()),
 ("S22", "normal", "Tie, my percentage higher", dict(gpa=3.9, position=1, tie=TIE[1]), ok(70)),
 ("S23", "normal", "Double tie: all tied students get it", dict(gpa=3.9, position=2, tie=TIE[2]), ok(40)),
 ("S24", "normal", "Not on regular course load", dict(gpa=3.9, position=1, regular_load=False), no()),
 ("S25", "normal", "Holds another scholarship and keeps that one", dict(gpa=3.9, position=1, other_scholarship=True, keep_merit=False), no()),
 ("S26", "normal", "Two-year MS/MPhil, 3rd semester, 1st", dict(gpa=3.9, position=1, level="Postgraduate", ms_mphil_2yr=True, ms_semester=3), ok(70)),
 ("S27", "invalid", "GPA 4.5", dict(gpa=4.5, position=1), TE_),
 ("S28", "invalid", "GPA -1", dict(gpa=-1, position=1), TE_),
 ("S29", "invalid", "position 0", dict(gpa=3.9, position=0), TE_),
 ("S30", "invalid", "position 1.5", dict(gpa=3.9, position=1.5), TE_),
 ("S31", "invalid", "GPA missing", dict(gpa=None, position=1), TE_),
 ("S32", "invalid", "MS/MPhil flag for an undergraduate", dict(gpa=3.9, position=1, ms_mphil_2yr=True, ms_semester=3), TE_),
 ("S33", "invalid", "unknown programme 'Medical'", dict(gpa=3.9, position=1, program="Medical"), TE_),
 ("S34", "invalid", "MS/MPhil semester 7", dict(gpa=3.9, position=1, level="Postgraduate", ms_mphil_2yr=True, ms_semester=7), TE_),
]


def run_tests():
    rows = []
    for mod, fn, cases in (("Attendance", assess_attendance, TEST_ATT), ("Scholarship", assess_scholarship, TEST_SCH)):
        for cid, kind, desc, kw, exp in cases:
            try:
                r = fn(**{k: (v() if callable(v) else v) for k, v in kw.items()})
                if exp.get("error"):
                    got, passed = "accepted (no error)", False
                else:
                    got = ("ELIGIBLE" + (f" {r['percentage']}%" if "percentage" in r and r["eligible"] else "")) if r["eligible"] else "NOT ELIGIBLE"
                    passed = (r["eligible"] == exp["eligible"] and ("pct" not in exp or r["percentage"] == exp["pct"])
                              and r["policy_consistent"])
            except InputError as e:
                got, passed = "error: " + "; ".join(e.messages), bool(exp.get("error"))
            want = "reject input" if exp.get("error") else ("ELIGIBLE" + (f" {exp['pct']}%" if "pct" in exp else "") if exp["eligible"] else "NOT ELIGIBLE")
            rows.append((cid, kind, desc, want, got, "PASS" if passed else "FAIL"))
    for r in rows:
        print(f"{r[0]:<4}{r[1]:<9}{r[2][:58]:<60}{r[3]:<18}{r[4][:40]:<42}{r[5]}")
    p = sum(r[5] == "PASS" for r in rows)
    print(f"\nTotal {len(rows)} queries, {p} passed, {len(rows) - p} failed, accuracy {100 * p / len(rows):.1f}%")
    for kind in ("normal", "boundary", "invalid"):
        k = [r for r in rows if r[1] == kind]
        print(f"  {kind}: {sum(r[5] == 'PASS' for r in k)}/{len(k)}")
    return rows


def print_samples():
    for code, mod, title, kw in EXAMPLES:
        kw = {k: (float(v) if isinstance(v, str) and k in ("attendance", "unforeseen", "tutorial", "gpa") else v) for k, v in kw.items()}
        if "position" in kw:
            kw["position"] = int(kw["position"])
        r = (assess_attendance if mod == "attendance" else assess_scholarship)(**kw)
        print(f"\n=== {code}: {title}\nInput: {kw}")
        for f in r["fired"]:
            print(f"  fired {f['id']:<9} strength {f['strength']:.2f}  IF {f['if']} THEN {f['then']}  [{f['source']}]")
        if mod == "attendance":
            print(f"Output: risk {r['risk_score']:.1f} ({r['risk_label']}), eligibility score {r['elig_score']:.1f} -> {r['decision']}")
        else:
            print(f"Output: score {r['elig_score']:.1f}, centroid {r['raw_percentage']:.1f}% -> {r['decision']} (GPA margin {r['gpa_margin']:+.2f}, {r['gpa_label']})")
        for a in r["advice"]:
            print("  advice:", a)


# =====================================================================  7. GUI (Streamlit)
def gui():
    import pandas as pd
    import streamlit as st
    # --- compatibility with old/new Streamlit and pandas versions ---
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
    WIDE = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
    st.set_page_config(page_title="Bahria Handbook Fuzzy Advisor", page_icon="🎓", layout="wide")
    st.markdown("""<style>
    .block-container{padding-top:1.6rem;max-width:1250px}
    .verdict{padding:14px 18px;border-radius:10px;font-size:1.25rem;font-weight:650;margin-bottom:.6rem}
    .ok{background:#e3f4e8;color:#14532d;border:1px solid #9fd5b0}.bad{background:#fbe5e3;color:#7f1d1d;border:1px solid #e7a8a3}
    .small{color:#555;font-size:.85rem}
    </style>""", unsafe_allow_html=True)

    DEFAULTS = {
        "att_pct": "85", "att_unf": "0", "att_tut": "0", "att_lvl": "Undergraduate", "att_buho": False,
        "att_clash": False, "att_relax": False, "att_mode": "Percentage", "att_hrs": "30", "att_tot": "40",
        "s_gpa": "3.85", "s_pos": "1", "s_lvl": "Undergraduate", "s_prog": "Non-Engineering",
        "s_sem": "Regular", "s_staff": False, "s_conc": False, "s_naval": False, "s_reg": True,
        "s_tie": "No tie", "s_other": False, "s_keep": True, "s_ms": False, "s_mssem": "1",
    }
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    KEYMAP = {"attendance": "att_pct", "unforeseen": "att_unf", "tutorial": "att_tut", "level": None,
              "buho_approved": "att_buho", "clash": "att_clash", "relaxation_request": "att_relax",
              "gpa": "s_gpa", "position": "s_pos", "program": "s_prog", "semester_type": "s_sem",
              "staff": "s_staff", "existing_concession": "s_conc", "naval": "s_naval", "tie": "s_tie"}


    def load_example():
        label = st.session_state.ex
        if label == "-":
            return
        ex = next(e for e in EXAMPLES if e[0] + " - " + e[2] == label)
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["module"] = "Class attendance (s.1.27)" if ex[1] == "attendance" else "Merit scholarship (s.4.4)"
        for k, v in ex[3].items():
            key = KEYMAP.get(k) or ("att_lvl" if ex[1] == "attendance" else "s_lvl")
            st.session_state[key] = v


    def rules_table(fired):
        df = pd.DataFrame(fired)
        if df.empty:
            return df
        df = df.rename(columns={"id": "Rule", "if": "IF", "then": "THEN", "source": "Handbook source", "strength": "Firing strength"})
        return df[["Rule", "IF", "THEN", "Firing strength", "Handbook source"]]


    def show_rules(fired, all_rules):
        st.markdown("##### Rules fired (with firing strength)")
        st.dataframe(rules_table(fired), hide_index=True, **WIDE, column_config={
            "Firing strength": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        notes = [f"**{r['id']}** - {r['note']}" for r in fired if r["note"]]
        if notes:
            with st.expander("Design notes on the fired rules"):
                st.markdown("\n\n".join(notes))
        with st.expander(f"All {len(all_rules)} rules (including those that did not fire)"):
            st.dataframe(rules_table(all_rules), hide_index=True, **WIDE)


    def show_fuzzification(fz):
        st.markdown("##### Fuzzification (membership degree of each input in each linguistic term)")
        cols = st.columns(len(fz))
        for c, (name, mu) in zip(cols, fz.items()):
            c.caption(name)
            c.bar_chart(pd.Series(mu), height=190, color="#2a6f97")


    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.title("🎓 Handbook Fuzzy Advisor")
        st.caption("Bahria University Student Handbook · KRR Assignment 2")
        module = st.radio("Policy area", ["Class attendance (s.1.27)", "Merit scholarship (s.4.4)"], key="module")
        st.selectbox("Load a worked example", ["-"] + [e[0] + " - " + e[2] for e in EXAMPLES], key="ex",
                     on_change=load_example,
                     help="Fills the form with one of the sample queries used in the report.")
        st.divider()
        st.markdown("<div class='small'>Mamdani inference · MIN/MAX operators · centroid defuzzification. "
                    "The handbook thresholds (75 %, 3.75 / 3.60) are binding: fuzziness grades the <i>risk and advice</i> around them, "
                    "never the official decision.</div>", unsafe_allow_html=True)

    tab_main, tab_mf, tab_surf, tab_rules = st.tabs(["Assessment", "Membership functions", "Output surface", "Rule base & traceability"])

    # ------------------------------------------------------------------ assessment
    with tab_main:
        left, right = st.columns([5, 7], gap="large")
        if module.startswith("Class"):
            with left:
                st.radio("Enter attendance as", ["Percentage", "Hours"], key="att_mode", horizontal=True)
                if st.session_state.att_mode == "Percentage":
                    st.text_input("Attendance (%)", key="att_pct", help="0-100. Handbook 1.27.1: at least 75 % is mandatory.")
                    att_value = st.session_state.att_pct
                else:
                    c1, c2 = st.columns(2)
                    c1.text_input("Hours attended", key="att_hrs"); c2.text_input("Total contact hours", key="att_tot")
                    try:
                        att_value = hours_to_percent(st.session_state.att_hrs, st.session_state.att_tot)
                        st.caption(f"= {att_value:.2f} %")
                    except InputError as e:
                        att_value = None; st.caption("; ".join(e.messages))
                st.selectbox("Level", LEVELS, key="att_lvl")
                st.text_input("Severity of unforeseen situation (0-10)", key="att_unf", help="0 = none, 10 = very severe (illness, bereavement, law & order…). Modelling scale, not an official rating.")
                st.markdown("**Sports / extracurricular conflict (1.27.2)**")
                st.checkbox("Classes clashed with a sports / extracurricular event", key="att_clash")
                st.checkbox("BUHO gave PRIOR approval", key="att_buho")
                st.text_input("Tutorials completed for the missed classes (%)", key="att_tut")
                st.checkbox("I am asking for relaxation beyond the 25 % (matters for PG, 1.27.3)", key="att_relax")
            with right:
                try:
                    r = assess_attendance(att_value, st.session_state.att_unf, st.session_state.att_tut,
                                          st.session_state.att_buho, st.session_state.att_clash,
                                          st.session_state.att_lvl, st.session_state.att_relax)
                except InputError as e:
                    st.error("Please correct the input:\n\n" + "\n".join(f"- {m}" for m in e.messages))
                    st.stop()
                st.markdown(f"<div class='verdict {'ok' if r['eligible'] else 'bad'}'>{r['decision']}</div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Attendance risk (crisp)", f"{r['risk_score']:.1f} / 100", r["risk_label"], delta_color="off")
                m2.metric("Eligibility score (crisp)", f"{r['elig_score']:.1f} / 100", r["elig_label"], delta_color="off")
                m3.metric("Margin vs 75 %", f"{r['margin']:+.2f} pts")
                st.markdown("##### Linguistic interpretation")
                fz = r["fuzzified"]["Attendance"]
                top = sorted(((v, k) for k, v in fz.items() if v > 0), reverse=True)
                terms = " and ".join(f"**{k}** ({v:.2f})" for v, k in top)
                side = "meets" if r["eligible"] else "does not meet"
                st.write(f"Attendance of {r['inputs']['attendance']:.2f} % is {terms}. Risk of being barred is **{r['risk_label']}**. "
                         f"The student {side} the handbook minimum of 75 % (1.27.1).")
                for a in r["advice"]:
                    st.info(a)
                if not r["policy_consistent"]:
                    st.warning("Fuzzy score and handbook gate disagree - the handbook decision above is the one that applies.")
            st.divider()
            show_rules(r["fired"], r["all_rules"])
            show_fuzzification(r["fuzzified"])
            p1, p2 = st.columns(2)
            p1.pyplot(fig_aggregate(RISK, r["agg"]["Attendance Risk"], r["risk_score"]))
            p2.pyplot(fig_aggregate(A_ELIG, r["agg"]["Examination Eligibility"], r["elig_score"]))
        else:
            with left:
                st.selectbox("Level", LEVELS, key="s_lvl")
                st.selectbox("Programme", PROGRAMS, key="s_prog", help="Engineering minimum GPA 3.60, others 3.75 (4.4.3.3).")
                st.text_input("Previous-semester GPA (0-4)", key="s_gpa")
                st.text_input("Position in batch (1 = topper)", key="s_pos", help="UG: top 3 awarded (70/40/30). PG: top 2 (70/40).")
                st.selectbox("Semester type", SEMESTERS, key="s_sem")
                st.selectbox("Tie on this position?", TIES, key="s_tie")
                st.checkbox("Following the regular course load / roadmap (4.4.3.5)", key="s_reg")
                c1, c2 = st.columns(2)
                c1.checkbox("BU faculty / staff", key="s_staff"); c2.checkbox("Already has a fee concession", key="s_conc")
                st.checkbox("Naval fee concession holder", key="s_naval")
                st.checkbox("Also receives another scholarship / waiver", key="s_other")
                if st.session_state.s_other:
                    st.checkbox("…and chooses the MERIT award (4.4.3.9)", key="s_keep")
                st.checkbox("Two-year MS / MPhil programme", key="s_ms")
                if st.session_state.s_ms:
                    st.text_input("MS/MPhil semester (1-4)", key="s_mssem")
            with right:
                try:
                    r = assess_scholarship(st.session_state.s_gpa, st.session_state.s_pos, st.session_state.s_lvl,
                                           st.session_state.s_prog, st.session_state.s_sem, st.session_state.s_staff,
                                           st.session_state.s_conc, st.session_state.s_naval, st.session_state.s_reg,
                                           st.session_state.s_tie, st.session_state.s_other, st.session_state.s_keep,
                                           st.session_state.s_ms, st.session_state.s_mssem)
                except InputError as e:
                    st.error("Please correct the input:\n\n" + "\n".join(f"- {m}" for m in e.messages))
                    st.stop()
                st.markdown(f"<div class='verdict {'ok' if r['eligible'] else 'bad'}'>{r['decision']}</div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Scholarship % (centroid → award)", f"{r['raw_percentage']:.1f} → {r['percentage']}%")
                m2.metric("Eligibility score (crisp)", f"{r['elig_score']:.1f} / 100", r["elig_label"], delta_color="off")
                m3.metric(f"GPA margin (GPA - {r['inputs']['T']:.2f})", f"{r['gpa_margin']:+.2f}", r["gpa_label"], delta_color="off")
                st.markdown("##### Linguistic interpretation")
                fz = r["fuzzified"]["GPA margin"]
                terms = " and ".join(f"**{k}** ({v:.2f})" for k, v in fz.items() if v > 0)
                st.write(f"GPA {r['inputs']['gpa']:.2f} vs required {r['inputs']['T']:.2f} is {terms}. "
                         + ("All handbook conditions are satisfied." if r["eligible"] else "Not awarded because: " + " ".join(r["reasons"])))
                for a in r["advice"]:
                    st.info(a)
                if not r["policy_consistent"]:
                    st.warning("Fuzzy output and handbook check disagree - the handbook decision applies.")
            st.divider()
            show_rules(r["fired"], r["all_rules"])
            show_fuzzification(r["fuzzified"])
            p1, p2 = st.columns(2)
            p1.pyplot(fig_aggregate(S_ELIG, r["agg"]["Scholarship Eligibility"], r["elig_score"]))
            p2.pyplot(fig_aggregate(PCT, r["agg"]["Scholarship Percentage"], r["raw_percentage"],
                                    f"Scholarship %: centroid {r['raw_percentage']:.1f} → authorised {r['percentage']}%"))

    # ------------------------------------------------------------------ membership functions
    with tab_mf:
        st.caption("Dashed line = the current input (if any). Red line = the handbook threshold.")
        c1, c2 = st.columns(2)
        c1.pyplot(fig_variable(ATT, vline=75, vline_label="75 % (1.27.1)"))
        c2.pyplot(fig_variable(UNF)); c1.pyplot(fig_variable(TUT))
        c2.pyplot(fig_variable(RISK)); c1.pyplot(fig_variable(A_ELIG, title="Examination / Scholarship Eligibility"))
        c2.pyplot(fig_variable(GPA, vline=0, vline_label="D = 0 (GPA = T)")); c1.pyplot(fig_variable(PCT))

    # ------------------------------------------------------------------ surface
    @st.cache_data(show_spinner="Computing surface…")
    def surface():
        xs = np.linspace(0, 100, 26); ys = np.linspace(0, 10, 11)
        z = np.array([[assess_attendance(x, y)["risk_score"] for x in xs] for y in ys])
        return xs, ys, z


    with tab_surf:
        xs, ys, z = surface()
        st.pyplot(fig_surface(xs, ys, z, "Attendance (%)", "Unforeseen severity (0-10)", "Attendance Risk (centroid)"))
        st.caption("Risk rises sharply as attendance falls through the 60-75 % band, and a severe situation pushes it up inside the borderline band (F-ATT-05).")

    # ------------------------------------------------------------------ rules
    with tab_rules:
        st.markdown("#### Rule base")
        allr = [("Attendance", r) for r in A_RULES] + [("Scholarship", r) for r in S_RULES]
        st.dataframe(pd.DataFrame([{"Rule": r.id, "Area": a, "IF": r.if_text, "THEN": r.then_text, "Source": r.source, "Note": r.note}
                                   for a, r in allr]), hide_index=True, **WIDE)
        st.markdown("#### Traceability (colours match the highlighted handbook and the report)")
        tr = TRACEABILITY
        df = pd.DataFrame([{"#": t["no"], "Colour": t["colour"], "Section": t["section"], "Original text": t["text"],
                            "Fuzzy rules": ", ".join(t["rules"])} for t in tr])
        st.dataframe((df.style.map if hasattr(df.style, "map") else df.style.applymap)(lambda v: f"background-color:{v};color:{v}", subset=["Colour"]), hide_index=True, **WIDE)


def _in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if all(r[5] == "PASS" for r in run_tests()) else 1)
    elif "--samples" in sys.argv:
        print_samples()
    elif _in_streamlit():
        gui()
    else:
        print(__doc__)
elif _in_streamlit():
    gui()
