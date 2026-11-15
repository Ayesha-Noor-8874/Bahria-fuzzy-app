# Bahria University Handbook – Fuzzy Inference System (single file)

One file, `bahria_fuzzy_app.py`, contains the fuzzy engine, both rule bases, the Streamlit GUI, the 60-query test set and the 14 worked examples.

| Area | Handbook section | What it answers |
|---|---|---|
| Class attendance | **1.27** | Am I eligible for the final exam? How risky is my attendance? |
| Merit scholarship for batch toppers | **4.4** | Do I get the 70 / 40 / 30 % merit scholarship? |

## 1. Run in VS Code
```bash
pip install numpy pandas matplotlib streamlit
streamlit run bahria_fuzzy_app.py        # opens http://localhost:8501
```
Other commands (no GUI needed):
```bash
python bahria_fuzzy_app.py --test        # runs the 60 test queries, prints pass/fail and accuracy
python bahria_fuzzy_app.py --samples     # prints the 14 worked examples (inputs, rules fired, outputs)
```

## 2. Run in Google Colab
Upload `bahria_fuzzy_app.py`, then:

**a) Tests, examples and plots (no GUI):**
```python
!pip -q install numpy pandas matplotlib
!python bahria_fuzzy_app.py --test
from bahria_fuzzy_app import *            # importing does NOT start the GUI
r = assess_attendance(74, unforeseen=0)
print(r["decision"], r["risk_score"], [f["id"] for f in r["fired"]])
fig_variable(ATT, vline=75, vline_label="75 %"); plt.show()
r = assess_scholarship(3.75, 1, program="Non-Engineering")
fig_aggregate(PCT, r["agg"]["Scholarship Percentage"], r["raw_percentage"]); plt.show()
```

**b) The Streamlit GUI (needs a tunnel because Colab has no browser):**
```python
!pip -q install streamlit
!streamlit run bahria_fuzzy_app.py --server.port 8501 --server.headless true &>/content/log.txt &
!curl -s ipv4.icanhazip.com        # copy this IP - it is the tunnel "password"
!npx -y localtunnel --port 8501    # open the printed URL and paste the IP when asked
```

## 3. Using the GUI
1. Choose a **policy area** in the sidebar, or load one of the **14 worked examples**.
2. Fill the form. Numeric boxes are free text on purpose, so invalid input (`abc`, `101`, empty) gives a clear error.
3. You get: the **verdict**, the **crisp outputs** (centroids), the **linguistic interpretation**, advice, the **rules fired with firing strengths and handbook source**, fuzzification bars and the aggregated output with the centroid line.
4. Tabs: *Membership functions*, *Output surface* (risk vs attendance × unforeseen severity), *Rule base & traceability*.

## 4. Design summary
* **Inference:** Mamdani. AND = min, OR = max, implication = min, aggregation = max, centroid defuzzification.
* **Binding thresholds:** 75 % attendance and GPA ≥ 3.75 (non-engineering) / 3.60 (engineering) are crisp in the handbook, so they are compared exactly (no rounding). Fuzzy sets around them grade *risk, closeness and advice*; they never override the official decision. Every run checks the fuzzy result against the handbook gate (`policy_consistent`).
* **GPA margin:** GPA is fuzzified as D = GPA − T so one set of terms serves both programmes.
* **Scholarship %:** centroid of the aggregated output, then snapped to the authorised awards {0, 30, 40, 70}.
* **Rule IDs:** `F-ATT-01…12` (attendance), `F-MS-00…19` (scholarship); each rule shows its handbook clause. Rules marked ADVISORY give advice flags, not Mamdani outputs.
* Rules **F-MS-16, 17, 18, 19** and the extra consequence in **F-ATT-10** were added to complete the rule base (handbook 4.4.1/4.4.2 range, 4.4.3.5, 4.4.3.9, fuzzy boundary advice). Update your report's rule table to match.

## 5. File map (inside `bahria_fuzzy_app.py`)
1. Engine (membership functions, rules, Mamdani inference, input validation)
2. Module 1 – attendance (variables, MFs, rules, `assess_attendance`)
3. Module 2 – merit scholarship (variables, MFs, rules, `assess_scholarship`)
4. Plot helpers  5. Examples and traceability data  6. Test set and `run_tests()`  7. GUI (`gui()`)

## 6. Known limitations
* The handbook is silent on rounding, so values are compared exactly as entered (74.99 % fails; GPA 3.7499 fails).
* Position and tie status are **inputs**; the system does not rank students.
* 4.4.3.10 (MS/MPhil coursework GPA in the 3rd semester) is ambiguous; outside semester 3 the standard GPA test applies with a warning.
* The unforeseen-situation scale (0–10) and tutorial % are modelling scales, not official quantities. An unforeseen situation never changes eligibility (1.27.1: no shortfall is condoned).

## 7. Running on any computer
Put these files in one folder: `bahria_fuzzy_app.py`, `requirements.txt`, `run_windows.bat`, `run_mac_linux.sh`.
* **Windows:** install Python 3.9+ from python.org (tick *Add Python to PATH*), then double-click `run_windows.bat`.
* **Mac / Linux:** `bash run_mac_linux.sh`
* **Manual:** `pip install -r requirements.txt` then `python -m streamlit run bahria_fuzzy_app.py`

The launchers create a private `.venv`, so the app does not depend on what is already installed on that computer. The code also adapts to older/newer Streamlit and pandas versions. If the browser does not open, go to http://localhost:8501.
