import pandas as pd
import streamlit as st

from scoring import build_lookup, classify_score, derive_inputs, load_artifacts, score_dataset, validate_inputs

st.set_page_config(page_title="Behavioral Credit Scorecard", layout="wide")

# ---- Variable documentation: keys are lower-case variable names from rules["inputs"] ----
# Add one entry per variable; anything missing falls back to the label and type in rules.
VARIABLE_DOCS = {
    "grade": {
        "description": "Risk grade assigned to the borrower by lending Club",
        "dtype": "Text",
        "values": "A = low-risk borrower, B = slightly riskier, C = moderate risk, and so on down to G = highest risk",
    },
    "term": {
        "description": "Loan repayment period",
        "dtype": "Text",
        "values": "36 Months = shorter loan, 60 Months = longer loan (generally riskier)",
    },
    "purpose": {
        "description": "Borrower's stated reason for the loan",
        "dtype": "Text",
        "values": "Credit Card, Car, Home Improvement, Major Purchase, Debt Consolidation, Vacation, House, Medical, Wedding, Moving, Renewable Energy, Small Business, Educational, Other",
    },
    "verification_status": {
        "description": "Whether the borrower's income was verified",
        "dtype": "Text",
        "values": "Not Verified = income not checked, Source Verified = income source confirmed, Verified = income and source fully verified",
    },
    "initial_list_status": {
        "description": "How the loan was first offered to investors",
        "dtype": "Text",
        "values": "w = whole loan, offered in full to a single large investor; f = fractional loan, offered in small pieces to many individual investors",
    },
    "loan_amnt": {
        "description": "Amount of the loan",
        "dtype": "Float",
        "values": "Higher means a larger loan and generally higher risk",
    },
    "annual_inc": {
        "description": "Borrower's annual income",
        "dtype": "Float",
        "values": "Higher is better, as the borrower has more capacity to repay",
    },
    "tot_coll_amt": {
        "description": "Total amount ever sent to collections",
        "dtype": "Float",
        "values": "Higher is worse; 0 means no collection history",
    },
    "total_rec_int": {
        "description": "Total interest received from the borrower so far",
        "dtype": "Float",
        "values": "Higher is better, as the borrower has been paying consistently",
    },
    "inq_last_6mths": {
        "description": "Number of credit inquiries in the last 6 months",
        "dtype": "Integer",
        "values": "Higher is worse, as many inquiries suggest the borrower is seeking more credit",
    },
    "credit_history_length": {
        "description": "Length of credit history in months",
        "dtype": "Integer",
        "values": "Higher is better, as a longer history means a more established borrower",
    },
    "revol_util": {
        "description": "Share of available revolving credit in use (%)",
        "dtype": "Float",
        "values": "Higher is worse, as the borrower is relying heavily on available credit",
    },
    "dti": {
        "description": "Debt-to-income ratio",
        "dtype": "Float",
        "values": "Monthly debt payments divided by monthly income; higher is worse, as more income goes to debt",
    },
}

DISPLAY_LABELS = {
    "three_year": "36 Months",
    "five_year": "60 Months",
    "credit_card": "Credit Card",
    # add more: "raw_value": "Display Text",
}


def display_label(v):
    return DISPLAY_LABELS.get(v, str(v).replace("_", " ").title())

def variable_docs_table(inputs, sc):
    rows = []
    for var, r in inputs.items():
        doc = VARIABLE_DOCS.get(var.lower(), {})
        if r["type"] == "categorical":
            default_dtype = "Text"
            default_values = ", ".join(map(str, sc["variables"][var]["bins"]))
        else:
            default_dtype = "Integer" if r.get("integer") else "Float"
            default_values = "Numeric value"
        rows.append({
            "Variable": var,
            "Description": doc.get("description", r["label"]),
            "Data type": doc.get("dtype", default_dtype),
            "What the values mean": doc.get("values", default_values),
        })
    return pd.DataFrame(rows)


def score_color(score, cutoff, thresholds):
    """Red below the cutoff (deeper red = riskier); amber to green above it (greener = safer)."""
    clamp = lambda x: max(0.0, min(1.0, x))
    lo = min(thresholds) if thresholds and min(thresholds) < cutoff else cutoff - 100
    hi = max(thresholds) if thresholds and max(thresholds) > cutoff else cutoff + 100
    if score < cutoff:
        t = clamp((cutoff - score) / (cutoff - lo))
        return f"hsl(0, 78%, {50 - 22 * t:.0f}%)"
    t = clamp((score - cutoff) / (hi - cutoff))
    return f"hsl({35 + 95 * t:.0f}, 70%, 38%)"


def score_circle(score, category, color):
    return f"""
    <div style="display:flex;justify-content:center;">
      <div style="width:190px;height:190px;border-radius:50%;background:{color};color:#fff;
                  display:flex;flex-direction:column;align-items:center;justify-content:center;
                  box-shadow:0 4px 14px rgba(0,0,0,.25);">
        <div style="font-size:52px;font-weight:700;line-height:1;">{score}</div>
        <div style="font-size:14px;margin-top:6px;opacity:.95;">{category}</div>
      </div>
    </div>
    """


@st.cache_resource
def artifacts():
    sc, rules = load_artifacts()
    return sc, rules, build_lookup(sc)


sc, rules, lookup = artifacts()
inputs = rules["inputs"]
bands = rules["risk"]["bands"]
cutoff = rules["risk"]["default_flag_below"]
thresholds = [b["below"] for b in bands if b["below"] is not None]

# ---- Header ----
st.title("Behavioral Credit Scorecard")
st.write("A behavioral scorecard turns a borrower's previous loan experience, income and repayment history data into a single risk score.")
st.write("Its objective is to identify existing borrowers who are likely to default so that action can be taken early.")
st.caption(f"Scorecard version {sc['version']}. Scores below {cutoff} are flagged as potential default.")

# ---- Development process ----
st.header("How the scorecard was built")
st.markdown(
    """
1. **Collect history:** gather past loans along with what eventually happened to each one (repaid or defaulted).
2. **Clean the data:** fix errors, handle missing values and define what counts as a default.
3. **Group values into bins:** put similar values together, for example low, medium and high income.
4. **Measure risk in each group:** check how often each group defaulted compared to the others.
5. **Select the best variables:** keep those that predict default well and drop those that repeat each other.
6. **Fit the model:** a statistical model (logistic regression) learns how much each variable matters.
7. **Convert to points:** model output is turned into an easy-to-read score, where higher means safer.
8. **Test and monitor:** check on data the model has not seen and keep tracking it over time.
"""
)

# Variables
st.header("Final Variables")
st.dataframe(variable_docs_table(inputs, sc), hide_index=True, width="stretch")

# Score Calculation
LABEL_HINTS = {
    "loan_amnt": " (e.g. Min-1000, Max-35000)",
    "annual_inc": " (e.g. 20000, 30000)",
    "tot_coll_amt": " (e.g. 0, 250)",
    "total_rec_int": " (e.g. 800, 2500.50)",
    "inq_last_6mths": " (e.g. 0, 2)",
    "credit_history_length": " (months, e.g. 60, 180)",
    "revol_util": " (%) (e.g. 35, 72.50)",
    "dti": " (%) (e.g. 5, 10.20)",
}


def field_label(var, r):
    return r["label"] + LABEL_HINTS.get(var, "")

st.header("Score a borrower")
with st.form("borrower"):
    cols = st.columns(3)
    record = {}
    for i, (var, r) in enumerate(inputs.items()):
        with cols[i % 3]:
            if r["type"] == "categorical":
                record[var] = st.selectbox(r["label"], list(sc["variables"][var]["bins"]),
                           index=None, placeholder="Select", format_func=display_label)
            else:
                integer = bool(r.get("integer"))
                record[var] = st.number_input(
                    field_label(var, r), value=None, step=1 if integer else 0.01,
                    format="%d" if integer else "%.2f",
                    help="Leave blank if not available." if r.get("allow_missing") else None,
                )
    submitted = st.form_submit_button("Calculate score")

if submitted:
    errors, warnings = validate_inputs(record, rules, sc)
    for w in warnings:
        st.warning(w)
    for e in errors:
        st.error(e)

    if not errors:
        row = derive_inputs(record, rules)
        try:
            scores, parts = score_dataset([row], lookup, return_breakdown=True)
        except (ValueError, KeyError) as e:
            st.error(f"Could not score this borrower: {e}")
            st.stop()

        score = round(scores[0])
        category = classify_score(score, bands)
        flagged = score < cutoff
        color = score_color(score, cutoff, thresholds)

        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            st.markdown(score_circle(score, category, color), unsafe_allow_html=True)
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Risk category", category)
            m2.metric("Flag", "Potential default" if flagged else "OK")
            if flagged:
                st.error(f"Potential default: score is below {cutoff}.")
            else:
                st.success(f"Not flagged: score is at or above {cutoff}.")

        def fmt(v):
            return "Missing" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

        table = pd.DataFrame({
            "Variable": ["Offset"] + list(parts[0]),
            "Value": [""] + [display_label(row[v]) if isinstance(row[v], str) else fmt(row[v]) for v in parts[0]],   # <- replaced line
            "Points": [lookup["offset"]] + list(parts[0].values()),
        })
        st.subheader("Points breakdown")
        st.dataframe(table.style.format({"Points": "{:.2f}"}), hide_index=True, width="stretch")

        with st.expander("Risk bands"):
            rows, prev = [], None
            for b in bands:
                if prev is None:
                    rng = f"below {b['below']}"
                elif b["below"] is None:
                    rng = f"{prev} and above"
                else:
                    rng = f"{prev} to {b['below'] - 1}"
                rows.append({"Category": b["name"], "Score": rng, "This borrower": "◄" if b["name"] == category else ""})
                prev = b["below"]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") 

