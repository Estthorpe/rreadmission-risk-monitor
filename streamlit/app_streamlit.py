from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import requests
import streamlit as st


# -----------------------------
# App Config
# -----------------------------
APP_TITLE = "Readmission Risk Monitor"
DEFAULT_HOSPITAL = "St. Catherine General Hospital"
DEFAULT_UNIT = "Medical Ward"
DEFAULT_API_URL = os.getenv("RRM_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)


# -----------------------------
# Small helpers
# -----------------------------
def _pill(label: str, value: str) -> str:
    return f"<span style='padding:6px 10px;border-radius:999px;background:#F0F7FF;border:1px solid #D6E8FF;font-size:0.9rem;'><b>{label}:</b> {value}</span>"


def post_predict(api_url: str, payload: Dict[str, Any], timeout_s: int = 30) -> Dict[str, Any]:
    url = api_url.rstrip("/") + "/predict"
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def get_health(api_url: str, timeout_s: int = 10) -> Dict[str, Any]:
    url = api_url.rstrip("/") + "/health"
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def risk_badge(tier: str) -> str:
    tier_l = (tier or "").lower()
    if tier_l == "high":
        return "🔴 High"
    if tier_l == "medium":
        return "🟠 Medium"
    if tier_l == "low":
        return "🟢 Low"
    return f"⚪ {tier}"


# -----------------------------
# Header / Branding
# -----------------------------
st.markdown(
    """
    <style>
      .brand-header {
        padding: 18px 20px;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(12,47,91,1) 0%, rgba(17,92,172,1) 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
      }
      .brand-title { font-size: 1.9rem; font-weight: 800; margin: 0; }
      .brand-sub { opacity: 0.9; margin-top: 6px; }
      .kpi-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: white;
        border: 1px solid #E9EEF5;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
      }
      .muted { color: #667085; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; }
      .divider { height: 1px; background: #E9EEF5; margin: 12px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brand-header">
      <div class="brand-title">🩺 {APP_TITLE}</div>
      <div class="brand-sub">Care Management Dashboard • 30-day readmission risk triage</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# -----------------------------
# Sidebar: Operational Context
# -----------------------------
with st.sidebar:
    st.header("🏥 Clinical context")

    hospital_name = st.text_input("Hospital / Facility", value=DEFAULT_HOSPITAL)
    unit_name = st.selectbox(
        "Unit / Department",
        options=[
            DEFAULT_UNIT,
            "Emergency Department",
            "Cardiology",
            "Endocrinology",
            "ICU / Critical Care",
            "Surgery",
            "Outpatient Clinic",
        ],
        index=0,
    )
    clinician = st.text_input("Clinician / Care Manager", value="A. Clinician")
    shift = st.selectbox("Shift", ["Day", "Evening", "Night"], index=0)

    st.divider()

    st.subheader("🔌 API connection")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)
    st.caption("Example: http://127.0.0.1:8000")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Check /health", use_container_width=True):
            try:
                health = get_health(api_url)
                st.success(f"Service OK • model={health.get('model_version')} • schema={health.get('schema_version')}")
            except Exception as e:
                st.error(f"Health check failed: {e}")
    with col_b:
        st.link_button("Open API Docs", api_url.rstrip("/") + "/docs", use_container_width=True)

    st.divider()

    st.subheader("⚙️ Request settings")
    timeout_s = st.slider("Timeout (seconds)", min_value=5, max_value=60, value=30, step=5)
    st.caption("If the model/bundle is large, increase timeout.")


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("🧾 Patient features")

    st.caption(
        "For Phase 5, we keep this **fully functional** by allowing clinicians to paste/enter feature values as JSON. "
        "In the next iteration, we’ll auto-build a guided form from the model’s feature list."
    )

    demo_features = {
        # Put a few realistic-looking examples. These will be ignored if not part of FEATURE_COLUMNS in the API.
        "AGE": 65,
        "RACE": "Caucasian",
        "GENDER": "Female",
        "ADMISSION_TYPE_ID": 1,
        "TIME_IN_HOSPITAL": 4,
        "NUM_LAB_PROCEDURES": 50,
        "NUM_MEDICATIONS": 15,
        "NUMBER_OUTPATIENT": 0,
        "NUMBER_EMERGENCY": 1,
        "NUMBER_INPATIENT": 0,
        "DIAG_1": "250.83",
    }

    if "features_json" not in st.session_state:
        st.session_state["features_json"] = json.dumps(demo_features, indent=2)

    st.session_state["features_json"] = st.text_area(
        "Paste / edit JSON feature values",
        value=st.session_state["features_json"],
        height=320,
        help="Must be valid JSON (object). Any keys not used by the model will be ignored safely.",
    )

    col1, col2, col3 = st.columns([0.34, 0.33, 0.33])
    with col1:
        predict_clicked = st.button("🚀 Predict risk", type="primary", use_container_width=True)
    with col2:
        if st.button("Load demo patient", use_container_width=True):
            st.session_state["features_json"] = json.dumps(demo_features, indent=2)
            st.rerun()
    with col3:
        if st.button("Clear", use_container_width=True):
            st.session_state["features_json"] = "{}"
            st.rerun()

    st.write("")

    st.subheader("📌 Encounter details (for audit trail)")
    c1, c2, c3 = st.columns(3)
    with c1:
        encounter_id = st.text_input("Encounter ID", value="ENC-0001")
    with c2:
        patient_ref = st.text_input("Patient Ref (local)", value="MRN-12345")
    with c3:
        discharge_date = st.date_input("Planned discharge date", value=None)

    st.caption("These fields are **context** for the UI and audit trail; they don’t change the model unless you include them in the JSON features.")


with right:
    st.subheader("📊 Risk output")

    # Context pills
    st.markdown(
        f"""
        {_pill("Hospital", hospital_name)}&nbsp;&nbsp;
        {_pill("Unit", unit_name)}&nbsp;&nbsp;
        {_pill("Shift", shift)}
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Result area
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None

    if predict_clicked:
        st.session_state["last_error"] = None
        try:
            features = json.loads(st.session_state["features_json"])
            if not isinstance(features, dict):
                raise ValueError("Features JSON must be an object (key-value map).")

            payload = {
                "request_id": f"{encounter_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "features": features,
            }

            with st.spinner("Calling model service..."):
                result = post_predict(api_url, payload, timeout_s=timeout_s)

            st.session_state["last_result"] = {
                "result": result,
                "payload": payload,
                "context": {
                    "hospital": hospital_name,
                    "unit": unit_name,
                    "clinician": clinician,
                    "shift": shift,
                    "encounter_id": encounter_id,
                    "patient_ref": patient_ref,
                },
                "created_utc": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.session_state["last_result"] = None

    if st.session_state["last_error"]:
        st.error(f"Prediction failed: {st.session_state['last_error']}")
        st.info("Tip: ensure FastAPI is running and the base URL is correct (e.g., http://127.0.0.1:8000).")

    if st.session_state["last_result"]:
        r = st.session_state["last_result"]["result"]

        risk = float(r.get("readmission_risk", 0.0))
        tier = str(r.get("risk_tier", "unknown"))
        latency_ms = float(r.get("latency_ms", 0.0))
        model_version = str(r.get("model_version", "unknown"))
        schema_version = str(r.get("schema_version", "unknown"))

        # KPI cards
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="muted">Readmission risk</div>
                  <div style="font-size:2.2rem;font-weight:800;">{risk*100:.1f}%</div>
                  <div class="muted">Tier: <b>{risk_badge(tier)}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="muted">Service</div>
                  <div style="font-size:1.2rem;font-weight:700;">Model v{model_version}</div>
                  <div class="muted">Schema v{schema_version}</div>
                  <div class="divider"></div>
                  <div class="muted">Latency</div>
                  <div style="font-size:1.6rem;font-weight:800;">{latency_ms:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.subheader("🧠 Care guidance")
        if tier.lower() == "high":
            st.warning(
                "High risk: recommend **early follow-up plan**, medication reconciliation, and a documented discharge call within 48 hours."
            )
        elif tier.lower() == "medium":
            st.info(
                "Medium risk: recommend **standard follow-up**, reinforce discharge instructions, and confirm outpatient scheduling."
            )
        else:
            st.success(
                "Low risk: continue with standard discharge process; provide routine self-management guidance."
            )

        st.write("")
        st.subheader("🧾 Raw response (for audit)")
        st.code(json.dumps(st.session_state["last_result"], indent=2), language="json")

    else:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="muted">No prediction yet</div>
              <div style="font-size:1.1rem;margin-top:6px;">
                Enter patient features and click <b>Predict risk</b>.
              </div>
              <div class="muted" style="margin-top:10px;">
                Ensure FastAPI is running at <span class="mono">/predict</span>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.write("")
st.caption("Phase 5 UI • Built for clinical triage workflows • Uses FastAPI /predict under the hood.")
