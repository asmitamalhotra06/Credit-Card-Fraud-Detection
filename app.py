import streamlit as st
import pandas as pd
import joblib
import textwrap


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

model = joblib.load("models/random_forest_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    textwrap.dedent("""
    <style>
    /* REMOVE STREAMLIT TOP HEADER */

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

    .stApp {
        background:
            radial-gradient(
                circle at 5% 35%,
                rgba(120, 170, 255, 0.18) 0%,
                transparent 18%
            ),
            radial-gradient(
                circle at 95% 70%,
                rgba(160, 120, 255, 0.15) 0%,
                transparent 18%
            ),
            #f7f9ff;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* HEADER */

    .header-card {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f1f3ff 55%,
            #e8edff 100%
        );

        border-radius: 24px;
        padding: 35px 45px;
        margin-bottom: 25px;

        border: 1px solid rgba(120, 140, 220, 0.18);

        box-shadow:
            0 10px 30px rgba(50, 70, 130, 0.08);
    }

    .header-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .header-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .header-title span {
        color: #5b3df5;
    }

    .header-subtitle {
        font-size: 17px;
        color: #64748b;
    }


    /* INFORMATION CARD */

    .info-card {
        background: linear-gradient(
            135deg,
            #edf6ff,
            #eef2ff
        );

        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 20px 25px;
        margin-bottom: 22px;

        color: #1e3a8a;
    }

    .info-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .info-text {
        font-size: 20px;
        color: #64748b;
    }


    /* BUTTON */

    div.stButton > button {
        width: 100%;

        background: linear-gradient(
            90deg,
            #3563f4,
            #6945e8
        );

        color: white;
        border: none;
        border-radius: 14px;

        padding: 15px 25px;

        font-size: 45px;
        font-weight: 700;

        box-shadow:
            0 8px 20px rgba(79, 70, 229, 0.25);

        transition: all 0.2s ease;
    }

    div.stButton > button p {
    font-size: 25px !important;
    font-weight: 700 !important;
}

    div.stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 25px rgba(79, 70, 229, 0.35);

        color: white;
    }


    /* TRANSACTION CARDS */

    .transaction-card {
        background: white;

        border-radius: 20px;

        padding: 25px 30px;

        border: 1px solid #e5e7eb;

        box-shadow:
            0 8px 25px rgba(50, 70, 130, 0.07);

        margin-top: 15px;
    }

    .card-label {
    font-size: 21px;
    font-weight: 700;
    color: #172554;
    margin-bottom: 8px;
}

.card-value {
    font-size: 32px;
    font-weight: 800;
    color: #172554;
}

    .card-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }


    /* ANALYSIS CARD */

    .analysis-card {
        background: white;

        border-radius: 22px;

        padding: 30px;

        margin-top: 25px;

        border: 1px solid #e5e7eb;

        box-shadow:
            0 8px 25px rgba(50, 70, 130, 0.07);
    }

    .analysis-title {
        font-size: 28px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 3px;
    }

    .analysis-subtitle {
        color: #64748b;
        font-size: 20px;
        margin-bottom: 22px;
    }


/* FRAUD RISK CARD */

.risk-card-fraud {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #fff7ed
    );

    border: 1px solid #fecaca;

    border-radius: 16px;

    padding: 20px 25px;

    margin-bottom: 18px;
}

.risk-card-fraud .risk-label {
    color: #dc2626;
    font-size: 25px;
    font-weight: 700;
}

.risk-card-fraud .risk-value {
    color: #dc2626;
    font-size: 36px;
    font-weight: 800;
    margin-top: 5px;
}


/* LEGITIMATE RISK CARD */

.risk-card-safe {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f0fdf4
    );

    border: 1px solid #a7f3d0;

    border-radius: 16px;

    padding: 20px 25px;

    margin-bottom: 18px;
}

.risk-card-safe .risk-label {
    color: #059669;
    font-size: 25px;
    font-weight: 700;
}

.risk-card-safe .risk-value {
    color: #059669;
    font-size: 36px;
    font-weight: 800;
    margin-top: 5px;
}


    /* APPROVED RESULT */

    .approved-card {
        background: linear-gradient(
            135deg,
            #ecfdf5,
            #f0fdf4
        );

        border: 1px solid #a7f3d0;

        border-radius: 16px;

        padding: 20px 25px;

        color: #047857;
    }

    .approved-title {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .approved-text {
        color: #475569;
        font-size: 20px;
    }


    /* FRAUD RESULT */

    .fraud-card {
        background: linear-gradient(
            135deg,
            #fff1f2,
            #fff7ed
        );

        border: 1px solid #fecaca;

        border-radius: 16px;

        padding: 20px 25px;

        color: #dc2626;
    }

    .fraud-title {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .fraud-text {
        color: #475569;
        font-size: 20px;
    }


    /* DISCLAIMER */

    .disclaimer {
        text-align: center;

        color: #94a3b8;

        font-size: 12px;

        margin-top: 35px;

        padding-top: 15px;

        border-top: 1px solid #e2e8f0;
    }

    /* Simulation section spacing */
.simulation-section {
    margin-top: 10px !important;
    margin-bottom: 15px !important;
    padding-top: 0 !important;
}

/* Simulation section text */
.simulation-title {
    color: #172554 !important;
    font-size: 25px !important;
    font-weight: 700 !important;
    margin-bottom: 5px !important;
}

.simulation-text {
    font-size: 25px;
    font-weight: 600;
    color: #173568;
}

    </style>
    """),
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.html("""
<div class="header-card">
    <div class="header-title">
        Credit Card Fraud
        <span>Detection System</span>
    </div>

    <div class="header-subtitle">
        Secure transaction risk assessment
    </div>
</div>
""")


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

st.html("""
<div class="info-card">

    <div class="info-title">
        ℹ️ Transaction Security
    </div>

    <div class="info-text">
        This system analyzes incoming credit card transactions
        and identifies potentially suspicious activity.
    </div>

</div>
""")

# --------------------------------------------------
# SIMULATION
# --------------------------------------------------

st.html("""
<div class="simulation-text">
    Click below to analyze an incoming credit card transaction.
</div>s
""")

col_left, col_center, col_right = st.columns([1, 1, 1])

with col_center:

    simulate_clicked = st.button(
        "Simulate Transaction",
        use_container_width=True
    )


if simulate_clicked:

    demo_data = pd.read_csv(
        "data/demo_transactions.csv"
    )

    transaction = demo_data.sample(
        1
    ).iloc[0]


    # --------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------


    col1, col2 = st.columns(2)


    with col1:

        st.html(f"""
<div class="transaction-card">
    <div class="card-label">
        Transaction ID
    </div>

    <div class="card-value">
        {int(transaction["Transaction_ID"])}
    </div>

</div>
""")


    with col2:

        st.html(f"""
<div class="transaction-card">
    <div class="card-label">
        Transaction Amount
    </div>

    <div class="card-value">
        ${transaction["Amount"]:.2f}
    </div>

</div>
""")


    # --------------------------------------------------
    # PREPARE TRANSACTION
    # --------------------------------------------------

    transaction_features = transaction[
        feature_names
    ]

    transaction_scaled = scaler.transform(
        transaction_features.to_frame().T
    )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(
        transaction_scaled
    )[0]

    probability = model.predict_proba(
        transaction_scaled
    )[0][1]


# --------------------------------------------------
# RISK ANALYSIS
# --------------------------------------------------

if prediction == 1:

    risk_card_class = "risk-card-fraud"
    risk_label = "⚠️ Fraud Risk"
    risk_color = "#991b1b"

else:

    risk_card_class = "risk-card-safe"
    risk_label = "✅ Legitimate Transaction"
    risk_color = "#065f46"


if prediction == 1:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #ffe4e4,
                #fecaca
            );
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #d1fae5,
                #a7f3d0
            );
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.html(f"""
<div class="analysis-card">

    <div class="analysis-title">
        Transaction Risk Analysis
    </div>

    <div class="analysis-subtitle">
        The transaction has been analyzed for potential risk.
    </div>

    <div class="{risk_card_class}">

        <div class="risk-label" style="color:{risk_color};">
            {risk_label}
        </div>

        <div class="risk-value" style="color:{risk_color};">
            {probability * 100:.2f}%
        </div>

    </div>

</div>
""")

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

if prediction == 1:

       st.html("""
<div class="fraud-card">

    <div class="fraud-title">
        🚨 TRANSACTION FLAGGED
    </div>

    <div class="fraud-text">
        This transaction has been identified as potentially
        suspicious. Manual review is recommended.
    </div>

</div>
""")

else:

       st.html("""
<div class="approved-card">

    <div class="approved-title">
        ✅ TRANSACTION APPROVED
    </div>

    <div class="approved-text">
        This transaction appears to be legitimate.
    </div>

</div>
""")



# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.markdown(
    textwrap.dedent("""
    <div class="disclaimer">
        ⚠️ Demonstration application — transaction risk predictions
        should not be used as the sole basis for real-world financial decisions.
    </div>
    """),
    unsafe_allow_html=True
)