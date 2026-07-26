import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==========================================
# DATA LOADING & MODEL TRAINING PIPELINE
# ==========================================
df = pd.read_csv('social_media_addiction_mental_wellbeing.csv')

# Handle missing values
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Mapping categorical features
binary_cols = ['Late_Night_Usage', 'Tried_To_Cut_Back', 'Failed_To_Cut_Back']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

if 'First_Check_Morning' in df.columns:
    df['First_Check_Morning'] = df['First_Check_Morning'].astype(str).str.strip()
    check_morning_mapping = {'Within 5 min': 0, '5-30 min': 1, 'After 30 min': 2}
    df['First_Check_Morning'] = df['First_Check_Morning'].map(check_morning_mapping).fillna(0)

df = pd.get_dummies(df, columns=['Primary_Platform'], drop_first=True)
encoded_platform_cols = [col for col in df.columns if col.startswith('Primary_Platform_')]

# 1. Feature Lists Definition
all_features_col = [
    'Platforms_Used_Count', 'Posts_Per_Week', 'Late_Night_Usage',
    'First_Check_Morning', 'Notifications_Per_Day', 'Daily_Usage_Hours',
    'Scroll_Without_Purpose', 'Social_Comparison_Score', 'Failed_To_Cut_Back',
    'FOMO_Score', 'Sleep_Quality_Score'
] + encoded_platform_cols

reduced_features_col = [
    'Daily_Usage_Hours', 'Notifications_Per_Day',
    'FOMO_Score', 'Sleep_Quality_Score', 'Scroll_Without_Purpose'
]

# Scalers for All Features vs Reduced Features
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(df[all_features_col])

scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(df[reduced_features_col])

linear_targets = [
    'Productivity_Loss_Score',
    'Mental_Wellbeing_Score',
    'Depression_Score',
    'Loneliness_Score',
    'Self_Esteem_Score'
]

# ==========================================
# MODEL TRAINING ACCORDING TO YOUR RULES
# ==========================================

# A. Standard Linear Regression for all 5 targets (Using ALL Features, No Regularization/Poly)
linear_models = {}
for target in linear_targets:
    y_lin = df[target]
    lin_reg = LinearRegression()
    lin_reg.fit(X_all_scaled, y_lin)
    linear_models[target] = lin_reg

# B. Standard Logistic Regression for Addiction Level (Using ALL Features)
Logistic_target = "Addiction_Level"
y_log = df[Logistic_target]
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_all_scaled, y_log)

# C. Random Forest Classifier for other targets (Using REDUCED 5 Features & Median Binary Threshold)
rf_classifier_models = {}
for target in linear_targets:
    threshold = df[target].median()
    y_bin = (df[target] > threshold).astype(int)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_red_scaled, y_bin)
    rf_classifier_models[target] = rf_clf


# ==========================================
# INPUT VALIDATION HELPERS
# ==========================================
# Server-side bounds so a direct POST (bypassing the UI's sliders/toggles)
# can't send an out-of-domain value that silently produces a garbage
# extrapolated prediction. Values are clamped rather than rejected, so the
# app still returns a best-effort result instead of erroring on edge cases.
FIELD_BOUNDS = {
    # Bounds below reflect the observed min/max in the training CSV, with a
    # small margin so the clamp doesn't reject legitimate edge-case users.
    'Daily_Usage_Hours': (0.0, 15.0),       # observed 0.5-12
    'Notifications_Per_Day': (0.0, 200.0),  # observed 5-165
    'Platforms_Used_Count': (1.0, 10.0),    # observed 1-8
    'Posts_Per_Week': (0.0, 20.0),          # observed 0-12
    'Scroll_Without_Purpose': (0.0, 10.0),  # observed 0-10
    'Social_Comparison_Score': (0.0, 10.0), # observed 0-10
    'FOMO_Score': (0.0, 10.0),              # observed 0-10
    'Sleep_Quality_Score': (0.0, 10.0),     # observed 0-10
}

VALID_CHOICES = {
    'Late_Night_Usage': {0, 1},
    'Failed_To_Cut_Back': {0, 1},
    'First_Check_Morning': {0, 1, 2},
}

# The exact platform strings the training data contains (get_dummies with
# drop_first=True drops 'Facebook' alphabetically, so a selection of
# 'Facebook' correctly produces an all-zero dummy vector below — that's
# expected one-hot behavior, not a bug). Anything outside this set (e.g. a
# hand-crafted POST) also safely falls back to the all-zero/reference vector.
KNOWN_PLATFORMS = {'Facebook', 'Instagram', 'Snapchat', 'TikTok', 'Twitter/X', 'YouTube'}


def get_clamped_float(form, field_name):
    """Parse a form field as float and clamp it to FIELD_BOUNDS[field_name]."""
    value = float(form[field_name])
    lo, hi = FIELD_BOUNDS[field_name]
    return max(lo, min(hi, value))


def get_valid_choice(form, field_name):
    """Parse a form field as int and ensure it's one of VALID_CHOICES[field_name].

    Falls back to the smallest valid option if an unexpected value is sent
    (e.g. a hand-crafted POST request outside the UI's fixed option set).
    """
    value = int(float(form[field_name]))
    allowed = VALID_CHOICES[field_name]
    if value not in allowed:
        return min(allowed)
    return value


# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form inputs capture (validated / clamped to sane domain ranges)
        platforms_count = get_clamped_float(request.form, 'Platforms_Used_Count')
        posts = get_clamped_float(request.form, 'Posts_Per_Week')
        late_night = get_valid_choice(request.form, 'Late_Night_Usage')
        first_check = float(get_valid_choice(request.form, 'First_Check_Morning'))
        notifications = get_clamped_float(request.form, 'Notifications_Per_Day')
        daily_hours = get_clamped_float(request.form, 'Daily_Usage_Hours')
        scroll_purpose = get_clamped_float(request.form, 'Scroll_Without_Purpose')
        social_comp = get_clamped_float(request.form, 'Social_Comparison_Score')
        failed_cut = get_valid_choice(request.form, 'Failed_To_Cut_Back')
        fomo = get_clamped_float(request.form, 'FOMO_Score')
        sleep_quality = get_clamped_float(request.form, 'Sleep_Quality_Score')

        # Primary_Platform: one-hot encode the selection to match the dummy
        # columns the models were trained on. If the selected platform was
        # the reference category dropped by drop_first=True, this correctly
        # leaves the whole dummy vector at zero.
        selected_platform = request.form.get('Primary_Platform', '').strip()
        platform_col_name = f'Primary_Platform_{selected_platform}'
        platform_dummies = [
            1 if col == platform_col_name else 0
            for col in encoded_platform_cols
        ]

        # 1. Build input array for ALL features
        input_all = [
            platforms_count, posts, late_night, first_check, notifications,
            daily_hours, scroll_purpose, social_comp, failed_cut, fomo, sleep_quality
        ] + platform_dummies

        input_all_scaled = scaler_all.transform([input_all])

        # 2. Build input array for REDUCED 5 features
        input_red = [
            daily_hours, notifications, fomo, sleep_quality, scroll_purpose
        ]
        input_red_scaled = scaler_red.transform([input_red])

        # Predictions Execution
        addiction_pred = log_model.predict(input_all_scaled)[0]

        results = {
            "Addiction_Level": str(addiction_pred),

            # Linear Regression Continuous Scores (Using All Features)
            "Productivity_Loss_Score": round(linear_models['Productivity_Loss_Score'].predict(input_all_scaled)[0], 2),
            "Mental_Wellbeing_Score": round(linear_models['Mental_Wellbeing_Score'].predict(input_all_scaled)[0], 2),
            "Depression_Score": round(linear_models['Depression_Score'].predict(input_all_scaled)[0], 2),
            "Loneliness_Score": round(linear_models['Loneliness_Score'].predict(input_all_scaled)[0], 2),
            "Self_Esteem_Score": round(linear_models['Self_Esteem_Score'].predict(input_all_scaled)[0], 2),

            # Random Forest Binary Predictions (Using Reduced 5 Features)
            "Productivity_Loss_RF": int(rf_classifier_models['Productivity_Loss_Score'].predict(input_red_scaled)[0]),
            "Mental_Wellbeing_RF": int(rf_classifier_models['Mental_Wellbeing_Score'].predict(input_red_scaled)[0]),
            "Depression_RF": int(rf_classifier_models['Depression_Score'].predict(input_red_scaled)[0]),
            "Loneliness_RF": int(rf_classifier_models['Loneliness_Score'].predict(input_red_scaled)[0]),
            "Self_Esteem_RF": int(rf_classifier_models['Self_Esteem_Score'].predict(input_red_scaled)[0])
        }

        return render_template('index.html', prediction_text=results)

    except Exception as e:
        return render_template('index.html', error_text=f"Error occurred: {str(e)}")


if __name__ == '__main__':
    # Debug mode is OFF by default. Set FLASK_DEBUG=1 in your local shell
    # (never in production) to enable the interactive debugger/auto-reload.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
