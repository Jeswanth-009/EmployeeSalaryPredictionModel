import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Fallback function to create model from scratch
def create_model_from_scratch():
    """Create and train a new model if the joblib files fail to load"""
    try:
        # Load and preprocess data
        data = pd.read_csv(r"adult 3.csv")
        data.replace({'?': 'Others'}, inplace=True)
        data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
        data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
        data.drop(columns=['education'], inplace=True)
        data = data[(data['age'] >= 17) & (data['age'] <= 75)]
        
        # Prepare features and target
        X = data.drop(columns=['income'])
        y = data['income']
        
        # Create label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Define categorical and numerical columns
        categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
        numerical_features = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        # Create pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])
        
        # Split data and train
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        pipeline.fit(X_train, y_train)
        
        # Make predictions for evaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        return (pipeline, label_encoder, data, X_test, y_test, y_pred, y_proba, 
                accuracy, roc_auc, conf_matrix, class_report)
    
    except Exception as e:
        st.error(f"Error creating model from scratch: {str(e)}")
        st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Encoder ---
@st.cache_resource # Cache the model loading for better performance
def load_artifacts():
    # Due to Python 3.13 compatibility issues with pre-trained models,
    # we'll create a fresh model to ensure compatibility
    st.info("Training model with current environment for compatibility...")
    return create_model_from_scratch()

model_pipeline, income_label_encoder, data_orig, X_test_for_eval, y_test_for_eval, \
y_pred_eval, y_proba_eval, eval_accuracy, eval_roc_auc, eval_conf_matrix, eval_classification_report = load_artifacts()

# --- Define Feature Options (Dynamically from loaded data_orig) ---
feature_columns = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 
                   'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 
                   'native-country']

workclass_options = sorted(data_orig['workclass'].unique().tolist())
marital_status_options = sorted(data_orig['marital-status'].unique().tolist())
occupation_options = sorted(data_orig['occupation'].unique().tolist())
relationship_options = sorted(data_orig['relationship'].unique().tolist())
race_options = sorted(data_orig['race'].unique().tolist())
gender_options = sorted(data_orig['gender'].unique().tolist())
native_country_options = sorted(data_orig['native-country'].unique().tolist())

# --- Title and Introduction ---
st.title("💼 Employee Income Prediction App")
st.markdown("""
Welcome! This application predicts whether an individual's income is **<=50K** or **>50K** 
based on various demographic and work-related attributes.
""")

st.markdown("---")

# --- Sidebar for Input Features ---
st.sidebar.header("Input Employee Details")
st.sidebar.markdown("Adjust the features below to get an income prediction.")

with st.sidebar:
    age = st.slider("Age", 17, 75, 30)
    workclass = st.selectbox("Workclass", workclass_options)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=150000)
    educational_num = st.slider("Education Years (educational-num)", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    occupation = st.selectbox("Occupation", occupation_options)
    relationship = st.selectbox("Relationship", relationship_options)
    race = st.selectbox("Race", race_options)
    gender = st.selectbox("Gender", gender_options)
    
    # Capital gain/loss can have very large ranges, so setting appropriate max/min is important
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Amount of capital gains.")
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, help="Amount of capital losses.")
    
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", native_country_options)

    st.markdown("---")
    predict_button = st.button("✨ Predict Income ✨")

# --- Main Content Area ---
st.header("🔍 Prediction Results")

if predict_button:
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week, # CORRECTED KEY HERE to match feature_columns
        'native-country': native_country
    }], columns=feature_columns)

    # Display input data (optional, for debugging/transparency)
    st.subheader("Input Data Summary:")
    st.dataframe(input_data)

    with st.spinner('Making prediction...'):
        prediction_encoded = model_pipeline.predict(input_data)
        prediction_proba = model_pipeline.predict_proba(input_data)

        predicted_income = income_label_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader("Your Predicted Income Category:")
    if predicted_income == '>50K':
        st.success(f"**Predicted Income: {predicted_income}** 🎉 (High Income)")
    else:
        st.info(f"**Predicted Income: {predicted_income}** (Lower Income)")

    st.write(f"Confidence (Probability of <=50K): `{prediction_proba[0][0]:.2f}`")
    st.write(f"Confidence (Probability of >50K): `{prediction_proba[0][1]:.2f}`")

st.markdown("---")

# --- Model Performance Overview ---
st.header("📊 Model Performance Overview")
st.write("""
Below are the key performance metrics of the deployed model, evaluated on a held-out test set. 
Given the imbalance in income categories (more '<=50K' samples), the model was trained with 
**class weights** to improve its ability to correctly identify the '>50K' (minority) income group.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Metrics")
    st.metric(label="Accuracy", value=f"{eval_accuracy:.3f}")
    st.metric(label="ROC AUC Score", value=f"{eval_roc_auc:.3f}", 
              help="Area Under the Receiver Operating Characteristic Curve. Higher is better, indicating good separation between classes.")

with col2:
    st.subheader("Class-wise Performance")
    df_report = pd.DataFrame(eval_classification_report).transpose()
    st.dataframe(df_report.loc[['<=50K', '>50K', 'weighted avg', 'macro avg'], 
                                ['precision', 'recall', 'f1-score', 'support']].style.format({
                                    'precision': "{:.2f}", 
                                    'recall': "{:.2f}", 
                                    'f1-score': "{:.2f}",
                                    'support': "{:.0f}"
                                }))
    st.markdown("""
    <small>
    - **Precision:** Of all samples predicted as a class, how many were actually that class?
    - **Recall:** Of all samples that truly belong to a class, how many did the model correctly identify?
    - **F1-Score:** The harmonic mean of precision and recall. It's especially useful in imbalanced classification to evaluate a model's performance on the minority class.
    </small>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Visualizations ---
st.header("📈 Detailed Model Performance Visualizations")

# Confusion Matrix Plot
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(eval_conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=income_label_encoder.classes_, yticklabels=income_label_encoder.classes_, ax=ax_cm)
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')
ax_cm.set_title(f'Confusion Matrix for Best Model')
st.pyplot(fig_cm)
st.markdown("""
<small>The confusion matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes.
- **True Negatives (Top-Left):** Correctly predicted <=50K.
- **False Positives (Top-Right):** Predicted >50K, but actually <=50K. (Type I Error)
- **False Negatives (Bottom-Left):** Predicted <=50K, but actually >50K. (Type II Error)
- **True Positives (Bottom-Right):** Correctly predicted >50K.
</small>
""", unsafe_allow_html=True)
plt.close(fig_cm) # Close plot to prevent display issues

st.subheader("Receiver Operating Characteristic (ROC) Curve")
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test_for_eval, y_proba_eval)
ax_roc.plot(fpr, tpr, label=f'Best Model (AUC = {eval_roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc='lower right')
ax_roc.grid(True)
st.pyplot(fig_roc)
st.markdown("""
<small>The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
The closer the curve follows the top-left corner, the better the model. An AUC of 0.5 suggests a random classifier, while 1.0 indicates a perfect classifier.
</small>
""", unsafe_allow_html=True)
plt.close(fig_roc) # Close plot to prevent display issues

st.markdown("---")

# --- New Section: Categorical Feature Insights ---
st.header("📊 Categorical Feature Insights")
st.write("Explore how different categories within features are distributed across the income ranges in the dataset.")

categorical_features_for_insights = [
    'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'
]

# Ensure 'income' column is present for analysis, it was dropped for model input but needed for insights
# We will use data_orig which still has the original 'income' column
for col in categorical_features_for_insights:
    with st.expander(f"Impact of **{col.replace('-', ' ').title()}** on Income"):
        # Calculate value counts for each income group
        income_counts = data_orig.groupby([col, 'income']).size().unstack(fill_value=0)
        
        # Ensure both <=50K and >50K columns exist, even if empty for some categories
        if '<=50K' not in income_counts.columns:
            income_counts['<=50K'] = 0
        if '>50K' not in income_counts.columns:
            income_counts['>50K'] = 0

        # Calculate total and percentage for >50K
        income_counts['Total'] = income_counts['<=50K'] + income_counts['>50K']
        income_counts['% >50K'] = (income_counts['>50K'] / income_counts['Total'] * 100).fillna(0)

        # Sort by '% >50K' for better insights
        income_counts_sorted = income_counts.sort_values(by='% >50K', ascending=False)
        
        st.dataframe(income_counts_sorted.style.format({
            '<=50K': "{:,.0f}", 
            '>50K': "{:,.0f}", 
            'Total': "{:,.0f}", 
            '% >50K': "{:.2f}%"
        }))

        # Optional: Add a bar chart for visual impact
        fig, ax = plt.subplots(figsize=(10, len(income_counts_sorted) * 0.4 + 2)) # Dynamic height
        sns.barplot(x='% >50K', y=income_counts_sorted.index, data=income_counts_sorted, palette='viridis', ax=ax)
        ax.set_title(f'Percentage of >50K Income by {col.replace("-", " ").title()}')
        ax.set_xlabel('% Income >50K')
        ax.set_ylabel(col.replace('-', ' ').title())
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close plot to prevent display issues

st.markdown("---")


# --- About the Model Section ---
st.header("💡 About This Model")
st.write("""
This application uses a Machine Learning pipeline that includes:
- **Preprocessing:** Categorical features are converted using One-Hot Encoding, and numerical features are scaled using MinMaxScaler. This ensures that the model can interpret all types of data effectively.
- **Class Imbalance Handling:** To account for the uneven distribution of income categories in the dataset, the chosen classification model (Logistic Regression or Random Forest) is trained with **balanced class weights**. This makes the model pay more attention to correctly classifying the minority '>50K' income group.
- **Model:** The best performing classifier (identified via extensive testing and hyperparameter tuning in the Jupyter notebook) is used for predictions. This typically results in a **RandomForestClassifier** or **GradientBoostingClassifier**, known for their robustness and accuracy.

The model was trained on a publicly available dataset ([Adult Census Income Dataset](https://archive.ics.uci.uci.edu/ml/datasets/Adult)), which contains demographic data from the 1994 Census.
""")

st.markdown("---")
st.markdown("Developed by Your Name/Team | [GitHub Repo Link](https://github.com/your-username/your-repo-name)")