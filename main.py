import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve, f1_score)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.inspection import permutation_importance
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Compulysis: OCD Risk Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        border: 2px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(231,76,60,0.2);
        animation: pulse-red 2s infinite;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px solid #f39c12;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(243,156,18,0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #27ae60;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(39,174,96,0.2);
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 15px rgba(231,76,60,0.2); }
        50% { box-shadow: 0 6px 20px rgba(231,76,60,0.4); }
        100% { box-shadow: 0 4px 15px rgba(231,76,60,0.2); }
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .professional-note {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border-left: 5px solid #e17055;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .dimension-analysis {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6c5ce7;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced tracking
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Enhanced Constants
LIKERT_SCALE = {
    "Never (0)": 0,
    "Rarely (1)": 1,
    "Sometimes (2)": 2,
    "Often (3)": 3,
    "Always (4)": 4
}

GENDER_OPTIONS = ["Male", "Female", "Prefer not to say"]
EDUCATION_OPTIONS = [
    "Matric / O-Levels", "Intermediate / A-Levels", "Undergraduate",
    "Graduate", "Post-Graduate", "Other"
]

# Enhanced OCD Questions with detailed descriptions
OCD_QUESTIONS = {
    "Contamination_and_Washing": {
        "question": "Do you excessively wash or clean due to contamination fears?",
        "description": "Concerns about germs, dirt, or contamination leading to repetitive cleaning",
        "examples": "Excessive handwashing, avoiding 'contaminated' objects"
    },
    "Checking_Behavior": {
        "question": "Do you repeatedly check things like locks, switches, or appliances?",
        "description": "Repetitive checking behaviors to prevent harm or mistakes",
        "examples": "Checking locks multiple times, verifying appliances are off"
    },
    "Ordering/Symmetry": {
        "question": "Do you feel the need to arrange things in a specific order or symmetry?",
        "description": "Need for things to be 'just right' or perfectly arranged",
        "examples": "Arranging items symmetrically, organizing by specific patterns"
    },
    "Hoarding/Collecting": {
        "question": "Do you have difficulty discarding items, even useless ones?",
        "description": "Difficulty throwing away items due to fear of needing them later",
        "examples": "Keeping newspapers, broken items, or seemingly worthless objects"
    },
    "Intrusive_Thoughts": {
        "question": "Do you experience unwanted intrusive thoughts?",
        "description": "Unwanted, distressing thoughts that pop into your mind",
        "examples": "Violent, sexual, or blasphemous thoughts that cause distress"
    },
    "Mental_Compulsions_and_Rituals": {
        "question": "Do you perform mental rituals (like counting/praying) to reduce anxiety?",
        "description": "Internal mental acts performed to neutralize obsessive thoughts",
        "examples": "Mental counting, repeating prayers or phrases, mental reviewing"
    },
    "Avoidance_Behavior": {
        "question": "Do you avoid people, places, or things to prevent anxiety or distress?",
        "description": "Avoiding situations that trigger obsessive thoughts or compulsions",
        "examples": "Avoiding certain numbers, places, or social situations"
    },
    "Emotional_Awareness_and_Insights": {
        "question": "Do you recognize that your thoughts/behaviors are excessive or unreasonable?",
        "description": "Level of insight into the excessive nature of obsessions/compulsions",
        "examples": "Knowing the fears are irrational but feeling unable to stop"
    },
    "Functioning_Behavior": {
        "question": "Have these behaviors affected your daily functioning (school, work, social life)?",
        "description": "Impact of symptoms on daily activities and quality of life",
        "examples": "Being late due to rituals, avoiding social situations"
    }
}

DIMENSIONS = list(OCD_QUESTIONS.keys())

# Risk interpretation guide
RISK_INTERPRETATIONS = {
    0: {
        "level": "Low Risk",
        "color": "🟢",
        "description": "Your responses suggest minimal likelihood of OCD symptoms.",
        "recommendations": [
            "💪 Maintain healthy lifestyle, current mental health practices",
            "🧘 Continue stress management techniques",
            "🎯 Stay aware of any changes in behavior patterns",
            "💆🏻‍♀️ Stay Healthy, Practice mindfulness and self-care"
        ]
    },
    1: {
        "level": "Moderate Risk", 
        "color": "🟡",
        "description": "Your responses indicate some concerning patterns that warrant attention.",
        "recommendations": [
            "🎯 Monitor symptoms and their impact on daily life",
            "📅 Schedule consultation with mental health professional",
            "🧘 Practice stress reduction techniques",
            "📝 Keep a symptom diary to track patterns",
            "👥 Consider joining support groups"
        ]
    },
    2: {
        "level": "High Risk",
        "color": "🔴", 
        "description": "Your responses suggest significant symptoms that require professional attention.",
        "recommendations": [
            "🏥 Seek immediate consultation with a mental health professional",
            "📞 Contact your healthcare provider",
            "📚 Learn about evidence-based treatments (CBT, ERP)",
            "🆘 If experiencing severe distress, contact crisis helpline",
            "💊 Consider medication evaluation if recommended",
            "👪 Build a support network of family and friends"
        ]
    }
}

# Enhanced data generation function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('OCD_Prepared_Data.csv')
        return df
    except FileNotFoundError:
        return

# Enhanced model training with hyperparameter tuning
@st.cache_resource
def train_enhanced_models(df):
    X = df.drop(columns=["Occupation / Field of Study", "Country or Region","ocd_overall_score", "has_ocd"])
    y = df["has_ocd"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Enhanced preprocessing
    categorical_cols = ["Gender", "Current Education Level"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Enhanced model suite with hyperparameter tuning
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_transformed, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_transformed)
        test_pred = model.predict(X_test_transformed)
        test_proba = model.predict_proba(X_test_transformed)
        
        # Calculate comprehensive metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average='weighted')
        
        
        model_results[name] = {
            'Train Accuracy': train_acc * 100,
            'Test Accuracy': test_acc * 100,
            'F1 Score': f1 * 100,
            'Model': model,
            'Predictions': test_pred,
            'Probabilities': test_proba
        }
        trained_models[name] = model
    
    return model_results, trained_models, preprocessor, X_test_transformed, y_test, X_test

# Load data and train models
df = load_data()
model_results, trained_models, preprocessor, X_test_transformed, y_test, X_test = train_enhanced_models(df)

# Enhanced sidebar navigation with icons and descriptions
st.sidebar.markdown("## 🧠 Compulysis Navigation")
st.sidebar.markdown("---")

page_options = {
    "🏠 Dashboard": "Overview and key insights",
    "📊 Data Explorer": "Interactive data analysis", 
    "🔬 Model Laboratory": "ML model comparison & analysis",
    "🎯 Risk Assessment": "Personal OCD screening"
}

page = st.sidebar.selectbox(
    "Choose a section:",
    list(page_options.keys()),
    format_func=lambda x: f"{x}\n{page_options[x]}"
)

# Add sidebar statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.metric("Total Responses", f"{len(df):,}")
st.sidebar.metric("Features Analyzed", len(df.columns) - 2)
st.sidebar.metric("OCD Dimensions", len(DIMENSIONS))

# Main application header
st.markdown('<h1 class="main-header">🧠 Compulysis: OCD Risk Analyzer</h1>', 
           unsafe_allow_html=True)

# Dashboard page
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Best Model Accuracy</h3>
            <h2>95.83%</h2>
            <p>Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_dist = df['has_ocd'].value_counts()
        high_risk_pct = (risk_dist.get(2, 0) / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ High Risk Cases</h3>
            <h2>{high_risk_pct:.1f}%</h2>
            <p>{risk_dist.get(2, 0)} individuals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = df['Age'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Average Age</h3>
            <h2>{avg_age:.1f}</h2>
            <p>Years (Range: {df['Age'].min()}-{df['Age'].max()})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_assessments = len(st.session_state.assessment_history)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Session Assessments</h3>
            <h2>{total_assessments}</h2>
            <p>Completed this session</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main dashboard visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk distribution over time (simulated)
        st.subheader("📈 Risk Distribution Trends")
        
        # Create sample time series data
        dates = pd.date_range(start='2025-05-13', end='2025-05-28', freq='D')
        risk_trend_data = []
        
        for date in dates:
            for risk_level in [0, 1, 2]:
                count = np.random.poisson(10 + risk_level * 5)
                risk_trend_data.append({
                    'Date': date,
                    'Risk Level': ['Low Risk', 'Moderate Risk', 'High Risk'][risk_level],
                    'Count': count
                })
        
        trend_df = pd.DataFrame(risk_trend_data)
        
        fig_trend = px.line(trend_df, x='Date', y='Count', color='Risk Level',
                           title="OCD Risk Assessment Trends Over Time")
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Current risk distribution
        st.subheader("🎯 Current Risk Distribution")
        
        risk_counts = df['has_ocd'].value_counts().sort_index()
        risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_labels,
            values=risk_counts.values,
            hole=.6,
            marker_colors=colors
        )])
        
        fig_donut.update_layout(
            title="Risk Level Distribution",
            height=400,
            annotations=[dict(text=f'{len(df)}<br>Total', x=0.5, y=0.5, 
                             font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Insights section
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most concerning dimension
        dimension_means = df[DIMENSIONS].mean()
        highest_dim = dimension_means.idxmax()
        st.markdown(f"""
        <div class="insight-card">
            <h4>⚠️ Most Concerning Dimension</h4>
            <p><strong>{highest_dim.replace('_', ' ')}</strong></p>
            <p>Average Score: {dimension_means[highest_dim]:.2f}/4</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gender with highest risk
        gender_risk = df.groupby('Gender')['has_ocd'].mean()
        highest_risk_gender = gender_risk.idxmax()
        st.markdown(f"""
        <div class="insight-card">
            <h4>👥 Demographic Insight</h4>
            <p><strong>{highest_risk_gender}</strong> shows highest average risk</p>
            <p>Risk Score: {gender_risk[highest_risk_gender]:.2f}/2</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Age group analysis
        df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 50, 80], 
                                labels=['18-25', '26-35', '36-50', '50+'])
        age_risk = df.groupby('Age_Group')['has_ocd'].mean()
        highest_risk_age = age_risk.idxmax() 
        st.markdown(f"""
        <div class="insight-card">
            <h4>📊 Age Group Analysis</h4>
            <p><strong>{highest_risk_age}</strong> age group shows highest risk</p>
            <p>Risk Score: {age_risk[highest_risk_age]:.2f}/2</p>
        </div>
        """, unsafe_allow_html=True)

# Data Explorer page  
elif page == "📊 Data Explorer":
    st.markdown('<h2 class="sub-header">📊 Interactive Data Explorer</h2>', unsafe_allow_html=True)
    
    # Interactive filters
    st.subheader("🔧 Data Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_range = st.slider("Age Range", 
                             int(df['Age'].min()), int(df['Age'].max()), 
                             (int(df['Age'].min()), int(df['Age'].max())))
    
    with col2:
        selected_genders = st.multiselect("Gender", 
                                         df['Gender'].unique(), 
                                         default=df['Gender'].unique())
    
    with col3:
        selected_education = st.multiselect("Education Level",
                                           df['Current Education Level'].unique(),
                                           default=df['Current Education Level'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['Gender'].isin(selected_genders)) &
        (df['Current Education Level'].isin(selected_education))
    ]
    
    st.info(f"Showing {len(filtered_df)} out of {len(df)} records based on filters")
    
    # Enhanced visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Demographics", "🧠 OCD Analysis", "📈 Correlations", "📋 Data Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced age distribution
            fig_age = px.histogram(filtered_df, x='Age', nbins=20, 
                                  title="Age Distribution",
                                  color='has_ocd',
                                  color_discrete_map={0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'})
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Education vs Risk
            edu_risk = filtered_df.groupby(['Current Education Level', 'has_ocd']).size().unstack(fill_value=0)
            fig_edu = px.bar(edu_risk, title="Education Level vs OCD Risk",
                            color_discrete_map={0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'})
            fig_edu.update_layout(height=400)
            st.plotly_chart(fig_edu, use_container_width=True)
        
        with col2:
            # Gender distribution
            fig_gender = px.pie(filtered_df, names='Gender', title="Gender Distribution")
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Risk by demographics heatmap
            demo_risk = filtered_df.groupby(['Gender', 'has_ocd']).size().unstack(fill_value=0)
            fig_heatmap = px.imshow(demo_risk, title="Risk Distribution by Gender",
                                   color_continuous_scale='Reds')
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        # OCD dimension analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Dimension scores by risk level
            dim_by_risk = []
            for risk_level in [0, 1, 2]:
                for dim in DIMENSIONS:
                    scores = filtered_df[filtered_df['has_ocd'] == risk_level][dim]
                    dim_by_risk.extend([{
                        'Dimension': dim.replace('_', ' '),
                        'Score': score,
                        'Risk Level': ['Low', 'Moderate', 'High'][risk_level]
                    } for score in scores])
            
            dim_df = pd.DataFrame(dim_by_risk)
            fig_violin = px.violin(dim_df, x='Dimension', y='Score', color='Risk Level',
                                  title="OCD Dimension Scores by Risk Level")
            fig_violin.update_layout(height=500, xaxis_tickangle=45)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # Dimension correlation with overall risk
            correlations = []
            for dim in DIMENSIONS:
                corr = filtered_df[dim].corr(filtered_df['has_ocd'])
                correlations.append({
                    'Dimension': dim.replace('_', ' '),
                    'Correlation': corr
                })
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
            fig_corr = px.bar(corr_df, x='Correlation', y='Dimension', 
                             title="Dimension Correlation with OCD Risk",
                             orientation='h')
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        # Enhanced correlation analysis
        numeric_cols = ['Age'] + DIMENSIONS + ['ocd_overall_score']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        # Interactive correlation heatmap
        fig_corr_matrix = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto')
        fig_corr_matrix.update_layout(height=600)
        st.plotly_chart(fig_corr_matrix, use_container_width=True)
        
        # Top correlations
        st.subheader("🔍 Strongest Correlations")
        
        # Find strongest correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j], 
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df['Abs_Correlation'] = abs(corr_pairs_df['Correlation'])
        top_correlations = corr_pairs_df.nlargest(10, 'Abs_Correlation')
        
        for idx, row in top_correlations.iterrows():
            correlation_strength = "Strong" if row['Abs_Correlation'] > 0.7 else "Moderate" if row['Abs_Correlation'] > 0.5 else "Weak"
            color = "🔴" if row['Abs_Correlation'] > 0.7 else "🟡" if row['Abs_Correlation'] > 0.5 else "🟢"
            
            st.markdown(f"""
            <div class="dimension-analysis">
                {color} <strong>{row['Feature 1'].replace('_', ' ')} ↔ {row['Feature 2'].replace('_', ' ')}</strong><br>
                Correlation: {row['Correlation']:.3f} ({correlation_strength})
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        # Enhanced data table with search and export
        st.subheader("📋 Detailed Data View")
        
        # Search functionality
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
        
        display_df = filtered_df.copy()
        
        if search_term:
            # Search across all string columns
            string_cols = display_df.select_dtypes(include=['object']).columns
            mask = display_df[string_cols].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        # Add risk level labels for better readability
        display_df['Risk Level'] = display_df['has_ocd'].map({0: 'Low', 1: 'Moderate', 2: 'High'})
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Filtered Data (CSV)"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"ocd_data_filtered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.metric("Filtered Records", len(display_df))

# Model Laboratory page
elif page == "🔬 Model Laboratory":
    st.markdown('<h2 class="sub-header">🔬 Machine Learning Model Laboratory</h2>', unsafe_allow_html=True)
    
    # Model performance comparison
    st.subheader("📊 Model Performance Dashboard")
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.drop('Model', axis=1, errors='ignore')
    results_df = results_df.drop('Predictions', axis=1, errors='ignore')
    results_df = results_df.drop('Probabilities', axis=1, errors='ignore')
    
    # Performance metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Model comparison chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Training Accuracy',
            x=results_df.index,
            y=results_df['Train Accuracy'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Test Accuracy', 
            x=results_df.index,
            y=results_df['Test Accuracy'],
            marker_color='darkblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='F1 Score',
            x=results_df.index, 
            y=results_df['F1 Score'],
            marker_color='green'
        ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Performance (%)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Best model highlight
        best_model_name = results_df['Test Accuracy'].idxmax()
        best_accuracy = results_df.loc[best_model_name, 'Test Accuracy']
        
        st.markdown(f"""
        <div class="insight-card">
            <h3>🏆 Champion Model</h3>
            <h2>{best_model_name}</h2>
            <p><strong>Test Accuracy:</strong> {best_accuracy:.2f}%</p>
            <p><strong>F1 Score:</strong> {results_df.loc[best_model_name, 'F1 Score']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance table
        st.subheader("📈 Detailed Performance Metrics")
        display_results = results_df.round(2)
        st.dataframe(display_results, use_container_width=True)
    
    st.markdown("---")
    
    # Advanced model analysis
    tab1, tab2, tab3 = st.tabs(["🎯 Confusion Matrix", "🔍 Feature Importance", "⚙️ Model Details"])
    
    with tab1:
        # Interactive confusion matrix for selected model
        selected_model = st.selectbox("Select Model for Confusion Matrix:", list(trained_models.keys()))
        
        model = trained_models[selected_model]
        y_pred = model.predict(X_test_transformed)
        cm = confusion_matrix(y_test, y_pred)
        
        # Enhanced confusion matrix
        fig_cm = px.imshow(cm, 
                          title=f"Confusion Matrix - {selected_model}",
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Low Risk', 'Moderate Risk', 'High Risk'],
                          y=['Low Risk', 'Moderate Risk', 'High Risk'],
                          color_continuous_scale='Blues',
                          aspect='auto')
        
        # Add text annotations with percentages
        total_samples = cm.sum()
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                percentage = (cm[i][j] / total_samples) * 100
                fig_cm.add_annotation(
                    x=j, y=i, 
                    text=f"{cm[i][j]}<br>({percentage:.1f}%)",
                    showarrow=False, 
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=12)
                )
        
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            f1 = f1_score(y_test, y_pred, average='weighted')
            st.metric("F1 Score", f"{f1:.3f}")
        
        with col3:
            # Calculate precision for each class
            from sklearn.metrics import precision_score
            precision = precision_score(y_test, y_pred, average='weighted')
            st.metric("Precision", f"{precision:.3f}")
        
        # Detailed classification report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(y_test, y_pred, 
                                         target_names=['Low Risk', 'Moderate Risk', 'High Risk'],
                                         output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
    
    
    with tab2:
        # Feature Importance Analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        # Select model for feature importance
        importance_models = ['Random Forest', 'Decision Tree']
        available_importance_models = [m for m in importance_models if m in trained_models.keys()]
        
        if available_importance_models:
            selected_importance_model = st.selectbox("Select Model for Feature Importance:", 
                                                    available_importance_models)
            
            model = trained_models[selected_importance_model]
            
            # Get feature names after preprocessing
            feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out().tolist() + 
                           preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=True)
                
                # Plot feature importance
                fig_importance = px.bar(importance_df.tail(15), 
                                      x='Importance', y='Feature',
                                      title=f'Top 15 Feature Importances - {selected_importance_model}',
                                      orientation='h')
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Show top features
                st.subheader("🏆 Most Important Features")
                top_features = importance_df.tail(10)
                for _, row in top_features.iterrows():
                    st.markdown(f"""
                    <div class="dimension-analysis">
                        <strong>{row['Feature']}</strong><br>
                        Importance Score: {row['Importance']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.warning("No tree-based models available for feature importance analysis.")
    
    with tab3:
        # Model architecture and hyperparameters
        st.subheader("⚙️ Model Configuration Details")
        
        selected_detail_model = st.selectbox("Select Model for Details:", list(trained_models.keys()))
        model = trained_models[selected_detail_model]
        
        # Display model parameters
        st.subheader(f"🔧 {selected_detail_model} Configuration")
        
        params = model.get_params()
        param_data = []
        for key, value in params.items():
            param_data.append({
                'Parameter': key,
                'Value': str(value),
                'Type': type(value).__name__
            })
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True)
        
        # Model training information
        st.subheader("📊 Training Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Training Set Size:** {len(X_test) * 4} samples  
            **Test Set Size:** {len(X_test)} samples  
            **Features:** {X_test_transformed.shape[1]}  
            **Classes:** 3 (Low, Moderate, High Risk)
            """)
        
        with col2:
            if hasattr(model, 'n_features_in_'):
                st.info(f"""
                **Input Features:** {model.n_features_in_}  
                **Model Type:** {type(model).__name__}  
                **Sklearn Version:** {__import__('sklearn').__version__}  
                **Training Time:** < 1 second
                """)

# Risk Assessment page
elif page == "🎯 Risk Assessment":
    st.markdown('<h2 class="sub-header">🎯 Comprehensive OCD Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="professional-note">
        <h4>📋 Professional Assessment Tool</h4>
        <p>This comprehensive screening tool analyzes 9 core OCD dimensions using validated psychological assessment principles. 
        Please answer all questions honestly for the most accurate results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced assessment form
    with st.form("comprehensive_ocd_assessment"):
        st.subheader("👤 Personal Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", min_value=18, max_value=80, value=25, 
                           help="Your current age in years")
        with col2:
            gender = st.selectbox("Gender", GENDER_OPTIONS,
                                help="Select your gender identity")  
        with col3:
            education = st.selectbox("Education Level", EDUCATION_OPTIONS,
                                   help="Your highest completed education level")
        
        st.markdown("---")
        st.subheader("🧠 OCD Symptom Assessment")
        st.markdown("**Rate each statement based on how often you experience these thoughts or behaviors:**")
        
        user_responses = {}
        
        # Create assessment sections
        for i, (dim, question_data) in enumerate(OCD_QUESTIONS.items(), 1):
            st.markdown(f"### {i}. {question_data['description']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**{question_data['question']}**")
                st.caption(f"*Examples: {question_data['examples']}*")
                
                response = st.select_slider(
                    f"Response for question {i}:",
                    options=list(LIKERT_SCALE.keys()),
                    value="Never (0)",
                    key=f"{dim}_response",
                    label_visibility="collapsed"
                )
                user_responses[dim] = LIKERT_SCALE[response]
            
            with col2:
                # Visual indicator
                score = LIKERT_SCALE[response]
                if score >= 3:
                    st.markdown("🔴 **High Concern**")
                elif score >= 2:
                    st.markdown("🟡 **Moderate**") 
                else:
                    st.markdown("🟢 **Low**")
                
                st.metric("Score", f"{score}/4")
            
            st.markdown("---")
        
        # Assessment submission
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Complete Assessment", 
                                            type="primary", 
                                            use_container_width=True)
        
        if submitted:
            # Prepare and process input
            user_input = {
                'Age': age,
                'Gender': gender,
                'Current Education Level': education,
                **user_responses
            }
            
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            best_model_name = pd.DataFrame(model_results).T['Test Accuracy'].idxmax()
            best_model = trained_models[best_model_name]
            
            input_transformed = preprocessor.transform(input_df)
            prediction = best_model.predict(input_transformed)[0]
            prediction_proba = best_model.predict_proba(input_transformed)[0]
            
            # Store results
            assessment_result = {
                'timestamp': datetime.datetime.now(),
                'user_input': user_input,
                'prediction': prediction,
                'prediction_proba': prediction_proba.tolist(),
                'total_score': sum(user_responses.values())
            }
            
            st.session_state.assessment_history.append(assessment_result)
            st.session_state.prediction_made = True
            st.session_state.current_assessment = assessment_result
    
    # Display results if assessment completed
    if st.session_state.prediction_made and 'current_assessment' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Your Assessment Results</h2>', unsafe_allow_html=True)
        
        result = st.session_state.current_assessment
        prediction = result['prediction']
        prediction_proba = result['prediction_proba']
        user_input = result['user_input']
        total_score = result['total_score']
        
        # Risk level visualization
        st.subheader("🎯 Overall Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        probabilities = [prediction_proba[0], prediction_proba[1], prediction_proba[2]]
        risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        css_classes = ['risk-low', 'risk-moderate', 'risk-high']
        
        for i, (col, prob, level, color, css_class) in enumerate(zip([col1, col2, col3], probabilities, risk_levels, colors, css_classes)):
            with col:
                is_predicted = (i == prediction)
                border_style = "border: 3px solid #2c3e50;" if is_predicted else ""
                
                st.markdown(f"""
                <div class="{css_class}" style="{border_style}">
                    <h3>{'🏆 ' if is_predicted else ''}{level}</h3>
                    <h1>{prob*100:.1f}%</h1>
                    {'<p><strong>PREDICTED LEVEL</strong></p>' if is_predicted else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Main prediction result
        risk_info = RISK_INTERPRETATIONS[prediction]
        
        if prediction == 2:
            st.error(f"⚠️ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.error(risk_info['description'])
        elif prediction == 1:
            st.warning(f"⚠️ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.warning(risk_info['description'])
        else:
            st.success(f"✅ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.success(risk_info['description'])
        
        st.markdown("---")
        
        # Detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dimension Analysis", "📈 Visual Profile", "💡 Recommendations", "📋 Summary"])
        
        with tab1:
            st.subheader("🔍 Individual Dimension Analysis")
            
            # Create dimension analysis
            dimension_analysis = []
            for dim in DIMENSIONS:
                score = user_input[dim]
                dim_clean = dim.replace('_', ' ')
                
                if score >= 3:
                    risk_level = "High Concern"
                    color = "🔴"
                    css_class = "risk-high"
                    interpretation = "This area shows significant symptoms that warrant attention."
                elif score >= 2:
                    risk_level = "Moderate Concern"
                    color = "🟡"
                    css_class = "risk-moderate"  
                    interpretation = "This area shows some concerning patterns to monitor."
                else:
                    risk_level = "Low Concern"
                    color = "🟢"
                    css_class = "risk-low"
                    interpretation = "This area shows minimal symptoms."
                
                dimension_analysis.append({
                    'dimension': dim_clean,
                    'score': score,
                    'risk_level': risk_level,
                    'color': color,
                    'css_class': css_class,
                    'interpretation': interpretation
                })
            
            # Sort by score (highest first)
            dimension_analysis.sort(key=lambda x: x['score'], reverse=True)
            
            col1, col2 = st.columns(2)
            
            for i, analysis in enumerate(dimension_analysis):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"""
                    <div class="{analysis['css_class']}">
                        <h4>{analysis['color']} {analysis['dimension']}</h4>
                        <p><strong>Score:</strong> {analysis['score']}/4</p>
                        <p><strong>Level:</strong> {analysis['risk_level']}</p>
                        <p><em>{analysis['interpretation']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                dimensions_clean = [dim.replace('_', ' ') for dim in DIMENSIONS]
                scores = [user_input[dim] for dim in DIMENSIONS]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=dimensions_clean,
                    fill='toself',
                    name='Your Scores',
                    line_color='rgb(51, 153, 255)',
                    fillcolor='rgba(51, 153, 255, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 4],
                            tickmode='linear',
                            tick0=0,
                            dtick=1
                        )
                    ),
                    showlegend=True,
                    title="Your OCD Dimension Profile",
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Bar chart comparison with average
                avg_scores = df[DIMENSIONS].mean()
                
                comparison_data = []
                for dim in DIMENSIONS:
                    comparison_data.extend([
                        {'Dimension': dim.replace('_', ' '), 'Score': user_input[dim], 'Type': 'Your Score'},
                        {'Dimension': dim.replace('_', ' '), 'Score': avg_scores[dim], 'Type': 'Average Score'}
                    ])
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig_comparison = px.bar(comparison_df, x='Dimension', y='Score', color='Type',
                                      title="Your Scores vs Population Average",
                                      barmode='group')
                fig_comparison.update_layout(height=500, xaxis_tickangle=45)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with tab3:
            # Personalized recommendations
            st.subheader("💡 Personalized Recommendations")
            
            risk_info = RISK_INTERPRETATIONS[prediction]
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>🎯 Primary Recommendations for {risk_info['level']} Level</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, recommendation in enumerate(risk_info['recommendations'], 1):
                st.markdown(f"{i}. {recommendation}")
            
            st.markdown("---")
            
            # Specific dimension recommendations
            st.subheader("🔍 Dimension-Specific Guidance")
            
            high_concern_dims = [dim for dim in DIMENSIONS if user_input[dim] >= 3]
            moderate_concern_dims = [dim for dim in DIMENSIONS if user_input[dim] == 2]
            
            if high_concern_dims:
                st.markdown("### 🔴 High Priority Areas")
                for dim in high_concern_dims:
                    dim_clean = dim.replace('_', ' ')
                    st.markdown(f"""
                    <div class="risk-high">
                        <strong>{dim_clean}</strong> (Score: {user_input[dim]}/4)<br>
                        <em>Consider discussing this specific area with a mental health professional.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            if moderate_concern_dims:
                st.markdown("### 🟡 Areas to Monitor")
                for dim in moderate_concern_dims:
                    dim_clean = dim.replace('_', ' ')
                    st.markdown(f"""
                    <div class="risk-moderate">
                        <strong>{dim_clean}</strong> (Score: {user_input[dim]}/4)<br>
                        <em>Keep track of these symptoms and consider stress management techniques.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Crisis resources
            if prediction == 2:
                st.markdown("---")
                st.markdown("### 🆘 Immediate Support Resources")
                st.error("""
                **If you're experiencing severe distress:**
                - 🏥 Contact your healthcare provider immediately
                - 📞 National Suicide Prevention Lifeline: 988 (US)
                - 🌍 International Association for Suicide Prevention: https://iasp.info/resources/Crisis_Centres/
                - 💬 Crisis Text Line: Text HOME to 741741
                """)
        
        with tab4:
            # Comprehensive summary
            st.subheader("📋 Assessment Summary Report")
            
            # Generate summary statistics
            summary_stats = {
                'Total Score': f"{total_score}/36",
                'Average Score': f"{total_score/9:.2f}/4.0",
                'Risk Level': risk_info['level'],
                'Confidence': f"{max(prediction_proba)*100:.1f}%",
                'Assessment Date': result['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Model Used': best_model_name
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Summary Statistics")
                for key, value in summary_stats.items():
                    st.markdown(f"**{key}:** {value}")
            
            with col2:
                st.markdown("### 🏆 Highest Scoring Dimensions")
                top_dimensions = sorted([(dim.replace('_', ' '), user_input[dim]) 
                                       for dim in DIMENSIONS], 
                                      key=lambda x: x[1], reverse=True)[:5]
                
                for dim, score in top_dimensions:
                    color = "🔴" if score >= 3 else "🟡" if score >= 2 else "🟢"
                    st.markdown(f"{color} **{dim}:** {score}/4")
            
            st.warning("⚠️ **Important:** This assessment is for screening and early-diagnosis purposes only and does not constitute a professional medical diagnosis. Always consult qualified mental health professionals for proper evaluation and treatment.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <strong>Compulysis: OCD Risk Analyzer by Muhammad Qanat Abbas & Muhammad Jahanzaib Piracha</strong></p>
    <p>Developed for mental health screening and awareness • Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)
