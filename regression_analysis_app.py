import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Regression Analysis Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .example-box {
        background-color: #f1f8e9;
        padding: 1rem;
        border: 2px solid #8bc34a;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .definition-box {
        background-color: #fffde7;
        padding: 1rem;
        border: 2px solid #fdd835;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .beginner-box {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-left: 5px solid #2196F3;
        border-radius: 4px;
        margin: 1rem 0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Helper Functions for Visualizations

def create_scatter_with_regression_line(x, y, y_pred, title, x_label, y_label):
    """Create scatter plot with regression line"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Actual Data',
                            marker=dict(size=8, color='#667eea', opacity=0.6)))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Regression Line',
                            line=dict(color='#f5576c', width=3)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                     template='plotly_white', height=400)
    return fig

def create_residual_plot(y_pred, residuals):
    """Create residual vs fitted plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                            marker=dict(size=8, color='#667eea', opacity=0.6)))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Plot", xaxis_title="Fitted Values",
                     yaxis_title="Residuals", template='plotly_white', height=400)
    return fig

def create_qq_plot(residuals):
    """Create Q-Q plot for normality check"""
    qq = stats.probplot(residuals, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                            marker=dict(size=8, color='#667eea', opacity=0.6),
                            name='Sample Quantiles'))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines',
                            line=dict(color='red', dash='dash'), name='Theoretical Line'))
    fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles",
                     yaxis_title="Sample Quantiles", template='plotly_white', height=400)
    return fig

def create_distribution_plot(data, column):
    """Create distribution plot"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[column], nbinsx=30, name='Distribution',
                              marker_color='#667eea', opacity=0.7))
    fig.update_layout(title=f"Distribution of {column}", xaxis_title=column,
                     yaxis_title="Frequency", template='plotly_white', height=400)
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    corr = data.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale='RdBu', zmid=0, text=corr.values.round(2),
                                    texttemplate='%{text}', textfont={"size":10}))
    fig.update_layout(title="Correlation Heatmap", height=500, template='plotly_white')
    return fig

def create_regression_comparison_chart():
    """Create regression types comparison visualization"""
    x = np.linspace(0, 10, 100)
    y_linear = 2 * x + 1
    y_poly = 0.5 * x**2 - 2 * x + 5
    y_log = 5 * np.log(x + 1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_linear, name='Linear', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_poly, name='Polynomial', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_log, name='Logarithmic', line=dict(color='green', width=2)))
    fig.update_layout(title="Comparison of Regression Types", xaxis_title="X", yaxis_title="Y",
                     template='plotly_white', height=400)
    return fig

def create_overfitting_demo(degree=1):
    """Demonstrate overfitting with polynomial regression"""
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 2, 20)
    
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(x_poly, y)
    
    x_line = np.linspace(0, 10, 100)
    x_line_poly = poly.transform(x_line.reshape(-1, 1))
    y_pred = model.predict(x_line_poly)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(size=10, color='blue')))
    fig.add_trace(go.Scatter(x=x_line, y=y_pred, mode='lines', name=f'Degree {degree}',
                            line=dict(color='red', width=2)))
    fig.update_layout(title=f"Polynomial Regression (Degree {degree})", xaxis_title="X",
                     yaxis_title="Y", template='plotly_white', height=400)
    return fig

def create_roc_curve_plot(fpr, tpr, auc_score):
    """Create ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_score:.3f})',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                            line=dict(color='red', dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                     yaxis_title="True Positive Rate", template='plotly_white', height=400)
    return fig

def create_vif_bar_chart(vif_data):
    """Create VIF bar chart"""
    fig = go.Figure(go.Bar(x=vif_data['Feature'], y=vif_data['VIF'],
                          marker_color=['red' if v > 10 else 'orange' if v > 5 else 'green'
                                       for v in vif_data['VIF']]))
    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderate (5)")
    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="High (10)")
    fig.update_layout(title="Variance Inflation Factor (VIF)", xaxis_title="Features",
                     yaxis_title="VIF Value", template='plotly_white', height=400)
    return fig

def create_metrics_comparison_chart(models_dict):
    """Compare metrics across multiple models"""
    model_names = list(models_dict.keys())
    r2_scores = [models_dict[m]['r2'] for m in model_names]
    rmse_scores = [models_dict[m]['rmse'] for m in model_names]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Score", "RMSE"))
    fig.add_trace(go.Bar(x=model_names, y=r2_scores, name='R²', marker_color='#667eea'), row=1, col=1)
    fig.add_trace(go.Bar(x=model_names, y=rmse_scores, name='RMSE', marker_color='#f5576c'), row=1, col=2)
    fig.update_layout(height=400, template='plotly_white', showlegend=False)
    return fig

def create_assumption_summary_plot(assumptions):
    """Create visual summary of assumption tests"""
    fig = go.Figure()
    categories = list(assumptions.keys())
    values = [1 if assumptions[k] else 0 for k in categories]
    colors = ['green' if v else 'red' for v in values]
    
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors,
                        text=['✓' if v else '✗' for v in values],
                        textposition='auto'))
    fig.update_layout(title="Regression Assumptions Summary", yaxis_title="Pass/Fail",
                     template='plotly_white', height=400, showlegend=False)
    fig.update_yaxis(range=[0, 1.2], tickvals=[0, 1], ticktext=['Fail', 'Pass'])
    return fig

# Sidebar Navigation
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "🎓 Beginner's Guide",
    "📊 Regression Basics",
    "⚠️ Common Pitfalls",
    "🧪 Regression Methods",
    "📈 Model Evaluation",
    "🔍 Advanced Concepts",
    "🎯 Model Builder",
    "🔬 Model Comparison"
])

# Main Content
if page == "🏠 Home":
    st.markdown('<div class="main-header"><h1>📊 Regression Analysis Hub</h1><p>Your Complete Guide to Regression Analysis</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 What You'll Learn")
        st.markdown("""
        - **Linear & Multiple Regression**
        - **Polynomial Regression**
        - **Logistic Regression**
        - **Ridge & Lasso Regression**
        - **Model Evaluation & Diagnostics**
        - **Assumption Testing**
        - **Model Comparison**
        """)
    
    with col2:
        st.markdown("### ⚡ Interactive Features")
        st.markdown("""
        - **Upload Your Data**
        - **Build Custom Models**
        - **Interactive Visualizations**
        - **Real-time Diagnostics**
        - **Compare Multiple Models**
        - **Download Results**
        """)
    
    with col3:
        st.markdown("### 📚 Key Concepts")
        st.markdown("""
        - **Regression Fundamentals**
        - **Coefficients & P-values**
        - **R² & Adjusted R²**
        - **Residual Analysis**
        - **Multicollinearity (VIF)**
        - **Autocorrelation**
        """)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 Regression Types Overview</h2></div>', unsafe_allow_html=True)
    
    fig = create_regression_comparison_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header"><h2>🚀 Quick Start Guide</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-box"><h4>👶 Beginners</h4><p>Start with <b>Beginner\'s Guide</b> for plain English explanations, then explore <b>Regression Basics</b> to understand core concepts.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="warning-box"><h4>🎓 Intermediate</h4><p>Jump to <b>Regression Methods</b> to explore different techniques, then use <b>Model Builder</b> to practice with your data.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="success-box"><h4>🚀 Advanced</h4><p>Dive into <b>Advanced Concepts</b> for assumption testing and diagnostics, use <b>Model Comparison</b> for complex analysis.</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 How to Use This Guide")
    with st.expander("🔍 Click to expand"):
        st.markdown("""
        1. **Navigate** using the sidebar menu
        2. **Upload your data** in the Model Builder section
        3. **Choose a regression type** based on your needs
        4. **Fit the model** and view results
        5. **Check assumptions** in Advanced Concepts
        6. **Compare models** to find the best fit
        7. **Download** your results and visualizations
        """)

elif page == "🎓 Beginner's Guide":
    st.markdown('<div class="main-header"><h1>🎓 Regression for Beginners</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="beginner-box"><h2>🤔 What is Regression?</h2><p style="font-size:18px;"><b>Imagine you\'re trying to guess the price of a house.</b> You notice that bigger houses cost more. Regression helps you find the exact relationship between house size and price, so you can predict prices for houses you haven\'t seen yet!</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 The Story of Sarah's Ice Cream Stand")
    with st.expander("📚 Read the complete example"):
        st.markdown("""
        **Sarah sells ice cream and wants to predict her daily sales.**
        
        She notices that sales seem related to temperature. On hot days (30°C), she sells about 200 cones. On mild days (20°C), she sells about 100 cones.
        
        Using **linear regression**, Sarah can:
        1. **Collect data**: Temperature and sales for 30 days
        2. **Find the pattern**: For every 1°C increase, sales go up by 10 cones
        3. **Make predictions**: Tomorrow is 25°C → predict 150 cones
        4. **Plan inventory**: Order enough supplies based on weather forecast
        
        **The Magic Formula**: Sales = 10 × Temperature + 50
        
        This is regression in action! It found the mathematical relationship between temperature (input) and sales (output).
        """)
    
    st.markdown("---")
    st.markdown("### 🔑 Key Terms Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="definition-box"><h4>📍 Dependent Variable (Y)</h4><p><b>Simple:</b> The thing you want to predict</p><p><b>Example:</b> Ice cream sales, house prices, test scores</p><p><b>Also called:</b> Target, outcome, response variable</p></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="definition-box"><h4>📍 Independent Variable (X)</h4><p><b>Simple:</b> The thing you use to make predictions</p><p><b>Example:</b> Temperature, house size, study hours</p><p><b>Also called:</b> Feature, predictor, explanatory variable</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="definition-box"><h4>📊 Coefficient</h4><p><b>Simple:</b> How much Y changes when X increases by 1</p><p><b>Example:</b> If coefficient is 10, each extra degree = 10 more cones</p><p><b>Also called:</b> Slope, weight, parameter</p></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="definition-box"><h4>📊 R² (R-squared)</h4><p><b>Simple:</b> How well your prediction works (0-100%)</p><p><b>Example:</b> R²=80% means predictions are 80% accurate</p><p><b>Good value:</b> Above 70% is usually good</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ❓ Common Beginner Questions")
    
    with st.expander("Q: When should I use regression?"):
        st.markdown("""
        Use regression when you want to **predict a number** based on other information:
        - Predict **sales** based on advertising spend
        - Predict **house prices** based on size, location, age
        - Predict **student grades** based on study time
        - Predict **weight loss** based on exercise and diet
        
        **Don't use it** when you want to categorize (use classification instead):
        - Is this email spam? (Yes/No) → Use **Logistic Regression**
        - What type of flower is this? (3 types) → Use **Classification**
        """)
    
    with st.expander("Q: How much data do I need?"):
        st.markdown("""
        **General rule**: At least 10-20 observations per variable
        
        - **1 predictor** (simple regression): 20+ observations
        - **5 predictors** (multiple regression): 100+ observations
        - **More is better!** 100+ observations gives more reliable results
        
        **Quality matters**: Clean, accurate data is more important than lots of messy data.
        """)
    
    with st.expander("Q: What makes a 'good' model?"):
        st.markdown("""
        A good model has:
        1. **High R²** (above 0.7 or 70%)
        2. **Low prediction errors** (RMSE close to 0)
        3. **Significant predictors** (p-values < 0.05)
        4. **Passes assumption tests** (check residual plots)
        5. **Makes sense** (coefficients have logical signs)
        
        ⚠️ **Warning**: Very high R² (>0.99) might mean overfitting!
        """)
    
    with st.expander("Q: Linear vs Multiple vs Polynomial - which one?"):
        st.markdown("""
        **Simple Linear Regression**: 1 predictor
        - Example: Predict price from size
        - Use when: You have one main factor
        
        **Multiple Linear Regression**: 2+ predictors
        - Example: Predict price from size, age, location
        - Use when: Multiple factors affect outcome
        
        **Polynomial Regression**: Curved relationships
        - Example: Profit vs advertising (diminishing returns)
        - Use when: Relationship isn't a straight line
        
        **Start simple** (linear) and add complexity only if needed!
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 Learning Path Forward")
    st.markdown("""
    **Your next steps:**
    1. ✅ You just learned the basics! (You're here)
    2. 📊 Go to **Regression Basics** → See it in action with visualizations
    3. 🧪 Explore **Regression Methods** → Learn each type in detail
    4. 🎯 Try **Model Builder** → Upload data and build your first model!
    5. 🔍 Check **Advanced Concepts** → Make sure your model is valid
    """)

elif page == "📊 Regression Basics":
    st.markdown('<div class="main-header"><h1>📊 Regression Basics & Fundamentals</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>📈 Simple Linear Regression</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Linear regression finds the best straight line through your data points. The formula is: <b>Y = β₀ + β₁X + ε</b></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Adjust Parameters")
        slope = st.slider("Slope (β₁)", -5.0, 5.0, 2.0, 0.1)
        intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 1.0, 0.5)
        noise = st.slider("Noise Level", 0.0, 5.0, 1.0, 0.1)
    
    np.random.seed(42)
    x_demo = np.linspace(0, 10, 50)
    y_demo = slope * x_demo + intercept + np.random.normal(0, noise, 50)
    y_pred_demo = slope * x_demo + intercept
    
    fig = create_scatter_with_regression_line(x_demo, y_demo, y_pred_demo,
                                              "Linear Regression Demo",
                                              "X Variable", "Y Variable")
    
    with col2:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **Current Equation**: Y = {intercept:.2f} + {slope:.2f}X
    - **Slope ({slope:.2f})**: For every 1 unit increase in X, Y changes by {slope:.2f}
    - **Intercept ({intercept:.2f})**: When X=0, Y={intercept:.2f}
    - **Noise ({noise:.2f})**: Random variation around the line
    """)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔄 Polynomial Regression</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="warning-box">Polynomial regression can model curved relationships, but beware of overfitting with high degrees!</div>', unsafe_allow_html=True)
    
    degree = st.slider("Polynomial Degree", 1, 10, 2, 1)
    fig_poly = create_overfitting_demo(degree)
    st.plotly_chart(fig_poly, use_container_width=True)
    
    if degree == 1:
        st.markdown("**Degree 1**: This is just linear regression - a straight line.")
    elif degree == 2:
        st.markdown("**Degree 2**: Quadratic - one curve. Good for U-shaped or inverted-U patterns.")
    elif degree <= 4:
        st.markdown(f"**Degree {degree}**: Can model {degree-1} curve(s). Be careful not to overfit!")
    else:
        st.markdown(f"⚠️ **Degree {degree}**: Very high! Likely overfitting - the model follows noise, not the true pattern.")
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🎯 Coefficients & Interpretation</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Example: House Price Prediction")
        st.markdown("""
        **Model**: Price = 50,000 + 200(SqFt) + 10,000(Bedrooms) - 5,000(Age)
        
        **Interpretation**:
        - **Intercept (50,000)**: Base price of a house
        - **SqFt (200)**: Each extra sq ft adds $200
        - **Bedrooms (10,000)**: Each bedroom adds $10,000
        - **Age (-5,000)**: Each year older reduces price by $5,000
        """)
    
    with col2:
        st.markdown("#### Coefficient Signs")
        st.markdown("""
        **Positive Coefficient (+)**:
        - As X increases, Y increases
        - Example: More bedrooms → Higher price ✅
        
        **Negative Coefficient (-)**:
        - As X increases, Y decreases
        - Example: Older house → Lower price ✅
        
        **Zero/Near-Zero**:
        - X has little/no effect on Y
        - Consider removing this variable
        """)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 R² and Model Fit</h2></div>', unsafe_allow_html=True)
    
    r2_example = st.slider("R² Value", 0.0, 1.0, 0.75, 0.05)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_example:.2f}")
    with col2:
        if r2_example >= 0.7:
            st.metric("Model Quality", "Good", "✅")
        elif r2_example >= 0.5:
            st.metric("Model Quality", "Moderate", "⚠️")
        else:
            st.metric("Model Quality", "Poor", "❌")
    with col3:
        st.metric("Variance Explained", f"{r2_example*100:.0f}%")
    
    if r2_example >= 0.9:
        st.markdown('<div class="warning-box">⚠️ <b>Very high R²!</b> Check for overfitting or data leakage.</div>', unsafe_allow_html=True)
    elif r2_example >= 0.7:
        st.markdown('<div class="success-box">✅ <b>Good fit!</b> The model explains most of the variance.</div>', unsafe_allow_html=True)
    elif r2_example >= 0.4:
        st.markdown('<div class="warning-box">⚠️ <b>Moderate fit.</b> Consider adding more features or trying different models.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">❌ <b>Poor fit.</b> This model doesn\'t explain the data well. Try a different approach.</div>', unsafe_allow_html=True)

elif page == "⚠️ Common Pitfalls":
    st.markdown('<div class="main-header"><h1>⚠️ Common Pitfalls & Mistakes</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 🚫 Mistake 1: Overfitting")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="warning-box"><h4>What is Overfitting?</h4><p>When your model learns the noise in training data instead of the actual pattern. It performs great on training data but terribly on new data.</p><p><b>Signs:</b> Very high R² (>0.99), complex model, poor test performance</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box"><h4>How to Avoid</h4><ul><li>Use simpler models (lower polynomial degrees)</li><li>Split data: train/test sets</li><li>Use regularization (Ridge/Lasso)</li><li>Cross-validation</li><li>More training data</li></ul></div>', unsafe_allow_html=True)
    
    degree_demo = st.slider("See Overfitting in Action - Polynomial Degree", 1, 15, 1)
    fig_overfit = create_overfitting_demo(degree_demo)
    st.plotly_chart(fig_overfit, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🚫 Mistake 2: Multicollinearity")
    st.markdown('<div class="warning-box"><b>Problem:</b> When predictor variables are highly correlated with each other (e.g., height in cm and height in inches)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Bad Example** 🚫:
        - Predictors: House SqFt, House SqMeters (correlated!)
        - Predictors: Income, Salary (same thing!)
        - Predictors: Age, YearBorn (perfect correlation)
        
        **Why it's bad**:
        - Unstable coefficients
        - Hard to interpret importance
        - Inflated standard errors
        """)
    
    with col2:
        st.markdown("""
        **How to Detect**:
        - Check correlation matrix (>0.8 is concerning)
        - Calculate VIF (>10 is problematic)
        - Look for unexpected coefficient signs
        
        **How to Fix**:
        - Remove one of correlated variables
        - Combine into single variable (PCA)
        - Use Ridge/Lasso regression
        """)
    
    st.markdown("---")
    st.markdown("### 🚫 Mistake 3: Ignoring Assumptions")
    
    assumptions_data = {
        'Assumption': ['Linearity', 'Independence', 'Normality', 'Homoscedasticity', 'No Multicollinearity'],
        'What It Means': [
            'Relationship between X and Y is linear',
            'Observations are independent (not correlated)',
            'Residuals are normally distributed',
            'Residuals have constant variance',
            'Predictors are not highly correlated'
        ],
        'How to Check': [
            'Residual vs Fitted plot',
            'Durbin-Watson test',
            'Q-Q plot, Shapiro-Wilk test',
            'Residual vs Fitted plot',
            'VIF scores, correlation matrix'
        ],
        'Impact if Violated': [
            'Wrong predictions, biased coefficients',
            'Invalid p-values and confidence intervals',
            'Invalid hypothesis tests',
            'Invalid standard errors and tests',
            'Unstable, unreliable coefficients'
        ]
    }
    
    st.dataframe(pd.DataFrame(assumptions_data), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🚫 Mistake 4: Extrapolation")
    st.markdown('<div class="warning-box"><b>Problem:</b> Making predictions outside the range of your training data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Example**:
    - Your data: Houses from 1,000 to 3,000 sq ft
    - You predict: A 10,000 sq ft mansion ❌
    
    **Why it's dangerous**: The relationship might change outside your data range!
    
    **Rule**: Only predict within the range of your training data (1,000 - 3,000 sq ft in this case)
    """)
    
    st.markdown("---")
    st.markdown("### 🚫 Mistake 5: Correlation ≠ Causation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="warning-box"><h4>Correlation (✓)</h4><p>Two variables move together</p><p><b>Example:</b> Ice cream sales and drowning deaths are correlated</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box"><h4>Causation (different!)</h4><p>One variable causes the other</p><p><b>Reality:</b> Both are caused by hot weather, not each other!</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Remember**: Regression finds relationships, not causes!
    - Just because X predicts Y doesn't mean X causes Y
    - There might be a hidden variable (confounding variable)
    - Need experiments or domain knowledge to establish causation
    """)
    
    st.markdown("---")
    st.markdown("### ✅ Best Practices Checklist")
    
    checklist = {
        'Best Practice': [
            '1. Explore your data first',
            '2. Check for outliers',
            '3. Split train/test data',
            '4. Start with simple models',
            '5. Check assumptions',
            '6. Validate on test data',
            '7. Interpret coefficients',
            '8. Document your process'
        ],
        'Why It Matters': [
            'Understand patterns before modeling',
            'Outliers can skew results dramatically',
            'Prevents overfitting, validates model',
            'Complex models often perform worse',
            'Ensures valid inference',
            'True measure of model quality',
            'Make sure results make sense',
            'Reproducibility and transparency'
        ]
    }
    
    st.dataframe(pd.DataFrame(checklist), hide_index=True, use_container_width=True)

elif page == "🧪 Regression Methods":
    st.markdown('<div class="main-header"><h1>🧪 Regression Methods Reference</h1></div>', unsafe_allow_html=True)
    
    methods_data = {
        'Method': ['Linear', 'Multiple', 'Polynomial', 'Logistic', 'Ridge', 'Lasso'],
        'Use Case': [
            '1 predictor, linear relationship',
            'Multiple predictors, linear relationships',
            'Curved/non-linear relationships',
            'Binary outcome (Yes/No, 0/1)',
            'Multiple predictors, multicollinearity present',
            'Multiple predictors, want feature selection'
        ],
        'Complexity': ['Low', 'Medium', 'Medium-High', 'Medium', 'Medium', 'Medium'],
        'Best For': [
            'Simple relationships, easy interpretation',
            'Most common use case',
            'When straight line doesn\'t fit',
            'Classification problems',
            'Preventing overfitting',
            'Automatic feature selection'
        ]
    }
    
    st.dataframe(pd.DataFrame(methods_data), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    method_choice = st.selectbox("Select a method to learn more:", 
                                 ['Linear Regression', 'Multiple Regression', 'Polynomial Regression', 
                                  'Logistic Regression', 'Ridge Regression', 'Lasso Regression'])
    
    if method_choice == 'Linear Regression':
        st.markdown('<div class="section-header"><h2>📈 Linear Regression</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Formula**: Y = β₀ + β₁X + ε
            
            **When to Use**:
            - One independent variable (X)
            - Linear relationship between X and Y
            - Continuous outcome (Y)
            
            **Assumptions**:
            - Linearity
            - Independence
            - Normality of residuals
            - Homoscedasticity
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - Predict salary from years of experience
            - Predict weight from height
            - Predict sales from advertising spend
            
            **Advantages**:
            - Simple and interpretable
            - Fast to train
            - Easy to explain
            
            **Disadvantages**:
            - Only works for linear relationships
            - Sensitive to outliers
            """)
        
        st.code("""
# Python Example
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
        """, language='python')
    
    elif method_choice == 'Multiple Regression':
        st.markdown('<div class="section-header"><h2>📊 Multiple Regression</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Formula**: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
            
            **When to Use**:
            - Multiple independent variables
            - Linear relationships
            - Want to control for multiple factors
            
            **Assumptions**:
            - Linearity
            - Independence
            - Normality of residuals
            - Homoscedasticity
            - No multicollinearity (low VIF)
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - Predict house price from size, age, location
            - Predict test score from study time, sleep, prior scores
            - Predict sales from multiple marketing channels
            
            **Advantages**:
            - Models multiple factors simultaneously
            - Can control for confounding variables
            
            **Disadvantages**:
            - Risk of multicollinearity
            - More complex interpretation
            """)
        
        st.code("""
# Python Example
from sklearn.linear_model import LinearRegression

X = df[['size', 'bedrooms', 'age']]  # Multiple features
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
        """, language='python')
    
    elif method_choice == 'Polynomial Regression':
        st.markdown('<div class="section-header"><h2>📈 Polynomial Regression</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Formula**: Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε
            
            **When to Use**:
            - Relationship is curved (not linear)
            - U-shaped or inverted-U patterns
            - Diminishing returns
            
            **Choosing Degree**:
            - Degree 2: One curve (parabola)
            - Degree 3: Two curves
            - Higher: More curves, but risk overfitting!
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - Advertising spend vs ROI (diminishing returns)
            - Age vs income (peaks mid-career)
            - Temperature vs energy usage (U-shaped)
            
            **Advantages**:
            - Models non-linear relationships
            - Flexible
            
            **Disadvantages**:
            - Easy to overfit
            - Harder to interpret
            - Extrapolation is dangerous
            """)
        
        st.code("""
# Python Example
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
predictions = model.predict(X_poly)
        """, language='python')
    
    elif method_choice == 'Logistic Regression':
        st.markdown('<div class="section-header"><h2>🎯 Logistic Regression</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Formula**: P(Y=1) = 1 / (1 + e^-(β₀ + β₁X))
            
            **When to Use**:
            - Binary outcome (Yes/No, 0/1, True/False)
            - Want probability of outcome
            - Classification problem
            
            **Output**:
            - Probability between 0 and 1
            - Threshold (usually 0.5) for classification
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - Will customer buy? (Yes/No)
            - Is email spam? (Yes/No)
            - Will patient recover? (Yes/No)
            
            **Evaluation Metrics**:
            - Accuracy
            - Precision & Recall
            - AUC-ROC
            - Confusion Matrix
            
            **Not the same as linear regression!**
            - Output is probability, not continuous value
            """)
        
        st.code("""
# Python Example
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
# Get classes (0 or 1)
predictions = model.predict(X_test)
        """, language='python')
    
    elif method_choice == 'Ridge Regression':
        st.markdown('<div class="section-header"><h2>🛡️ Ridge Regression (L2 Regularization)</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What It Does**:
            - Adds penalty for large coefficients
            - Shrinks coefficients toward zero
            - Helps with multicollinearity
            
            **When to Use**:
            - Many correlated predictors
            - Risk of overfitting
            - Unstable coefficient estimates
            
            **Alpha Parameter**:
            - α = 0: Same as linear regression
            - α = small: Light regularization
            - α = large: Strong regularization
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - Many correlated features
            - More predictors than observations
            - High-dimensional data
            
            **Advantages**:
            - Reduces overfitting
            - Handles multicollinearity
            - All features kept
            
            **Disadvantages**:
            - Doesn't remove features (all coefficients > 0)
            - Need to tune alpha
            """)
        
        st.code("""
# Python Example
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # Try different alphas
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Coefficients:", model.coef_)
        """, language='python')
    
    elif method_choice == 'Lasso Regression':
        st.markdown('<div class="section-header"><h2>✂️ Lasso Regression (L1 Regularization)</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What It Does**:
            - Adds penalty for large coefficients
            - Can shrink coefficients to exactly zero
            - Performs automatic feature selection
            
            **When to Use**:
            - Many features, want to select important ones
            - Suspect some features are irrelevant
            - Want sparse model (few non-zero coefficients)
            
            **Alpha Parameter**:
            - α = small: Keeps more features
            - α = large: Keeps fewer features
            """)
        
        with col2:
            st.markdown("""
            **Example Use Cases**:
            - High-dimensional data
            - Feature selection
            - Interpretable models
            
            **Advantages**:
            - Automatic feature selection
            - Reduces overfitting
            - Simpler, more interpretable models
            
            **Disadvantages**:
            - Might remove useful features
            - Need to tune alpha
            - With correlated features, picks one arbitrarily
            """)
        
        st.code("""
# Python Example
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)  # Try different alphas
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Features with non-zero coefficients
important_features = X.columns[model.coef_ != 0]
print("Selected features:", important_features)
        """, language='python')

elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-header"><h1>📈 Model Evaluation & Metrics</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>📊 Regression Metrics</h2></div>', unsafe_allow_html=True)
    
    metrics_data = {
        'Metric': ['R²', 'Adjusted R²', 'MSE', 'RMSE', 'MAE'],
        'Range': ['0 to 1', '0 to 1', '0 to ∞', '0 to ∞', '0 to ∞'],
        'Better': ['Higher', 'Higher', 'Lower', 'Lower', 'Lower'],
        'Interpretation': [
            'Proportion of variance explained',
            'R² adjusted for number of predictors',
            'Average squared error',
            'Average error in original units',
            'Average absolute error'
        ],
        'Good Value': [
            '> 0.7',
            '> 0.7',
            'Close to 0',
            'Close to 0',
            'Close to 0'
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Interactive Metrics Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input Sample Values")
        actual_vals = st.text_area("Actual Values (comma-separated)", "10, 20, 30, 40, 50")
        predicted_vals = st.text_area("Predicted Values (comma-separated)", "12, 19, 28, 42, 51")
        
        if st.button("Calculate Metrics"):
            try:
                y_actual = np.array([float(x.strip()) for x in actual_vals.split(',')])
                y_pred = np.array([float(x.strip()) for x in predicted_vals.split(',')])
                
                if len(y_actual) != len(y_pred):
                    st.error("Arrays must have the same length!")
                else:
                    r2 = r2_score(y_actual, y_pred)
                    mse = mean_squared_error(y_actual, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_actual, y_pred)
                    
                    # Adjusted R² (assuming 1 predictor for simplicity)
                    n = len(y_actual)
                    p = 1
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    
                    st.session_state.metrics_calculated = True
                    st.session_state.calc_r2 = r2
                    st.session_state.calc_adj_r2 = adj_r2
                    st.session_state.calc_mse = mse
                    st.session_state.calc_rmse = rmse
                    st.session_state.calc_mae = mae
                    st.session_state.y_actual = y_actual
                    st.session_state.y_pred = y_pred
            except:
                st.error("Please enter valid comma-separated numbers!")
    
    with col2:
        if 'metrics_calculated' in st.session_state and st.session_state.metrics_calculated:
            st.markdown("#### Results")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R²", f"{st.session_state.calc_r2:.4f}")
                st.metric("MSE", f"{st.session_state.calc_mse:.4f}")
                st.metric("MAE", f"{st.session_state.calc_mae:.4f}")
            with col_b:
                st.metric("Adjusted R²", f"{st.session_state.calc_adj_r2:.4f}")
                st.metric("RMSE", f"{st.session_state.calc_rmse:.4f}")
            
            if st.session_state.calc_r2 >= 0.7:
                st.markdown('<div class="success-box">✅ Good model fit! (R² ≥ 0.7)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Consider improving the model (R² < 0.7)</div>', unsafe_allow_html=True)
    
    if 'metrics_calculated' in st.session_state and st.session_state.metrics_calculated:
        fig = create_scatter_with_regression_line(
            st.session_state.y_actual, 
            st.session_state.y_actual,
            st.session_state.y_pred,
            "Actual vs Predicted",
            "Actual Values", "Predicted Values"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📉 Diagnostic Plots</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Residual Plot")
    st.markdown('<div class="info-box">Shows residuals (errors) vs fitted values. Should show random scatter with no pattern.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What to Look For**:
        - ✅ Random scatter around 0
        - ❌ Patterns (curved, funnel-shaped)
        - ❌ Outliers
        
        **Common Patterns**:
        - **Curved**: Non-linear relationship → Try polynomial
        - **Funnel**: Heteroscedasticity → Transform Y
        - **Clusters**: Different groups → Add categorical variable
        """)
    
    with col2:
        if 'metrics_calculated' in st.session_state and st.session_state.metrics_calculated:
            residuals = st.session_state.y_actual - st.session_state.y_pred
            fig_resid = create_residual_plot(st.session_state.y_pred, residuals)
            st.plotly_chart(fig_resid, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 2. Q-Q Plot")
    st.markdown('<div class="info-box">Checks if residuals are normally distributed. Points should follow the diagonal line.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if 'metrics_calculated' in st.session_state and st.session_state.metrics_calculated:
            residuals = st.session_state.y_actual - st.session_state.y_pred
            fig_qq = create_qq_plot(residuals)
            st.plotly_chart(fig_qq, use_container_width=True)
    
    with col2:
        st.markdown("""
        **What to Look For**:
        - ✅ Points follow the red line
        - ❌ Points deviate from line
        
        **Interpretations**:
        - **S-curve**: Heavy tails (outliers)
        - **Bend up**: Right-skewed
        - **Bend down**: Left-skewed
        
        **If violated**: 
        - Transform Y (log, sqrt)
        - Remove outliers
        - Use robust regression
        """)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🎯 Logistic Regression Metrics</h2></div>', unsafe_allow_html=True)
    
    log_metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Formula': [
            '(TP + TN) / Total',
            'TP / (TP + FP)',
            'TP / (TP + FN)',
            '2 × (Precision × Recall) / (Precision + Recall)',
            'Area under ROC curve'
        ],
        'When to Use': [
            'Balanced classes',
            'Cost of false positives is high',
            'Cost of false negatives is high',
            'Balance precision and recall',
            'Overall model quality'
        ],
        'Good Value': ['> 0.8', '> 0.7', '> 0.7', '> 0.7', '> 0.8']
    }
    
    st.dataframe(pd.DataFrame(log_metrics_data), hide_index=True, use_container_width=True)
    
    st.markdown("""
    **TP** = True Positives, **TN** = True Negatives, **FP** = False Positives, **FN** = False Negatives
    """)

elif page == "🔍 Advanced Concepts":
    st.markdown('<div class="main-header"><h1>🔍 Advanced Concepts & Assumption Testing</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>✅ Regression Assumptions</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Linear regression makes several assumptions. Violating these can lead to biased or unreliable results!</div>', unsafe_allow_html=True)
    
    assumptions_detail = {
        'Assumption': ['Linearity', 'Independence', 'Normality', 'Homoscedasticity', 'No Multicollinearity'],
        'Test Method': [
            'Residual plot',
            'Durbin-Watson test',
            'Q-Q plot, Shapiro-Wilk',
            'Residual plot, Breusch-Pagan',
            'VIF, Correlation matrix'
        ],
        'Pass Criteria': [
            'Random scatter in residual plot',
            'DW ≈ 2 (1.5 - 2.5)',
            'Points follow line in Q-Q plot',
            'Constant spread in residual plot',
            'VIF < 10 for all variables'
        ],
        'If Violated': [
            'Transform variables, add polynomial terms',
            'Add time lags, cluster std errors',
            'Transform Y, use robust methods',
            'Transform Y (log, sqrt), WLS',
            'Remove variables, PCA, Ridge/Lasso'
        ]
    }
    
    st.dataframe(pd.DataFrame(assumptions_detail), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔢 Variance Inflation Factor (VIF)</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><b>VIF</b> measures how much multicollinearity inflates the variance of coefficient estimates.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Interpretation**:
        - **VIF = 1**: No correlation
        - **VIF = 1-5**: Moderate correlation (OK)
        - **VIF = 5-10**: High correlation (concerning)
        - **VIF > 10**: Severe multicollinearity (fix it!)
        
        **Formula**: VIF = 1 / (1 - R²ⱼ)
        
        Where R²ⱼ is from regressing variable j on all other variables
        """)
    
    with col2:
        # Example VIF data
        vif_example = pd.DataFrame({
            'Feature': ['Size', 'Bedrooms', 'Age', 'Location_Score'],
            'VIF': [2.3, 8.5, 1.8, 12.4]
        })
        fig_vif = create_vif_bar_chart(vif_example)
        st.plotly_chart(fig_vif, use_container_width=True)
    
    st.markdown("""
    In this example:
    - ✅ **Size** (2.3) and **Age** (1.8) are fine
    - ⚠️ **Bedrooms** (8.5) is concerning - might correlate with size
    - ❌ **Location_Score** (12.4) is problematic - highly correlated with other variables
    """)
    
    st.code("""
# Calculate VIF in Python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Assuming X is your feature matrix
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
# Remove features with VIF > 10
    """, language='python')
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 Durbin-Watson Test</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><b>Durbin-Watson</b> tests for autocorrelation in residuals (common in time series data).</div>', unsafe_allow_html=True)
    
    dw_value = st.slider("Durbin-Watson Statistic", 0.0, 4.0, 2.0, 0.1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("DW Value", f"{dw_value:.2f}")
    
    with col2:
        if 1.5 <= dw_value <= 2.5:
            st.metric("Autocorrelation", "None", "✅")
        elif dw_value < 1.5:
            st.metric("Autocorrelation", "Positive", "⚠️")
        else:
            st.metric("Autocorrelation", "Negative", "⚠️")
    
    with col3:
        if 1.5 <= dw_value <= 2.5:
            st.metric("Status", "Pass", "✅")
        else:
            st.metric("Status", "Fail", "❌")
    
    st.markdown("""
    **Interpretation**:
    - **DW = 2**: No autocorrelation (perfect!)
    - **DW < 2**: Positive autocorrelation (consecutive residuals are similar)
    - **DW > 2**: Negative autocorrelation (consecutive residuals are opposite)
    - **Rule of thumb**: 1.5 < DW < 2.5 is acceptable
    
    **If violated**:
    - Add time lags as predictors
    - Use time series models (ARIMA)
    - Cluster standard errors
    """)
    
    st.code("""
# Calculate Durbin-Watson in Python
from statsmodels.stats.stattools import durbin_watson

# residuals from your model
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.2f}")

if 1.5 <= dw <= 2.5:
    print("No autocorrelation detected")
else:
    print("Autocorrelation detected - violation!")
    """, language='python')
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📈 P-values & Statistical Significance</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What is a P-value?**
        
        The probability that the observed relationship happened by random chance.
        
        **Interpretation**:
        - **p < 0.001**: Extremely significant (***)
        - **p < 0.01**: Very significant (**)
        - **p < 0.05**: Significant (*)
        - **p >= 0.05**: Not significant (no relationship)
        
        **Example**:
        - p = 0.002 → 0.2% chance this is random → Significant!
        - p = 0.30 → 30% chance this is random → Not significant
        """)
    
    with col2:
        st.markdown("""
        **Confidence Intervals**
        
        Range where true coefficient likely falls (usually 95%).
        
        **Example**: Coefficient = 10, CI = [5, 15]
        - True value is likely between 5 and 15
        - If CI doesn't contain 0 → Significant!
        - Narrow CI → Precise estimate
        - Wide CI → Uncertain estimate
        
        **What if CI includes 0?**
        - Example: CI = [-2, 8]
        - Could be positive or negative
        - Not significant (can't rule out zero effect)
        """)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🎯 Advanced Techniques</h2></div>', unsafe_allow_html=True)
    
    advanced_techniques = {
        'Technique': [
            'Cross-Validation',
            'Regularization',
            'Feature Engineering',
            'Ensemble Methods',
            'Robust Regression'
        ],
        'Purpose': [
            'Validate model on multiple data splits',
            'Prevent overfitting (Ridge/Lasso)',
            'Create new features from existing ones',
            'Combine multiple models',
            'Handle outliers and violations'
        ],
        'When to Use': [
            'Always! Gold standard for validation',
            'High-dimensional data, multicollinearity',
            'Non-linear relationships, domain knowledge',
            'Maximize prediction accuracy',
            'Data has outliers, assumptions violated'
        ],
        'Difficulty': ['Medium', 'Medium', 'High', 'High', 'Medium']
    }
    
    st.dataframe(pd.DataFrame(advanced_techniques), hide_index=True, use_container_width=True)
    
    with st.expander("🔍 Learn More: Cross-Validation"):
        st.markdown("""
        **K-Fold Cross-Validation**:
        1. Split data into K parts (e.g., 5 folds)
        2. Train on K-1 parts, test on remaining part
        3. Repeat K times, rotating test fold
        4. Average performance across all folds
        
        **Benefits**:
        - Uses all data for both training and testing
        - More reliable performance estimate
        - Detects overfitting
        
        **Code**:
        ```python
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"Average R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
        ```
        """)

elif page == "🎯 Model Builder":
    st.markdown('<div class="main-header"><h1>🎯 Interactive Model Builder</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Upload your data, select variables, choose a model type, and get instant results with diagnostics!</div>', unsafe_allow_html=True)
    
    # Step 1: Data Upload
    st.markdown('<div class="section-header"><h2>📁 Step 1: Upload Data</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded! Shape: {st.session_state.data.shape}")
    
    with col2:
        if st.button("Use Sample Data"):
            np.random.seed(42)
            sample_size = 100
            st.session_state.data = pd.DataFrame({
                'Size': np.random.uniform(1000, 3000, sample_size),
                'Bedrooms': np.random.randint(1, 6, sample_size),
                'Age': np.random.uniform(0, 50, sample_size),
                'Price': np.random.uniform(100000, 500000, sample_size)
            })
            # Create realistic relationship
            st.session_state.data['Price'] = (
                200 * st.session_state.data['Size'] + 
                10000 * st.session_state.data['Bedrooms'] - 
                2000 * st.session_state.data['Age'] + 
                np.random.normal(0, 20000, sample_size)
            )
            st.success("✅ Sample data generated!")
    
    # Step 2: Data Preview & Cleaning
    if st.session_state.data is not None:
        st.markdown('<div class="section-header"><h2>📊 Step 2: Preview & Clean Data</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🔍 Data Info", "📊 Distributions"])
        
        with tab1:
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", st.session_state.data.shape[0])
            with col2:
                st.metric("Columns", st.session_state.data.shape[1])
            with col3:
                st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
            
            st.markdown("**Data Types:**")
            st.dataframe(pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Type': st.session_state.data.dtypes,
                'Missing': st.session_state.data.isnull().sum(),
                'Unique': st.session_state.data.nunique()
            }), hide_index=True, use_container_width=True)
        
        with tab3:
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                col_to_plot = st.selectbox("Select column to visualize", numeric_cols)
                fig = create_distribution_plot(st.session_state.data, col_to_plot)
                st.plotly_chart(fig, use_container_width=True)
            
            if len(numeric_cols) >= 2:
                st.markdown("**Correlation Heatmap**")
                fig_corr = create_correlation_heatmap(st.session_state.data)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data Cleaning Options
        with st.expander("🧹 Data Cleaning Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.data.isnull().sum().sum() > 0:
                    st.markdown("**Handle Missing Values:**")
                    missing_method = st.radio("Method", ["Drop rows", "Fill with mean", "Fill with median"])
                    if st.button("Apply Missing Value Treatment"):
                        if missing_method == "Drop rows":
                            st.session_state.data = st.session_state.data.dropna()
                        elif missing_method == "Fill with mean":
                            st.session_state.data = st.session_state.data.fillna(st.session_state.data.mean())
                        elif missing_method == "Fill with median":
                            st.session_state.data = st.session_state.data.fillna(st.session_state.data.median())
                        st.success("✅ Missing values handled!")
                        st.rerun()
            
            with col2:
                st.markdown("**Filter Columns:**")
                cols_to_keep = st.multiselect("Select columns to keep", 
                                             st.session_state.data.columns.tolist(),
                                             default=st.session_state.data.columns.tolist())
                if st.button("Apply Column Filter"):
                    st.session_state.data = st.session_state.data[cols_to_keep]
                    st.success("✅ Columns filtered!")
                    st.rerun()
        
        # Step 3: Model Setup
        st.markdown('<div class="section-header"><h2>⚙️ Step 3: Model Setup</h2></div>', unsafe_allow_html=True)
        
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric columns for regression!")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model_type = st.selectbox("Regression Type", [
                    "Linear Regression",
                    "Multiple Linear Regression",
                    "Polynomial Regression",
                    "Ridge Regression",
                    "Lasso Regression"
                ])
            
            with col2:
                target_var = st.selectbox("Target Variable (Y)", numeric_cols)
            
            with col3:
                feature_vars = st.multiselect("Features (X)", 
                                             [col for col in numeric_cols if col != target_var],
                                             default=[col for col in numeric_cols if col != target_var][:1])
            
            if model_type == "Polynomial Regression":
                poly_degree = st.slider("Polynomial Degree", 1, 5, 2)
            
            if model_type in ["Ridge Regression", "Lasso Regression"]:
                alpha_value = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
            
            # Step 4: Fit Model
            st.markdown('<div class="section-header"><h2>🚀 Step 4: Fit Model</h2></div>', unsafe_allow_html=True)
            
            if len(feature_vars) == 0:
                st.warning("⚠️ Please select at least one feature variable!")
            elif st.button("🚀 Fit Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        X = st.session_state.data[feature_vars].values
                        y = st.session_state.data[target_var].values
                        
                        # Remove any rows with NaN
                        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                        X = X[mask]
                        y = y[mask]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Fit model
                        if model_type == "Linear Regression" or model_type == "Multiple Linear Regression":
                            model = LinearRegression()
                            X_train_model = X_train
                            X_test_model = X_test
                        
                        elif model_type == "Polynomial Regression":
                            poly = PolynomialFeatures(degree=poly_degree)
                            X_train_model = poly.fit_transform(X_train)
                            X_test_model = poly.transform(X_test)
                            model = LinearRegression()
                        
                        elif model_type == "Ridge Regression":
                            model = Ridge(alpha=alpha_value)
                            X_train_model = X_train
                            X_test_model = X_test
                        
                        elif model_type == "Lasso Regression":
                            model = Lasso(alpha=alpha_value)
                            X_train_model = X_train
                            X_test_model = X_test
                        
                        model.fit(X_train_model, y_train)
                        
                        # Predictions
                        y_train_pred = model.predict(X_train_model)
                        y_test_pred = model.predict(X_test_model)
                        
                        # Metrics
                        train_r2 = r2_score(y_train, y_train_pred)
                        test_r2 = r2_score(y_test, y_test_pred)
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        
                        # Adjusted R²
                        n = len(y_test)
                        p = X_test.shape[1]
                        adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
                        
                        # Store results
                        st.session_state.current_model = {
                            'model': model,
                            'model_type': model_type,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'y_train_pred': y_train_pred,
                            'y_test_pred': y_test_pred,
                            'feature_names': feature_vars,
                            'target_name': target_var,
                            'metrics': {
                                'train_r2': train_r2,
                                'test_r2': test_r2,
                                'adj_r2': adj_r2,
                                'train_rmse': train_rmse,
                                'test_rmse': test_rmse,
                                'test_mae': test_mae
                            }
                        }
                        
                        st.success("✅ Model fitted successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Step 5: Results
            if 'current_model' in st.session_state:
                st.markdown('<div class="section-header"><h2>📊 Step 5: Results</h2></div>', unsafe_allow_html=True)
                
                results = st.session_state.current_model
                
                # Metrics
                st.markdown("### 📈 Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("R²", f"{results['metrics']['test_r2']:.4f}")
                with col2:
                    st.metric("Adjusted R²", f"{results['metrics']['adj_r2']:.4f}")
                with col3:
                    st.metric("RMSE", f"{results['metrics']['test_rmse']:.2f}")
                with col4:
                    st.metric("MAE", f"{results['metrics']['test_mae']:.2f}")
                with col5:
                    overfitting = results['metrics']['train_r2'] - results['metrics']['test_r2']
                    st.metric("Train-Test Gap", f"{overfitting:.4f}")
                
                if overfitting > 0.1:
                    st.markdown('<div class="warning-box">⚠️ <b>Possible overfitting!</b> Training R² is much higher than test R². Consider regularization or simpler model.</div>', unsafe_allow_html=True)
                elif results['metrics']['test_r2'] >= 0.7:
                    st.markdown('<div class="success-box">✅ <b>Good model!</b> High R² and low overfitting.</div>', unsafe_allow_html=True)
                
                # Coefficients
                st.markdown("### 🔢 Model Coefficients")
                
                if hasattr(results['model'], 'coef_'):
                    coef_data = pd.DataFrame({
                        'Feature': results['feature_names'],
                        'Coefficient': results['model'].coef_[:len(results['feature_names'])],
                        'Abs_Coefficient': np.abs(results['model'].coef_[:len(results['feature_names'])])
                    }).sort_values('Abs_Coefficient', ascending=False)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(coef_data[['Feature', 'Coefficient']], hide_index=True, use_container_width=True)
                        st.markdown(f"**Intercept**: {results['model'].intercept_:.4f}")
                    
                    with col2:
                        fig = go.Figure(go.Bar(
                            x=coef_data['Coefficient'],
                            y=coef_data['Feature'],
                            orientation='h',
                            marker_color=['red' if c < 0 else 'green' for c in coef_data['Coefficient']]
                        ))
                        fig.update_layout(title="Feature Importance", xaxis_title="Coefficient Value",
                                        template='plotly_white', height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Visualizations
                st.markdown("### 📊 Diagnostic Plots")
                
                tab1, tab2, tab3 = st.tabs(["Predictions", "Residuals", "Q-Q Plot"])
                
                with tab1:
                    if len(results['feature_names']) == 1:
                        fig = create_scatter_with_regression_line(
                            results['X_test'].flatten(),
                            results['y_test'],
                            results['y_test_pred'],
                            f"Predictions: {results['target_name']} vs {results['feature_names'][0]}",
                            results['feature_names'][0],
                            results['target_name']
                        )
                    else:
                        fig = create_scatter_with_regression_line(
                            results['y_test'],
                            results['y_test'],
                            results['y_test_pred'],
                            "Actual vs Predicted",
                            "Actual Values",
                            "Predicted Values"
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    residuals = results['y_test'] - results['y_test_pred']
                    fig = create_residual_plot(results['y_test_pred'], residuals)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Check for patterns
                    st.markdown("**Residual Analysis:**")
                    if np.std(residuals) < results['metrics']['test_rmse'] * 1.1:
                        st.markdown("✅ Residuals appear random (good!)")
                    else:
                        st.markdown("⚠️ Residuals show patterns - consider non-linear model")
                
                with tab3:
                    residuals = results['y_test'] - results['y_test_pred']
                    fig = create_qq_plot(residuals)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Normality test
                    from scipy.stats import shapiro
                    if len(residuals) <= 5000:
                        stat, p_value = shapiro(residuals)
                        if p_value > 0.05:
                            st.markdown(f"✅ Shapiro-Wilk test: p={p_value:.4f} (residuals are normal)")
                        else:
                            st.markdown(f"⚠️ Shapiro-Wilk test: p={p_value:.4f} (residuals may not be normal)")
                
                # Save Model Option
                st.markdown("### 💾 Save Model")
                model_name = st.text_input("Model Name", f"{model_type}_{len(st.session_state.model_results) + 1}")
                if st.button("Save Model for Comparison"):
                    st.session_state.model_results[model_name] = {
                        'r2': results['metrics']['test_r2'],
                        'adj_r2': results['metrics']['adj_r2'],
                        'rmse': results['metrics']['test_rmse'],
                        'mae': results['metrics']['test_mae'],
                        'model_type': model_type,
                        'features': results['feature_names']
                    }
                    st.success(f"✅ Model '{model_name}' saved for comparison!")

elif page == "🔬 Model Comparison":
    st.markdown('<div class="main-header"><h1>🔬 Model Comparison</h1></div>', unsafe_allow_html=True)
    
    if len(st.session_state.model_results) == 0:
        st.markdown('<div class="info-box">No models saved yet! Go to <b>Model Builder</b> to create and save models for comparison.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box">You have <b>{len(st.session_state.model_results)}</b> saved model(s) to compare!</div>', unsafe_allow_html=True)
        
        # Comparison Table
        st.markdown('<div class="section-header"><h2>📊 Performance Comparison</h2></div>', unsafe_allow_html=True)
        
        comparison_df = pd.DataFrame({
            'Model': list(st.session_state.model_results.keys()),
            'Type': [st.session_state.model_results[m]['model_type'] for m in st.session_state.model_results],
            'R²': [st.session_state.model_results[m]['r2'] for m in st.session_state.model_results],
            'Adjusted R²': [st.session_state.model_results[m]['adj_r2'] for m in st.session_state.model_results],
            'RMSE': [st.session_state.model_results[m]['rmse'] for m in st.session_state.model_results],
            'MAE': [st.session_state.model_results[m]['mae'] for m in st.session_state.model_results],
            'Features': [', '.join(st.session_state.model_results[m]['features']) for m in st.session_state.model_results]
        })
        
        st.dataframe(comparison_df.style.highlight_max(subset=['R²', 'Adjusted R²'], color='lightgreen')
                                      .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                    hide_index=True, use_container_width=True)
        
        # Best Model
        best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        st.markdown(f'<div class="success-box">🏆 <b>Best Model (by R²):</b> {best_model_name}</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<div class="section-header"><h2>📈 Visual Comparison</h2></div>', unsafe_allow_html=True)
        
        if len(st.session_state.model_results) >= 2:
            fig = create_metrics_comparison_chart(st.session_state.model_results)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### R² Scores")
            fig_r2 = go.Figure(go.Bar(
                x=list(st.session_state.model_results.keys()),
                y=[st.session_state.model_results[m]['r2'] for m in st.session_state.model_results],
                marker_color='#667eea'
            ))
            fig_r2.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            st.markdown("### RMSE Scores")
            fig_rmse = go.Figure(go.Bar(
                x=list(st.session_state.model_results.keys()),
                y=[st.session_state.model_results[m]['rmse'] for m in st.session_state.model_results],
                marker_color='#f5576c'
            ))
            fig_rmse.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Clear Models
        st.markdown("---")
        if st.button("🗑️ Clear All Saved Models"):
            st.session_state.model_results = {}
            st.success("All models cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Regression Analysis Hub</b> | Built with Streamlit 📊</p>
    <p>A comprehensive tool for regression analysis, model building, and evaluation</p>
</div>
""", unsafe_allow_html=True)

