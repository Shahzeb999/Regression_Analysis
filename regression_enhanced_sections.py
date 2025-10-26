# ENHANCED SECTIONS FOR REGRESSION ANALYSIS APP
# Copy these sections into regression_analysis_app.py after the existing pages

# ==============================
# CORRELATION ANALYSIS SECTION
# ==============================

"""
elif page == "📈 Correlation Analysis":
    st.markdown('<div class="main-header"><h1>📈 Correlation Analysis</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Correlation measures the strength and direction of the linear relationship between two variables.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>🔍 Understanding Correlation</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        **Pearson Correlation Coefficient (r)**
        
        **Range**: -1 to +1
        
        **Interpretation**:
        - **r = +1**: Perfect positive correlation
        - **r = +0.7 to +1**: Strong positive
        - **r = +0.3 to +0.7**: Moderate positive
        - **r = -0.3 to +0.3**: Weak/No correlation
        - **r = -0.7 to -0.3**: Moderate negative
        - **r = -1 to -0.7**: Strong negative
        - **r = -1**: Perfect negative correlation
        ''')
    
    with col2:
        st.markdown('''
        **Important Notes**:
        - Correlation ≠ Causation!
        - Only measures **linear** relationships
        - Sensitive to outliers
        - Doesn't capture non-linear patterns
        
        **Formula**: r = Σ[(X-X̄)(Y-Ȳ)] / √[Σ(X-X̄)²Σ(Y-Ȳ)²]
        ''')
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🎮 Interactive Correlation Demo</h2></div>', unsafe_allow_html=True)
    
    n_points = st.slider("Number of data points", 20, 200, 50)
    correlation_strength = st.slider("Correlation strength", -1.0, 1.0, 0.7, 0.1)
    noise_level = st.slider("Noise level", 0.0, 5.0, 1.0, 0.1)
    
    np.random.seed(42)
    x_demo = np.random.normal(0, 1, n_points)
    y_demo = correlation_strength * x_demo + np.random.normal(0, noise_level, n_points)
    
    actual_corr = np.corrcoef(x_demo, y_demo)[0, 1]
    
    fig = create_correlation_scatter(x_demo, y_demo, "Correlation Demonstration", "X Variable", "Y Variable")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Correlation", f"{correlation_strength:.3f}")
    with col2:
        st.metric("Actual Correlation", f"{actual_corr:.3f}")
    with col3:
        p_value = stats.pearsonr(x_demo, y_demo)[1]
        st.metric("P-value", f"{p_value:.4f}")
    
    if p_value < 0.05:
        st.markdown('<div class="success-box">✅ <b>Significant correlation</b> (p < 0.05)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <b>Not significant</b> (p >= 0.05)</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 Correlation with Your Data</h2></div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Matrix")
            corr_matrix = st.session_state.data[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate='%{text}',
                textfont={"size":10},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(title="Correlation Heatmap", height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Pairwise Correlation Analysis")
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1", numeric_cols)
            with col2:
                var2 = st.selectbox("Variable 2", [c for c in numeric_cols if c != var1])
            
            if var1 and var2:
                x_data = st.session_state.data[var1].values
                y_data = st.session_state.data[var2].values
                
                # Remove NaN
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[mask]
                y_data = y_data[mask]
                
                corr, p_val = stats.pearsonr(x_data, y_data)
                
                fig = create_correlation_scatter(x_data, y_data, f"{var1} vs {var2}", var1, var2)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pearson r", f"{corr:.4f}")
                with col2:
                    st.metric("R²", f"{corr**2:.4f}")
                with col3:
                    st.metric("P-value", f"{p_val:.4f}")
                with col4:
                    if abs(corr) >= 0.7:
                        st.metric("Strength", "Strong", "✅")
                    elif abs(corr) >= 0.3:
                        st.metric("Strength", "Moderate", "⚠️")
                    else:
                        st.metric("Strength", "Weak", "❌")
                
                st.markdown(f"""
                **Interpretation**:
                - The correlation coefficient is **{corr:.4f}**
                - This indicates a **{"positive" if corr > 0 else "negative"}** relationship
                - The relationship is **{"strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak"}**
                - **{corr**2*100:.1f}%** of variance in {var2} is explained by {var1}
                - The correlation is **{"statistically significant" if p_val < 0.05 else "not statistically significant"}** (α = 0.05)
                """)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis!")
    else:
        st.info("📁 Upload data in Model Builder to perform correlation analysis!")
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📚 Correlation Types</h2></div>', unsafe_allow_html=True)
    
    corr_types = {
        'Method': ['Pearson', 'Spearman', 'Kendall'],
        'Measures': ['Linear relationship', 'Monotonic relationship', 'Ordinal association'],
        'When to Use': [
            'Continuous data, linear relationship',
            'Ordinal data or non-linear monotonic',
            'Small sample, ordinal data'
        ],
        'Assumptions': [
            'Normal distribution, linear',
            'Monotonic relationship',
            'Ordinal or continuous'
        ]
    }
    
    st.dataframe(pd.DataFrame(corr_types), hide_index=True, use_container_width=True)
"""

# ==============================
# STATISTICAL INFERENCE SECTION
# ==============================

"""
elif page == "🎯 Statistical Inference":
    st.markdown('<div class="main-header"><h1>🎯 Statistical Inference</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Statistical inference allows us to make conclusions about populations based on sample data.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>📊 Hypothesis Testing for Regression</h2></div>', unsafe_allow_html=True)
    
    st.markdown('''
    ### Testing Individual Coefficients
    
    **Null Hypothesis (H₀)**: β₁ = 0 (no relationship)
    **Alternative Hypothesis (H₁)**: β₁ ≠ 0 (relationship exists)
    
    **Test Statistic**: t = β̂₁ / SE(β̂₁)
    
    **Decision Rule**:
    - If p-value < α (typically 0.05) → Reject H₀ → Coefficient is significant
    - If p-value ≥ α → Fail to reject H₀ → Coefficient is not significant
    ''')
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔢 Confidence Intervals</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        **For Coefficients (β)**:
        
        95% CI = β̂ ± t(α/2, n-2) × SE(β̂)
        
        **Interpretation**:
        - We are 95% confident the true coefficient lies within this range
        - If CI doesn't contain 0 → Coefficient is significant
        - Narrow CI → Precise estimate
        - Wide CI → Uncertain estimate
        ''')
    
    with col2:
        st.markdown('''
        **For Predictions (Ŷ)**:
        
        **Confidence Interval** (for mean):
        - CI for average Y at given X
        - Narrower band
        
        **Prediction Interval** (for individual):
        - PI for single observation
        - Wider band (accounts for individual variation)
        ''')
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🧮 Interactive Calculator</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### Confidence Interval Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        coef_estimate = st.number_input("Coefficient Estimate (β̂)", value=2.5)
        std_error = st.number_input("Standard Error", value=0.5, min_value=0.01)
    
    with col2:
        n_obs = st.number_input("Sample Size (n)", value=100, min_value=3)
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    # Calculate
    df = n_obs - 2
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df)
    margin_error = t_critical * std_error
    ci_lower = coef_estimate - margin_error
    ci_upper = coef_estimate + margin_error
    
    # T-statistic
    t_stat = coef_estimate / std_error
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    with col3:
        st.metric("T-statistic", f"{t_stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
    
    st.markdown(f'''
    **Results:**
    - **Point Estimate**: {coef_estimate:.4f}
    - **{confidence_level*100:.0f}% Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]
    - **Margin of Error**: ±{margin_error:.4f}
    - **Interpretation**: We are {confidence_level*100:.0f}% confident the true coefficient is between {ci_lower:.4f} and {ci_upper:.4f}
    ''')
    
    if p_value < 0.05:
        st.markdown('<div class="success-box">✅ <b>Significant!</b> The coefficient is statistically significant (p < 0.05)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <b>Not significant</b> (p >= 0.05). Cannot reject null hypothesis.</div>', unsafe_allow_html=True)
    
    if 0 < ci_lower or 0 > ci_upper:
        st.markdown('<div class="success-box">✅ The confidence interval does NOT contain zero → Coefficient is significant</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ The confidence interval CONTAINS zero → Coefficient may not be significant</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 F-Test for Overall Model Significance</h2></div>', unsafe_allow_html=True)
    
    st.markdown('''
    **Purpose**: Tests if at least one predictor is useful
    
    **Hypotheses**:
    - H₀: β₁ = β₂ = ... = βₚ = 0 (no predictors are useful)
    - H₁: At least one βⱼ ≠ 0 (at least one predictor is useful)
    
    **Test Statistic**: F = (SSR/p) / (SSE/(n-p-1))
    
    Where:
    - SSR = Sum of Squares Regression
    - SSE = Sum of Squares Error
    - p = number of predictors
    - n = sample size
    ''')
    
    st.markdown("### F-Test Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        r_squared = st.slider("R² value", 0.0, 1.0, 0.75, 0.01)
        n_predictors = st.number_input("Number of predictors (p)", value=3, min_value=1)
        sample_size = st.number_input("Sample size (n)", value=100, min_value=10)
    
    # Calculate F-statistic
    f_stat = (r_squared / n_predictors) / ((1 - r_squared) / (sample_size - n_predictors - 1))
    df1 = n_predictors
    df2 = sample_size - n_predictors - 1
    f_p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    with col2:
        st.metric("F-statistic", f"{f_stat:.4f}")
        st.metric("P-value", f"{f_p_value:.6f}")
        st.metric("Degrees of Freedom", f"({df1}, {df2})")
    
    if f_p_value < 0.05:
        st.markdown('<div class="success-box">✅ <b>Model is significant!</b> At least one predictor is useful (p < 0.05)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <b>Model is not significant</b> (p >= 0.05). Predictors may not be useful.</div>', unsafe_allow_html=True)
"""

# ==============================
# INFLUENTIAL POINTS SECTION
# ==============================

"""
elif page == "🔬 Influential Points":
    st.markdown('<div class="main-header"><h1>🔬 Influential Points & Outliers</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Influential points can dramatically affect regression results. It\'s crucial to identify and handle them appropriately.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>📚 Types of Unusual Points</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="definition-box"><h4>🎯 Outliers</h4><p><b>Definition</b>: Points with unusual Y values</p><p><b>Detection</b>: Standardized residuals > 2 or < -2</p><p><b>Effect</b>: Affects R² and error metrics</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="definition-box"><h4>⚖️ High Leverage</h4><p><b>Definition</b>: Points with unusual X values</p><p><b>Detection</b>: Leverage > 2p/n</p><p><b>Effect</b>: Potential to influence slope</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="definition-box"><h4>💥 Influential</h4><p><b>Definition</b>: Points that greatly affect regression line</p><p><b>Detection</b>: Cook\'s Distance > 1</p><p><b>Effect</b>: Changes coefficients significantly</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔍 Detection Methods</h2></div>', unsafe_allow_html=True)
    
    detection_methods = {
        'Method': ['Standardized Residuals', 'Leverage (h)', "Cook's Distance", 'DFBETAS', 'DFFITS'],
        'Purpose': [
            'Identify outliers in Y',
            'Identify unusual X values',
            'Identify influential points',
            'Change in coefficients',
            'Change in fitted values'
        ],
        'Threshold': [
            '> 2 or < -2',
            '> 2p/n',
            '> 1 (or > 4/n)',
            '> 2/√n',
            '> 2√(p/n)'
        ],
        'Interpretation': [
            'Large residual = poor fit',
            'Far from mean of X',
            'Removing changes results a lot',
            'Removing changes coefficient',
            'Removing changes prediction'
        ]
    }
    
    st.dataframe(pd.DataFrame(detection_methods), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🎮 Interactive Demonstration</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### Create Data with Influential Point")
    
    add_outlier = st.checkbox("Add outlier (unusual Y)", value=False)
    add_leverage = st.checkbox("Add high leverage point (unusual X)", value=False)
    add_influential = st.checkbox("Add influential point (both)", value=True)
    
    # Generate data
    np.random.seed(42)
    x = np.random.uniform(0, 10, 30)
    y = 2 * x + 5 + np.random.normal(0, 2, 30)
    
    if add_outlier:
        x = np.append(x, 5)
        y = np.append(y, 30)  # Way above the line
    
    if add_leverage:
        x = np.append(x, 20)  # Far from other X values
        y = np.append(y, 2 * 20 + 5)  # But on the line
    
    if add_influential:
        x = np.append(x, 20)  # Far from other X values
        y = np.append(y, 10)  # And way off the line
    
    # Fit model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    
    # Create plot
    fig = create_scatter_with_regression_line(x, y, y_pred, 
                                              "Influential Points Demonstration",
                                              "X", "Y")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 Diagnostic Plots</h2></div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None and 'current_model' in st.session_state:
        results = st.session_state.current_model
        
        try:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import OLSInfluence
            
            X = results['X_train']
            y = results['y_train']
            
            # Fit OLS model for diagnostics
            X_with_const = sm.add_constant(X)
            ols_model = sm.OLS(y, X_with_const).fit()
            influence = OLSInfluence(ols_model)
            
            # Cook's Distance
            st.markdown("### 1. Cook's Distance")
            cooks_d = influence.cooks_distance[0]
            threshold_cooks = 1.0
            
            fig_cooks = create_cooks_distance_plot(cooks_d, threshold_cooks)
            st.plotly_chart(fig_cooks, use_container_width=True)
            
            n_influential = np.sum(cooks_d > threshold_cooks)
            if n_influential > 0:
                st.markdown(f'<div class="warning-box">⚠️ Found <b>{n_influential}</b> influential point(s) with Cook\'s D > {threshold_cooks}</div>', unsafe_allow_html=True)
                influential_indices = np.where(cooks_d > threshold_cooks)[0]
                st.markdown(f"**Influential observation indices**: {influential_indices.tolist()}")
            else:
                st.markdown('<div class="success-box">✅ No highly influential points detected</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Leverage
            st.markdown("### 2. Leverage")
            leverage = influence.hat_matrix_diag
            threshold_leverage = 2 * X.shape[1] / len(X)
            
            fig_leverage = create_leverage_plot(leverage, threshold_leverage)
            st.plotly_chart(fig_leverage, use_container_width=True)
            
            n_leverage = np.sum(leverage > threshold_leverage)
            if n_leverage > 0:
                st.markdown(f'<div class="warning-box">⚠️ Found <b>{n_leverage}</b> high leverage point(s) > {threshold_leverage:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ No high leverage points detected</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Standardized Residuals
            st.markdown("### 3. Standardized Residuals")
            std_resid = influence.resid_studentized_internal
            
            fig = go.Figure()
            colors = ['red' if abs(r) > 2 else '#667eea' for r in std_resid]
            fig.add_trace(go.Scatter(x=list(range(len(std_resid))), y=std_resid,
                                    mode='markers', marker=dict(size=8, color=colors)))
            fig.add_hline(y=2, line_dash="dash", line_color="red")
            fig.add_hline(y=-2, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_dash="solid", line_color="gray")
            fig.update_layout(title="Standardized Residuals", xaxis_title="Observation Index",
                            yaxis_title="Standardized Residual", template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            n_outliers = np.sum(np.abs(std_resid) > 2)
            if n_outliers > 0:
                st.markdown(f'<div class="warning-box">⚠️ Found <b>{n_outliers}</b> potential outlier(s) with |residual| > 2</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ No outliers detected</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error calculating diagnostics: {str(e)}")
            st.info("Build a model in Model Builder first!")
    else:
        st.info("📁 Build a model in Model Builder to see influential point diagnostics!")
    
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🛠️ What to Do About Influential Points</h2></div>', unsafe_allow_html=True)
    
    st.markdown('''
    ### Decision Framework:
    
    1. **Investigate First!**
       - Is it a data entry error? → Fix it
       - Is it a measurement error? → Remove it
       - Is it a valid observation? → Keep it (usually)
    
    2. **Run Sensitivity Analysis**
       - Fit model with and without the point
       - Compare results
       - If results change dramatically → Point is influential
    
    3. **Options:**
       - **Keep it**: If it's valid data (most common)
       - **Remove it**: If it's an error (rare)
       - **Transform**: Apply log or other transformation
       - **Robust regression**: Use methods less sensitive to outliers
       - **Report it**: Note the influence in your analysis
    
    4. **⚠️ Never automatically remove points!**
       - Each case is unique
       - Removing valid data introduces bias
       - Transparency is key
    ''')
"""

print("Enhanced sections created successfully!")
print("Now adding these sections to the main app...")

