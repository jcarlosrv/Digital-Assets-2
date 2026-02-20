import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Quant16 Portfolio Optimization - Digital Assets", layout="wide", page_icon="📊")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0F0F0F;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #222222;
    }
    
    h1, h2, h3 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Define colors
COLORS = {
    'primary': '#00D9A3',
    'secondary': '#FF8C42',
    'danger': '#FF4444',
    'background': '#0F0F0F',
    'card': '#1E1E1E'
}

# === CARD HELPER ===
def metric_card(label, value, subtitle="", accent=None):
    """Consistent metric card with top accent bar."""
    c = accent or COLORS['primary']
    sub_html = f"<p style='margin: 0; color: {c}; font-size: 12px; font-weight: 600;'>{subtitle}</p>" if subtitle else ""
    return f"""
    <div style='
        background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%);
        border-radius: 10px;
        border-top: 3px solid {c};
        padding: 18px 16px 14px 16px;
        height: 115px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center;
    '>
        <p style='margin: 0 0 4px 0; color: #D0D0D0; font-size: 10px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;'>{label}</p>
        <p style='margin: 0; color: #FFFFFF; font-size: 28px; font-weight: 700; line-height: 1.1;'>{value}</p>
        {sub_html}
    </div>
    """

def build_sankey_capped(df_in, levels, value_col, max_per_level=10):
    """Build Sankey nodes/links from a dataframe with hierarchical levels, capping total unique nodes per level."""
    import plotly.graph_objects as go
    
    # Build aggregated flows between each pair of adjacent levels
    all_links = []  # list of (source_label, target_label, value)
    
    for i in range(len(levels) - 1):
        parent_col = levels[i]
        child_col = levels[i + 1]
        
        grouped = df_in.groupby([parent_col, child_col])[value_col].sum().reset_index()
        
        # Cap TOTAL unique children across all parents at max_per_level
        child_totals = grouped.groupby(child_col)[value_col].sum().sort_values(ascending=False)
        top_children = child_totals.head(max_per_level).index.tolist()
        
        top_rows = grouped[grouped[child_col].isin(top_children)]
        rest_rows = grouped[~grouped[child_col].isin(top_children)]
        
        # Add top rows as-is
        for _, row in top_rows.iterrows():
            src = f"L{i}_{row[parent_col]}"
            tgt = f"L{i+1}_{row[child_col]}"
            all_links.append((src, tgt, row[value_col]))
        
        # Roll up the rest into "Others" per parent
        if len(rest_rows) > 0:
            others_label = f"Others"
            for parent_val in rest_rows[parent_col].unique():
                others_val = rest_rows[rest_rows[parent_col] == parent_val][value_col].sum()
                if others_val > 0:
                    src = f"L{i}_{parent_val}"
                    tgt = f"L{i+1}_{others_label}"
                    all_links.append((src, tgt, others_val))
    
    # Build unique node list preserving order
    seen = set()
    node_labels = []
    for src, tgt, _ in all_links:
        for n in [src, tgt]:
            if n not in seen:
                seen.add(n)
                node_labels.append(n)
    
    node_map = {label: i for i, label in enumerate(node_labels)}
    
    # Display labels (strip prefix)
    display_labels = [l.split('_', 1)[1][:35] if '_' in l else l for l in node_labels]
    
    # Color by level
    level_colors = [COLORS['primary'], COLORS['secondary'], '#9D4EDD', 'rgba(255,255,255,0.4)', '#4ECDC4']
    link_alphas  = ['rgba(0,217,163,0.25)', 'rgba(255,140,66,0.25)', 'rgba(157,78,221,0.2)', 'rgba(78,205,196,0.2)']
    
    node_colors_list = []
    for nl in node_labels:
        lvl = int(nl[1]) if nl[1].isdigit() else 0
        node_colors_list.append(level_colors[min(lvl, len(level_colors)-1)])
    
    sources = [node_map[s] for s, _, _ in all_links]
    targets = [node_map[t] for _, t, _ in all_links]
    values  = [v for _, _, v in all_links]
    link_colors_list = []
    for s, _, _ in all_links:
        lvl = int(s[1]) if s[1].isdigit() else 0
        link_colors_list.append(link_alphas[min(lvl, len(link_alphas)-1)])
    
    return display_labels, node_colors_list, sources, targets, values, link_colors_list

# Load data
@st.cache_data
def load_data():
    # Read main asset list
    df = pd.read_excel('AssetList.xlsx', sheet_name='AssetList')
    
    # Read component-level scores
    criticality_df = pd.read_excel('AssetList.xlsx', sheet_name='Criticality Score')
    specificity_df = pd.read_excel('AssetList.xlsx', sheet_name='Specificity Score')
    
    # Read AI/IA data
    ia_time_df = pd.read_excel('AssetList.xlsx', sheet_name='IA Engagement Time')
    ai_potential_df = pd.read_excel('AssetList.xlsx', sheet_name='AI Automation Potential')
    ia_structure_df = pd.read_excel('AssetList.xlsx', sheet_name='IA Structure')
    
    # Calculate average scores across all components for each app
    app_names = criticality_df.iloc[1:, 0].values  # Skip header row
    
    # Get all component scores for each app (excluding app name column)
    criticality_scores = criticality_df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    specificity_scores = specificity_df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Calculate mean across all components for each app
    avg_criticality = criticality_scores.mean(axis=1).values
    avg_specificity = specificity_scores.mean(axis=1).values
    
    # Create mapping from app name to average scores
    score_mapping = pd.DataFrame({
        'App Name': app_names,
        'Avg Criticality': avg_criticality,
        'Avg Specificity': avg_specificity
    })
    
    # Merge with main dataframe
    df = df.merge(score_mapping, on='App Name', how='left')
    
    # Calculate Combined Score from component averages
    df['Combined Score'] = (df['Avg Specificity'] + df['Avg Criticality']) / 2
    
    # Savings formula: Higher combined score = Lower savings (harder to optimize)
    # Score 1.4 → 70% savings, Score 5.0 → 20% savings
    df['Savings %'] = 0.70 - (0.50 / 3.6) * (df['Combined Score'] - 1.4)
    df['Savings %'] = df['Savings %'].clip(lower=0.20, upper=0.70)
    
    # Calculate financial metrics
    df['Base Spend'] = df['Development Cost']
    df['Potential Savings'] = df['Base Spend'] * df['Savings %']
    df['Optimized Spend'] = df['Base Spend'] - df['Potential Savings']
    
    # Calculate AI Impact based on IA Engagement Time × AI Automation Potential
    # For each app, calculate weighted AI automation across all activities
    activity_columns = [col for col in ia_time_df.columns if col != 'App Name']
    
    ai_savings_list = []
    for idx, row in df.iterrows():
        app_name = row['App Name']
        
        # Get engagement time and automation potential for this app
        ia_time = ia_time_df[ia_time_df['App Name'] == app_name]
        ai_pot = ai_potential_df[ai_potential_df['App Name'] == app_name]
        
        if len(ia_time) > 0 and len(ai_pot) > 0:
            # Calculate weighted AI savings: sum of (time × potential) across all activities
            total_ai_impact = 0
            for activity in activity_columns:
                time_pct = ia_time[activity].values[0]
                automation_pct = ai_pot[activity].values[0]
                total_ai_impact += time_pct * automation_pct
            
            ai_savings_list.append(total_ai_impact)
        else:
            ai_savings_list.append(0)
    
    df['AI Savings %'] = ai_savings_list
    df['AI Savings % Raw'] = df['AI Savings %'].copy()  # Store raw for reference
    
    return df, ia_time_df, ai_potential_df, ia_structure_df

df, ia_time_df, ai_potential_df, ia_structure_df = load_data()

# Convert Department to string to handle mixed types
df['Department'] = df['Department'].astype(str)

# Sidebar - Global Filters
st.sidebar.header("Filters")

sector_options = sorted([str(s) for s in df['Sector'].unique() if pd.notna(s)])
sidebar_sector = st.sidebar.selectbox(
    "Sector",
    options=['All'] + sector_options,
    index=0,
    help="Filter all tabs by sector"
)

dept_options = sorted([str(d) for d in df['Department'].unique() if pd.notna(d)])
sidebar_dept = st.sidebar.selectbox(
    "Department",
    options=['All'] + dept_options,
    index=0,
    help="Filter all tabs by department"
)

fy_values = sorted(df['Fiscal Year'].dropna().unique().tolist())
if len(fy_values) >= 2:
    fy_min, fy_max = st.sidebar.slider(
        "Fiscal Year Range",
        min_value=int(min(fy_values)),
        max_value=int(max(fy_values)),
        value=(int(min(fy_values)), int(max(fy_values))),
        step=1,
        help="Filter all tabs by fiscal year range"
    )
else:
    fy_min = fy_max = int(fy_values[0]) if fy_values else 2020

# Apply global filters
if sidebar_sector != 'All':
    df = df[df['Sector'].astype(str) == sidebar_sector]
if sidebar_dept != 'All':
    df = df[df['Department'].astype(str) == sidebar_dept]
df = df[(df['Fiscal Year'] >= fy_min) & (df['Fiscal Year'] <= fy_max)]

st.sidebar.markdown("---")

# Sidebar - Savings adjustment parameter
st.sidebar.header("Savings Parameter")

savings_factor = st.sidebar.slider(
    "Component Optimization Realization (%)",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
    help="Adjust the overall savings potential from 0% (no savings) to 100% (maximum savings)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### AI Impact")
ai_factor = st.sidebar.slider(
    "AI Cost Reduction (%)",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
    help="Adjust AI-driven cost reduction from 0% to 100%"
)

app_cost_share = st.sidebar.slider(
    "Applications Cost Share (%)",
    min_value=0,
    max_value=100,
    value=65,
    step=5,
    help="% of app cost allocated to Applications components. Remainder goes to Infrastructure."
)
infra_cost_share = 100 - app_cost_share

st.sidebar.markdown("---")
st.sidebar.markdown("#### Projections")
enable_projections = st.sidebar.checkbox("Enable Projections", value=False)
projection_years = st.sidebar.slider(
    "Projection Years",
    min_value=1,
    max_value=7,
    value=5,
    step=1,
    help="Number of years to project forward"
)
noise_level = st.sidebar.slider(
    "Noise Level (%)",
    min_value=0,
    max_value=20,
    value=5,
    step=1,
    help="Random variation in projections"
)

# Apply savings factor
df['Adjusted Savings %'] = df['Savings %'] * (savings_factor / 100)
df['Potential Savings'] = df['Base Spend'] * df['Adjusted Savings %']
df['Optimized Spend'] = df['Base Spend'] - df['Potential Savings']

# Calculate AI Impact using the pre-calculated AI Savings % from IA data
# AI Savings % is already calculated as weighted sum of (Engagement Time × Automation Potential)
df['AI Savings % Adjusted'] = df['AI Savings %'] * (ai_factor / 100)
df['AI Cost Reduction'] = df['Optimized Spend'] * df['AI Savings % Adjusted']
df['Total AI Cost'] = df['Optimized Spend'] - df['AI Cost Reduction']

# Generate projections if enabled
def generate_projections(historical_df, years, noise_pct):
    import numpy as np
    
    # Aggregate by fiscal year
    fy_agg = historical_df.groupby('Fiscal Year').agg({
        'Base Spend': 'sum',
        'Optimized Spend': 'sum',
        'Total AI Cost': 'sum'
    }).reset_index()
    
    # Calculate trends (linear regression on log scale for growth)
    from numpy.polynomial import Polynomial
    
    years_data = fy_agg['Fiscal Year'].values
    base_data = fy_agg['Base Spend'].values
    opt_data = fy_agg['Optimized Spend'].values
    ai_data = fy_agg['Total AI Cost'].values
    
    # Fit polynomial (degree 1 = linear trend)
    base_poly = Polynomial.fit(years_data, base_data, 1)
    opt_poly = Polynomial.fit(years_data, opt_data, 1)
    ai_poly = Polynomial.fit(years_data, ai_data, 1)
    
    # Generate future years
    last_year = years_data.max()
    future_years = np.arange(last_year + 1, last_year + years + 1)
    
    # Project with noise
    np.random.seed(42)  # For reproducibility
    projections = []
    
    for idx, year in enumerate(future_years):
        # Base trend projection
        base_proj = base_poly(year)
        opt_proj = opt_poly(year)
        ai_proj = ai_poly(year)
        
        # Add noise (random variation)
        noise_factor = 1 + (np.random.uniform(-noise_pct, noise_pct) / 100)
        
        # Calculate Potential Spending with gradual transition
        # Year 0 (first projection): 66% base, 34% optimized
        # Year 1 (second projection): 34% base, 66% optimized
        # Year 2+: 100% optimized
        if idx == 0:  # First projection year
            potential_spend = (base_proj * 0.66 + opt_proj * 0.34) * noise_factor
        elif idx == 1:  # Second projection year
            potential_spend = (base_proj * 0.34 + opt_proj * 0.66) * noise_factor
        else:  # Third year onwards - full optimized
            potential_spend = opt_proj * noise_factor
        
        projections.append({
            'Fiscal Year': int(year),
            'Base Spend': base_proj * noise_factor,
            'Optimized Spend': opt_proj * noise_factor,
            'Total AI Cost': ai_proj * noise_factor,
            'Potential Spend': potential_spend,
            'Is Projection': True
        })
    
    # Mark historical data - Potential Spend = Base Spend for historical
    fy_agg['Is Projection'] = False
    fy_agg['Potential Spend'] = fy_agg['Base Spend']
    
    # Combine historical and projections
    proj_df = pd.DataFrame(projections)
    combined = pd.concat([fy_agg, proj_df], ignore_index=True)
    
    return combined

if enable_projections:
    fy_data = generate_projections(df, projection_years, noise_level)
else:
    fy_data = df.groupby('Fiscal Year').agg({
        'Base Spend': 'sum',
        'Optimized Spend': 'sum',
        'Total AI Cost': 'sum'
    }).reset_index()
    fy_data['Is Projection'] = False
    # Potential Spend only defined from 2025 onwards
    fy_data['Potential Spend'] = fy_data.apply(
        lambda row: row['Base Spend'] if row['Fiscal Year'] < 2025 else row['Base Spend'],
        axis=1
    )

# Header
st.markdown("""
<div style='background: linear-gradient(90deg, #00D9A3 0%, #00A87E 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <div style='display: flex; align-items: center; gap: 15px;'>
        <div style='background-color: #0F0F0F; border-radius: 8px; padding: 8px 12px; font-weight: 700; font-size: 18px; color: #00D9A3;'>
            Q16
        </div>
        <div>
            <h1 style='margin: 0; color: #0F0F0F; font-size: 28px; font-weight: 700;'>
                Quant16 Portfolio Optimization Model - Digital Assets
            </h1>
            <p style='margin: 5px 0 0 0; color: #0F0F0F; font-size: 14px;'>
                Cloud-Enabled Low-Code Platform Migration Savings Scenario
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info about scoring methodology
st.markdown(f"""
<div style='background-color: #222222; padding: 12px; border-radius: 6px; margin-bottom: 20px; border-left: 3px solid {COLORS['primary']}'>
    <span style='color: #D0D0D0; font-size: 12px;'>
    💡 <b>Savings Calculation:</b> Based on average of 40+ component-level scores (Criticality + Specificity) per application
    </span>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab5, tab6, tab7 = st.tabs([
    "📊 Portfolio Overview", 
    "📈 Portfolio Variance", 
    "🤖 AI Impact",
    "📋 Asset Inventory",
    "📐 Vendor Cost Benchmarking",
    "🎯 Vendor Recommendation"
])

with tab1:
    # Savings scenario indicator
    st.markdown(f"""
    <div style='background-color: #222222; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-top: 3px solid {COLORS['primary']}'>
        <span style='color: #D0D0D0; font-size: 13px;'>SAVINGS SCENARIO:</span>
        <span style='color: {COLORS['primary']}; font-size: 16px; font-weight: 700; margin-left: 10px;'>{savings_factor}% of Maximum Potential</span>
    </div>
    """, unsafe_allow_html=True)

    # Four cards - compact sizing to prevent overflow
    col1, col2, col3, col4 = st.columns(4)

    base_spend_total = df['Base Spend'].sum()
    optimized_spend_total = df['Optimized Spend'].sum()
    optimization_savings = df['Potential Savings'].sum()
    ai_savings_total = df['AI Cost Reduction'].sum()
    total_savings = optimization_savings + ai_savings_total
    combined_savings_pct = total_savings / base_spend_total
    
    # Calculate Potential Future Savings (Baseline vs Potential Spending)
    # Only for 2025 and ahead, including AI savings
    if enable_projections:
        projected_data = fy_data[(fy_data['Is Projection'] == True) | (fy_data['Fiscal Year'] >= 2025)]
        if len(projected_data) > 0:
            projected_baseline = projected_data['Base Spend'].sum()
            # Potential Spending should include AI savings
            projected_optimized = projected_data['Optimized Spend'].sum()
            # Apply AI savings to optimized
            projected_ai = projected_optimized * (df['AI Cost Reduction'].sum() / df['Optimized Spend'].sum())
            projected_potential = projected_optimized - projected_ai
            
            # Apply transition logic
            potential_spending_values = []
            for idx, row in projected_data.iterrows():
                if row['Is Projection']:
                    year_offset = row['Fiscal Year'] - 2025
                    if year_offset <= 0:
                        potential = row['Base Spend']
                    elif year_offset == 1:
                        potential = 0.5 * row['Base Spend'] + 0.5 * (row['Optimized Spend'] - row['Optimized Spend'] * (ai_savings_total / optimized_spend_total))
                    else:
                        potential = row['Optimized Spend'] - row['Optimized Spend'] * (ai_savings_total / optimized_spend_total)
                    potential_spending_values.append(potential)
                else:
                    potential_spending_values.append(row['Base Spend'])
            
            projected_potential = sum(potential_spending_values)
            potential_future_savings = projected_baseline - projected_potential
            potential_future_savings_pct = potential_future_savings / projected_baseline if projected_baseline > 0 else 0
        else:
            potential_future_savings = 0
            potential_future_savings_pct = 0
    else:
        potential_future_savings = 0
        potential_future_savings_pct = 0

    with col1:
        st.markdown(metric_card("BASE SPEND", f"${base_spend_total/1e6:.1f}M", f"{len(df)} apps", COLORS['secondary']), unsafe_allow_html=True)

    with col2:
        optimization_pct = optimization_savings / base_spend_total
        st.markdown(metric_card("COMPONENT OPT.", f"${optimization_savings/1e6:.1f}M", f"{optimization_pct:.1%}", COLORS['secondary']), unsafe_allow_html=True)

    with col3:
        ai_savings_pct = ai_savings_total / optimized_spend_total if optimized_spend_total > 0 else 0
        st.markdown(metric_card("AI SAVINGS", f"${ai_savings_total/1e6:.1f}M", f"{ai_savings_pct:.1%}", '#9D4EDD'), unsafe_allow_html=True)

    with col4:
        st.markdown(metric_card("COMBINED", f"${total_savings/1e6:.1f}M", f"{combined_savings_pct:.1%}", COLORS['primary']), unsafe_allow_html=True)
    
    # Potential Future Savings card (only show if projections enabled)
    if enable_projections and potential_future_savings > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color: {COLORS['card']}; padding: 30px; border-radius: 10px; border: 2px solid {COLORS['primary']}; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h4 style='margin: 0; color: #D0D0D0; font-size: 14px; font-weight: 500; letter-spacing: 1px;'>POTENTIAL FUTURE SAVINGS</h4>
                    <p style='margin: 5px 0 0 0; color: #B0B0B0; font-size: 12px;'>Baseline vs Potential Spending (Projected Years)</p>
                </div>
                <div style='text-align: right;'>
                    <h2 style='margin: 0; color: {COLORS['primary']}; font-size: 42px; font-weight: 700;'>${potential_future_savings/1e6:.2f}M</h2>
                    <p style='margin: 5px 0 0 0; color: {COLORS['primary']}; font-size: 16px; font-weight: 600;'>{potential_future_savings_pct:.1%} reduction</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Line chart - Baseline vs Optimized by Fiscal Year with Projections
    st.markdown("<br>", unsafe_allow_html=True)

    # Separate historical and projected data
    fy_historical = fy_data[fy_data['Is Projection'] == False].sort_values('Fiscal Year')
    fy_projected = fy_data[fy_data['Is Projection'] == True].sort_values('Fiscal Year')

    # Create line chart
    fig = go.Figure()
    
    # Helper: build smooth curve using scipy
    from scipy.interpolate import make_interp_spline
    
    def smooth_line(years, values, n_points=300):
        """Create smooth interpolation across all points."""
        if len(years) < 3:
            return np.array(years, dtype=float), np.array(values, dtype=float)
        x = np.array(years, dtype=float)
        y = np.array(values, dtype=float)
        try:
            k = min(3, len(x) - 1)
            spline = make_interp_spline(x, y, k=k)
            x_smooth = np.linspace(x.min(), x.max(), n_points)
            y_smooth = spline(x_smooth)
            return x_smooth, y_smooth
        except Exception:
            return x, y
    
    last_hist_year = fy_historical['Fiscal Year'].max() if len(fy_historical) > 0 else 0
    
    # === BASELINE ===
    all_years_base = fy_historical['Fiscal Year'].tolist()
    all_vals_base = (fy_historical['Base Spend'] / 1e6).tolist()
    if enable_projections and len(fy_projected) > 0:
        all_years_base += fy_projected['Fiscal Year'].tolist()
        all_vals_base += (fy_projected['Base Spend'] / 1e6).tolist()
    xs_base, ys_base = smooth_line(all_years_base, all_vals_base)
    
    hist_mask_b = xs_base <= last_hist_year + 0.01
    proj_mask_b = xs_base >= last_hist_year - 0.01
    
    # Baseline shadow (fill to zero)
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs_base[hist_mask_b], xs_base[hist_mask_b][::-1]]),
        y=np.concatenate([ys_base[hist_mask_b], np.zeros(hist_mask_b.sum())]),
        fill='toself', fillcolor='rgba(255, 140, 66, 0.07)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    # Baseline historical (solid)
    fig.add_trace(go.Scatter(
        x=xs_base[hist_mask_b], y=ys_base[hist_mask_b],
        mode='lines', name='Baseline',
        line=dict(color=COLORS['secondary'], width=3),
        legendgroup='baseline'
    ))
    fig.add_trace(go.Scatter(
        x=fy_historical['Fiscal Year'], y=fy_historical['Base Spend'] / 1e6,
        mode='markers', marker=dict(size=7, color=COLORS['secondary']),
        legendgroup='baseline', showlegend=False
    ))
    
    # === OPTIMIZED ===
    all_years_opt = fy_historical['Fiscal Year'].tolist()
    all_vals_opt = (fy_historical['Optimized Spend'] / 1e6).tolist()
    if enable_projections and len(fy_projected) > 0:
        all_years_opt += fy_projected['Fiscal Year'].tolist()
        all_vals_opt += (fy_projected['Optimized Spend'] / 1e6).tolist()
    xs_opt, ys_opt = smooth_line(all_years_opt, all_vals_opt)
    
    hist_mask_o = xs_opt <= last_hist_year + 0.01
    proj_mask_o = xs_opt >= last_hist_year - 0.01
    
    # Optimized shadow
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs_opt[hist_mask_o], xs_opt[hist_mask_o][::-1]]),
        y=np.concatenate([ys_opt[hist_mask_o], np.zeros(hist_mask_o.sum())]),
        fill='toself', fillcolor='rgba(0, 217, 163, 0.07)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=xs_opt[hist_mask_o], y=ys_opt[hist_mask_o],
        mode='lines', name='Optimized',
        line=dict(color=COLORS['primary'], width=3),
        legendgroup='optimized'
    ))
    fig.add_trace(go.Scatter(
        x=fy_historical['Fiscal Year'], y=fy_historical['Optimized Spend'] / 1e6,
        mode='markers', marker=dict(size=7, color=COLORS['primary']),
        legendgroup='optimized', showlegend=False
    ))

    # === POTENTIAL SPENDING (same scipy approach, connected to baseline) ===
    fy_potential_data = fy_data[fy_data['Fiscal Year'] >= 2025]
    
    if len(fy_potential_data) > 0:
        # Include last 2 baseline historical points for smooth transition
        n_bridge = min(2, len(fy_historical))
        pot_years = fy_historical['Fiscal Year'].tail(n_bridge).tolist() + fy_potential_data['Fiscal Year'].tolist()
        pot_vals = (fy_historical['Base Spend'].tail(n_bridge) / 1e6).tolist() + (fy_potential_data['Potential Spend'] / 1e6).tolist()
        
        xs_pot, ys_pot = smooth_line(pot_years, pot_vals)
        
        # Only show from 2025 onwards
        pot_show = xs_pot >= 2025 - 0.5
        
        # Potential shadow
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs_pot[pot_show], xs_pot[pot_show][::-1]]),
            y=np.concatenate([ys_pot[pot_show], np.zeros(pot_show.sum())]),
            fill='toself', fillcolor='rgba(157, 78, 221, 0.06)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=xs_pot[pot_show], y=ys_pot[pot_show],
            mode='lines', name='Potential Spending',
            line=dict(color='#9D4EDD', width=3, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=fy_potential_data['Fiscal Year'], y=fy_potential_data['Potential Spend'] / 1e6,
            mode='markers', marker=dict(size=7, symbol='star', color='#9D4EDD'),
            showlegend=False
        ))

    # === PROJECTED PORTIONS (dashed continuation) ===
    if enable_projections and len(fy_projected) > 0:
        # Baseline projected shadow
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs_base[proj_mask_b], xs_base[proj_mask_b][::-1]]),
            y=np.concatenate([ys_base[proj_mask_b], np.zeros(proj_mask_b.sum())]),
            fill='toself', fillcolor='rgba(255, 140, 66, 0.04)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=xs_base[proj_mask_b], y=ys_base[proj_mask_b],
            mode='lines', line=dict(color=COLORS['secondary'], width=3, dash='dash'),
            opacity=0.7, legendgroup='baseline', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=fy_projected['Fiscal Year'], y=fy_projected['Base Spend'] / 1e6,
            mode='markers', marker=dict(size=6, symbol='diamond', color=COLORS['secondary']),
            opacity=0.7, legendgroup='baseline', showlegend=False
        ))
        
        # Optimized projected shadow
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs_opt[proj_mask_o], xs_opt[proj_mask_o][::-1]]),
            y=np.concatenate([ys_opt[proj_mask_o], np.zeros(proj_mask_o.sum())]),
            fill='toself', fillcolor='rgba(0, 217, 163, 0.04)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=xs_opt[proj_mask_o], y=ys_opt[proj_mask_o],
            mode='lines', line=dict(color=COLORS['primary'], width=3, dash='dash'),
            opacity=0.7, legendgroup='optimized', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=fy_projected['Fiscal Year'], y=fy_projected['Optimized Spend'] / 1e6,
            mode='markers', marker=dict(size=6, symbol='diamond', color=COLORS['primary']),
            opacity=0.7, legendgroup='optimized', showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card'],
        font=dict(color='#FFFFFF', family='Arial'),
        title=dict(
            text='Baseline vs Optimized Spend by Fiscal Year' + (' with Projections' if enable_projections else ''),
            font=dict(size=20, color='#FFFFFF')
        ),
        xaxis=dict(
            title='Fiscal Year',
            gridcolor='#333333',
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            title='Spend ($M)',
            gridcolor='#333333'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        height=550,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # === VISUAL 2: Structure (ECharts Collapsible Tree) ===
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌐 Structure")
    
    # Value selector
    col1, col2 = st.columns([1, 3])
    with col1:
        network_value = st.radio(
            "Display Value:",
            ["Savings", "Actual Portfolio", "Optimized Portfolio"],
            horizontal=False
        )
    
    try:
        from streamlit_echarts import st_echarts
        
        # Determine value field
        if network_value == "Savings":
            value_field = lambda x: x['Potential Savings'] + x['AI Cost Reduction']
        elif network_value == "Actual Portfolio":
            value_field = lambda x: x['Base Spend']
        else:
            value_field = lambda x: x['Total AI Cost']
        
        # Build tree: Portfolio → Sector → Department → Vendor → Apps
        root_value = df.apply(value_field, axis=1).sum()
        MAX_CHILDREN = 8
        
        def fmt(val):
            if val >= 1e6:
                return f"${val/1e6:.1f}M"
            elif val >= 1e3:
                return f"${val/1e3:.0f}K"
            else:
                return f"${val:.0f}"
        
        def build_children(sub_df, group_col, color, next_builder=None):
            """Build capped child list grouped by group_col."""
            grouped = sub_df.groupby(group_col).apply(
                lambda x: pd.Series({'Value': x.apply(value_field, axis=1).sum(), 'df': x})
            )
            # Sort and extract values
            val_series = sub_df.groupby(group_col).apply(
                lambda x: x.apply(value_field, axis=1).sum()
            ).sort_values(ascending=False)
            
            children = []
            for i, (name, val) in enumerate(val_series.items()):
                if i >= MAX_CHILDREN:
                    # Others
                    others_val = val_series.iloc[MAX_CHILDREN:].sum()
                    if others_val > 0:
                        children.append({
                            "name": "Others",
                            "value": round(others_val / 1e6, 2),
                            "itemStyle": {"color": "#666666"},
                            "label": {"formatter": f"Others\n{fmt(others_val)}"}
                        })
                    break
                
                node = {
                    "name": str(name)[:30],
                    "value": round(val / 1e6, 2),
                    "itemStyle": {"color": color},
                    "label": {"formatter": f"{str(name)[:25]}\n{fmt(val)}"}
                }
                
                if next_builder:
                    child_df = sub_df[sub_df[group_col] == name]
                    sub_children = next_builder(child_df)
                    if sub_children:
                        node["children"] = sub_children
                
                children.append(node)
            
            return children
        
        def build_apps(sub_df):
            """Build app leaf nodes (capped)."""
            app_vals = sub_df.apply(value_field, axis=1)
            app_data = pd.DataFrame({'App': sub_df['App Name'].values, 'Value': app_vals.values})
            app_data = app_data.sort_values('Value', ascending=False)
            
            children = []
            for i, (_, row) in enumerate(app_data.iterrows()):
                if i >= MAX_CHILDREN:
                    others_val = app_data.iloc[MAX_CHILDREN:]['Value'].sum()
                    if others_val > 0:
                        children.append({
                            "name": "Others",
                            "value": round(others_val / 1e6, 2),
                            "itemStyle": {"color": "#666666"},
                            "label": {"formatter": f"Others\n{fmt(others_val)}"}
                        })
                    break
                children.append({
                    "name": str(row['App'])[:30],
                    "value": round(row['Value'] / 1e6, 2),
                    "itemStyle": {"color": "rgba(255,255,255,0.4)"},
                    "label": {"formatter": f"{str(row['App'])[:25]}\n{fmt(row['Value'])}"}
                })
            return children
        
        def build_vendors(sub_df):
            return build_children(sub_df, 'Vendor', '#4ECDC4', next_builder=build_apps)
        
        def build_departments(sub_df):
            return build_children(sub_df, 'Department', '#9D4EDD', next_builder=build_vendors)
        
        sector_children = build_children(df, 'Sector', '#FF8C42', next_builder=build_departments)
        
        tree_data = [{
            "name": "Portfolio",
            "value": round(root_value / 1e6, 2),
            "itemStyle": {"color": "#00D9A3"},
            "label": {"formatter": f"Portfolio\n{fmt(root_value)}"},
            "children": sector_children
        }]
        
        option = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
                "formatter": "{b}: ${c}M"
            },
            "series": [{
                "type": "tree",
                "data": tree_data,
                "top": "2%",
                "left": "10%",
                "bottom": "2%",
                "right": "25%",
                "symbolSize": 18,
                "orient": "LR",
                "label": {
                    "position": "right",
                    "verticalAlign": "middle",
                    "align": "left",
                    "fontSize": 13,
                    "color": "#FFFFFF",
                    "distance": 10,
                    "fontWeight": "bold"
                },
                "leaves": {
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "fontSize": 11,
                        "color": "#D0D0D0"
                    }
                },
                "lineStyle": {
                    "color": "#555555",
                    "width": 1.5,
                    "curveness": 0.5
                },
                "emphasis": {
                    "focus": "descendant"
                },
                "expandAndCollapse": True,
                "initialTreeDepth": 1,
                "animationDuration": 550,
                "animationDurationUpdate": 750
            }]
        }
        
        st_echarts(option, height="700px", key="structure_tree")
        
        st.caption("🖱️ **Click nodes to expand/collapse** | 🟢 Portfolio → 🟠 Sector → 🟣 Department → 🔵 Vendor → ⚪ Apps")
        
    except ImportError:
        st.warning("📦 Install streamlit-echarts: `pip install streamlit-echarts`")
        st.info("The collapsible tree provides an interactive hierarchical view of portfolio structure.")

with tab2:
    st.markdown("### Portfolio Variance Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Smaller cards with centered values (like Tab 1)
    col1, col2, col3 = st.columns(3)
    
    portfolio_baseline = df['Base Spend'].sum()
    portfolio_optimized = df['Optimized Spend'].sum()
    portfolio_savings = df['Potential Savings'].sum()
    portfolio_savings_pct = portfolio_savings / portfolio_baseline
    
    with col1:
        st.markdown(metric_card("BASELINE", f"${portfolio_baseline/1e6:.1f}M", "Original costs", COLORS['secondary']), unsafe_allow_html=True)
    
    with col2:
        st.markdown(metric_card("OPTIMIZATION", f"${portfolio_savings/1e6:.1f}M", f"{portfolio_savings_pct:.1%}", COLORS['primary']), unsafe_allow_html=True)
    
    with col3:
        st.markdown(metric_card("OPTIMIZED", f"${portfolio_optimized/1e6:.1f}M", f"{(portfolio_optimized/portfolio_baseline*100):.1f}%", COLORS['secondary']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === Baseline vs Optimized by Application (like Tab 3 layout) ===
    st.markdown("### Baseline vs Optimized by Application")
    
    df_sorted_var_top = df.nlargest(20, 'Base Spend').sort_values('Base Spend', ascending=True)
    df_sorted_var_base = df.sort_values('Base Spend', ascending=False).copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Baseline vs Optimized (Top 20 by Base Cost)")
        
        fig_bar_var = go.Figure()
        
        fig_bar_var.add_trace(go.Bar(
            name='Baseline Cost',
            x=df_sorted_var_top['Base Spend'] / 1000,
            y=df_sorted_var_top['App Name'],
            orientation='h',
            marker=dict(color=COLORS['secondary'])
        ))
        
        fig_bar_var.add_trace(go.Bar(
            name='Optimized Cost',
            x=df_sorted_var_top['Optimized Spend'] / 1000,
            y=df_sorted_var_top['App Name'],
            orientation='h',
            marker=dict(color=COLORS['primary'])
        ))
        
        fig_bar_var.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=9),
            xaxis=dict(title='Cost ($K)', gridcolor='#333333'),
            yaxis=dict(title='', gridcolor='#333333'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=550,
            barmode='group',
            hovermode='y unified',
            margin=dict(l=200, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig_bar_var, use_container_width=True)
        

    
    with col2:
        st.markdown("#### Cumulative Baseline vs Optimized (Sorted by Largest)")
        
        df_sorted_var_base['Cumulative Baseline'] = df_sorted_var_base['Base Spend'].cumsum()
        df_sorted_var_base['Cumulative Optimized'] = df_sorted_var_base['Optimized Spend'].cumsum()
        df_sorted_var_base['App Number'] = range(1, len(df_sorted_var_base) + 1)
        
        fig_cum_var = go.Figure()
        
        fig_cum_var.add_trace(go.Scatter(
            x=df_sorted_var_base['App Number'],
            y=df_sorted_var_base['Cumulative Baseline'] / 1e6,
            mode='lines',
            name='Cumulative Baseline',
            line=dict(color=COLORS['secondary'], width=3),
            fill='tonexty',
            fillcolor='rgba(255, 140, 66, 0.1)'
        ))
        
        fig_cum_var.add_trace(go.Scatter(
            x=df_sorted_var_base['App Number'],
            y=df_sorted_var_base['Cumulative Optimized'] / 1e6,
            mode='lines',
            name='Cumulative Optimized',
            line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 163, 0.1)'
        ))
        
        fig_cum_var.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial'),
            xaxis=dict(title='Number of Applications', gridcolor='#333333'),
            yaxis=dict(title='Cumulative Cost ($M)', gridcolor='#333333'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cum_var, use_container_width=True)
        
        final_bl = df_sorted_var_base['Cumulative Baseline'].iloc[-1]
        final_opt_v = df_sorted_var_base['Cumulative Optimized'].iloc[-1]
        opt_savings = final_bl - final_opt_v

    # Summary cards row (full width)
    mean_savings_per_app = df['Potential Savings'].mean() if len(df) > 0 else 0
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['primary']}'>
            <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>TOTAL OPTIMIZATION SAVINGS</p>
            <h3 style='margin: 5px 0; color: {COLORS['primary']}; font-size: 24px;'>${opt_savings/1e6:.2f}M</h3>
            <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>{(opt_savings/final_bl*100):.1f}% reduction from baseline</p>
        </div>
        """, unsafe_allow_html=True)
    with card_col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['secondary']}'>
            <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>MEAN SAVINGS PER APP</p>
            <h3 style='margin: 5px 0; color: {COLORS['secondary']}; font-size: 24px;'>${mean_savings_per_app/1e3:.0f}K</h3>
            <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>Across {len(df)} applications</p>
        </div>
        """, unsafe_allow_html=True)

    # === Top and Bottom Apps by Savings (always visible) ===
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Top & Bottom 10 Apps by Savings", expanded=False):
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### Top 10 Apps by Savings %")
    
            top_apps = df.nlargest(10, 'Adjusted Savings %')[['App Name', 'Base Spend', 'Optimized Spend', 'Potential Savings', 'Adjusted Savings %']].sort_values('Adjusted Savings %', ascending=True)
        
            fig_top = go.Figure()
        
            fig_top.add_trace(go.Bar(
                name='Baseline Cost',
                x=top_apps['Base Spend'] / 1e6,
                y=top_apps['App Name'],
                orientation='h',
                marker=dict(color=COLORS['secondary']),
                hovertemplate='<b>%{y}</b><br>Baseline: $%{x:.2f}M<extra></extra>'
            ))
        
            fig_top.add_trace(go.Bar(
                name='Optimized Cost',
                x=top_apps['Optimized Spend'] / 1e6,
                y=top_apps['App Name'],
                orientation='h',
                marker=dict(color=COLORS['primary']),
                hovertemplate='<b>%{y}</b><br>Optimized: $%{x:.2f}M<extra></extra>'
            ))
        
            fig_top.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                xaxis=dict(title='Cost ($M)', gridcolor='#333333'),
                yaxis=dict(title='', gridcolor='#333333'),
                barmode='group',
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
        
            st.plotly_chart(fig_top, use_container_width=True)
        
            # Top savings summary card
            top_total = top_apps['Potential Savings'].sum()
            top_avg_pct = top_apps['Adjusted Savings %'].mean() * 100
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['primary']}'>
                <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>TOP 10 TOTAL SAVINGS</p>
                <h3 style='margin: 5px 0; color: {COLORS['primary']}; font-size: 24px;'>${top_total/1e6:.2f}M</h3>
                <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>Avg savings rate: {top_avg_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
        with col2:
            st.markdown("#### Bottom 10 Apps by Savings")
        
            bottom_apps = df.nsmallest(10, 'Potential Savings')[['App Name', 'Base Spend', 'Optimized Spend', 'Potential Savings', 'Adjusted Savings %']].sort_values('Potential Savings', ascending=True)
        
            fig_bottom = go.Figure()
        
            fig_bottom.add_trace(go.Bar(
                name='Baseline Cost',
                x=bottom_apps['Base Spend'] / 1e6,
                y=bottom_apps['App Name'],
                orientation='h',
                marker=dict(color=COLORS['secondary']),
                hovertemplate='<b>%{y}</b><br>Baseline: $%{x:.2f}M<extra></extra>'
            ))
        
            fig_bottom.add_trace(go.Bar(
                name='Optimized Cost',
                x=bottom_apps['Optimized Spend'] / 1e6,
                y=bottom_apps['App Name'],
                orientation='h',
                marker=dict(color=COLORS['danger']),
                hovertemplate='<b>%{y}</b><br>Optimized: $%{x:.2f}M<extra></extra>'
            ))
        
            fig_bottom.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                xaxis=dict(title='Cost ($M)', gridcolor='#333333'),
                yaxis=dict(title='', gridcolor='#333333'),
                barmode='group',
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
        
            st.plotly_chart(fig_bottom, use_container_width=True)
        
            # Bottom savings summary card
            bottom_total = bottom_apps['Potential Savings'].sum()
            bottom_avg_pct = bottom_apps['Adjusted Savings %'].mean() * 100
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['danger']}'>
                <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>BOTTOM 10 TOTAL SAVINGS</p>
                <h3 style='margin: 5px 0; color: {COLORS['danger']}; font-size: 24px;'>${bottom_total/1e6:.2f}M</h3>
                <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>Avg savings rate: {bottom_avg_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # === Component Structure Sankey (at end of tab) ===
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🏗️ Component Structure", expanded=False):
    
        structure_df = pd.read_excel('AssetList.xlsx', sheet_name='Structure')
        component_to_category = dict(zip(structure_df['Component Lowest'], structure_df['Component lvl 1']))
        component_to_lvl2 = dict(zip(structure_df['Component Lowest'], structure_df['Component lvl 2']))
    
        criticality_df = pd.read_excel('AssetList.xlsx', sheet_name='Criticality Score')
        component_names = list(criticality_df.iloc[0, 1:].values)
    
        cat_counts = {}
        for comp_name in component_names:
            cat = component_to_category.get(comp_name, 'Infrastructure')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
        comp_sankey_data = []
        for comp_name in component_names:
            lvl1 = component_to_category.get(comp_name, 'Infrastructure')
            lvl2 = component_to_lvl2.get(comp_name, 'Unknown')
            if lvl1 == 'Applications':
                cat_share = app_cost_share / 100
            else:
                cat_share = infra_cost_share / 100
            n_in_cat = cat_counts.get(lvl1, 1)
            comp_value = (portfolio_baseline * cat_share) / n_in_cat
            comp_sankey_data.append({'Category': lvl1, 'Type': lvl2, 'Component': comp_name, 'Value': comp_value})
    
        comp_sankey_df = pd.DataFrame(comp_sankey_data)
    
        if len(comp_sankey_df) > 0 and comp_sankey_df['Value'].sum() > 0:
            labels, ncolors, sources, targets, values, lcolors = build_sankey_capped(
                comp_sankey_df, ['Category', 'Type', 'Component'], 'Value', max_per_level=10
            )
            fig_comp_sankey = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color='#333333', width=0.5),
                          label=labels, color=ncolors,
                          hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<extra></extra>'),
                link=dict(source=sources, target=targets, value=values, color=lcolors,
                          hovertemplate='%{source.label} → %{target.label}<br>$%{value:,.0f}<extra></extra>')
            )])
            fig_comp_sankey.update_layout(
                paper_bgcolor=COLORS['background'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                height=550, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_comp_sankey, use_container_width=True)
        else:
            st.info("No component data available for the current filters.")


with tab3:
    st.markdown("### AI Impact Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Three AI cards — compact style matching Tab 1
    col1, col2, col3 = st.columns(3)
    
    ai_reduction_total = df['AI Cost Reduction'].sum()
    optimized_spend_total = df['Optimized Spend'].sum()
    total_ai_cost = df['Total AI Cost'].sum()
    ai_reduction_pct = ai_reduction_total / optimized_spend_total if optimized_spend_total > 0 else 0
    total_ai_pct = total_ai_cost / df['Base Spend'].sum()
    
    with col1:
        st.markdown(metric_card("AI COST REDUCTION", f"${ai_reduction_total/1e6:.1f}M", f"{ai_reduction_pct:.1%} reduction", COLORS['primary']), unsafe_allow_html=True)
    
    with col2:
        st.markdown(metric_card("TOTAL AI COST", f"${total_ai_cost/1e6:.1f}M", f"{total_ai_pct:.1%} of baseline", COLORS['secondary']), unsafe_allow_html=True)
    
    with col3:
        combined_savings = (df['Base Spend'].sum() - total_ai_cost)
        combined_savings_pct = combined_savings / df['Base Spend'].sum() if df['Base Spend'].sum() > 0 else 0
        st.markdown(metric_card("COMBINED SAVINGS", f"${combined_savings/1e6:.1f}M", f"{combined_savings_pct:.1%} total reduction", COLORS['primary']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sort for AI charts
    df_sorted_ai_opt = df.nlargest(20, 'Total AI Cost').sort_values('Total AI Cost', ascending=True)
    df_sorted_ai_base = df.sort_values('Optimized Spend', ascending=False)
    
    st.markdown("### AI Cost by Application")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Optimized vs AI Cost (Top 20 by AI Cost)")
    
        # Bar chart - Optimized vs AI Cost
        fig_bar_ai = go.Figure()
        
        fig_bar_ai.add_trace(go.Bar(
            name='Optimized Cost',
            x=df_sorted_ai_opt['Optimized Spend'] / 1000,
            y=df_sorted_ai_opt['App Name'],
            orientation='h',
            marker=dict(color=COLORS['secondary'])
        ))
        
        fig_bar_ai.add_trace(go.Bar(
            name='AI Cost',
            x=df_sorted_ai_opt['Total AI Cost'] / 1000,
            y=df_sorted_ai_opt['App Name'],
            orientation='h',
            marker=dict(color=COLORS['primary'])
        ))
        
        fig_bar_ai.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=9),
            xaxis=dict(
                title='Cost ($K)',
                gridcolor='#333333'
            ),
            yaxis=dict(
                title='',
                gridcolor='#333333'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=550,
            barmode='group',
            hovermode='y unified',
            margin=dict(l=200, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig_bar_ai, use_container_width=True)
    
    with col2:
        st.markdown("#### Cumulative AI Cost (Sorted by Largest First)")
    
        # Calculate cumulative sums (sorted by optimized spend descending)
        df_sorted_ai_base['Cumulative Optimized'] = df_sorted_ai_base['Optimized Spend'].cumsum()
        df_sorted_ai_base['Cumulative AI'] = df_sorted_ai_base['Total AI Cost'].cumsum()
        df_sorted_ai_base['App Number'] = range(1, len(df_sorted_ai_base) + 1)
        
        # Cumulative line chart for AI
        fig_cum_ai = go.Figure()
        
        fig_cum_ai.add_trace(go.Scatter(
            x=df_sorted_ai_base['App Number'],
            y=df_sorted_ai_base['Cumulative Optimized'] / 1e6,
            mode='lines',
            name='Cumulative Optimized',
            line=dict(color=COLORS['secondary'], width=3),
            fill='tonexty',
            fillcolor='rgba(255, 140, 66, 0.1)'
        ))
        
        fig_cum_ai.add_trace(go.Scatter(
            x=df_sorted_ai_base['App Number'],
            y=df_sorted_ai_base['Cumulative AI'] / 1e6,
            mode='lines',
            name='Cumulative AI Cost',
            line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 163, 0.1)'
        ))
        
        fig_cum_ai.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial'),
            xaxis=dict(
                title='Number of Applications',
                gridcolor='#333333'
            ),
            yaxis=dict(
                title='Cumulative Cost ($M)',
                gridcolor='#333333'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cum_ai, use_container_width=True)
        
        final_opt = df_sorted_ai_base['Cumulative Optimized'].iloc[-1]
        final_ai = df_sorted_ai_base['Cumulative AI'].iloc[-1]
        final_ai_savings = final_opt - final_ai

    # Summary cards row (full width)
    mean_ai_savings_per_app = df['AI Cost Reduction'].mean() if len(df) > 0 else 0
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['primary']}'>
            <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>TOTAL AI COST REDUCTION</p>
            <h3 style='margin: 5px 0; color: {COLORS['primary']}; font-size: 24px;'>${final_ai_savings/1e6:.2f}M</h3>
            <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>{(final_ai_savings/final_opt*100):.1f}% reduction from optimized</p>
        </div>
        """, unsafe_allow_html=True)
    with card_col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['secondary']}'>
            <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>MEAN AI SAVINGS PER APP</p>
            <h3 style='margin: 5px 0; color: {COLORS['secondary']}; font-size: 24px;'>${mean_ai_savings_per_app/1e3:.0f}K</h3>
            <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>Across {len(df)} applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Activity Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤖 AI Automation by Activity", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            activity_columns = [col for col in ai_potential_df.columns if col != 'App Name']
            avg_automation = ai_potential_df[activity_columns].mean().sort_values(ascending=False)
            
            activity_type_map = dict(zip(ia_structure_df['IA Component Lowest'], ia_structure_df['IA Component Lvl1']))
            activity_types = [activity_type_map.get(act, 'Unknown') for act in avg_automation.index]
            
            fig_activity = go.Figure()
            colors_map = {'Soft': COLORS['primary'], 'Hard': COLORS['secondary']}
            
            fig_activity.add_trace(go.Bar(
                x=avg_automation.values * 100,
                y=avg_automation.index,
                orientation='h',
                marker=dict(color=[colors_map.get(t, '#B0B0B0') for t in activity_types]),
                text=[f"{v*100:.1f}%" for v in avg_automation.values],
                textposition='outside',
                hovertemplate='%{y}<br>Automation: %{x:.1f}%<extra></extra>'
            ))
            
            fig_activity.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=11),
                title='Average AI Automation Potential by Activity',
                xaxis=dict(title='Automation Potential (%)', gridcolor='#333333', range=[0, 100]),
                yaxis=dict(title='', gridcolor='#333333'),
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_activity, use_container_width=True)
        
        with col2:
            soft_activities = [act for act, typ in zip(avg_automation.index, activity_types) if typ == 'Soft']
            hard_activities = [act for act, typ in zip(avg_automation.index, activity_types) if typ == 'Hard']
            
            avg_soft = ai_potential_df[soft_activities].mean().mean()
            avg_hard = ai_potential_df[hard_activities].mean().mean()
            
            fig_type = go.Figure()
            
            fig_type.add_trace(go.Bar(
                x=['Soft Activities', 'Hard Activities'],
                y=[avg_soft * 100, avg_hard * 100],
                marker=dict(color=[COLORS['primary'], COLORS['secondary']]),
                text=[f"{avg_soft*100:.1f}%", f"{avg_hard*100:.1f}%"],
                textposition='outside',
                textfont=dict(size=16)
            ))
            
            fig_type.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial'),
                title='AI Automation: Soft vs Hard Activities',
                yaxis=dict(title='Average Automation Potential (%)', gridcolor='#333333', range=[0, 100]),
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig_type, use_container_width=True)

        # Aligned automation cards
        top_activity = avg_automation.index[0]
        top_activity_pct = avg_automation.values[0] * 100
        least_activity = avg_automation.index[-1]
        least_activity_pct = avg_automation.values[-1] * 100
        
        card_col1, card_col2 = st.columns(2)
        with card_col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['primary']}'>
                <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>HIGHEST AUTOMATION POTENTIAL</p>
                <h3 style='margin: 5px 0; color: {COLORS['primary']}; font-size: 20px;'>{top_activity}</h3>
                <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>{top_activity_pct:.1f}% avg automation potential</p>
            </div>
            """, unsafe_allow_html=True)
        with card_col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 15px; border-radius: 8px; border-top: 3px solid {COLORS['secondary']}'>
                <p style='margin: 0; color: #D0D0D0; font-size: 12px;'>LOWEST AUTOMATION POTENTIAL</p>
                <h3 style='margin: 5px 0; color: {COLORS['secondary']}; font-size: 20px;'>{least_activity}</h3>
                <p style='margin: 0; color: #B0B0B0; font-size: 11px;'>{least_activity_pct:.1f}% avg automation potential</p>
            </div>
            """, unsafe_allow_html=True)

    # === AI Structure Sankey: Sector → Soft/Hard → Activity ===
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔀 AI Cost Flow — Sector → Activity Structure", expanded=False):
        activity_columns_sankey = [col for col in ai_potential_df.columns if col != 'App Name']
        activity_type_map_sankey = dict(zip(ia_structure_df['IA Component Lowest'], ia_structure_df['IA Component Lvl1']))
        
        ai_sankey_rows = []
        for _, row in df.iterrows():
            app_name = row['App Name']
            sector = row['Sector']
            ai_total = row['AI Cost Reduction']
            
            if ai_total <= 0 or pd.isna(sector):
                continue
            
            app_pot = ai_potential_df[ai_potential_df['App Name'] == app_name]
            app_time = ia_time_df[ia_time_df['App Name'] == app_name]
            
            if len(app_pot) > 0 and len(app_time) > 0:
                weights = {}
                for act in activity_columns_sankey:
                    t = app_time[act].values[0] if act in app_time.columns else 0
                    p = app_pot[act].values[0] if act in app_pot.columns else 0
                    w = t * p
                    if w > 0:
                        weights[act] = w
                
                total_w = sum(weights.values()) if weights else 1
                
                for act, w in weights.items():
                    ia_type = activity_type_map_sankey.get(act, 'Soft')
                    ai_sankey_rows.append({
                        'Sector': str(sector),
                        'IA Type': ia_type,
                        'Activity': act,
                        'Value': ai_total * (w / total_w)
                    })
        
        ai_sankey_df = pd.DataFrame(ai_sankey_rows)
        
        if len(ai_sankey_df) > 0 and ai_sankey_df['Value'].sum() > 0:
            labels, ncolors, sources, targets, values, lcolors = build_sankey_capped(
                ai_sankey_df, ['Sector', 'IA Type', 'Activity'], 'Value', max_per_level=10
            )
            
            fig_ai_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color='#333333', width=0.5),
                    label=labels, color=ncolors,
                    hovertemplate='<b>%{label}</b><br>AI Savings: $%{value:,.0f}<extra></extra>'
                ),
                link=dict(
                    source=sources, target=targets, value=values, color=lcolors,
                    hovertemplate='%{source.label} → %{target.label}<br>$%{value:,.0f}<extra></extra>'
                )
            )])
            
            fig_ai_sankey.update_layout(
                paper_bgcolor=COLORS['background'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                height=550,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            
            st.plotly_chart(fig_ai_sankey, use_container_width=True)
        else:
            st.info("No AI savings data available for the current filters.")



with tab5:
    st.markdown("### Asset Inventory")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search and filter
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_term = st.text_input("🔍 Search Assets", "")
    
    with col2:
        sector_filter = st.selectbox("Filter by Sector", ['All'] + sorted([str(s) for s in df['Sector'].unique() if pd.notna(s)]))
    
    with col3:
        dept_filter = st.selectbox("Filter by Department", ['All'] + sorted([str(d) for d in df['Department'].unique() if pd.notna(d)]))
    
    with col4:
        vendor_filter = st.selectbox("Filter by Vendor", ['All'] + sorted([str(v) for v in df['Vendor'].unique() if pd.notna(v)]))
    
    # Apply filters
    inventory_df = df.copy()
    
    if search_term:
        inventory_df = inventory_df[inventory_df['App Name'].str.contains(search_term, case=False, na=False)]
    
    if sector_filter != 'All':
        inventory_df = inventory_df[inventory_df['Sector'] == sector_filter]
    
    if dept_filter != 'All':
        inventory_df = inventory_df[inventory_df['Department'] == dept_filter]
    
    if vendor_filter != 'All':
        inventory_df = inventory_df[inventory_df['Vendor'] == vendor_filter]
    
    # Summary metrics - update to show both savings types
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Apps", len(inventory_df))
    with col2:
        st.metric("Base Cost", f"${inventory_df['Base Spend'].sum()/1e6:.1f}M")
    with col3:
        st.metric("Optimization", f"${inventory_df['Potential Savings'].sum()/1e6:.1f}M")
    with col4:
        st.metric("AI Savings", f"${inventory_df['AI Cost Reduction'].sum()/1e6:.1f}M")
    with col5:
        total_combined = inventory_df['Potential Savings'].sum() + inventory_df['AI Cost Reduction'].sum()
        st.metric("Total Savings", f"${total_combined/1e6:.1f}M")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # === TWO VISUALS ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Savings % vs Base Cost")
        
        # Calculate mean of specificity and criticality for color
        if 'Avg Specificity' in inventory_df.columns and 'Avg Criticality' in inventory_df.columns:
            color_vals = (inventory_df['Avg Specificity'] + inventory_df['Avg Criticality']) / 2
        else:
            color_vals = inventory_df['Combined Score'] if 'Combined Score' in inventory_df.columns else None
        
        fig_scatter_inv = go.Figure()
        
        fig_scatter_inv.add_trace(go.Scatter(
            x=inventory_df['Base Spend'] / 1e6,
            y=inventory_df['Adjusted Savings %'] * 100,
            mode='markers',
            marker=dict(
                size=10,
                color=color_vals,
                colorscale=[[0, COLORS['danger']], [0.5, COLORS['secondary']], [1, '#4ECDC4']],
                reversescale=True,
                showscale=True,
                colorbar=dict(title='Specificity'),
                opacity=0.75
            ),
            text=inventory_df['App Name'],
            customdata=np.stack([
                inventory_df['Vendor'],
                inventory_df['Department']
            ], axis=-1),
            hovertemplate='<b>%{text}</b><br>Base: $%{x:.2f}M<br>Savings: %{y:.1f}%<br>Vendor: %{customdata[0]}<br>Dept: %{customdata[1]}<extra></extra>'
        ))
        
        fig_scatter_inv.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=10),
            xaxis=dict(title='Base Cost ($M)', gridcolor='#333333'),
            yaxis=dict(title='Savings %', gridcolor='#333333'),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_scatter_inv, use_container_width=True)
    
    with col2:
        st.markdown("#### Top 10 Apps by Combined Savings")
        
        inv_top_savings = inventory_df.nlargest(10, 'Potential Savings').sort_values('Potential Savings', ascending=True)
        
        fig_top_inv = go.Figure()
        
        fig_top_inv.add_trace(go.Bar(
            name='Optimization',
            x=inv_top_savings['Potential Savings'] / 1000,
            y=inv_top_savings['App Name'],
            orientation='h',
            marker=dict(color=COLORS['secondary']),
            hovertemplate='<b>%{y}</b><br>Optimization: $%{x:.0f}K<extra></extra>'
        ))
        
        fig_top_inv.add_trace(go.Bar(
            name='AI Savings',
            x=inv_top_savings['AI Cost Reduction'] / 1000,
            y=inv_top_savings['App Name'],
            orientation='h',
            marker=dict(color='#9D4EDD'),
            hovertemplate='<b>%{y}</b><br>AI: $%{x:.0f}K<extra></extra>'
        ))
        
        fig_top_inv.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=10),
            xaxis=dict(title='Savings ($K)', gridcolor='#333333'),
            yaxis=dict(title='', gridcolor='#333333'),
            barmode='stack',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_top_inv, use_container_width=True)

    # === Portfolio Cost Flow ===
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Portfolio Cost Flow")
    
    flow_metric = st.radio(
        "Display Value:",
        ["Baseline Cost", "Optimized + AI", "Total Savings"],
        horizontal=True,
        key="flow_metric_v5"
    )
    
    flow_df = inventory_df.copy()
    if flow_metric == "Baseline Cost":
        flow_df['_fv'] = flow_df['Base Spend']
    elif flow_metric == "Optimized + AI":
        flow_df['_fv'] = flow_df['Total AI Cost']
    else:
        flow_df['_fv'] = flow_df['Potential Savings'] + flow_df['AI Cost Reduction']
    
    flow_df = flow_df[flow_df['_fv'] > 0]
    
    if len(flow_df) > 0:
        # Build sunburst data manually for full control
        sb_ids = ["Portfolio"]
        sb_labels = ["Portfolio"]
        sb_parents = [""]
        sb_values = [0]  # root gets 0 (auto-summed)
        sb_colors = [COLORS['primary']]
        
        for sector in flow_df['Sector'].dropna().unique():
            sdf = flow_df[flow_df['Sector'] == sector]
            sid = f"s_{sector}"
            sb_ids.append(sid)
            sb_labels.append(str(sector)[:25])
            sb_parents.append("Portfolio")
            sb_values.append(0)
            sb_colors.append('#FF8C42')
            
            for dept in sdf['Department'].dropna().unique():
                ddf = sdf[sdf['Department'] == dept]
                did = f"d_{sector}_{dept}"
                sb_ids.append(did)
                sb_labels.append(str(dept)[:22])
                sb_parents.append(sid)
                sb_values.append(0)
                sb_colors.append('#9D4EDD')
                
                for _, row in ddf.iterrows():
                    aid = f"a_{row['App Name']}"
                    sb_ids.append(aid)
                    sb_labels.append(str(row['App Name'])[:20])
                    sb_parents.append(did)
                    sb_values.append(round(row['_fv'], 2))
                    sb_colors.append('#4ECDC4')
        
        fig_sun = go.Figure(go.Sunburst(
            ids=sb_ids,
            labels=sb_labels,
            parents=sb_parents,
            values=sb_values,
            branchvalues='total',
            marker=dict(colors=sb_colors, line=dict(width=1, color=COLORS['background'])),
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<extra></extra>',
            textinfo='label',
            insidetextorientation='radial'
        ))
        
        fig_sun.update_layout(
            paper_bgcolor=COLORS['background'],
            font=dict(color='#FFFFFF', family='Arial', size=11),
            height=550,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig_sun, use_container_width=True)
        st.caption("🖱️ **Click a ring to drill down** | Click center to go back")
    else:
        st.info("No data available for the current filters and metric.")

    # === APPLICATION TABLE ===
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Asset Details", expanded=False):
    
        # Display table with both savings types
        inventory_df['Combined Savings'] = inventory_df['Potential Savings'] + inventory_df['AI Cost Reduction']
    
        display_cols = ['App Name', 'Sector', 'Department', 'Vendor', 'Fiscal Year', 'Base Spend', 
                        'Adjusted Savings %', 'Potential Savings', 'AI Savings % Adjusted', 
                        'AI Cost Reduction', 'Total AI Cost', 'Combined Savings']
    
        inventory_display = inventory_df[display_cols].copy()
    
        # Format columns
        inventory_display['Base Spend'] = inventory_display['Base Spend'].apply(lambda x: f'${x/1000:.0f}K')
        inventory_display['Adjusted Savings %'] = inventory_display['Adjusted Savings %'].apply(lambda x: f'{x*100:.1f}%')
        inventory_display['Potential Savings'] = inventory_display['Potential Savings'].apply(lambda x: f'${x/1000:.0f}K')
        inventory_display['AI Savings % Adjusted'] = inventory_display['AI Savings % Adjusted'].apply(lambda x: f'{x*100:.1f}%')
        inventory_display['AI Cost Reduction'] = inventory_display['AI Cost Reduction'].apply(lambda x: f'${x/1000:.0f}K')
        inventory_display['Combined Savings'] = inventory_display['Combined Savings'].apply(lambda x: f'${x/1000:.0f}K')
        inventory_display['Total AI Cost'] = inventory_display['Total AI Cost'].apply(lambda x: f'${x/1000:.0f}K')
    
        inventory_display.columns = ['Application', 'Sector', 'Department', 'Vendor', 'FY', 'Base Cost', 
                                      'Opt %', 'Opt $', 'AI %', 'AI $', 'Optimized Cost', 'Total Savings']
    
        st.dataframe(
            inventory_display,
            use_container_width=True,
            height=600
        )
    
        # Download button
        csv = inventory_df[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Inventory as CSV",
            data=csv,
            file_name="app_inventory.csv",
            mime="text/csv"
        )

with tab6:
    st.markdown("### Vendor Cost Benchmarking")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Vendor analysis - now with component breakdown
    # First, we need to calculate savings by component type for each vendor
    
    # Load structure to categorize components
    structure_df_vendors = pd.read_excel('AssetList.xlsx', sheet_name='Structure')
    component_to_category_vendor = dict(zip(structure_df_vendors['Component Lowest'], structure_df_vendors['Component lvl 1']))
    
    # Get component-level data
    criticality_vendor = pd.read_excel('AssetList.xlsx', sheet_name='Criticality Score')
    specificity_vendor = pd.read_excel('AssetList.xlsx', sheet_name='Specificity Score')
    
    # Get actual component names from first row (header row in data)
    actual_component_names = criticality_vendor.iloc[0, 1:].tolist()
    
    # Calculate savings for Applications vs Infrastructure components per app
    # Count components per category for equal split within category
    vendor_cat_counts = {}
    for cn in actual_component_names:
        cat = component_to_category_vendor.get(cn, 'Infrastructure')
        vendor_cat_counts[cat] = vendor_cat_counts.get(cat, 0) + 1
    
    app_component_savings = []
    for idx, row in df.iterrows():
        app_name = row['App Name']
        vendor = row['Vendor']
        base_cost = row['Base Spend']
        
        # Get this app's scores (skip first row which is headers)
        app_crit = criticality_vendor[criticality_vendor.iloc[:, 0] == app_name]
        app_spec = specificity_vendor[specificity_vendor.iloc[:, 0] == app_name]
        
        if len(app_crit) > 0 and len(app_spec) > 0:
            # Calculate savings for each component
            for i, comp_name in enumerate(actual_component_names):
                crit_score = pd.to_numeric(app_crit.iloc[0, i+1], errors='coerce')
                spec_score = pd.to_numeric(app_spec.iloc[0, i+1], errors='coerce')
                
                if pd.notna(crit_score) and pd.notna(spec_score):
                    combined = (crit_score + spec_score) / 2
                    savings_pct = 0.70 - (0.50 / 3.6) * (combined - 1.4)
                    savings_pct = max(0.20, min(0.70, savings_pct))
                    
                    # Determine component category
                    category = component_to_category_vendor.get(comp_name, 'Infrastructure')
                    
                    # Cost split: app_cost_share% to Applications, rest to Infrastructure
                    if category == 'Applications':
                        cat_share = app_cost_share / 100
                    else:
                        cat_share = infra_cost_share / 100
                    n_in_cat = vendor_cat_counts.get(category, 1)
                    component_base_cost = (base_cost * cat_share) / n_in_cat
                    component_savings = component_base_cost * savings_pct * (savings_factor / 100)
                    
                    app_component_savings.append({
                        'Vendor': vendor,
                        'Category': category,
                        'Savings': component_savings
                    })
    
    component_savings_df = pd.DataFrame(app_component_savings)
    vendor_component_summary = component_savings_df.groupby(['Vendor', 'Category'])['Savings'].sum().reset_index()
    
    # Regular vendor summary
    vendor_summary = df.groupby('Vendor').agg({
        'App Name': 'count',
        'Base Spend': 'sum',
        'Potential Savings': 'sum',
        'Optimized Spend': 'sum'
    }).reset_index()
    
    vendor_summary.columns = ['Vendor', 'App Count', 'Base Spend', 'Savings', 'Optimized Spend']
    vendor_summary['Savings %'] = vendor_summary['Savings'] / vendor_summary['Base Spend']
    vendor_summary = vendor_summary.sort_values('Savings', ascending=False)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two visuals side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Total Savings by Vendor
        st.markdown("#### Total Savings by Vendor")
        
        vendor_top15 = vendor_summary.head(15).sort_values('Savings', ascending=True)
            
        fig_vendor_savings = go.Figure()
        
        fig_vendor_savings.add_trace(go.Bar(
            x=vendor_top15['Savings'] / 1e6,
            y=vendor_top15['Vendor'],
            orientation='h',
            marker=dict(color=COLORS['primary']),
            text=[f"${v/1e6:.1f}M" for v in vendor_top15['Savings']],
            textposition='outside'
        ))
        
        fig_vendor_savings.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=9),
            xaxis=dict(title='Savings ($M)', gridcolor='#333333'),
            yaxis=dict(title='', gridcolor='#333333'),
            height=450,
            margin=dict(l=120, r=60, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_vendor_savings, use_container_width=True)
    
    with col2:
        # Vendor Cost per App (scatter: bubble size = app count, x = avg cost, y = savings %)
        st.markdown("#### Vendor Cost Efficiency")
        
        fig_vendor_eff = go.Figure()
        
        fig_vendor_eff.add_trace(go.Scatter(
            x=vendor_summary['Base Spend'] / vendor_summary['App Count'] / 1e3,
            y=vendor_summary['Savings %'] * 100,
            mode='markers+text',
            marker=dict(
                size=vendor_summary['App Count'] * 3 + 10,
                color=vendor_summary['Savings %'],
                colorscale=[[0, COLORS['danger']], [0.5, COLORS['secondary']], [1, COLORS['primary']]],
                opacity=0.8,
                line=dict(width=1, color='#333333')
            ),
            text=vendor_summary['Vendor'].apply(lambda x: str(x)[:12]),
            textposition='top center',
            textfont=dict(size=8, color='#CCCCCC'),
            hovertemplate='<b>%{hovertext}</b><br>Avg Cost/App: $%{x:.0f}K<br>Savings: %{y:.1f}%<br><extra></extra>',
            hovertext=vendor_summary['Vendor']
        ))
        
        fig_vendor_eff.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=9),
            xaxis=dict(title='Avg Cost per App ($K)', gridcolor='#333333'),
            yaxis=dict(title='Savings %', gridcolor='#333333'),
            height=450,
            margin=dict(l=50, r=20, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_vendor_eff, use_container_width=True)
    
    # =====================================================
    # BENCHMARK DATA CALCULATION (shared across expanders)
    # =====================================================
    criticality_bench = pd.read_excel('AssetList.xlsx', sheet_name='Criticality Score')
    specificity_bench = pd.read_excel('AssetList.xlsx', sheet_name='Specificity Score')
    structure_bench = pd.read_excel('AssetList.xlsx', sheet_name='Structure')
    
    actual_comp_names = criticality_bench.iloc[0, 1:].tolist()
    component_to_category_bench = dict(zip(structure_bench['Component Lowest'], structure_bench['Component lvl 1']))
    
    # Per-component benchmark (median score-weighted cost across all apps)
    # First: compute score-weighted component costs for every app
    all_app_comp_costs = []  # list of {App, Component, Category, Cost}
    for _, app_row in df.iterrows():
        app_name = app_row['App Name']
        base_cost = app_row['Base Spend']
        app_crit = criticality_bench[criticality_bench.iloc[:, 0] == app_name]
        app_spec = specificity_bench[specificity_bench.iloc[:, 0] == app_name]
        if len(app_crit) > 0 and len(app_spec) > 0:
            # Get scores and categories for valid components
            comp_info = []
            for idx_c, comp_name in enumerate(actual_comp_names):
                cs = pd.to_numeric(app_crit.iloc[0, idx_c+1], errors='coerce')
                ss = pd.to_numeric(app_spec.iloc[0, idx_c+1], errors='coerce')
                if pd.notna(cs) and pd.notna(ss):
                    cat = component_to_category_bench.get(comp_name, 'Infrastructure')
                    comp_info.append((comp_name, cat, (cs + ss) / 2))
            
            # Group by category, then score-weight within each category
            from collections import defaultdict
            cat_comps = defaultdict(list)
            for cn, cat, score in comp_info:
                cat_comps[cat].append((cn, score))
            
            for cat, comps in cat_comps.items():
                if cat == 'Applications':
                    cat_budget = base_cost * (app_cost_share / 100)
                else:
                    cat_budget = base_cost * (infra_cost_share / 100)
                
                total_score = sum(s for _, s in comps)
                if total_score == 0:
                    total_score = 1
                
                for cn, score in comps:
                    comp_cost = cat_budget * (score / total_score)
                    all_app_comp_costs.append({
                        'App Name': app_name,
                        'Component': cn,
                        'Category': cat,
                        'Cost': comp_cost
                    })
    
    all_comp_costs_df = pd.DataFrame(all_app_comp_costs)
    
    # Derive benchmark per component (median across all apps)
    component_benchmarks = []
    for comp_name in actual_comp_names:
        comp_data = all_comp_costs_df[all_comp_costs_df['Component'] == comp_name]
        if len(comp_data) > 0:
            category = component_to_category_bench.get(comp_name, 'Infrastructure')
            component_benchmarks.append({
                'Component': comp_name,
                'Category': category,
                'Benchmark Cost': comp_data['Cost'].median(),
                'Min Cost': comp_data['Cost'].min(),
                'Max Cost': comp_data['Cost'].max()
            })
    benchmark_df = pd.DataFrame(component_benchmarks)
    
    # Per-vendor overall variance from benchmark
    vendor_bench_data = []
    for vendor_name in df['Vendor'].unique():
        vendor_apps = df[df['Vendor'] == vendor_name]
        vendor_actual = vendor_apps['Base Spend'].mean()  # avg cost per app
        portfolio_avg = df['Base Spend'].median()  # portfolio median per app
        variance_pct = ((vendor_actual - portfolio_avg) / portfolio_avg) * 100
        vendor_bench_data.append({
            'Vendor': vendor_name,
            'Avg Cost/App': vendor_actual,
            'Benchmark': portfolio_avg,
            'Variance %': variance_pct,
            'App Count': len(vendor_apps),
            'Total Spend': vendor_apps['Base Spend'].sum()
        })
    vendor_bench_df = pd.DataFrame(vendor_bench_data).sort_values('Variance %', ascending=True)
    
    # Per-vendor per-component costs (reuse score-weighted allocation from above)
    vendor_comp_df = all_comp_costs_df.merge(
        df[['App Name', 'Vendor']].drop_duplicates(), on='App Name', how='left'
    )
    vendor_comp_df = vendor_comp_df.drop(columns=['App Name'])
    
    # =====================================================
    # EXPANDER 1: Overall Vendor Benchmark
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Vendor Benchmark — Overall", expanded=False):
        st.markdown("#### Diverging Bar — Variance from Portfolio Benchmark")
        st.caption("Zero line = portfolio median cost per app. Red = above benchmark (overpriced), Green = below (competitive).")
        
        # Diverging bar chart
        fig_diverge = go.Figure()
        
        vb_sorted = vendor_bench_df.sort_values('Variance %', ascending=True)
        bar_colors = [COLORS['danger'] if v > 0 else COLORS['primary'] for v in vb_sorted['Variance %']]
        
        fig_diverge.add_trace(go.Bar(
            x=vb_sorted['Variance %'],
            y=vb_sorted['Vendor'],
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{v:+.1f}%" for v in vb_sorted['Variance %']],
            textposition='outside',
            customdata=np.stack([
                vb_sorted['Avg Cost/App'] / 1e6,
                vb_sorted['Benchmark'] / 1e6,
                vb_sorted['App Count']
            ], axis=-1),
            hovertemplate='<b>%{y}</b><br>Avg Cost/App: $%{customdata[0]:.2f}M<br>Benchmark: $%{customdata[1]:.2f}M<br>Apps: %{customdata[2]:.0f}<br>Variance: %{x:+.1f}%<extra></extra>'
        ))
        
        fig_diverge.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=10),
            xaxis=dict(
                title='Variance from Benchmark (%)',
                gridcolor='#333333',
                zeroline=True,
                zerolinecolor='#FFFFFF',
                zerolinewidth=2
            ),
            yaxis=dict(title='', gridcolor='#333333'),
            height=max(400, len(vb_sorted) * 25),
            showlegend=False,
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig_diverge, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Dumbbell — Actual vs Benchmark Cost per App")
        st.caption("Each row shows vendor average cost per app (dot) vs portfolio benchmark (line). Gap = over/under-spend.")
        
        # Dumbbell chart — top 20 vendors by total spend for readability
        vb_top = vendor_bench_df.nlargest(20, 'Total Spend').sort_values('Avg Cost/App', ascending=True)
        
        fig_dumbbell = go.Figure()
        
        # Connector lines
        for _, row in vb_top.iterrows():
            line_color = COLORS['danger'] if row['Avg Cost/App'] > row['Benchmark'] else COLORS['primary']
            fig_dumbbell.add_trace(go.Scatter(
                x=[row['Benchmark'] / 1e6, row['Avg Cost/App'] / 1e6],
                y=[row['Vendor'], row['Vendor']],
                mode='lines',
                line=dict(color=line_color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Benchmark dots
        fig_dumbbell.add_trace(go.Scatter(
            x=vb_top['Benchmark'] / 1e6,
            y=vb_top['Vendor'],
            mode='markers',
            name='Benchmark (Median)',
            marker=dict(color='#FFFFFF', size=10, symbol='line-ns-open', line=dict(width=2, color='#FFFFFF')),
            hovertemplate='<b>%{y}</b><br>Benchmark: $%{x:.2f}M<extra></extra>'
        ))
        
        # Actual dots
        dot_colors = [COLORS['danger'] if row['Avg Cost/App'] > row['Benchmark'] else COLORS['primary'] for _, row in vb_top.iterrows()]
        fig_dumbbell.add_trace(go.Scatter(
            x=vb_top['Avg Cost/App'] / 1e6,
            y=vb_top['Vendor'],
            mode='markers',
            name='Actual Avg Cost/App',
            marker=dict(color=dot_colors, size=12, symbol='circle'),
            hovertemplate='<b>%{y}</b><br>Actual: $%{x:.2f}M<extra></extra>'
        ))
        
        fig_dumbbell.update_layout(
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['card'],
            font=dict(color='#FFFFFF', family='Arial', size=10),
            xaxis=dict(title='Cost per App ($M)', gridcolor='#333333'),
            yaxis=dict(title='', gridcolor='#333333'),
            height=max(400, len(vb_top) * 28),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig_dumbbell, use_container_width=True)
    
    # =====================================================
    # EXPANDER 2: Component-Level Benchmark
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Vendor Benchmark — By Component", expanded=False):
        
        # Component level selector
        comp_level_col, _ = st.columns([1, 3])
        with comp_level_col:
            bench_comp_level = st.selectbox(
                "Component Level:",
                ['Component', 'Category'],
                key='bench_comp_level'
            )
        
        # Aggregate vendor × component data
        if bench_comp_level == 'Category':
            group_col = 'Category'
        else:
            group_col = 'Component'
        
        # Calculate vendor avg and benchmark per component group
        vendor_comp_agg = vendor_comp_df.groupby(['Vendor', group_col])['Cost'].mean().reset_index()
        benchmark_by_comp = vendor_comp_df.groupby(group_col)['Cost'].median().to_dict()
        vendor_comp_agg['Benchmark'] = vendor_comp_agg[group_col].map(benchmark_by_comp)
        vendor_comp_agg['Variance %'] = ((vendor_comp_agg['Cost'] - vendor_comp_agg['Benchmark']) / vendor_comp_agg['Benchmark']) * 100
        
        # --- HEATMAP ---
        st.markdown("#### Heatmap — Vendor × Component Variance (%)")
        st.caption("Red = above benchmark (overpriced), Green = below benchmark (competitive), White = at benchmark.")
        
        # Use top 15 vendors by spend and top 15 components by total cost
        top_vendors_for_heatmap = vendor_bench_df.nlargest(15, 'Total Spend')['Vendor'].tolist()
        top_comps_for_heatmap = vendor_comp_df.groupby(group_col)['Cost'].sum().nlargest(15).index.tolist()
        
        heatmap_data = vendor_comp_agg[
            (vendor_comp_agg['Vendor'].isin(top_vendors_for_heatmap)) &
            (vendor_comp_agg[group_col].isin(top_comps_for_heatmap))
        ]
        
        pivot_hm = heatmap_data.pivot_table(index='Vendor', columns=group_col, values='Variance %')
        
        if len(pivot_hm) > 0:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_hm.values,
                x=[str(c)[:25] for c in pivot_hm.columns],
                y=pivot_hm.index.tolist(),
                colorscale=[
                    [0, COLORS['primary']],
                    [0.5, '#FFFFFF'],
                    [1, COLORS['danger']]
                ],
                zmid=0,
                text=np.round(pivot_hm.values, 1),
                texttemplate='%{text:.0f}%',
                textfont={"size": 9},
                colorbar=dict(title="Variance %", ticksuffix='%')
            ))
            
            fig_heatmap.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                xaxis=dict(title='', tickangle=-45, gridcolor='#333333'),
                yaxis=dict(title='', gridcolor='#333333'),
                height=max(400, len(pivot_hm) * 35 + 100),
                margin=dict(l=200, b=150)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Not enough data to build heatmap.")
        
        # --- PER-VENDOR DROPDOWN DETAIL ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Per-Vendor Component Breakdown")
        
        selected_vendor = st.selectbox(
            "Select a vendor to see component-level variance:",
            vendor_bench_df.sort_values('Total Spend', ascending=False)['Vendor'].tolist(),
            key='bench_vendor_select'
        )
        
        if selected_vendor:
            vendor_detail = vendor_comp_agg[vendor_comp_agg['Vendor'] == selected_vendor].copy()
            vendor_detail = vendor_detail.sort_values('Variance %', ascending=True)
            
            # Diverging bar for this vendor's components
            fig_vendor_comp = go.Figure()
            
            comp_bar_colors = [COLORS['danger'] if v > 0 else COLORS['primary'] for v in vendor_detail['Variance %']]
            
            fig_vendor_comp.add_trace(go.Bar(
                x=vendor_detail['Variance %'],
                y=vendor_detail[group_col],
                orientation='h',
                marker=dict(color=comp_bar_colors),
                text=[f"{v:+.1f}%" for v in vendor_detail['Variance %']],
                textposition='outside',
                customdata=np.stack([
                    vendor_detail['Cost'] / 1000,
                    vendor_detail['Benchmark'] / 1000
                ], axis=-1),
                hovertemplate='<b>%{y}</b><br>Actual: $%{customdata[0]:.0f}K<br>Benchmark: $%{customdata[1]:.0f}K<br>Variance: %{x:+.1f}%<extra></extra>'
            ))
            
            fig_vendor_comp.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                title=f'{selected_vendor} — Component Variance from Benchmark',
                xaxis=dict(
                    title='Variance from Benchmark (%)',
                    gridcolor='#333333',
                    zeroline=True,
                    zerolinecolor='#FFFFFF',
                    zerolinewidth=2
                ),
                yaxis=dict(title='', gridcolor='#333333'),
                height=max(400, len(vendor_detail) * 22),
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_vendor_comp, use_container_width=True)
            
            # Summary stats for this vendor
            above = len(vendor_detail[vendor_detail['Variance %'] > 0])
            below = len(vendor_detail[vendor_detail['Variance %'] <= 0])
            avg_var = vendor_detail['Variance %'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Above Benchmark", f"{above} components", delta=None)
            with col2:
                st.metric("Below Benchmark", f"{below} components", delta=None)
            with col3:
                delta_color = "inverse" if avg_var > 0 else "normal"
                st.metric("Avg Variance", f"{avg_var:+.1f}%", delta=f"{'overpriced' if avg_var > 0 else 'competitive'}", delta_color=delta_color)
    
    # Vendor Cost Analysis & Negotiation Strategy
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🎯 Vendor Cost Analysis & Negotiation Strategy", expanded=False):

        vendor_cost_analysis = df.groupby('Vendor').agg({
            'Base Spend': 'sum',
            'Optimized Spend': 'sum',
            'App Name': 'count'
        }).reset_index()
        vendor_cost_analysis.columns = ['Vendor', 'Actual Cost', 'Should-Cost', 'App Count']
        
        # Calculate benchmark (average cost per app across portfolio)
        avg_cost_per_app = df['Base Spend'].sum() / len(df) if len(df) > 0 else 0
        vendor_cost_analysis['Benchmark Cost'] = vendor_cost_analysis['App Count'] * avg_cost_per_app
        vendor_cost_analysis['Negotiation Margin'] = vendor_cost_analysis['Actual Cost'] - vendor_cost_analysis['Should-Cost']
        vendor_cost_analysis['Negotiation %'] = (vendor_cost_analysis['Negotiation Margin'] / vendor_cost_analysis['Actual Cost']) * 100
        vendor_cost_analysis = vendor_cost_analysis.sort_values('Negotiation Margin', ascending=False)
        
        negotiation_analysis = vendor_cost_analysis.copy()
    
        # Negotiation Priority cards
        st.markdown("#### Negotiation Priority")
    
        for idx, row in negotiation_analysis.head(5).iterrows():
            priority = "🔴 HIGH" if row['Negotiation %'] > 40 else "🟡 MEDIUM" if row['Negotiation %'] > 25 else "🟢 LOW"
            
            st.markdown(f"""
            <div style='background-color: {COLORS['card']}; padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                    <span style='color: #FFFFFF; font-weight: 600; font-size: 14px;'>{row['Vendor']}</span>
                    <span style='font-size: 12px;'>{priority}</span>
                </div>
                <div style='font-size: 11px; color: #B0B0B0;'>
                    💰 Margin: ${row['Negotiation Margin']/1e6:.2f}M ({row['Negotiation %']:.1f}%)
                </div>
                <div style='font-size: 11px; color: #B0B0B0;'>
                    📊 Apps: {int(row['App Count'])} | Target: ${row['Should-Cost']/1e6:.1f}M
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        st.markdown("""
        <div style='background-color: #222222; padding: 12px; border-radius: 6px; margin-top: 15px;'>
            <p style='margin: 0; color: #D0D0D0; font-size: 11px;'>
                <b>Priority Levels:</b><br>
                🔴 HIGH: >40% over target<br>
                🟡 MEDIUM: 25-40% over target<br>
                🟢 LOW: <25% over target
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Complete negotiation table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Complete Negotiation Summary")
        
        summary_display = negotiation_analysis[['Vendor', 'App Count', 'Actual Cost', 'Should-Cost', 
                                                'Negotiation Margin', 'Negotiation %']].head(20).copy()
        summary_display['Actual Cost'] = summary_display['Actual Cost'].apply(lambda x: f'${x/1e6:.2f}M')
        summary_display['Should-Cost'] = summary_display['Should-Cost'].apply(lambda x: f'${x/1e6:.2f}M')
        summary_display['Negotiation Margin'] = summary_display['Negotiation Margin'].apply(lambda x: f'${x/1e6:.2f}M')
        summary_display['Negotiation %'] = summary_display['Negotiation %'].apply(lambda x: f'{x:.1f}%')
    
        st.dataframe(summary_display, use_container_width=True, height=400)


with tab7:
    st.markdown("### Vendor Recommendation Engine")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load component data
    structure_rec = pd.read_excel('AssetList.xlsx', sheet_name='Structure')
    criticality_rec = pd.read_excel('AssetList.xlsx', sheet_name='Criticality Score')
    specificity_rec = pd.read_excel('AssetList.xlsx', sheet_name='Specificity Score')
    
    comp_names_rec = list(criticality_rec.iloc[0, 1:].values)
    comp_to_cat_rec = dict(zip(structure_rec['Component Lowest'], structure_rec['Component lvl 1']))
    
    # === SECTION A: Quick Glance — Best Vendor by Sector × Department ===
    st.markdown("#### Best Vendor by Segment")
    st.caption("Lowest average cost per app for each Sector × Department combination")
    
    segment_data = df.groupby(['Sector', 'Department', 'Vendor']).agg(
        Avg_Cost=('Base Spend', 'mean'),
        App_Count=('App Name', 'count')
    ).reset_index()
    
    # Find best vendor (lowest avg cost) per segment
    best_vendors = segment_data.loc[segment_data.groupby(['Sector', 'Department'])['Avg_Cost'].idxmin()]
    best_vendors = best_vendors.sort_values(['Sector', 'Department'])
    
    # Display as styled table
    bv_display = best_vendors[['Sector', 'Department', 'Vendor', 'Avg_Cost', 'App_Count']].copy()
    bv_display['Avg_Cost'] = bv_display['Avg_Cost'].apply(lambda x: f'${x/1e3:.0f}K')
    bv_display.columns = ['Sector', 'Department', 'Best Vendor', 'Avg Cost/App', 'Apps']
    st.dataframe(bv_display, use_container_width=True, height=min(400, len(bv_display) * 35 + 40))
    
    # === SECTION B: Component-Weighted Vendor Scoring ===
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Component-Weighted Vendor Scoring")
    st.markdown("Select which components matter most for your project. The engine scores each vendor based on historical cost-per-component performance.")
    
    # Group components by category
    app_comps = [c for c in comp_names_rec if comp_to_cat_rec.get(c, '') == 'Applications']
    infra_comps = [c for c in comp_names_rec if comp_to_cat_rec.get(c, '') != 'Applications']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Applications Components")
        selected_app_comps = st.multiselect("Select:", app_comps, default=app_comps[:3], key="rec_app_comps")
    with col2:
        st.markdown("##### Infrastructure Components")
        selected_infra_comps = st.multiselect("Select:", infra_comps, default=infra_comps[:3], key="rec_infra_comps")
    
    selected_comps = selected_app_comps + selected_infra_comps
    
    if len(selected_comps) > 0:
        # Build per-vendor per-component cost matrix
        vendor_comp_costs = []
        for _, row in df.iterrows():
            app_name = row['App Name']
            vendor = row['Vendor']
            base_cost = row['Base Spend']
            
            app_crit = criticality_rec[criticality_rec.iloc[:, 0] == app_name]
            app_spec = specificity_rec[specificity_rec.iloc[:, 0] == app_name]
            
            if len(app_crit) > 0 and len(app_spec) > 0:
                # Score-weighted cost within categories
                comp_info = []
                for idx_c, cn in enumerate(comp_names_rec):
                    cs = pd.to_numeric(app_crit.iloc[0, idx_c+1], errors='coerce')
                    ss = pd.to_numeric(app_spec.iloc[0, idx_c+1], errors='coerce')
                    if pd.notna(cs) and pd.notna(ss):
                        cat = comp_to_cat_rec.get(cn, 'Infrastructure')
                        comp_info.append((cn, cat, (cs + ss) / 2))
                
                from collections import defaultdict
                cat_groups = defaultdict(list)
                for cn, cat, score in comp_info:
                    cat_groups[cat].append((cn, score))
                
                for cat, comps in cat_groups.items():
                    if cat == 'Applications':
                        budget = base_cost * (app_cost_share / 100)
                    else:
                        budget = base_cost * (infra_cost_share / 100)
                    total_s = sum(s for _, s in comps)
                    if total_s == 0: total_s = 1
                    for cn, s in comps:
                        if cn in selected_comps:
                            vendor_comp_costs.append({
                                'Vendor': vendor,
                                'Component': cn,
                                'Cost': budget * (s / total_s)
                            })
        
        if vendor_comp_costs:
            vcc_df = pd.DataFrame(vendor_comp_costs)
            
            # Average cost per component per vendor
            vendor_scores = vcc_df.groupby('Vendor').agg(
                Total_Cost=('Cost', 'sum'),
                Avg_Comp_Cost=('Cost', 'mean')
            ).reset_index()
            
            # Count apps per vendor
            vendor_app_counts = df.groupby('Vendor')['App Name'].count().reset_index()
            vendor_app_counts.columns = ['Vendor', 'App Count']
            vendor_scores = vendor_scores.merge(vendor_app_counts, on='Vendor')
            
            # Normalize: cost per app for selected components
            vendor_scores['Cost Per App'] = vendor_scores['Total_Cost'] / vendor_scores['App Count']
            vendor_scores = vendor_scores.sort_values('Cost Per App')
            
            # Score: lower cost = better score (0-100)
            max_cost = vendor_scores['Cost Per App'].max()
            min_cost = vendor_scores['Cost Per App'].min()
            if max_cost > min_cost:
                vendor_scores['Score'] = 100 * (1 - (vendor_scores['Cost Per App'] - min_cost) / (max_cost - min_cost))
            else:
                vendor_scores['Score'] = 100
            
            # Top vendor card
            best = vendor_scores.iloc[0]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS['card']} 0%, #252525 100%); padding: 20px; border-radius: 10px; border-top: 3px solid {COLORS['primary']}; margin: 15px 0;'>
                <p style='margin: 0; color: #D0D0D0; font-size: 12px; letter-spacing: 1px;'>🏆 RECOMMENDED VENDOR</p>
                <h2 style='margin: 5px 0; color: {COLORS['primary']}; font-size: 28px;'>{best['Vendor']}</h2>
                <p style='margin: 0; color: #B0B0B0; font-size: 13px;'>Score: {best['Score']:.0f}/100 | Cost/App: ${best['Cost Per App']/1e3:.0f}K | {best['App Count']:.0f} historical apps</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Ranked bar chart — top 10 only
            st.markdown("##### Vendor Ranking by Component Cost Efficiency")
            
            vendor_top10 = vendor_scores.head(10).sort_values('Score', ascending=True)
            
            fig_rec = go.Figure()
            
            colors_rec = [COLORS['primary'] if s == vendor_top10['Score'].max() else (COLORS['secondary'] if s >= vendor_top10['Score'].quantile(0.7) else '#666666') for s in vendor_top10['Score']]
            
            fig_rec.add_trace(go.Bar(
                x=vendor_top10['Score'],
                y=vendor_top10['Vendor'],
                orientation='h',
                marker=dict(color=colors_rec),
                text=[f"Score: {s:.0f} | ${c/1e3:.0f}K/app" for s, c in zip(vendor_top10['Score'], vendor_top10['Cost Per App'])],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>'
            ))
            
            fig_rec.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['card'],
                font=dict(color='#FFFFFF', family='Arial', size=10),
                xaxis=dict(title='Efficiency Score (100 = Best)', gridcolor='#333333', range=[0, 115]),
                yaxis=dict(title='', gridcolor='#333333'),
                height=380,
                margin=dict(l=150, r=100, t=20, b=40),
                showlegend=False
            )
            
            st.plotly_chart(fig_rec, use_container_width=True)
            
            # Component breakdown for top 3 vendors — in expander
            with st.expander("🔬 Component Cost Breakdown — Top 3 Vendors", expanded=False):
            
                top3 = vendor_scores.head(3)['Vendor'].tolist()
                top3_data = vcc_df[vcc_df['Vendor'].isin(top3)]
            
                pivot = top3_data.groupby(['Vendor', 'Component'])['Cost'].mean().reset_index()
            
                fig_comp_rec = go.Figure()
            
                comp_colors = [COLORS['primary'], COLORS['secondary'], '#9D4EDD']
                for i, v in enumerate(top3):
                    v_data = pivot[pivot['Vendor'] == v].sort_values('Cost', ascending=True)
                    fig_comp_rec.add_trace(go.Bar(
                        name=v,
                        x=v_data['Cost'] / 1000,
                        y=v_data['Component'],
                        orientation='h',
                        marker=dict(color=comp_colors[i])
                    ))
            
                fig_comp_rec.update_layout(
                    paper_bgcolor=COLORS['background'],
                    plot_bgcolor=COLORS['card'],
                    font=dict(color='#FFFFFF', family='Arial', size=9),
                    xaxis=dict(title='Avg Cost per Component ($K)', gridcolor='#333333'),
                    yaxis=dict(title='', gridcolor='#333333'),
                    barmode='group',
                    height=max(400, len(selected_comps) * 35),
                    margin=dict(l=200, r=20, t=30, b=40),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
            
                st.plotly_chart(fig_comp_rec, use_container_width=True)
        else:
            st.info("No component cost data available for the selected components.")
    else:
        st.info("Please select at least one component to generate vendor recommendations.")
