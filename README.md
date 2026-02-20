# Quant16 Portfolio Optimization Dashboard

A comprehensive Streamlit dashboard for analyzing application portfolio optimization with component-level cost analysis, AI impact assessment, and interactive visualizations.

## Features

- **Portfolio Overview**: Track baseline, optimization, and AI savings across fiscal years
- **Component Analysis**: Drill down into 40+ technical components with complexity scoring
- **AI Impact Assessment**: Measure automation potential across 10 IA activities
- **Vendor Benchmarking**: Compare costs against component-level benchmarks
- **Department & Sector Views**: Analyze spending by organizational structure
- **Interactive Hierarchy**: Explore portfolio spending with markmap visualization
- **Top/Bottom Performers**: Identify best and worst optimization opportunities
- **Application Inventory**: Searchable/filterable database with export capability

## Installation

### 1. Install Python Dependencies

```bash
pip install streamlit pandas plotly openpyxl numpy streamlit-markmap
```

### 2. File Structure

Ensure these files are in the same directory:
```
your-project/
├── streamlit_app.py
└── AssetList.xlsx
```

### 3. Data Requirements

`AssetList.xlsx` should contain these sheets:
- **AssetList**: App Name, Sector, Department, Vendor, Fiscal Year, Development Cost
- **Structure**: Component lvl 1, Component lvl 2, Component Lowest
- **Criticality Score**: Component criticality ratings (1-5)
- **Specificity Score**: Component specificity ratings (1-5)
- **IA Engagement Time**: Time allocation per activity (decimal %)
- **AI Automation Potential**: AI automation potential per activity (decimal %)
- **IA Structure**: Activity categorization (Soft vs Hard)

## Usage

### Run the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Tabs

1. **📊 Overview**
   - 4 metric cards: Base Spend, Optimization, AI Savings, Combined Total
   - Fiscal year trends with projections
   - Department and Sector analysis
   - Interactive spending hierarchy markmap

2. **📈 Portfolio Variance**
   - Component-level savings analysis with drilldown (Applications vs Infrastructure)
   - Application comparison charts (scrollable)
   - Cumulative savings visualization
   - Component complexity distribution

3. **🤖 AI Impact**
   - AI cost reduction breakdown
   - Activity-level automation analysis (Soft vs Hard)
   - AI savings by application
   - Cumulative AI impact

4. **🏆 Top/Bottom Apps**
   - Top 3 highest savings applications
   - Bottom 3 lowest savings applications
   - Combined optimization + AI breakdown
   - Stacked comparison chart

5. **📋 App Inventory**
   - Searchable application database
   - Filters: Sector, Department, Vendor
   - Detailed savings columns (Optimization + AI)
   - CSV export functionality

6. **🏢 Savings by Vendor**
   - Top 5 vendors with component breakdown (Applications vs Infrastructure)
   - Component cost benchmark analysis
   - Top apps variance from benchmark
   - Detailed component comparison
   - Vendor benchmarking methodology

## Key Formulas

### Optimization Savings
```
Combined Score = (Avg Criticality + Avg Specificity) / 2
Savings % = 0.80 - ((Combined Score - 1) × 0.20)
Optimized Spend = Base Spend - (Base Spend × Savings %)
```

### AI Impact
```
AI Savings % = Σ(IA Engagement Time × AI Automation Potential)
AI Cost Reduction = Optimized Spend × AI Savings %
Final Cost = Optimized Spend - AI Cost Reduction
```

### Component Benchmarks
```
Benchmark Cost = Median(Component Cost across all apps)
Variance % = (Actual Cost - Benchmark Cost) / Benchmark Cost × 100
```

## Sidebar Controls

- **Savings Adjustment**: Scale optimization savings (0-100%)
- **AI Cost Reduction**: Scale AI impact (0-100%)
- **Enable Projections**: Toggle fiscal year projections
- **Projection Years**: Set forecast period (1-7 years)
- **Noise Level**: Add variance to projections (0-20%)

## Customization

### Colors
Edit the `COLORS` dictionary in `streamlit_app.py`:
```python
COLORS = {
    'primary': '#00D9A3',      # Green - savings, optimized
    'secondary': '#FF8C42',    # Orange - baseline, costs
    'danger': '#FF444',        # Red - warnings
    'background': '#0F0F0F',   # Black
    'card': '#1E1E1E'          # Dark gray
}
```

### Component Filter
In Tab 2 (Portfolio Variance), use the dropdown to filter by:
- All components
- Applications only
- Infrastructure only

## Data Export

Download filtered inventory data as CSV from the App Inventory tab using the "📥 Download Inventory as CSV" button.

## Interactive Markmap

The Portfolio Spending Hierarchy uses `streamlit-markmap` for interactive visualization. Click any node to expand/collapse the hierarchy showing:

```
Portfolio → Sector → Department → Component Lvl 1 → Lvl 2 → Lowest
```

## Troubleshooting

### Issue: Charts not scrolling
- Reduce browser zoom level
- Charts dynamically adjust height based on number of apps

### Issue: Markmap not displaying
- Ensure `streamlit-markmap` is installed: `pip install streamlit-markmap`
- Check browser console for errors
- Fallback text view is available if package not installed

### Issue: Department filter showing errors
- Ensure Department column has no mixed data types
- Dashboard automatically converts to string format

### Issue: Component savings showing zero
- Verify Criticality Score and Specificity Score sheets have data
- Check that component names in first row match Structure sheet

## Performance Notes

- Dashboard caches data loading for faster performance
- Recommended for portfolios up to 200 applications
- Component calculations process 40+ components × 100 apps = 4,000+ data points

## Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- Plotly 5.0+
- NumPy 1.24+
- OpenPyXL 3.0+
- streamlit-markmap 0.3+

## License

Proprietary - Quant16 Portfolio Optimization Model

## Support

For questions or issues, contact your system administrator.

---

**Version**: 2.0  
**Last Updated**: February 2026  
**Developed by**: Quant16 Team
