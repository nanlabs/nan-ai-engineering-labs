# Practice 02 — Advanced Data Visualization

## 🎯 Objectives

- Create effective visualizations with Matplotlib and Seaborn
- Design informative dashboards
- Apply storytelling principles with Data
- Generate interactive graphics

______________________________________________________________________

## 📚 Part 1: Guided Exercises

### Exercise 1.1: Distribution Graphs

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Dataset de example
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'sales': np.random.lognormal(10, 0.5, n),
    'customers': np.random.poisson(50, n),
    'satisfaction': np.random.normal(4.2, 0.8, n).clip(1, 5),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
    'month': np.random.choice(range(1, 13), n)
})

# Multiple distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma
axes[0, 0].hist(df['sales'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Sales ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Sales')
axes[0, 0].axvline(df['sales'].median(), color='red', linestyle='--', label='Median')
axes[0, 0].legend()

# KDE Plot
sns.kdeplot(data=df, x='satisfaction', ax=axes[0, 1], fill=True, color='coral')
axes[0, 1].set_xlabel('Satisfaction Score')
axes[0, 1].set_title('Customer Satisfaction Distribution')

# Box Plot por category
sns.boxplot(data=df, x='category', y='sales', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].set_title('Sales by Category')
axes[1, 0].tick_params(axis='x', rotation=45)

# Violin Plot
sns.violinplot(data=df, x='category', y='satisfaction', ax=axes[1, 1], palette='muted')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Satisfaction Score')
axes[1, 1].set_title('Satisfaction by Category')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('distributions_multi.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Graphics de distribution creados")
```

### Exercise 1.2: Relationship Graphs

```python
# Correlation y scatter plots

# Add variable correlacionada
df['revenue'] = df['sales'] * df['customers'] / 10 + np.random.normal(0, 1000, n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot basic
axes[0].scatter(df['sales'], df['revenue'], alpha=0.5, c='steelblue')
axes[0].set_xlabel('Sales ($)')
axes[0].set_ylabel('Revenue ($)')
axes[0].set_title('Sales vs Revenue')
axes[0].grid(True, alpha=0.3)

# Scatter plot con categorical hue
for category in df['category'].unique():
    subset = df[df['category'] == category]
    axes[1].scatter(subset['sales'], subset['revenue'], alpha=0.6, label=category, s=50)

axes[1].set_xlabel('Sales ($)')
axes[1].set_ylabel('Revenue ($)')
axes[1].set_title('Sales vs Revenue by Category')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Heatmap de correlation
numeric_cols = ['sales', 'customers', 'satisfaction', 'revenue']
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[2], square=True, linewidths=1)
axes[2].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('relationships.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Correlaciones ===")
print(corr_matrix)
```

### Exercise 1.3: Time series visualization

```python
# Add componente temporal
df_ts = df.groupby('month').agg({
    'sales': 'sum',
    'customers': 'sum',
    'satisfaction': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Line plot con area
axes[0].plot(df_ts['month'], df_ts['sales'], marker='o',
             linewidth=2, markersize=8, color='steelblue', label='Sales')
axes[0].fill_between(df_ts['month'], df_ts['sales'], alpha=0.3, color='steelblue')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Sales ($)')
axes[0].set_title('Monthly Sales Trend')
axes[0].set_xticks(range(1, 13))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dual axis
ax1 = axes[1]
ax2 = ax1.twinx()

ax1.bar(df_ts['month'], df_ts['customers'], alpha=0.7, color='coral', label='Customers')
ax2.plot(df_ts['month'], df_ts['satisfaction'], marker='s',
         linewidth=2, color='green', label='Satisfaction', markersize=6)

ax1.set_xlabel('Month')
ax1.set_ylabel('Total Customers', color='coral')
ax2.set_ylabel('Avg Satisfaction', color='green')
ax1.set_title('Monthly Customers vs Satisfaction')
ax1.set_xticks(range(1, 13))
ax1.tick_params(axis='y', labelcolor='coral')
ax2.tick_params(axis='y', labelcolor='green')

# Combiner legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('timeseries_viz.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualizaciones temporales creadas")
```

### Exercise 1.4: Dashboard complete

```python
# Dashboard de 6 paneles

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: KPI (Sales)
ax1 = fig.add_subplot(gs[0, 0])
total_sales = df['sales'].sum()
ax1.text(0.5, 0.5, f'${total_sales:,.0f}',
         ha='center', va='center', fontsize=32, fontweight='bold', color='steelblue')
ax1.text(0.5, 0.2, 'Total Sales', ha='center', va='center', fontsize=14, color='gray')
ax1.axis('off')

# Panel 2: KPI (Customers)
ax2 = fig.add_subplot(gs[0, 1])
total_customers = df['customers'].sum()
ax2.text(0.5, 0.5, f'{total_customers:,}',
         ha='center', va='center', fontsize=32, fontweight='bold', color='coral')
ax2.text(0.5, 0.2, 'Total Customers', ha='center', va='center', fontsize=14, color='gray')
ax2.axis('off')

# Panel 3: KPI (Satisfaction)
ax3 = fig.add_subplot(gs[0, 2])
avg_satisfaction = df['satisfaction'].mean()
ax3.text(0.5, 0.5, f'{avg_satisfaction:.2f}/5',
         ha='center', va='center', fontsize=32, fontweight='bold', color='green')
ax3.text(0.5, 0.2, 'Avg Satisfaction', ha='center', va='center', fontsize=14, color='gray')
ax3.axis('off')

# Panel 4: Top categories
ax4 = fig.add_subplot(gs[1, :2])
category_sales = df.groupby('category')['sales'].sum().sort_values(ascending=True)
ax4.barh(category_sales.index, category_sales.values, color='steelblue', alpha=0.7)
ax4.set_xlabel('Total Sales ($)')
ax4.set_title('Sales by Category', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Panel 5: Distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df['satisfaction'], bins=20, color='green', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Satisfaction')
ax5.set_ylabel('Frequency')
ax5.set_title('Satisfaction Distribution', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Monthly trend
ax6 = fig.add_subplot(gs[2, :])
monthly_sales = df.groupby('month')['sales'].sum()
ax6.plot(monthly_sales.index, monthly_sales.values, marker='o',
         linewidth=2, markersize=8, color='steelblue')
ax6.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.3, color='steelblue')
ax6.set_xlabel('Month')
ax6.set_ylabel('Total Sales ($)')
ax6.set_title('Monthly Sales Trend', fontweight='bold')
ax6.set_xticks(range(1, 13))
ax6.grid(True, alpha=0.3)

plt.suptitle('Sales Analytics Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('dashboard_complete.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Dashboard complete creado")
```

______________________________________________________________________

## 🚀 Part 2: Suggested Exercises

### Exercise 2.1: Small Multiples

**Statement:**
Create a "small multiples" (facet grid) Visualization that shows:

- Sales distribution by each category (4 subplots)
- Use same scale on X for comparability
- Add mean and median in each subplot

**Validation:**

```python
# Debe generar 1 figura con 4 subplots (2x2)
# Cada subplot must tener qualification con nombre de category
# Lines verticales para mean y median en each subplot
```

### Exercise 2.2: Stacked Bar Chart

**Statement:**
View the sales composition by month and category:

- X axis: months (1-12)
- Y axis: total sales
- Bars stacked by category with different colors
- Legend sorted by total sales descent

### Exercise 2.3: Sunburst/Treemap (with matplotlib)

**Statement:**
Create a treemap that shows:

- Hierarchy: Category → Month → Sales
- Size of each rectangle proportional to sales
- Colors by category
- Labels with percentage of the total

### Exercise 2.4: Visualization of outliers

**Statement:**
Create an outliers diagnostic panel:

1. Boxplot with individual outlier points marked
1. Scatter plot marking outliers in red
1. Histogram with outliers in different bars
1. Table with top 5 outliers and their values

### Exercise 2.5: Animated Plot (export frames)

**Statement:**
Create a sequence of graphs that show time evolution:

- One PNG per month showing accumulated sales up to that month
- Progressive bar or incremental line plot
- Export 12 Images: `month_01.png` to `month_12.png`
- Bonus: use `imageio` to create GIF

______________________________________________________________________

## ✅ Skills Checklist

After completing this Practice, you should be able to:

- [ ] Create histograms, KDE plots, boxplots and violin plots
- [ ] Design scatter plots with multiple dimensions (color, size, shape)
- [ ] Generate readable correlation heatmaps
- [ ] Visualize time series with trends and seasonality
- [ ] Build informative multi-panel dashboards
- [ ] Apply design principles (color, layout, hierarchy)
- [ ] Export visualizations in high resolution
- [ ] Choose appropriate Chart Type for each message

______________________________________________________________________

## 📚 Additional Resources

**Inspiration Galleries:**

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Python Graph Gallery](https://python-graph-gallery.com/)

**Interactive libraries:**

- `plotly`: Interactive plots
- `bokeh`: Web Dashboards
- `altair`: Grammar of graphics

**Design principles:**

- [Data Visualization catalog](https://datavizcatalogue.com/)
- [Color Brewer](https://colorbrewer2.org/) for accessible palettes
