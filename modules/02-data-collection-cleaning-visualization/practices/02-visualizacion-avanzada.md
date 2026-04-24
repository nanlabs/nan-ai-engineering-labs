# Práctica 02 — Visualización de Datos Avanzada

## 🎯 Objetivos

- Crear visualizaciones efectivas con Matplotlib y Seaborn
- Diseñar dashboards informativos
- Aplicar principios de storytelling con datos
- Generar gráficos interactivos

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Gráficos de distribución

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Dataset de ejemplo
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'sales': np.random.lognormal(10, 0.5, n),
    'customers': np.random.poisson(50, n),
    'satisfaction': np.random.normal(4.2, 0.8, n).clip(1, 5),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
    'month': np.random.choice(range(1, 13), n)
})

# Múltiples distribuciones
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

# Box Plot por categoría
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

print("✅ Gráficos de distribución creados")
```

### Ejercicio 1.2: Gráficos de relación

```python
# Correlación y scatter plots

# Agregar variable correlacionada
df['revenue'] = df['sales'] * df['customers'] / 10 + np.random.normal(0, 1000, n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot básico
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

# Heatmap de correlación
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

### Ejercicio 1.3: Time series visualization

```python
# Agregar componente temporal
df_ts = df.groupby('month').agg({
    'sales': 'sum',
    'customers': 'sum',
    'satisfaction': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Line plot con área
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

# Combinar legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('timeseries_viz.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualizaciones temporales creadas")
```

### Ejercicio 1.4: Dashboard completo

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

print("✅ Dashboard completo creado")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Small Multiples

**Enunciado:**
Crea una visualización "small multiples" (facet grid) que muestre:

- Sales distribution por cada categoría (4 subplots)
- Usar misma escala en X para comparabilidad
- Añadir media y mediana en cada subplot

**Validación:**

```python
# Debe generar 1 figura con 4 subplots (2x2)
# Cada subplot debe tener título con nombre de categoría
# Lines verticales para mean y median en cada subplot
```

### Ejercicio 2.2: Stacked Bar Chart

**Enunciado:**
Visualiza la composición de ventas por mes y categoría:

- Eje X: meses (1-12)
- Eje Y: total sales
- Barras apiladas por categoría con colores diferentes
- Leyenda ordenada por total sales descendente

### Ejercicio 2.3: Sunburst/Treemap (con matplotlib)

**Enunciado:**
Crea un treemap que muestre:

- Jerarquía: Category → Month → Sales
- Tamaño de cada rectángulo proporcional a sales
- Colores por categoría
- Labels con porcentaje del total

### Ejercicio 2.4: Visualización de Outliers

**Enunciado:**
Crea un panel de diagnóstico de outliers:

1. Boxplot con puntos individuales de outliers marcados
1. Scatter plot marcando outliers en rojo
1. Histograma con outliers en barra diferente
1. Tabla con top 5 outliers y sus valores

### Ejercicio 2.5: Animated Plot (exportar frames)

**Enunciado:**
Crea una secuencia de gráficos que muestren evolución temporal:

- Un PNG por mes mostrando sales acumuladas hasta ese mes
- Barra progresiva o line plot incremental
- Exportar 12 imágenes: `month_01.png` a `month_12.png`
- Bonus: usar `imageio` para crear GIF

______________________________________________________________________

## ✅ Checklist de Competencias

Después de completar esta práctica, deberías poder:

- [ ] Crear histogramas, KDE plots, boxplots y violin plots
- [ ] Diseñar scatter plots con múltiples dimensiones (color, size, shape)
- [ ] Generar heatmaps de correlación legibles
- [ ] Visualizar series temporales con trends y seasonality
- [ ] Construir dashboards multi-panel informativos
- [ ] Aplicar principios de diseño (color, layout, jerarquía)
- [ ] Exportar visualizaciones en alta resolución
- [ ] Elegir tipo de gráfico apropiado para cada mensaje

______________________________________________________________________

## 📚 Recursos Adicionales

**Gallerías de inspiración:**

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Python Graph Gallery](https://python-graph-gallery.com/)

**Librerías interactivas:**

- `plotly`: Gráficos interactivos
- `bokeh`: Dashboards web
- `altair`: Grammar of graphics

**Principios de diseño:**

- [Data Visualization catalog](https://datavizcatalogue.com/)
- [Color Brewer](https://colorbrewer2.org/) para paletas accesibles
