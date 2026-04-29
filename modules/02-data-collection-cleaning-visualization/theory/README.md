# Theory — Data Collection, Cleaning & Visualization

## Why this module matters

The quality of the Data determines the upper limit of performance of any ML Model. Without clean, structured and well-understood data, even the best Algorithm will fail. This Module gives you tools to transform raw data into reliable datasets ready to train Models.

______________________________________________________________________

## 1. Data Types

### Classification by Structure

- **Structured data:** SQL tables, CSVs, Excel (rows and columns).
- **Semi-structured data:** JSON, XML, logs.
- **Unstructured data:** free text, Images, audio, video.

### Implications for ML

- Traditional models (regression, trees) work with structured tabular data.
- Deep learning efficiently handles unstructured data (Images, text).

______________________________________________________________________

## 2. Data Sources

### Obtaining strategies

- **APIs:** Data from web services (Twitter API, GitHub API, OpenWeather).
- **Web scraping:** extract Data from HTML using BeautifulSoup or Scrapy.
- **Databases:** SQL (PostgreSQL, MySQL), NoSQL (MongoDB).
- **Files:** CSV, JSON, Parquet, Excel.
- **Public datasets:** Kaggle, UCI ML Repository, Google Dataset Search.

### Critical Considerations

- Always verify licenses and terms of Data Usage.
- Evaluate update frequency and source reliability.
- Consider volume necessary to train robust Models.

📹 **Videos recommended:**

1. [Working with APIs in Python - Corey Schafer](https://www.youtube.com/watch?v=tb8gHvYlCFs) - 18 min
1. [Web Scraping with BeautifulSoup - freeCodeCamp](https://www.youtube.com/watch?v=XVv6mJpFOb0) - 2 horas
1. [Pandas for Data Analysis - Keith Galli](https://www.youtube.com/watch?v=vmEHCJofslg) - 1 hour

______________________________________________________________________

## 3. Data Quality and Cleaning

### Common problems in real data

- **Nuls values:** Missing data (NaN, None, null, empty spaces).
- **Duplicates:** repeated rows that artificially inflate Metrics.
- **outliers:** outlier values ​​that may be legitimate errors or extreme cases.
- **Inconsistent formats:** dates in multiple formats, mixed uppercase/lowercase.
- **Incorrect Data Types:** numbers saved as strings, categories as numbers.

### Cleaning Techniques by Problem

**For values ​​nulls:**

- Remove rows/columns with many nulls (when % is high, >50%).
- Impute with mean, median, mode (according to distribution and context).
- Forward fill / backward fill for time series.
- Use predictive models to impute (KNN Imputer, regression).

**For duplicates:**

- Detect with `.duplicated()` and remove with `.drop_duplicates()`.
- Decide which row to keep (first, last, or custom).

**For outliers:**

- IQR method (interquartile range): valuesoutside \[Q1 - 1.5*IQR, Q3 + 1.5*IQR\].
- Z-score: values ​​with |z| > 3 are potential outliers.
- Visualization with boxplots for informed decision.

**For formats:**

- Normalize strings: `.str.lower()`, `.str.strip()`, remove special characters.
- Date parsing: `pd.to_datetime()` with explicit format.
- Types conversion: `.astype()` with prior Validation.

📹 **Videos recommended:**

1. [Data Cleaning with Pandas - Corey Schafer](https://www.youtube.com/watch?v=eMOA1pPVUc4) - 20 min
1. [Handling Missing Data - StatQuest](https://www.youtube.com/watch?v=jb3CVnBYgQc) - 10 min
1. [Complete Data Cleaning Tutorial - Alex The Analyst](https://www.youtube.com/watch?v=bDhvCp3_lYw) - 30 min

📚 **Resources written:**

- [Pandas Data Cleaning Guide](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [DataCamp - Data Cleaning with Python](https://www.datacamp.com/tutorial/data-cleaning-python)
- [Kaggle - Data Cleaning Course](https://www.kaggle.com/learn/data-cleaning)

______________________________________________________________________

## 4. Feature Engineering

### Concepts clave

**Feature engineering** is creating new features from existing ones to improve the predictive capacity of the Model.

### common transformations

- **Logarithms:** compress large ranges, normalize asymmetric distributions.
- **Square root / powers:** adjust non-linear relationships.
- **Binning:** convert continuous variables into categories (age → age ranges).
- **Interactions:** multiply related features (area = width × height).

### Encoding of categorical variables

- **One-Hot Encoding:** create binary columns for each category.
- **Label Encoding:** assign integers to categories (only if there is order).
- **Target Encoding:** replace category by means of the target in that category.
- **Frequency Encoding:** replace with frequency of occurrence.

### Feature Scaling (Normalization)

- **Min-Max Scaling:** scale to range \[0, 1\]: `(x - min) / (max - min)`.
- **Standardization (Z-score):** media 0, std 1: `(x - mean) / std`.
- **When use each uno:**
- Min-Max: when you need limited range (neural networks).
  - Standardization: when distribution matters (regression, SVM).

### Examples applied

- From a date extract: day of the week, month, quarter, is_end_of_week.
- From text extract: length, number of words, presence of keywords.
- Transactions: average amount in the last 30 days, standard deviation, maximum.

📹 **Videos recommended:**

1. [Feature Engineering Course - Applied AI](https://www.youtube.com/watch?v=2N7hCn40YdY) - 1 hora
1. [Feature Scaling Explained - Krish Naik](https://www.youtube.com/watch?v=mnKm3YP56PY) - 15 min
1. [Categorical Encoding - Data Science Dojo](https://www.youtube.com/watch?v=irHhDMbw3xo) - 20 min

📚 **Resources written:**

- [Feature Engineering - Google ML Guide](https://developers.google.com/machine-learning/data-prep/transform/introduction)
- [Feature Engineering for Machine Learning - Towards Data Science](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

______________________________________________________________________

## 5. Exploratory Data Analysis (EDA)

### Objective of the EDA

Understand the Structure, distribution and relationships in your Data **BEFORE** training Models. EDA prevents costly errors and reveals insights for feature engineering.

### Typical EDA steps

**1. Initial statistical summary:**

```python
df.info()          # types, nulls
df.describe()      # statistics numerical
df.value_counts()  # frequencies categorical
```

**2. Analysis of distributions:**

- Histograms for numerical variables.
- KDE plots for probability density.
- Countplots for categorical.

**3. Relationship analysis:**

- Scatter plots for relationship between two numerical variables.
- Grouped box plots to compare distributions by category.
- Correlation matrix (heatmap) to detect multi-collinearity.

**4. Pattern detection:**

- Natural groupings (clusters).
- Temporal trends.
- outliers and anomalies.

### Visualization Tools

**Matplotlib:** highly customizable base plots.
**Seaborn:** Elegant statistical visualizations with less code.
**Plotly:** Interactive plots for dynamic Exploration.

📹 **Videos recommended:**

1. [EDA with Pandas - Corey Schafer](https://www.youtube.com/watch?v=Wb2Tp35dZ-I) - 35 min
1. [Data Visualization with Seaborn - Keith Galli](https://www.youtube.com/watch?v=6GUZXDef2U0) - 1 hour
1. [Complete EDA Tutorial - Ken Jee](https://www.youtube.com/watch?v=QWgg4w1SpJ8) - 40 min

📚 **Resources written:**

- [From Data to Viz](https://www.data-to-viz.com/) - chart selection guide
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Python Graph Gallery](https://www.python-graph-gallery.com/)
- [Kaggle EDA Notebooks](https://www.kaggle.com/code?tags=13204) - Examples real

______________________________________________________________________

## 6. Buenas Practices

### During Data preparation

- **Document transformations:** record all applied steps.
- **Do not contaminate test set:** separate Data BEFORE any transformation.
- **Reproducibility:** set random seeds when splitting train/test.
- **Automated pipelines:** use `sklearn.pipeline.Pipeline` to chain transformations.
- **Version datasets:** save snapshots of processed data.

### Quality verification (checklist)

- ✅ Check % of nulls per column.
- ✅ Validate expected ranges of numerical values.
- ✅ Confirm that there are no unintentional duplicates.
- ✅ Check consistency of date/time formats.
- ✅ Detect features with perfect correlation (redundant).
- ✅ Validate balance of classes in Classification Problems.

📚 **Additional resources:**

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Project structure
- [Great Expectations](https://greatexpectations.io/) - Automated Data Validation
- [Pandas Profiling](https://github.com/ydataai/ydata-profiling) - automatic EDA

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Load and explore a real Kaggle dataset using Pandas.
- ✅ Detect and handle null values ​​with at least 2 different justified strategies.
- ✅ Identify outliers and decide whether to eliminate them, keep them or treat them according to context.
- ✅ Perform complete EDA with visualizations that answer specific business questions.
- ✅ Apply feature engineering basic (encoding, scaling, new derived features).
- ✅ Create a reproducible step-by-step documented preprocessing pipeline.
- ✅ Explain why the quality of Data directly impacts the performance of the Model.

If you answered "yes" to all, you're ready to model some solid foundations.
