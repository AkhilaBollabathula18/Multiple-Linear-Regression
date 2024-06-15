### Report on Python Programs for Data Analysis and Linear Regression Modeling:


*** Introduction:

These Python programs analyze how advertising budgets allocated to TV, Radio, and Newspaper channels influence sales. By exploring relationships and using 
linear regression modeling, the programs aim to provide insights into optimizing marketing strategies for improved sales performance.

 *** Data Loading and Initial Exploration:

The program begins by importing necessary libraries (`pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`) and loading a dataset named "Advertising.csv" 
using `pd.read_csv()`. This dataset presumably contains information related to advertising expenditures (TV, Radio, Newspaper) and their impact on sales.

 *** Exploratory Data Analysis:

Upon loading the dataset, several exploratory steps were undertaken:
- **`df.head()`**: Displays the first few rows of the dataset to understand its structure and contents.
- **`df.describe()`**: Provides summary statistics (count, mean, min, max, etc.) for numeric columns like TV, Radio, Newspaper, and Sales.
- **`df.info()`**: Gives an overview of the dataset including column names and data types.
- **Missing Values Check**: `df.isnull().sum()` ensures no missing values disrupt the analysis.
- **Shape of the Dataset**: `df.shape` confirms the dataset's dimensions.

*** Visualization of Data:

To visually explore the relationship between advertising channels (TV, Radio) and sales:
- A 3D scatter plot (`plt.figure()` with `projection='3d'`) was created using `mpl_toolkits.mplot3d.Axes3D`, depicting the relationship between TV, Radio
  advertising expenditures, and Sales (`ax.scatter(tv, radio, sales, color="k", marker="^", label="Actual values")`).

- Additionally, a 2D scatter plot using Seaborn (`sns.scatterplot(data=df, x="TV", y="Sales")`) was utilized to visualize the direct relationship between TV 
advertising expenditure and Sales.

 *** Data Preparation and Model Training:

For the linear regression model:
- **Feature and Target Variables**: Features (`x`) were selected as a combination of TV and Newspaper advertising expenditures (`x = df[["TV", "Newspaper"]].
   values`), while Sales (`y`) was chosen as the target variable (`y = df[["Sales"]].values`).

- **Train-Test Split**: The dataset was split into training and testing sets (`train_test_split` from `sklearn.model_selection`) with a test size of 20% and
    a random state of 2 to ensure reproducibility.

*** Linear Regression Modeling:

Using `LinearRegression` from `sklearn.linear_model`:
- **Model Training**: `reg.fit(x_train, y_train)` trains the linear regression model on the training data.
- **Model Evaluation**: `reg.score(x_test, y_test)` computes the R-squared score of the model on the test data, indicating its predictive accuracy.
- **Coefficients and Intercept**: `reg.coef_` provides coefficients for TV and Newspaper expenditures, and `reg.intercept_` gives the intercept of the linear model.

*** Visualization of Model Results:

To visualize the model's performance:
- A 3D scatter plot (`fig.add_subplot(projection='3d')`) was used to display both actual sales values (`y_train`) and predicted sales values (`y_pred`) based
  on TV and Radio advertising expenditures (`ax.scatter(x_train[:,0], x_train[:,1], y_train, color="k", marker="^", label="Actual values")`,
   `ax.scatter(x_test[:,0], x_test[:,1], y_pred, color="r", marker="*", label="Predicted values")`).

*** Conclusion:

This Python program effectively demonstrates the process of:
- Loading and exploring a dataset related to advertising expenditures and sales.
- Visualizing relationships between variables using both 3D and 2D scatter plots.
- Preparing data for a linear regression model and evaluating its performance.
- Providing insights into how different advertising channels impact sales, as well as the predictive capability of the linear regression model.
