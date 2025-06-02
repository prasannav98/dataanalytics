import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1. Distribution Plot for a Numerical Column
def plot_distribution(df, column):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    st.pyplot(plt.gcf())
    plt.clf()

# 2. Box Plot for a Numerical Column (optionally grouped)
def plot_box(df, column, by=None):
    plt.figure(figsize=(8, 4))
    if by:
        sns.boxplot(x=by, y=column, data=df)
        plt.title(f'{column} by {by}')
    else:
        sns.boxplot(y=column, data=df)
        plt.title(f'Box Plot of {column}')
    st.pyplot(plt.gcf())
    plt.clf()

# 3. Correlation Heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(plt.gcf())
    plt.clf()

# 4. Pair Plot (for top N numeric columns)
def plot_pairplot(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='number').columns[:4]  # default to top 4
    sns.pairplot(df[columns])
    st.pyplot(plt.gcf())
    plt.clf()

# 5. Bar Plot for a Categorical Column
def plot_bar(df, column):
    plt.figure(figsize=(8, 4))
    df[column].value_counts().plot(kind='bar')
    plt.title(f'Frequency of {column}')
    plt.ylabel('Count')
    plt.xlabel(column)
    st.pyplot(plt.gcf())
    plt.clf()

# 6. Scatter Plot between Two Numerical Columns
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    st.pyplot(plt.gcf())
    plt.clf()

def show_plot_interface(df):
    st.subheader("Plot Generator")

    plot_type = st.selectbox("Choose a plot type", [
        "Distribution Plot",
        "Box Plot",
        "Bar Plot",
        "Scatter Plot",
        "Correlation Heatmap",
        "Pair Plot"
    ])

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if plot_type == "Distribution Plot":
        selected_col = st.selectbox("Select numerical column", numeric_cols)
        if selected_col and st.button("Generate Plot"):
            plot_distribution(df, selected_col)

    elif plot_type == "Box Plot":
        num_col = st.selectbox("Select numerical column", numeric_cols)
        group_col = st.selectbox("Group by (optional)", ["None"] + cat_cols)
        if num_col and st.button("Generate Plot"):
            plot_box(df, num_col, None if group_col == "None" else group_col)

    elif plot_type == "Bar Plot":
        selected_col = st.selectbox("Select categorical column", cat_cols)
        if selected_col and st.button("Generate Plot"):
            plot_bar(df, selected_col)

    elif plot_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_options = [col for col in numeric_cols if col != x_col]
        y_col = st.selectbox("Select Y-axis column", y_options)
        if x_col and y_col and st.button("Generate Plot"):
            plot_scatter(df, x_col, y_col)

    elif plot_type == "Correlation Heatmap":
        st.info("This will show correlation among all numeric features.")
        if st.button("Generate Plot"):
            plot_correlation_heatmap(df)

    elif plot_type == "Pair Plot":
        selected_cols = st.multiselect("Select 2–4 numeric columns", numeric_cols, default=numeric_cols[:2])
        if len(selected_cols) >= 2 and len(selected_cols) <= 4 and st.button("Generate Plot"):
            plot_pairplot(df, selected_cols)