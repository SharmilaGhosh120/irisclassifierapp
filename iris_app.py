
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# Set title
st.title("🌼 Iris Flower Classification")
st.markdown("A simple ML app to classify iris flowers using Random Forest and visualize performance.")

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names
feature_names = iris.feature_names

# Sidebar: user input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict user input
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = clf.predict(input_data)[0]
st.subheader("🌸 Prediction")
st.write(f"The predicted iris species is **{target_names[prediction]}**.")

# Show model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("📊 Model Accuracy")
st.write(f"{accuracy * 100:.2f}%")

# Confusion matrix
st.subheader("🔷 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
st.pyplot(fig)

# Pairplot
st.subheader("🔍 Feature Pairplot")
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Series([target_names[i] for i in y])
fig2 = sns.pairplot(df, hue='species')
st.pyplot(fig2)
