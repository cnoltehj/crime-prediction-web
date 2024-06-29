import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

# Function to create and display SHAP plots
def display_shap_plots(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the current figure for further plots

    # Force plot for a specific instance
    instance_index = 0  # You can choose any instance index
    force_plot = shap.force_plot(explainer.expected_value, shap_values[instance_index, :], X_test.iloc[instance_index, :])
    st_shap(force_plot)

# Helper function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
