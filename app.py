# gradio app for insurance dataset

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# =====================
# Load the trained model
# =====================
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# =====================
# Prediction function
# =====================
def predict_charge(age, sex, bmi, children, smoker, region):

    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=[
        "age", "sex", "bmi", "children", "smoker", "region"
    ])

    prediction = model.predict(input_df)[0]

    return f"Predicted Insurance Charge: ${prediction:,.2f}"

# =====================
# Gradio Interface
# =====================
inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=25.0),
    gr.Slider(0, 4, step=1, label="Number of Children"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(
        ["northeast", "northwest", "southeast", "southwest"],
        label="Region"
    )
]

app = gr.Interface(
    fn=predict_charge,
    inputs=inputs,
    outputs="text",
    title="Insurance Charge Predictor",
    description="Predict medical insurance cost using ML"
)

app.launch(share=True)
