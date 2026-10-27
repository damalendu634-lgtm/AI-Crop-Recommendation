
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load the saved AI model
saved_data = joblib.load("crop_recommendation_model.pkl")
model = saved_data["model"]
feature_columns = saved_data["feature_columns"]


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    values = [N, P, K, temperature, humidity, ph, rainfall]

    if not all(np.isfinite(float(v)) for v in values):
        return "Enter valid numbers", ""

    input_data = pd.DataFrame(
        [values], columns=feature_columns
    )

    crop = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]

    details = "\n".join(
        f"{classes[i]}: {probabilities[i] * 100:.2f}%"
        for i in top_indices
    )

    return str(crop), details


with gr.Blocks(title="AI Crop Recommendation System") as demo:
    gr.Markdown("# AI-Based Crop Recommendation System")
    gr.Markdown("Enter soil and weather information below.")

    with gr.Row():
        N = gr.Number(label="Nitrogen (N)", value=90)
        P = gr.Number(label="Phosphorus (P)", value=42)
        K = gr.Number(label="Potassium (K)", value=43)

    with gr.Row():
        temperature = gr.Number(label="Temperature (°C)", value=25.5)
        humidity = gr.Number(label="Humidity (%)", value=80)

    with gr.Row():
        ph = gr.Number(label="Soil pH", value=6.5)
        rainfall = gr.Number(label="Rainfall (mm)", value=200)

    button = gr.Button("Recommend Crop")
    result = gr.Textbox(label="Recommended Crop")
    details = gr.Textbox(label="Top 3 Predictions")

    button.click(
        predict_crop,
        inputs=[N, P, K, temperature, humidity, ph, rainfall],
        outputs=[result, details]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )