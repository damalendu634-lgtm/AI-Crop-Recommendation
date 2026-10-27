
import os
import joblib
import pandas as pd
import gradio as gr
from google import genai

# Load the existing crop recommendation model
print("Loading crop recommendation model...", flush=True)
saved_data = joblib.load("crop_recommendation_model.pkl")
model = saved_data["model"]
feature_columns = saved_data["feature_columns"]
print("Crop recommendation model loaded successfully.", flush=True)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    try:
        values = [N, P, K, temperature, humidity, ph, rainfall]

        if any(v is None for v in values):
            return "Please enter all seven values.", ""

        if not all(
            pd.notna(float(v)) and float(v) >= 0
            for v in values
        ):
            return "Please enter valid non-negative values.", ""

        input_data = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        crop = model.predict(input_data)[0]
        result = f"## 🌱 Recommended Crop: {str(crop).title()}"

        details = ""

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = getattr(model, "classes_", [])

            top_three = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            details = "### 📊 Top Predictions\n\n"
            for name, probability in top_three:
                details += (
                    f"- **{str(name).title()}**: "
                    f"{probability * 100:.1f}%\n"
                )
        else:
            details = (
                "The model does not provide prediction probabilities."
            )

        return result, details

    except Exception as error:
        return f"Prediction error: {error}", ""


def ask_gemini(message, history):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "in your Render Environment settings."
        )

    try:
        client = genai.Client(api_key=api_key)

        conversation = []
        for item in (history or [])[-10:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")

                if role in ("user", "assistant") and isinstance(content, str):
                    conversation.append(
                        ("User" if role == "user" else "Assistant")
                        + ": " + content
                    )

        prompt = f"""
You are AgriSmart AI, a friendly AI assistant.

Answer questions clearly and simply about agriculture,
soil, crops, education, Python, technology, and general topics.

Give careful farming guidance. Do not guarantee crop yields.
For important pesticide or disease decisions, suggest consulting
a local agricultural expert.

Previous conversation:
{chr(10).join(conversation)}

User's question:
{message}
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return response.text or "Sorry, no answer was generated."

    except Exception as error:
        print("Gemini error:", repr(error), flush=True)
        return f"Gemini API error: {error}"


CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #064e3b, #15803d, #84cc16);
    color: white;
    margin-bottom: 18px;
}
.hero h1 {
    color: white !important;
    font-size: 38px !important;
}
.hero p {
    color: white !important;
    font-size: 17px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="lime"
    ),
    css=CSS,
    title="AgriSmart AI"
) as demo:

    gr.HTML("""
    <div class="hero">
        <h1>🌱 AgriSmart AI</h1>
        <p>Smart crop recommendations and your personal Gemini AI assistant.</p>
        <p>🌾 Crop Insights &nbsp; • &nbsp; 💬 AI Chat &nbsp; • &nbsp; 📊 Predictions</p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🌾 Crop Recommendation"):
            gr.Markdown("## Enter Soil and Weather Values")

            with gr.Row():
                N = gr.Number(label="Nitrogen (N)", value=83)
                P = gr.Number(label="Phosphorus (P)", value=46)
                K = gr.Number(label="Potassium (K)", value=44)

            with gr.Row():
                temperature = gr.Number(
                    label="Temperature (°C)", value=25
                )
                humidity = gr.Number(
                    label="Humidity (%)", value=85
                )

            with gr.Row():
                ph = gr.Number(label="Soil pH", value=7)
                rainfall = gr.Number(
                    label="Rainfall (mm)", value=150
                )

            predict_button = gr.Button(
                "🌱 Recommend Crop",
                variant="primary"
            )
            crop_output = gr.Markdown()
            probability_output = gr.Markdown()

            predict_button.click(
                fn=predict_crop,
                inputs=[
                    N, P, K, temperature,
                    humidity, ph, rainfall
                ],
                outputs=[crop_output, probability_output]
            )

        with gr.Tab("💬 Ask Gemini AI"):
            gr.Markdown(
                "## Your Personal AI Assistant\n"
                "Ask about farming, education, Python, or any general topic."
            )

            gr.ChatInterface(
                fn=ask_gemini,
                title="AgriSmart Assistant",
                description="Powered by the Gemini API",
                examples=[
                    "How can I improve soil fertility?",
                    "Explain crop rotation simply.",
                    "What is artificial intelligence?",
                    "How does crop recommendation work?"
                ],
                chatbot=gr.Chatbot(height=430),
                textbox=gr.Textbox(
                    placeholder="Type your question and press Enter...",
                    lines=1,
                    max_lines=1
                )
            )

    gr.Markdown(
        "---\n"
        "Built with Python, Gradio, and Gemini AI. "
        "Verify important farming decisions with local experts."
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "10000"))
)
