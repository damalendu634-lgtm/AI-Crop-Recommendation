
import os
import pickle
from pathlib import Path

import gradio as gr
from openai import OpenAI

# --------------------------------------------------
# 1. LOAD YOUR EXISTING CROP RECOMMENDATION MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_recommendation_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# 2. CROP RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    try:
        values = [N, P, K, temperature, humidity, ph, rainfall]

        if any(value is None for value in values):
            return "Please fill in all seven fields.", ""

        if not 0 <= ph <= 14:
            return "Soil pH must be between 0 and 14.", ""

        # Keep the feature order used when training your model.
        features = [[
            float(N),
            float(P),
            float(K),
            float(temperature),
            float(humidity),
            float(ph),
            float(rainfall)
        ]]

        prediction = model.predict(features)[0]
        result = f"🌱 Recommended crop: **{str(prediction).title()}**"

        top_three = "Top predictions are not available for this model."

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            ranked = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            top_three = "\n".join(
                f"{i}. **{str(crop).title()}** — {prob * 100:.2f}%"
                for i, (crop, prob) in enumerate(ranked, start=1)
            )

        return result, top_three

    except Exception as error:
        print("Crop prediction error:", error)
        return (
            "Prediction failed. Please check your inputs and model.",
            "Check the Render logs for the technical error."
        )


# --------------------------------------------------
# 3. CHATGPT-LIKE AI CHATBOT
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are FarmWise AI, a friendly and helpful conversational assistant.

Help users with:
- Crop selection and growing methods
- Soil health, NPK nutrients and soil pH
- Irrigation, rainfall and weather-related farming
- Fertilizers, pests and plant diseases
- Sustainable farming and agricultural education
- General questions in clear, simple language

Answer politely and explain technical terms simply.
When giving farming advice, consider the user's crop and location
if they provide them. Do not invent facts, weather readings, or
guaranteed crop yields. For serious plant diseases or pesticide
decisions, recommend advice from a qualified local agriculture
expert. Make it clear when information is uncertain.
"""

def chat_with_ai(message, history):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "The AI chatbot has not been connected yet. "
            "The website owner needs to add OPENAI_API_KEY "
            "in the Render environment settings."
        )

    try:
        conversation = []

        # Include recent messages so the assistant can follow the context.
        for item in (history or [])[-10:]:
            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and isinstance(content, str):
                conversation.append({
                    "role": role,
                    "content": content
                })

        conversation.append({
            "role": "user",
            "content": message
        })

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            instructions=SYSTEM_INSTRUCTIONS,
            input=conversation,
            max_output_tokens=600
        )

        return response.output_text or (
            "Sorry, I couldn't generate a response. Please try again."
        )

    except Exception as error:
        print("AI chatbot error:", error)
        return (
            "Sorry, the AI assistant could not respond. "
            "Please check the API key, API account billing, "
            "model access, and Render logs."
        )


# --------------------------------------------------
# 4. MODERN GRAPHICS AND WEBSITE DESIGN
# --------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 65%);
}

.hero {
    padding: 30px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(120deg, #14532d, #16a34a, #65a30d);
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(22, 101, 52, 0.18);
}

.hero h1 {
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 16px;
    line-height: 1.6;
}

.info-card {
    padding: 18px;
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    background: #f0fdf4;
    margin: 8px 0;
}

footer {
    visibility: hidden;
}
"""

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="lime",
    neutral_hue="slate"
)

with gr.Blocks(
    theme=theme,
    css=custom_css,
    title="FarmWise AI | Crop Recommendation"
) as demo:

    gr.HTML("""
        <div class="hero">
            <h1>🌾 FarmWise AI</h1>
            <p>
                Smart crop recommendations and an AI farming assistant
                in one place. Get crop predictions and ask questions
                about agriculture in simple language.
            </p>
        </div>
    """)

    with gr.Tab("🌱 Crop Recommendation"):
        gr.Markdown(
            "### Find a suitable crop\n"
            "Enter your soil nutrient and weather information."
        )

        with gr.Row():
            nitrogen = gr.Number(
                label="Nitrogen (N)", value=83
            )
            phosphorus = gr.Number(
                label="Phosphorus (P)", value=46
            )
            potassium = gr.Number(
                label="Potassium (K)", value=44
            )

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

        recommend_button = gr.Button(
            "🌿 Recommend My Crop",
            variant="primary",
            size="lg"
        )

        crop_output = gr.Markdown(
            label="Recommended Crop"
        )

        top_output = gr.Markdown(
            label="Top 3 Predictions"
        )

        recommend_button.click(
            fn=recommend_crop,
            inputs=[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ],
            outputs=[crop_output, top_output]
        )

        gr.Markdown(
            "ℹ️ Predictions depend on your trained model and "
            "the quality of the input data."
        )

    with gr.Tab("🤖 Ask FarmWise AI"):
        gr.Markdown(
            "### Your farming and general-purpose AI assistant\n"
            "Ask a question below and continue the conversation."
        )

        gr.ChatInterface(
            fn=chat_with_ai,
            title="FarmWise AI Chat",
            description=(
                "Ask about crops, fertilizers, irrigation, "
                "soil health, plant diseases, or other topics."
            ),
            examples=[
                "Which fertilizer helps rice grow?",
                "How can I improve soil health?",
                "Explain nitrogen, phosphorus and potassium.",
                "How can farmers save water?"
            ],
            chatbot=gr.Chatbot(height=430),
            textbox=gr.Textbox(
                placeholder="Ask FarmWise AI anything...",
                lines=2
            ),
            save_history=True
        )

    with gr.Row():
        gr.Markdown("""
        <div class="info-card">
            🌱 <b>Crop Intelligence</b><br>
            Uses your saved machine-learning model.
        </div>
        """)
        gr.Markdown("""
        <div class="info-card">
            🤖 <b>AI Assistant</b><br>
            Answers questions using an AI API.
        </div>
        """)
        gr.Markdown("""
        <div class="info-card">
            🌍 <b>Built for Farmers</b><br>
            Designed for clear and practical guidance.
        </div>
        """)

if __name__ == "__main__":
    demo.launch()
