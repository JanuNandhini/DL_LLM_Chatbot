
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Student Roll No: 24BAD041")

model_name = "google/flan-t5-large"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


def chatbot(message, history):

    prompt = (
        "You are a helpful computer science tutor. "
        "Answer clearly and accurately in simple English.\n"
    )

    for item in history[-4:]:
        if isinstance(item, dict):
            prompt += f"{item['role']}: {item['content']}\n"

    prompt += f"user: {message}\nassistant:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            do_sample=False
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


demo = gr.ChatInterface(
    fn=chatbot,
    title="LLM Based AI Chatbot",
    description="Student Roll No: 24BAD041",
    type="messages"
)

demo.launch()
