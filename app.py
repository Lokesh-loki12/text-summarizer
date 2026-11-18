from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Initialize FastAPI
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# Load trained model and tokenizer
MODEL_PATH = "./saved_summary_model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Select available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Using device: {device}")

# HTML templates
templates = Jinja2Templates(directory=".")


# Input model
class DialogueInput(BaseModel):
    dialogue: str


# Clean input text
def clean_data(text: str) -> str:
    text = re.sub(r"\r?\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    return text.strip().lower()


# Generate summary
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter some text to summarize.")

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move input tensors to the model's device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate summary
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Convert token IDs to text
    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# API endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    try:
        summary = summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary}

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Summarization error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary. Check the server terminal."
        )


# Homepage
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )
