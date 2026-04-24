from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./byt5_g2p"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

model.eval()

def predict(word):
    inputs = tokenizer(word, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=32,
            num_beams=5
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(predict("Februar"))
print(predict("Haus"))
print(predict("Deutschland"))