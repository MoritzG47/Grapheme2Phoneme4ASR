"""
Alle paar Steps speichern nicht erst nach epochen
"""


from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
df = pd.read_csv("traindata/word_ipa_no_duplicate.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(batch):
    words = batch["word"]
    ipas = batch["ipa"]

    # Filter ungültige Einträge
    words = [w if w is not None else "" for w in words]
    ipas = [i if i is not None else "" for i in ipas]

    return tokenizer(
        text=words,
        text_target=ipas,
        padding="max_length",
        truncation=True,
        max_length=32,
    )

if __name__ == "__main__":
    tokenized = dataset.map(
        preprocess,
        batched=True,       
        batch_size=500,     
        num_proc=6,
        load_from_cache_file=True,
    )

    training_args = TrainingArguments(
            output_dir="./byt5_g2p",
            learning_rate=6e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,

            save_strategy="steps",
            save_steps=500,            

            eval_strategy="steps",     
            eval_steps=500,

            logging_dir="./logs",
            logging_steps=100,
            load_best_model_at_end=True,
            fp16=False,
            dataloader_num_workers=8,
            push_to_hub=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    trainer.save_model("./byt5_g2p")
    tokenizer.save_pretrained("./byt5_g2p")

    def predict(word):
        inputs = tokenizer(word, return_tensors="pt")

        outputs = model.generate(**inputs, max_length=32)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(predict("Februar"))