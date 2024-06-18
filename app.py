from flask import Flask
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import pandas as pd


app = Flask(__name__)

@app.route("/")
def index():
    return "Velkommen! legg til følgende for å spørre meg /analyser/(din query)"


@app.route("/analyser/<tekst>")
def analyserTekst(tekst):
    print("INIT MODEL")

    base_model_path = 'RuterNorway/Llama-2-7b-chat-norwegian'
    lora_path = './fine_tuned_lora'

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to("cpu")
    peft_config = PeftConfig.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(base_model, lora_path).to("cpu")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    # Set the pad_token to be the same as the eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    print("DONE INIT")
    print("SKAL ANALYSERE TEKST " + str(tekst))
    inputs = tokenizer(tekst, return_tensors="pt")
    # Generate text
    with torch.no_grad():
        print("JOBBER")
        outputs = model.generate(inputs.input_ids,max_length=300)
        print("DECODER TEKST")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("FERDIG...")
    return generated_text

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8080")
