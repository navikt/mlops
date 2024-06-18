from flask import Flask
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/app/fine_tuned_lora'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Set the pad_token to be the same as the eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
torch.cuda.empty_cache()
model.to("cpu")
app = Flask(__name__)


@app.route("/")
def hello_world(tekst):
    return "Velkommen! legg til følgende for å spørre meg /(din query)"

@app.route("/<tekst>")
def hello_world(tekst):

    inputs = tokenizer(tekst, return_tensors="pt").to("cpu")
    # Generate text
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids,max_length=300)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
