from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

import torch
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return "Velkommen! legg til følgende for å spørre meg /analyser/(din query)"

@app.route("/health")
def health():
    return {"health":"OK"}, 200

# {"instances":[{"text":"hva heter Norges hovedstad?"}]}
@app.route("/predict", methods=["POST"])
def predict():
  # Get the JSON data from the request body
  data = request.get_json()

  # Check if data is present
  if not data:
      return jsonify({'error': 'Missing data in request'}), 400

  # Access the "instances" list
  instances = data.get("instances")

  # Check if "instances" list exists
  if not instances:
      return jsonify({'error': 'Missing "instances" list in request data'}), 400

  # Get only the first instance
  first_instance = instances[0]
  input_text = first_instance.get("text")
  print("INIT MODEL")
  base_model_path = 'RuterNorway/Llama-2-7b-chat-norwegian'
  lora_path = './fine_tuned_lora/'
  base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to("cpu")
  peft_config = PeftConfig.from_pretrained(lora_path)
  model = PeftModel.from_pretrained(base_model, lora_path).to("cpu")
  model = model.merge_and_unload()
  tokenizer = AutoTokenizer.from_pretrained(lora_path)

  # Set the pad_token to be the same as the eos_token
  tokenizer.pad_token = tokenizer.eos_token
  model.config.use_cache = False
  print("DONE DOWNLOADING MODEL!!!")
    
  print("SKAL ANALYSERE TEKST " + str(input_text))    
  inputs = tokenizer(input_text, return_tensors="pt")
  # Generate text
  with torch.no_grad():
    print("JOBBER")
    outputs = model.generate(inputs.input_ids,max_length=300)
    print("DECODER TEKST")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("FERDIG...")
    return jsonify({"generated_text": generated_text})  
  return jsonify({"generated_text": generated_text})
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8080")#to timer timeout
