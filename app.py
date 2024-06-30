from flask import Flask, request, jsonify

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from google.cloud import storage
import os
import torch
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return "Velkommen! legg til følgende for å spørre meg /analyser/(din query)"

@app.route("/health")
def health():
    return {"health":"OK"}, 200



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if not os.path.exists(os.path.dirname(destination_file_name)):
        os.makedirs(os.path.dirname(destination_file_name))

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {source_blob_name} downloaded to {destination_file_name}."
    )

def download_folder(bucket_name, folder_path, local_destination):
    """Downloads all the blobs in the folder from the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_path)

    for blob in blobs:
        # Remove the folder path from the blob name to get the relative path
        relative_path = os.path.relpath(blob.name, folder_path)
        local_file_path = os.path.join(local_destination, relative_path)

        download_blob(bucket_name, blob.name, local_file_path)


print("SKAL LASTE NED --- LORA --- fra bucket")
bucket_name = "tiltak-mlops"
folder_path = "fine_tuned_lora"
local_destination = "fine_tuned_lora"

download_folder(bucket_name, folder_path, local_destination)
print("skal starte å laste ---- RUTER - HOVED ---- modellen.")
base_model_path = 'RuterNorway/Llama-2-7b-chat-norwegian'
lora_path = './fine_tuned_lora/'
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
m = PeftModel.from_pretrained(m, adapters_name)
m = m.merge_and_unload()
tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [0]
print("MODEL er FERDIG LASTet!!!")

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
  
    
  print("SKAL ANALYSERE TEKST " + str(input_text))    
  inputs = tok("### instruct: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Answers in the language you are written to. You are trained to answer relative to the principles of sensitive personal information which are: \nHelseopplysninger og annen informasjon som sier noe om en persons helsetilstand.\nRasemessig eller etnisk opprinnelse,\nPolitisk oppfatning, religion, filosofisk overbevisning eller fagforeningsmedlemskap,\nGenetiske opplysninger og biometriske opplysninger med det formål å entydig identifisere en fysisk person\nOpplysninger om en fysisk persons seksuelle forhold eller seksuelle orientering.\nSome placeholders will be used in the texts: <Name> is a placeholder for name. <Sted> is placeholder for origin. <Religion> for religions.\n\n\nHvilken type av sensitiv personlig informasjon innholder denne teksten?\n ### Input: " + input_text + " \n ### output: ", return_tensors="pt")
  # Generate text
  with torch.no_grad():
    print("JOBBER")
    outputs = m.generate(inputs.input_ids,max_length=300)
    print("DECODER TEKST")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("FERDIG...")
    return jsonify({"svar": generated_text}) 
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8080")
