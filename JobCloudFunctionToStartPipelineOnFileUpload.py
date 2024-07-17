#functions-framework==3.*
#google-api-python-client>=1.7.8,<2
#google-cloud-aiplatform
#PyYAML

import functions_framework
import base64
import json
from google.cloud import aiplatform

PROJECT_ID = '368669329223'                  
REGION = 'europe-west4'                            
PIPELINE_ROOT = 'gs://test-repo/mlops7' 

#https://cloud.google.com/vertex-ai/docs/pipelines/trigger-pubsub
# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def startTreningAvTryggtekst2(cloud_event):
  print("##### STARTER")
  data = cloud_event.data

  event_id = cloud_event["id"]
  event_type = cloud_event["type"]

  bucket = data["bucket"]
  name = data["name"]
  if(bucket and name):
      print(f"Bucket: {bucket}")
      print(f"File: {name}")
      
      search_text = "data".lower()

      if search_text in name:
          print("Text found!")
          #metageneration = data["metageneration"]
          #timeCreated = data["timeCreated"]
          #updated = data["updated"]

          #print(f"Event ID: {event_id}")
          #print(f"Event type: {event_type}")

          #print(f"Metageneration: {metageneration}")
          #print(f"Created: {timeCreated}")
          #print(f"Updated: {updated}")
          # Create a PipelineJob using the compiled pipeline from pipeline_spec_uri
          aiplatform.init(
              project=PROJECT_ID,
              location=REGION,
          )
          job = aiplatform.PipelineJob(
              display_name='startTreningAvTryggtekst',
              template_path="gs://tiltak-mlops/pipeline-template/mlops_pipeline.json",
              #pipeline_root=PIPELINE_ROOT,
              enable_caching=False
              #parameter_values=parameter_values
          )

          # Submit the PipelineJob
          job.submit()
  else:
      print("Nothing todo here")