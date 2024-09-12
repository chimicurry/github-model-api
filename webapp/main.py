import json
import os
from os.path import dirname, abspath, join
# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
# Azure AI imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Phi-3-mini-128k-instruct"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


app = FastAPI()


class Body(BaseModel):
    text: str


@app.get('/')
def root():
    """
    Allows to open the API documentation in the browser directly instead of
    requiring to open the /docs path.
    """
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
  response = client.complete(
      messages=[
          SystemMessage(content="You are a helpful assistant."),
          UserMessage(content=body.text),
      ],
      model=model_name,
      temperature=1.,
      max_tokens=1000,
      top_p=1.
  )
  
  return {"response": response.choices[0].message.content}


