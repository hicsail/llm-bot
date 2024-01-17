import os
from typing import Callable
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException, Query, Body, Form

from urllib.parse import unquote_plus

# Global Instance stuff
from global_state import global_instance

### LLM stuff
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.output_parsers import StrOutputParser

# For LLama 2
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp # The magic file, enables CPU based predictions with small mem and cpu usage footprint

router = APIRouter()

def bootstrap_llm(model_path: str):
	"""
	Bootstraps LLM of LLama 2 by Meta. Returns Void.

	Chain is so far:
	Prompt Template -> LLM -> String Output Parser
	"""
	# Prompt
	# We want to swap out prompts if needed
	prompt = PromptTemplate(
	    input_variables=["headline", "body"],
	    template="""<<SYS>> \n You are an assistant tasked with classifying whether the given \
	headline and body is related to China or not. \n <</SYS>> \n\n [INST] Generate a SHORT response \
	if the given headline and article body is related to China. The output should be either Yes or No. \n\n
	Headline: \n\n {headline} \n\n Body: {body} \n\n [/INST]""",
	)

	# We need to swap models, *** GPT4All seems better than Llama.cpp in optimization ***
	llm = LlamaCpp(
	    model_path=model_path,
	    n_gpu_layers=1,
	    n_batch=1024,
	    n_ctx=2048,
	    f16_kv=True,
	    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
	    verbose=True,
	)

	# Our Output parser
	output_parser = StrOutputParser()

	# Our Chain
	chain = prompt | llm | output_parser

	global_instance.update_data("chain", chain);

	return

@router.post("/identify_article")
async def identify_article(
	headline: str = Form(...),
    body: str = Form(...)
    ):
	try:
		#model_path = "./app/models/llama-2-7b-chat_Q4_K_M.gguf" # Production
		model_path = "./models/llama-2-7b-chat_Q4_K_M.gguf" # Local
		chain = global_instance.get_data("chain")

		if (chain == None):
			print("LLM was not detected. Creating one now!")
			os.chdir("/Users/zacharyg/Documents/GitHub/llm-bot/containers/src") # For Local testing
			#print(os.getcwd()); # For Local testing
			bootstrap_llm(model_path)
			chain = global_instance.get_data("chain")

		decoded_headline = str(unquote_plus(headline))
		print("Headline:", decoded_headline)

		decoded_body = str(unquote_plus(body))
		print("Body:", decoded_body)

		llm_response = chain.invoke({
		    "headline": decoded_headline,
		    "body": decoded_body
		})

		print(f"LLM RESPONSE: {llm_response.trim()}")

		return JSONResponse(content={"Result": llm_response.strip(), "status": "String Successfully Uploaded"}, status_code=200)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")




