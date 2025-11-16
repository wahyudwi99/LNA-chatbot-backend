import os
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from transformers import AutoModel, AutoTokenizer
from module import construct_response


# Define global environment vairables
BACKEND_API_SECRET_KEY=os.getenv("BACKEND_API_SECRET_KEY")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("LIST_CORS_HOST_ORIGINS")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

thread_executors = ThreadPoolExecutor(max_workers=int(os.getenv("THREAD_NUMBERS")))


# Define embedding models as global variables (to avoid loading time on every request)
embedding_model = AutoModel.from_pretrained("./models/embedding_model")
embedding_tokenizer = AutoTokenizer.from_pretrained("./models/embedding_tokenizer")


@app.post("/chatbot-response")
async def chatbot_response(data: dict = Body(...),
                           credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """
    Function as an API endpoint to response the user's chat from front-end
    """
    async_loop = asyncio.get_running_loop()

    # Check credentials
    token_bearer = credentials.credentials
    if str(token_bearer) != BACKEND_API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized request")

    try:
        result = await async_loop.run_in_executor(
            thread_executors,
            construct_response,
            data["input_prompt"],
            embedding_model,
            embedding_tokenizer
        )

        json_result = {
            "status": 200,
            "result": result
        }

        return json_result
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed processing llm")