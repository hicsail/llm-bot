import os
from typing import Callable
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException, Query, Body, Form

from urllib.parse import unquote_plus

# Global Instance stuff
from global_state import global_instance

router=APIRouter()

@router.post("/dummy_endpoint")
async def dummy_endpoint(payload: str = Form(...)):
    try: 
        decoded_payload = str(unquote_plus(payload)) 
        return JSONResponse(content={"Result": decoded_payload, "status": "String Successfully Uploaded"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    return
