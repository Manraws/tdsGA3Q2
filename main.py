import os
import json
import hashlib
import uuid
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel 
import google.generativeai as genai
import os

app = FastAPI()

# ---------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------
A2A_BASE_URL = os.getenv("A2A_BASE_URL", "https://delegate.example/a2a/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# 1. Configure your API key
# Ensure you have set the GEMINI_API_KEY environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Instantiate the Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

# 3. Prompt the model
prompt = "Write a haiku about writing Python code."
response = model.generate_content(prompt)

# 4. Output the result
print(response.text)

# In-memory storage (In production, use Redis or a DB)
# tasks_db[token][task_id] = task_data
tasks_db: Dict[str, Dict[str, Any]] = {}
# idempotency_db[token][message_hash] = task_id
idempotency_db: Dict[str, Dict[str, str]] = {}
# package_cache[package_hash] = proposal_dict
package_cache: Dict[str, dict] = {}

# ---------------------------------------------------------
# Dependencies & Security
# ---------------------------------------------------------
def get_token(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return authorization.split(" ")[1]

def require_a2a_headers(a2a_version: str = Header(None), content_type: str = Header(None)):
    if a2a_version != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported A2A-Version")
    if content_type != "application/a2a+json":
        raise HTTPException(status_code=400, detail="Unsupported Content-Type")

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def hash_message(message: dict) -> str:
    """Hash recursively key-sorted compact JSON of the message only."""
    encoded = json.dumps(message, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def hash_package(package: dict) -> str:
    """Hash a package for canonical caching of AI decisions."""
    encoded = json.dumps(package, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def mock_or_ai_decide(package: dict) -> dict:
    """
    Your AI Logic Goes Here.
    For now, it returns a safe fallback to ensure the protocol works.
    """
    pkg_hash = hash_package(package)
    if pkg_hash in package_cache:
        return package_cache[pkg_hash]
        
    # --- AI INTEGRATION POINT ---
    # In a real scenario, you would prompt Gemini here.
    # prompt = f"Analyze this invoice package: {json.dumps(package)}..."
    # response = gemini.chat.completions.create(...)
    
    # Dummy fallback response mimicking AI output
    decision = {
        "actionId": "ACT-" + str(uuid.uuid4()).replace("-", "")[:12],
        "action": "open_exception", # Safe fallback
        "facts": {
            "vendorName": "Unknown Vendor",
            "invoiceNumber": "INV-000",
            "amountMinor": 0,
            "currency": "USD"
        },
        "evidenceRefs": ["[P1-S1]", "[P2-S2]", "[P3-S3]"],
        "rationale": "Fallback exception triggered due to missing AI integration. Three refs cited."
    }
    
    package_cache[pkg_hash] = decision
    return decision

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/.well-known/agent-card.json")
def get_agent_card():
    """Origin-level agent card discovery."""
    return JSONResponse(content={
        "name": "Invoice Action Agent",
        "description": "Evaluates invoice packages and safely executes payment actions.",
        "version": "1.0.0",
        "capabilities": {
            "invoice_action_agent": {
                "name": "Invoice Action Agent",
                "description": "Decides invoice actions based on AI reasoning",
                "tags": ["finance", "invoices", "approval"]
            }
        },
        "supportedInterfaces": [{
            "url": A2A_BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }, headers={"Content-Type": "application/json"})

@app.post("/a2a/message:send")
async def send_message(
    request: Request,
    token: str = Depends(get_token),
    _=Depends(require_a2a_headers)
):
    body = await request.json()
    message = body.get("message", {})
    msg_hash = hash_message(message)
    
    # Initialize user storage
    if token not in tasks_db:
        tasks_db[token] = {}
        idempotency_db[token] = {}
        
    # Idempotency check
    if msg_hash in idempotency_db[token]:
        task_id = idempotency_db[token][msg_hash]
        return JSONResponse(content={"task": tasks_db[token][task_id]})
    
    # Check if a messageId is reused with different semantic content
    msg_id = message.get("messageId")
    for existing_msg_hash, existing_task_id in idempotency_db[token].items():
        if tasks_db[token][existing_task_id]["history"][-1]["messageId"] == msg_id:
            raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")

    parts = message.get("parts", [])
    if not parts:
        raise HTTPException(status_code=400, detail="Missing message parts")

    part = parts[0]
    media_type = part.get("mediaType")
    
    # ---------------------------------------------------------
    # Flow 1: Initial Invoice Batch (Proposals)
    # ---------------------------------------------------------
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        data = part.get("data", {})
        batch_id = data.get("batchId")
        
        proposals = []
        for pkg in data.get("packages", []):
            decision = mock_or_ai_decide(pkg)
            decision["packageId"] = pkg.get("packageId")
            proposals.append(decision)
            
        task_id = "TSK-" + str(uuid.uuid4())
        context_id = "CTX-" + str(uuid.uuid4())
        
        task = {
            "taskId": task_id,
            "contextId": context_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "parts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {
                    "batchId": batch_id,
                    "proposals": proposals
                }
            }]
        }
        
        # Save state
        tasks_db[token][task_id] = task
        idempotency_db[token][msg_hash] = task_id
        return JSONResponse(content={"task": task})

    # ---------------------------------------------------------
    # Flow 2: Grader Results (Receipts)
    # ---------------------------------------------------------
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = message.get("taskId")
        
        if task_id not in tasks_db[token]:
            raise HTTPException(status_code=404, detail="Task not found")
            
        task = tasks_db[token][task_id]
        
        if task["state"] != "TASK_STATE_INPUT_REQUIRED":
            raise HTTPException(status_code=409, detail="Task is not in INPUT_REQUIRED state")
            
        if task["contextId"] != message.get("contextId"):
            raise HTTPException(status_code=400, detail="Context ID mismatch")
            
        data = part.get("data", {})
        batch_id = data.get("batchId")
        
        # Extract original proposals to match against results
        original_proposals = task["parts"][0]["data"]["proposals"]
        prop_map = {p["packageId"]: p for p in original_proposals}
        
        executions = []
        for result in data.get("results", []):
            if result.get("outcome") == "ACCEPTED":
                pkg_id = result.get("packageId")
                prop = prop_map.get(pkg_id)
                
                if not prop or prop["actionId"] != result["actionId"] or prop["action"] != result["action"]:
                    raise HTTPException(status_code=400, detail="Result does not match proposal")
                    
                executions.append({
                    "packageId": pkg_id,
                    "actionId": prop["actionId"],
                    "action": prop["action"],
                    "receiptNonce": result.get("receiptNonce"),
                    "facts": prop["facts"],
                    "evidenceRefs": prop["evidenceRefs"]
                })
                
        # Update Task to COMPLETED
        task["history"].append(message)
        task["state"] = "TASK_STATE_COMPLETED"
        task["parts"].append({
            "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
            "data": {
                "batchId": batch_id,
                "executions": executions
            }
        })
        
        tasks_db[token][task_id] = task
        idempotency_db[token][msg_hash] = task_id
        return JSONResponse(content={"task": task})

    else:
        raise HTTPException(status_code=400, detail="Unsupported mediaType")

@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, token: str = Depends(get_token), _=Depends(require_a2a_headers)):
    if token not in tasks_db or task_id not in tasks_db[token]:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content={"task": tasks_db[token][task_id]})

@app.get("/a2a/tasks")
async def list_tasks(token: str = Depends(get_token), _=Depends(require_a2a_headers)):
    if token not in tasks_db:
        return JSONResponse(content={"tasks": []})
    tasks = list(tasks_db[token].values())
    return JSONResponse(content={"tasks": tasks})

@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, token: str = Depends(get_token), _=Depends(require_a2a_headers)):
    if token not in tasks_db or task_id not in tasks_db[token]:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = tasks_db[token][task_id]
    
    # Cancel/Receipt race condition handling
    if task["state"] in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
        raise HTTPException(status_code=409, detail="Task is terminal and cannot be canceled")
        
    task["state"] = "TASK_STATE_CANCELED"
    return JSONResponse(content={"task": task})