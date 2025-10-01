import os
import io
import pickle
import json
import re
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv


from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import google.generativeai as genai


os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

load_dotenv()

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"CRITICAL ERROR: Failed to configure Gemini. Check your GOOGLE_API_KEY. Error: {e}")


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
REDIRECT_URI = os.getenv('REDIRECT_URI', 'http://127.0.0.1:8888/oauth2callback')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    query: str

def get_google_flow():
    if 'GOOGLE_CREDENTIALS' in os.environ:
        creds_json = json.loads(os.environ['GOOGLE_CREDENTIALS'])
        return Flow.from_client_config(creds_json, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    else:
        return Flow.from_client_secrets_file(CREDENTIALS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI)


def get_credentials():

    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def get_drive_service():
    creds = get_credentials()
    if not creds or not creds.valid:
        return None
    return build('drive', 'v3', credentials=creds)

# --- CORE AGENT LOGIC ---

def get_intent_from_query(user_query: str):
    print("Step 1: Analyzing user query...")
    model = genai.GenerativeModel('models/gemini-flash-latest')

    prompt = f"""
    Analyze the user's query: "{user_query}"
    Your goal is to extract the single most important noun or topic that could be part of a filename.
    Determine the primary intent and the main search keyword.
    The intent must be one of: 'search_content' or 'list_files'.
    The keyword should be the primary, simplified topic (e.g., for 'neuroscience text file', the keyword is 'neuroscience').
    Respond with ONLY a valid JSON object with two keys: "intent" and "keyword".
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        print(f"Intent Analysis Complete: {result}")
        return result
    except Exception as e:
        print(f"ERROR in get_intent_from_query: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing query intent: {e}")

def execute_drive_action(service, intent: str, keyword: str):
    print(f"Step 2: Executing Drive action: '{intent}' with keyword '{keyword}'")
    try:
        if not keyword.strip():
             return "I couldn't identify a keyword in your request. Please try again."

        # If the intent is to list files, search Drive and return a list of names
        if intent == "list_files":
            drive_query = f"name contains '{keyword}' and mimeType != 'application/vnd.google-apps.folder'"
            results = service.files().list(q=drive_query, pageSize=10, fields="files(id, name)").execute()
            items = results.get('files', [])
            if not items:
                return "I couldn't find any files matching that keyword."
            file_names = [item['name'] for item in items]
            return "Here is a list of files I found:\n" + "\n".join(f"- {name}" for name in file_names)

        # If the intent is to get content, find the top file and return its content
        elif intent == "search_content":
            drive_query = f"name contains '{keyword}' and mimeType != 'application/vnd.google-apps.folder'"
            results = service.files().list(q=drive_query, pageSize=1, fields="files(id, name, mimeType)").execute()
            items = results.get('files', [])
            if not items:
                return "I couldn't find a file matching that keyword."
            
            top_file = items[0]
            file_id = top_file['id']
            file_name = top_file['name']
            mime_type = top_file.get('mimeType', '')
            print(f"Found file '{file_name}' with type '{mime_type}'. Fetching its content.")

            # Download or export the file content based on its type
            request = None
            if 'google-apps.document' in mime_type:
                # For Google Docs, export as plain text
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            else:
                # For other files (PDF, TXT, etc.), download directly
                request = service.files().get_media(fileId=file_id)
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            return fh.getvalue().decode('utf-8', errors='ignore')
        else:
            return "I'm sorry, I didn't understand that request."
            
    except HttpError as e:
        print(f"ERROR in execute_drive_action (Google API): {e}")
        raise HTTPException(status_code=500, detail="A Google Drive API error occurred.")
    except Exception as e:
        print(f"ERROR in execute_drive_action: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while accessing Drive.")

def clean_text(text: str) -> str:
    text = re.sub(r'[\*#]', '', text)
    return text.strip()

def get_final_answer(user_query: str, context: str):
    print("Step 3: Synthesizing final answer...")
    model = genai.GenerativeModel('models/gemini-flash-latest')
    system_prompt = "You are an intelligent assistant. Based *only* on the provided context, answer the user's question clearly and concisely. Do not use any markdown formatting."
    prompt = f"CONTEXT:\n---\n{context}\n---\n\nBased on the context above, please answer this question: {user_query}"
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = clean_text(response.text)
        return cleaned_response
    except Exception as e:
        print(f"ERROR in get_final_answer: {e}")
        raise HTTPException(status_code=500, detail="Error getting final answer from AI.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    creds = get_credentials()
    return templates.TemplateResponse("index.html", {"request": request, "logged_in": creds and creds.valid})

@app.post("/ask")
async def process_query(query: Query):
    """Orchestrates the agent's logic when a user asks a question."""
    service = get_drive_service()
    if not service:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in again.")
    
    user_query = query.query
    
    try:
        analysis = get_intent_from_query(user_query)
        intent = analysis.get("intent")
        keyword = analysis.get("keyword", "")

        if not intent:
            return {"answer": "I had trouble understanding your request. Please try rephrasing."}

        context = execute_drive_action(service, intent, keyword)
        final_answer = get_final_answer(user_query, context)
        return {"answer": final_answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"UNEXPECTED ERROR in process_query: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

# --- Authentication Endpoints ---

@app.get("/authorize")
async def authorize():
    flow = get_google_flow()
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    return RedirectResponse(authorization_url)

@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    flow = get_google_flow()
    flow.fetch_token(authorization_response=str(request.url))
    with open(TOKEN_FILE, 'wb') as token:
        pickle.dump(flow.credentials, token)
    return RedirectResponse(url='/')

