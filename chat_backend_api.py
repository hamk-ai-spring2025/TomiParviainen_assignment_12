# llm_chat_project/chat_backend_api.py

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv # For loading .env file

# --- STEP 1: CRITICAL - Load environment variables ---
# This MUST be called BEFORE trying to access os.getenv() for your API keys.
# Ensure your .env file is in the SAME DIRECTORY as this chat_backend_api.py file.
load_dotenv()
print("--- Backend Startup: Attempted to load .env file ---")

# --- STEP 2: Retrieve API Keys and Initialize Clients ---
# We'll try to get the keys now. If they are not found, the client objects will be None.

OPENAI_API_KEY_FROM_ENV = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY_FROM_ENV = os.getenv("ANTHROPIC_API_KEY")

print(f"--- Backend Startup: Value of OPENAI_API_KEY from env: '{OPENAI_API_KEY_FROM_ENV[:5] + '...' * (len(OPENAI_API_KEY_FROM_ENV) > 5 if OPENAI_API_KEY_FROM_ENV else 0)}' (shows first 5 chars if set)")
print(f"--- Backend Startup: Value of ANTHROPIC_API_KEY from env: '{ANTHROPIC_API_KEY_FROM_ENV[:5] + '...' * (len(ANTHROPIC_API_KEY_FROM_ENV) > 5 if ANTHROPIC_API_KEY_FROM_ENV else 0)}' (shows first 5 chars if set)")

# --- OpenAI Client Initialization ---
openai_client = None
if OPENAI_API_KEY_FROM_ENV:
    try:
        from openai import OpenAI # Import here to catch if not installed
        openai_client = OpenAI(api_key=OPENAI_API_KEY_FROM_ENV)
        print("--- Backend Startup: OpenAI client INITIALIZED successfully. ---")
    except ImportError:
        print("--- Backend Startup WARNING: 'openai' library not installed. OpenAI models will not work. ---")
        print("--- Please run: pip install openai ---")
    except Exception as e:
        print(f"--- Backend Startup WARNING: Error initializing OpenAI client: {e}. OpenAI models might not work. ---")
else:
    print("--- Backend Startup CRITICAL WARNING: OPENAI_API_KEY was NOT FOUND in environment. OpenAI models will NOT work. ---")
    print("--- Please ensure OPENAI_API_KEY is correctly set in your .env file. ---")

# --- Anthropic Client Initialization ---
anthropic_client = None
if ANTHROPIC_API_KEY_FROM_ENV:
    try:
        import anthropic # Import here
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY_FROM_ENV)
        print("--- Backend Startup: Anthropic client INITIALIZED successfully. ---")
    except ImportError:
        print("--- Backend Startup WARNING: 'anthropic' library not installed. Anthropic models will not work. ---")
        print("--- Please run: pip install anthropic ---")
    except Exception as e:
        print(f"--- Backend Startup CRITICAL WARNING: Error initializing Anthropic client: {e}. ---")
        print(f"--- This often means the ANTHROPIC_API_KEY is invalid, malformed, or your account has issues. Double-check the key in your .env file. ---")
        # The "Could not resolve authentication method" typically occurs here if the key is present but invalid/rejected by Anthropic.
else:
    print("--- Backend Startup CRITICAL WARNING: ANTHROPIC_API_KEY was NOT FOUND in environment. Anthropic models will NOT work. ---")
    print("--- Please ensure ANTHROPIC_API_KEY is correctly set in your .env file. ---")


# --- FastAPI App Setup ---
app = FastAPI()

origins = [
    "http://localhost",  # If you serve HTML via a local server on default port
    "http://127.0.0.1", # Common localhost address
    "null",             # For 'file://' origins when opening HTML directly in browser
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint ---
@app.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json()
    except Exception as e:
        # If JSON is malformed from frontend
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")

    model_name = data.get("model")
    message_content = data.get("message")

    if not message_content:
        return {"reply": "⚠️ Please provide a message."} # Should be handled by frontend too

    response_text = ""

    try:
        if model_name == "OpenAI GPT-4":
            if not openai_client: # Check if client was initialized
                return {"reply": "⚠️ OpenAI client is not configured on the server. Check server startup logs for API key issues."}
            
            # Client already has the key, no need to pass os.getenv() again
            chat_response = openai_client.chat.completions.create(
                model="gpt-4", # or "gpt-3.5-turbo" or other compatible models
                messages=[{"role": "user", "content": message_content}]
            )
            response_text = chat_response.choices[0].message.content.strip()

        elif model_name == "Claude 3 Sonnet":
            if not anthropic_client: # Check if client was initialized
                return {"reply": "⚠️ Anthropic client is not configured on the server. Check server startup logs for API key issues (ANTHROPIC_API_KEY)."}

            # Client already has the key. The error you're seeing likely means this call fails.
            chat_response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": message_content}]
            )
            response_text = chat_response.content[0].text.strip()
        else:
            response_text = f"⚠️ Invalid model selected: {model_name}"

        return {"reply": response_text}

    # Specific error handling for Anthropic API, including AuthenticationError
    except anthropic.APIError as e: # Catches various Anthropic API errors
        print(f"--- Backend CHAT ERROR (Anthropic APIError): {type(e).__name__} - {e} ---")
        error_detail = str(e)
        # AuthenticationError is a subclass of APIError. The message you see is typical for it.
        if isinstance(e, anthropic.AuthenticationError) or "authentication" in str(e).lower():
             return {"reply": f"⚠️ Anthropic Authentication Error: {error_detail}. Please URGENTLY double-check your ANTHROPIC_API_KEY in the .env file AND ensure the server was RESTARTED after any changes."}
        return {"reply": f"⚠️ Anthropic API Error: {error_detail}"}
    
    # General error handling for other unexpected issues during chat processing
    except Exception as e:
        print(f"--- Backend CHAT ERROR (General Exception): {type(e).__name__} - {e} ---")
        return {"reply": f"⚠️ An unexpected error occurred on the server: {str(e)}"}

# --- To Run (from the directory containing this file and your .env file) ---
# 1. Ensure you have a .env file with:
#    OPENAI_API_KEY="sk-your_openai_key"
#    ANTHROPIC_API_KEY="sk-ant-api03-your_anthropic_key"
#    (Replace with your actual keys!)
# 2. Install dependencies: pip install fastapi uvicorn openai anthropic python-dotenv
# 3. In your terminal, navigate to this project directory.
# 4. Run the server: uvicorn chat_backend_api:app --reload --host 127.0.0.1 --port 8000
#    OR, if you want to run directly with Python (uncomment the block below):
#    python chat_backend_api.py

if __name__ == "__main__":
    import uvicorn
    print("--- Backend Main: Starting server with `python chat_backend_api.py` ---")
    print("--- Make sure your .env file is correctly set up in the SAME directory as this script. ---")
    # The global variables OPENAI_API_KEY_FROM_ENV and ANTHROPIC_API_KEY_FROM_ENV were set at the top
    print(f"--- Backend Main: Detected OPENAI_API_KEY at startup? {'YES, key starts with ' + OPENAI_API_KEY_FROM_ENV[:5] if OPENAI_API_KEY_FROM_ENV else 'NO - OpenAI WILL LIKELY FAIL'}")
    print(f"--- Backend Main: Detected ANTHROPIC_API_KEY at startup? {'YES, key starts with ' + ANTHROPIC_API_KEY_FROM_ENV[:5] if ANTHROPIC_API_KEY_FROM_ENV else 'NO - Anthropic WILL LIKELY FAIL'}")
    uvicorn.run("chat_backend_api:app", host="127.0.0.1", port=8000, reload=True)