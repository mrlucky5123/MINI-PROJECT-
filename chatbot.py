import os
import chromadb
import cohere
from dotenv import load_dotenv

# --- 1. Load API keys and connect to Cohere ---
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if api_key:
    api_key = api_key.strip()
else:
    print("Error: COHERE_API_KEY not found in .env file.")
    exit()

co = cohere.Client(api_key)

# --- 2. Initialize Global Persistent Vector Database Client ---
# Initialized once globally to avoid connection overhead lag during active chat
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="rag_collection_pdfs")
    print("Successfully connected to 'rag_collection_pdfs' on disk.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    print("Did you remember to run 'ingest_data.py' successfully first?")
    exit()

# --- 3. Session Management Store (In-Memory Tracking) ---
chat_history_db = {}

def add_to_history(session_id, user_message, bot_response):
    """
    Appends a multi-turn chat sequence to a specific user session ID.
    Format maps directly to Cohere's Chat API specifications.
    """
    if session_id not in chat_history_db:
        chat_history_db[session_id] = []
    
    # Store using the specific API-compliant casing structure
    chat_history_db[session_id].append({"role": "USER", "message": user_message})
    chat_history_db[session_id].append({"role": "CHATBOT", "message": bot_response})

def get_history(session_id):
    """
    Retrieves the conversational history array for a user session.
    """
    return chat_history_db.get(session_id, [])

# --- 4. Retrieval and Generation Logic ---
def get_response(user_input, session_id):
    # A. Retrieve past conversational context for conversational memory
    conversation_history = get_history(session_id)
    
    try:
        # B. Convert incoming user query into vector embedding space
        user_query_embedding = co.embed(
            texts=[user_input],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        # C. Retrieve top matching text snippets from local ChromaDB vector store
        results = collection.query(
            query_embeddings=[user_query_embedding],
            n_results=2
        )
        
        # D. Format retrieved text results cleanly for the LLM documentation matrix
        formatted_documents = []
        if results['documents'] and results['documents'][0]:
            for doc_text in results['documents'][0]:
                formatted_documents.append({"snippet": doc_text})
        
        # E. Prompt isolation engineering (mitigates logic hallucinations)
        # E. Prompt isolation engineering (Optimized for Senior CP Engineer Profile)
        preamble_prompt = (
            "You are an elite Competitive Programming Coach and Lead Systems Architect.\n\n"
            "OPERATIONAL MANDATES:\n"
            "1. Synthesize responses utilizing ONLY the provided document snippets and conversation history.\n"
            "2. If the retrieved context lacks sufficient logical depth to confidently answer, strictly respond with: "
            "'I cannot extract an accurate engineering solution from the provided localized context.' Do not fabricate details.\n"
            "3. Absolute restriction: Do NOT leverage external knowledge thresholds or baseline assumptions outside the context.\n\n"
            "RESPONSE EXCELLENCE GUIDELINES:\n"
            "- Maintain an authoritative, highly technical, and precise tone. Eliminate generic introductory fluff.\n"
            "- Prioritize structural performance: Always mention Time Complexity and Space Complexity using Big-O notation where applicable.\n"
            "- Code Quality: Provide highly optimized, modular, and cleanly indented C++ syntax. Use modern competitive programming best practices (e.g., efficient I/O concepts, proper data types to avoid overflow).\n"
            "- Structural Layout: Use crisp Markdown headers, clean bullet points, and code fragments to make the response highly scannable."
        )
        
        # F. Dispatch context-locked call payload to Cohere model endpoint
        response = co.chat(
            model="command-r-08-2024",
            message=user_input,
            chat_history=conversation_history, # Maintains multi-turn context tracking
            preamble=preamble_prompt,
            documents=formatted_documents
        )

        bot_response = response.text
        
        # G. Log current interaction turn back into state storage
        add_to_history(session_id, user_input, bot_response)
        return bot_response

    except Exception as e:
        return f"An operational error occurred during processing: {e}"

# --- 5. Main Chatbot Interface Loop ---
if __name__ == "__main__":
    session_id = "default_user_session"
    print("\n" + "="*60)
    print("📚 Conversational RAG-Tutor Chatbot Online!")
    print("Ask any question from the Handbook. Type 'exit' to cleanly close.")
    print("="*60 + "\n")
    
    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == 'exit':
            print("\nChatbot: Happy coding! Goodbye!")
            break
        
        response = get_response(user_message, session_id)
        print(f"Chatbot: {response}\n")
        print("-" * 60)