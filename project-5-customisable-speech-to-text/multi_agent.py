import requests
import json
import logging 
import os 
from dotenv import load_dotenv
from threading import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

session_id = None
base_url = os.getenv("MULTI_AGENT_URL")

session_data = {}

def health():
    url = f'{base_url}/health'
    response = requests.get(url)
    response_data = response.json()

    logger.info(response_data)

def create_call():
    global session_id
    url = f'{base_url}/create_session'
    response = requests.post(url)
    response.raise_for_status()
    response_data = response.json()
    session_id = response_data['session_id']
    logger.info(f"create_session response data: {response_data}")
    return response_data['greeting']

def end_call():
    global session_id
    if session_id:
        url = f'{base_url}/end_session/{session_id}'
        response = requests.post(url)
        response_data = response.json()
        response.raise_for_status()
        logger.info(f"Session ended - Response data: {response_data}")
        return response_data['message']
    else:
        print("No active session. Please create a session first.")
        return

def send_request(textInput: str):

    global session_id
    if session_id is None:
        logger.info("No active session. Creating a new session.")
        create_call()

    while True:
        if textInput.lower() == 'exit':  # Check if the user wants to exit
            end_call()  # End the session
            break
        
        url = f'{base_url}/update_session/{session_id}'
        headers = {'Content-Type': 'application/json'}
        data = {
            "sentence": textInput
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            response_data = response.json()

            closing_mes = response_data['closing_message']
            if closing_mes != "":
                logger.info(f"Closing message before end_call: {closing_mes}")
                data = get_session()
                save_data(data)
                end_response = closing_mes
                logger.info(f"Closing message after end_call: {closing_mes}")

                return closing_mes, end_response
            
            logger.info("Caller: " + textInput)
            
            if "error" in response_data:
                print("Error: " + response_data["error"])
                if response_data["error"] == "Session not active. Please create a new session.":
                    create_call()
                    continue
            elif "response" in response_data:
                logger.info("AI: " + response_data["response"])
                logger.info("Execution time: {}".format(response_data.get("execution_time", "N/A")))
                logger.info("Extracted info: {}".format(json.dumps(response_data.get("extracted_info", {}), indent=4)))
                return response_data["response"], None
            else:
                logger.info("Unexpected response format. Full response:")
                logger.info(json.dumps(response_data, indent=4))
            
        except requests.exceptions.RequestException as e:
            logger.info(f"An error occurred: {e}")
            end_response = end_call()
            break
        except json.JSONDecodeError:
            logger.info("Error decoding JSON response")
        except KeyError as e:
            logger.info(f"Expected key not found in response: {e}")
        
        print("\n")



def save_data(response: dict, ttl: int = 120):
    global session_data  # Declare as global to modify the global variable

    # Extract session_id from the response
    session_id = response.get('session_id', 'default_session')

    # Initialize the session_id entry if not already present
    if session_id not in session_data:
        session_data[session_id] = {
            "session_id": session_id,  # Store the session_id at the top level
            "data": []  # Create an empty list for the session's data
        }

    # Append the response (excluding the session_id) to the "data" list under the session_id
    session_data[session_id]["data"].append({
        "convo_hist": response.get("convo_hist", ""),
        "extract_info": response.get("extracted_info", ""),
    })

    # Start a timer to delete the session data after the TTL
    t = Timer(ttl, delete_session_data, [session_id])
    t.start()


def get_session_data(session_id: str):
    global session_data

    data = session_data.get(session_id)
    if data:
        del session_data[session_id]
        logging.info(f"Delete {session_id} sucessfully")
    else:
        logging.info(f"Session {session_id} not found")
    return data

def delete_session_data(session_id: str):
    """
    Automatically delete session data after TTL expires.
    """
    global session_data
    if session_id in session_data:
        del session_data[session_id]
        logger.info(f"Session {session_id} deleted after timeout.")
    else:
        logger.info(f"Session {session_id} not found for delete automatically.")


def get_session():
    global session_id
    if session_id:
        url = f'{base_url}/get_session/{session_id}'
        response = requests.get(url)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Session info:")
        logger.info(f"Session_id: {response_data['session_id']}")
        logger.info(f"Extracted info: {json.dumps(response_data['extracted_info'], indent=4)}")
        logger.info(f"Conversation history: {response_data['convo_hist']}")
        return response_data
    else:
        logger.info("No active session.")
        return

# Update config
def update_config(data: dict, base_url=base_url):
    # The new configuration to be updated
    new_config = data
    url = f"{base_url}/config"
    headers = {"Content-Type": "application/json"}

    try:
        # Send PATCH request to update config
        response = requests.patch(url, headers=headers, data=json.dumps(new_config))
        response.raise_for_status()
        return {
            "status": "success",
            "status_code": response.status_code,
            "data": response.json()
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": str(e),
            "status_code": getattr(e.response, 'status_code', None)
        }