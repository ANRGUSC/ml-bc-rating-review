import requests
import base64

def get_auth_token(client_id, client_secret):
    client_creds = f"{client_id}:{client_secret}"
    client_creds_b64 = base64.b64encode(client_creds.encode())

    token_url = "https://accounts.spotify.com/api/token"
    method = "POST"
    token_data = {
        "grant_type": "client_credentials"
    }
    token_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {client_creds_b64.decode()}"
    }

    r = requests.post(token_url, data=token_data, headers=token_headers)
    valid_request = r.status_code in range(200, 299)
    
    if valid_request:
        token_response_data = r.json()
        return token_response_data['access_token']
    else:
        return None
