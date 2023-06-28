import requests
import json
from pprint import pprint

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_TEMPERATURE = 0.5


class AgentInstance:
    def __init__(self,
                 api_key: str,
                 agent_prompt: str,
                 model: str = DEFAULT_MODEL,
                 endpoint: str = DEFAULT_ENDPOINT,
                 temperature: float = DEFAULT_TEMPERATURE) -> None:
        assert temperature >= 0 and temperature <= 2, "Temperature must be in the range [0,2]"
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._temperature = temperature
        self._messages = [{"role": "system",
                          "content": agent_prompt}]

    def query(self, prompt: str) -> str:
        self._messages.append({"role": "user",
                               "content": prompt})
        try:
            response = self._raw_query()['choices'][0]['message']['content']
            self._messages.append({"role": "assistant",
                                   "content": response})
            return response
        except Exception:
            raise

    def print_history(self):
        pprint(self._messages, sort_dicts=False)

    def _raw_query(self) -> str:
        data = {
            "model": self._model,
            "messages": self._messages,
            "temperature": self._temperature,
        }
        response = requests.post(self._endpoint,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {self._api_key}",
                                          },
                                 data=json.dumps(data))
        if response.status_code == 200:
            response_json = response.json()
            return response_json
        else:
            raise Exception(f"Error querying GPT-3 API: "
                            f"{response.status_code} - {response.text}")
