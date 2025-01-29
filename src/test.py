import requests

url = "http://127.0.0.1:8000/chatbot"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "prompt": "Hello, my name is Javier.",
    "thread": 123
}

response = requests.post(url, headers=headers, json=data)

print("Status code:", response.status_code)
print("Response body:", response.text)