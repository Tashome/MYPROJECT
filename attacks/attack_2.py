import requests
API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "samples/original/original.png"
with open(IMAGE_PATH, "rb") as image:

    files = {
        "file": (
            "original.png",
            image,
            "image/png"
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )
print("HTTP status:")
print(response.status_code)

print("\nAPI response:")
print(response.text)
