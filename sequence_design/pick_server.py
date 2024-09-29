import random
import string
import time

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Completion API
stream = False


def generate_random_string(length):
    # Choose from letters and digits
    characters = string.ascii_letters + string.digits
    # Randomly choose characters from the set to form the string
    random_string = "".join(random.choice(characters) for i in range(length))
    return random_string


while True:
    time.sleep(1)
    completion = client.completions.create(
        model=model,
        prompt=generate_random_string(128),
        echo=False,
        n=2,
        stream=stream,
        logprobs=3,
    )
