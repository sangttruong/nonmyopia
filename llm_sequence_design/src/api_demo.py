import os

import uvicorn

from llmtuner import ChatModel, create_app


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 1337)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 1337)), workers=1)


if __name__ == "__main__":
    main()
