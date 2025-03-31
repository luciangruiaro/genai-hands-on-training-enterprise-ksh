from service.responder import generate_response


def start_cli():
    print("Running in CLI mode. Type your message:")
    while True:
        message = input("> ")
        if message.lower() in ["exit", "quit"]:
            break
        print(generate_response(message))
