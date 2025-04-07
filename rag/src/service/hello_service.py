from helpers.logger import setup_logger
from integrations.llm.llm_interface import LLMClient


class HelloService:
    def __init__(self, config):
        self.logger = setup_logger("app")
        self.constants = config.get("constants", {})
        self.llm = LLMClient(config)

    def say_hello(self):
        self.logger.debug("Processing GET /hello")
        return {"greeting": self.constants.get("greeting_message", "Hello!")}

    def process_data(self, data):
        self.logger.info(f"Processing POST /hello with data: {data}")
        return {
            "received": data,
            "message": self.constants.get("post_ack", "Data received.") +
                       self.llm.ask(
                           str(data) + " Te rog sa imi raspunzi la urmatoarea intrebare bazandu-te pe contextul dat sau sa facei o deductie daca consideri relevant. De asemenea, te rog sa raspunzi la obiect, fara a adauga niciun alt detaliu. Context: Tudor este specialist software la Keysight")
        }
