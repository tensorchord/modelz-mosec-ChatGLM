from typing import List

from transformers import AutoTokenizer, AutoModel  # type: ignore

from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

logger = get_logger()


class ChatGLM(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        self.model = model.eval()

    def forward(self, data):
        response = self.model.chat(self.tokenizer, data, history=[])
        return response


if __name__ == "__main__":
    server = Server()
    server.append_worker(ChatGLM, num=1, max_batch_size=4)
    server.run()
