
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

class QABot:
    def __init__(self, retriever):
        self.retriever = retriever
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def ask(self, query):
        return self.chain.run(query)
