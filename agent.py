from uagents import Agent, Context
from chat_proto import chat_proto

agent = Agent(
    name="EfficientViT SAM Agent",
    port=8000,
    mailbox=True,
    publish_agent_details=True,
    readme_path="README.md"
)

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()