import os
from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from uagents import Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    MetadataContent,
    Resource,
    ResourceContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from pydantic.v1 import UUID4
from uagents_core.storage import ExternalStorage

from evitsam import get_image

STORAGE_URL = os.getenv("AGENTVERSE_URL", "https://agentverse.ai") + "/v1/storage"
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")
if AGENTVERSE_API_KEY is None:
    raise ValueError("You need to provide an API_TOKEN.")

external_storage = ExternalStorage(api_token=AGENTVERSE_API_KEY, storage_url=STORAGE_URL)

def create_text_chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )

def create_metadata(metadata: dict[str, str]) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[MetadataContent(
            type="metadata",
            metadata=metadata,
        )],
    )

def create_resource_chat(asset_id: str, uri: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[
            ResourceContent(
                type="resource",
                resource_id=UUID4(asset_id),
                resource=Resource(
                    uri=uri,
                    metadata={
                        "mime_type": "image/png",
                        "role": "generated-image"
                    }
                )
            )
        ]
    )

chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.storage.set(str(ctx.session), sender)
    ctx.logger.info(f"Got a message from {sender}")
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id
        ),
    )

    image_content = []
    text_content = []
    
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Got a start session message from {sender}")
            await ctx.send(sender, create_metadata({"attachments": "true"}))
        elif isinstance(item, TextContent):
            ctx.logger.info(f"Got a message from {sender}: {item.text}")
            text_content.append({"text": item.text, "type": "text"})
        elif isinstance(item, ResourceContent):
            try:
                ctx.logger.info(f"Got a resource from {sender}")
                external_storage = ExternalStorage(
                    identity=ctx.agent.identity,
                    storage_url=STORAGE_URL,
                )
                data = external_storage.download(str(item.resource_id))
                image_content.append({
                    "type": "resource",
                    "mime_type": data["mime_type"],
                    "contents": data["contents"],
                })

            except Exception as ex:
                ctx.logger.error(f"Failed to download resource: {ex}")
                await ctx.send(sender, create_text_chat("Failed to download resource."))
        else:
            ctx.logger.warning(f"Got unexpected content from {sender}")

    if image_content:
        try:
            # Process the image with SAM
            segmented_image, analysis = await get_image(image_content)
            
            if segmented_image:
                # Store the segmented image
                asset_id = str(uuid4())
                external_storage = ExternalStorage(
                    api_token=AGENTVERSE_API_KEY,
                    storage_url=STORAGE_URL,
                )
                asset_id = external_storage.create_asset(
                    name=f"segmented_{asset_id}",
                    content=segmented_image,
                    mime_type="image/png"
                )
                
                ctx.logger.info(f"Asset created with ID: {asset_id}")
                # Get the URL for the segmented image
                asset_uri = f"agent-storage://{STORAGE_URL}/{asset_id}"
                
                external_storage.set_permissions(asset_id=asset_id, agent_address=sender)
                ctx.logger.info(f"Asset permissions set to: {sender}")

                # Send the analysis text if available
                if analysis:
                    await ctx.send(sender, create_text_chat(analysis))
                
                # Send the segmented image
                await ctx.send(sender, create_resource_chat(asset_id, asset_uri))
            else:
                await ctx.send(sender, create_text_chat(analysis or "Failed to process image."))
                
        except Exception as e:
            ctx.logger.error(f"Error processing image: {e}")
            await ctx.send(sender, create_text_chat("Error processing image."))
    elif text_content:
        await ctx.send(sender, create_text_chat("Please send an image to analyze."))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}"
    )