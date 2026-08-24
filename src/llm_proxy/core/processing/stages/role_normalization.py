"""Role normalization utility.

This is a pure function, not a PipelineStage, because it is applied on-demand
during retry when a provider returns a role error, not during the initial
linear pipeline pass.

Post-parse mutation contract: the transform diverges the parsed conversation
from the stashed raw protocol body, so it sets ``native_request_disabled`` to
keep every raw-reuse path (native passthrough and the wire-compatible fast
path) from resurrecting the rejected roles on the wire.
"""

from llm_proxy.models import InternalRequest
from llm_proxy.models.conversation import SystemMessage


def normalize_developer_roles(request: InternalRequest) -> bool:
    """Transform developer roles to system. Returns True if any were transformed."""
    transformed = False

    for i, msg in enumerate(request.conversation.system_messages):
        if msg.role == "developer":
            request.conversation.system_messages[i] = SystemMessage(
                role="system",
                content=msg.content,
                name=msg.name,
            )
            transformed = True

    developer_msgs = [msg for msg in request.conversation.messages if msg.role == "developer"]
    if developer_msgs:
        for msg in developer_msgs:
            request.conversation.system_messages.append(
                SystemMessage.from_text(
                    role="system",
                    text=msg.text_content,
                    name=msg.name,
                )
            )
            request.conversation.messages.remove(msg)
        transformed = True

    if transformed:
        # The parsed conversation no longer matches ``_raw_protocol_data``:
        # raw-reuse tiers (native passthrough, wire reuse) would
        # send the rejected developer roles again and the retry would fail
        # identically. Force the rebuilt path for the rest of this request.
        request.native_request_disabled = True

    return transformed
