"""Unit tests for internal models.

Tests cover:
1. Import of InternalRequest, InternalResponse
2. Import of all types from models package
3. Import of content blocks
4. Instantiation smoke tests for all models
"""


class TestInternalModelsImport:
    """Test import of InternalRequest and InternalResponse."""

    def test_import_internal_request(self):
        """Test that InternalRequest can be imported."""
        from llm_proxy.models import InternalRequest

        assert InternalRequest is not None

    def test_import_internal_response(self):
        """Test that InternalResponse can be imported."""
        from llm_proxy.models import InternalResponse

        assert InternalResponse is not None

    def test_import_request_metadata(self):
        """Test that RequestMetadata can be imported."""
        from llm_proxy.models import RequestMetadata

        assert RequestMetadata is not None

    def test_import_internal_models_directly(self):
        """Test direct import from internal module."""
        from llm_proxy.models.internal import (
            InternalRequest,
            InternalResponse,
            RequestMetadata,
        )

        assert InternalRequest is not None
        assert InternalResponse is not None
        assert RequestMetadata is not None


class TestAllModelsImport:
    """Test import of all types from models package."""

    def test_import_content_blocks(self):
        """Test import of all content block types."""
        from llm_proxy.models import (
            AudioBlock,
            ContentBlock,
            CustomToolUseBlock,
            DocumentBlock,
            FileBlock,
            ImageBlock,
            RedactedThinkingBlock,
            RefusalBlock,
            ServerToolUseBlock,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            BashCodeExecutionToolResultBlock,
            CacheControl,
            Caller,
            Citation,
            CitationCharLocation,
            CitationContentBlockLocation,
            CitationPageLocation,
            CitationSearchResultLocation,
            CitationWebSearchResultLocation,
            CodeExecutionToolResultBlock,
            ContainerUploadBlock,
            SearchResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolReferenceBlock,
            ToolSearchToolResultBlock,
            WebFetchToolResultBlock,
            WebSearchResultContentBlock,
            WebSearchToolResultBlock,
        )

        assert AudioBlock is not None
        assert BashCodeExecutionToolResultBlock is not None
        assert CacheControl is not None
        assert Caller is not None
        assert Citation is not None
        assert CitationCharLocation is not None
        assert CitationContentBlockLocation is not None
        assert CitationPageLocation is not None
        assert CitationSearchResultLocation is not None
        assert CitationWebSearchResultLocation is not None
        assert CodeExecutionToolResultBlock is not None
        assert ContainerUploadBlock is not None
        assert ContentBlock is not None
        assert CustomToolUseBlock is not None
        assert DocumentBlock is not None
        assert FileBlock is not None
        assert ImageBlock is not None
        assert RedactedThinkingBlock is not None
        assert RefusalBlock is not None
        assert SearchResultBlock is not None
        assert ServerToolUseBlock is not None
        assert TextBlock is not None
        assert TextEditorCodeExecutionToolResultBlock is not None
        assert ThinkingBlock is not None
        assert ToolReferenceBlock is not None
        assert ToolResultBlock is not None
        assert ToolSearchToolResultBlock is not None
        assert ToolUseBlock is not None
        assert WebFetchToolResultBlock is not None
        assert WebSearchResultContentBlock is not None
        assert WebSearchToolResultBlock is not None

    def test_import_conversation_types(self):
        """Test import of conversation types."""
        from llm_proxy.models import ConversationContext, Message, SystemMessage

        assert ConversationContext is not None
        assert Message is not None
        assert SystemMessage is not None

    def test_import_embedding_types(self):
        """Test import of embedding types."""
        from llm_proxy.models import (
            EmbeddingData,
            InternalEmbeddingRequest,
            InternalEmbeddingResponse,
        )

        assert EmbeddingData is not None
        assert InternalEmbeddingRequest is not None
        assert InternalEmbeddingResponse is not None

    def test_import_image_types(self):
        """Test import of image generation types."""
        from llm_proxy.models import (
            ImageData,
            ImageSize,
            InternalImageRequest,
            InternalImageResponse,
            Usage,
        )

        assert ImageData is not None
        assert ImageSize is not None
        assert Usage is not None
        assert InternalImageRequest is not None
        assert InternalImageResponse is not None

    def test_import_param_types(self):
        """Test import of parameter types."""
        from llm_proxy.models import (
            AnthropicSpecificParams,
            CommonParams,
            GeminiSpecificParams,
            GenerationParams,
            OpenAISpecificParams,
        )

        assert AnthropicSpecificParams is not None
        assert CommonParams is not None
        assert GeminiSpecificParams is not None
        assert GenerationParams is not None
        assert OpenAISpecificParams is not None

    def test_import_tool_types(self):
        """Test import of tool definition types."""
        from llm_proxy.models import (
            AllowedToolsConfig,
            CustomTool,
            FunctionTool,
            ToolChoice,
            ToolChoiceAllowedTools,
            ToolChoiceCustom,
            ToolChoiceFunction,
            ToolChoiceNamed,
            ToolChoiceSpec,
            ToolDefinition,
        )
        from llm_proxy.models.tools.anthropic_builtin import (
            BashTool,
            CodeExecutionTool,
            MemoryTool,
            TextEditorTool,
            ToolSearchTool,
            WebFetchTool,
            WebSearchTool,
        )

        assert AllowedToolsConfig is not None
        assert BashTool is not None
        assert CodeExecutionTool is not None
        assert CustomTool is not None
        assert FunctionTool is not None
        assert MemoryTool is not None
        assert TextEditorTool is not None
        assert ToolChoice is not None
        assert ToolChoiceAllowedTools is not None
        assert ToolChoiceCustom is not None
        assert ToolChoiceFunction is not None
        assert ToolChoiceNamed is not None
        assert ToolChoiceSpec is not None
        assert ToolDefinition is not None
        assert ToolSearchTool is not None
        assert WebFetchTool is not None
        assert WebSearchTool is not None

    def test_import_supporting_types(self):
        """Test import of supporting types."""
        from llm_proxy.models import (
            Annotation,
            AudioSource,
            CacheCreation,
            ChoiceLogprobs,
            CitationsConfig,
            CompletionTokensDetails,
            Container,
            DocumentSource,
            FinishReason,
            ImageSource,
            PromptTokensDetails,
            ResponseFormat,
            ResponseStatus,
            ServerToolUsage,
            StreamOptions,
            ThinkingConfig,
            TokenLogprob,
            UrlCitation,
            Usage,
        )

        assert Annotation is not None
        assert AudioSource is not None
        assert CacheCreation is not None
        assert ChoiceLogprobs is not None
        assert CitationsConfig is not None
        assert CompletionTokensDetails is not None
        assert Container is not None
        assert DocumentSource is not None
        assert FinishReason is not None
        assert ImageSource is not None
        assert PromptTokensDetails is not None
        assert ResponseFormat is not None
        assert ResponseStatus is not None
        assert ServerToolUsage is not None
        assert StreamOptions is not None
        assert ThinkingConfig is not None
        assert TokenLogprob is not None
        assert UrlCitation is not None
        assert Usage is not None


class TestContentBlocksInstantiation:
    """Test instantiation of content block types."""

    def test_text_block_instantiation(self):
        """Test TextBlock can be instantiated."""
        from llm_proxy.models import TextBlock

        block = TextBlock(text="Hello, world!")
        assert block.text == "Hello, world!"

    def test_image_block_instantiation(self):
        """Test ImageBlock can be instantiated."""
        from llm_proxy.models import ImageBlock, ImageSource

        source = ImageSource(type="url", data="https://example.com/image.png", media_type=None)
        block = ImageBlock(source=source)
        assert block.source.type == "url"

    def test_audio_block_instantiation(self):
        """Test AudioBlock can be instantiated."""
        from llm_proxy.models import AudioBlock, AudioSource

        source = AudioSource(type="base64", data="abc123", media_type="audio/mp3")
        block = AudioBlock(source=source)
        assert block.source.type == "base64"

    def test_document_block_instantiation(self):
        """Test DocumentBlock can be instantiated."""
        from llm_proxy.models import DocumentBlock, DocumentSource

        source = DocumentSource(type="base64", data="abc123", media_type="application/pdf")
        block = DocumentBlock(source=source, title="Test Document")
        assert block.title == "Test Document"

    def test_tool_use_block_instantiation(self):
        """Test ToolUseBlock can be instantiated."""
        from llm_proxy.models import ToolUseBlock

        block = ToolUseBlock(id="tool_123", name="get_weather", input={"location": "NYC"})
        assert block.id == "tool_123"
        assert block.name == "get_weather"

    def test_tool_result_block_instantiation(self):
        """Test ToolResultBlock can be instantiated."""
        from llm_proxy.models import ToolResultBlock

        block = ToolResultBlock(tool_use_id="tool_123", content="Sunny, 72F")
        assert block.tool_use_id == "tool_123"
        assert block.content == "Sunny, 72F"

    def test_thinking_block_instantiation(self):
        """Test ThinkingBlock can be instantiated."""
        from llm_proxy.models import ThinkingBlock

        block = ThinkingBlock(thinking="Let me think about this...")
        assert block.thinking == "Let me think about this..."

    def test_refusal_block_instantiation(self):
        """Test RefusalBlock can be instantiated."""
        from llm_proxy.models import RefusalBlock

        block = RefusalBlock(refusal="I cannot help with that request.")
        assert block.refusal == "I cannot help with that request."

    def test_file_block_instantiation(self):
        """Test FileBlock can be instantiated."""
        from llm_proxy.models import FileBlock

        block = FileBlock(file_id="file_123", filename="document.pdf")
        assert block.file_id == "file_123"
        assert block.filename == "document.pdf"

    def test_search_result_block_instantiation(self):
        """Test SearchResultBlock can be instantiated."""
        from llm_proxy.models import TextBlock
        from llm_proxy.models.content_blocks.anthropic_builtin import SearchResultBlock

        block = SearchResultBlock(
            file_id="file_123",
            title="Search Result",
            content=[TextBlock(text="Result text")],
        )
        assert block.file_id == "file_123"

    def test_web_search_tool_result_block_instantiation(self):
        """Test WebSearchToolResultBlock can be instantiated."""
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        block = WebSearchToolResultBlock(tool_use_id="tool_123", content="Search results")
        assert block.tool_use_id == "tool_123"

    def test_server_tool_use_block_instantiation(self):
        """Test ServerToolUseBlock can be instantiated."""
        from llm_proxy.models import ServerToolUseBlock

        block = ServerToolUseBlock(id="server_tool_123", name="web_search", input={"query": "test"})
        assert block.id == "server_tool_123"


class TestInternalModelsInstantiation:
    """Test instantiation of internal request/response models."""

    def test_request_metadata_instantiation(self):
        """Test RequestMetadata can be instantiated."""
        from llm_proxy.models import RequestMetadata

        metadata = RequestMetadata(request_id="req_123", user="user_456")
        assert metadata.request_id == "req_123"
        assert metadata.user == "user_456"

    def test_internal_request_instantiation(self):
        """Test InternalRequest can be instantiated."""
        from llm_proxy.models import ConversationContext, InternalRequest

        conversation = ConversationContext()
        request = InternalRequest(model="gpt-4", conversation=conversation)
        assert request.model == "gpt-4"
        assert request.request_type == "chat"
        assert request.stream is False

    def test_internal_response_instantiation(self):
        """Test InternalResponse can be instantiated."""
        from llm_proxy.models import InternalResponse, TextBlock

        response = InternalResponse(id="resp_123", model="gpt-4", output=[TextBlock(text="Hello!")])
        assert response.id == "resp_123"
        assert response.model == "gpt-4"
        assert len(response.output) == 1

    def test_internal_response_get_thinking_content(self):
        """Test InternalResponse.get_thinking_content method."""
        from llm_proxy.models import InternalResponse, TextBlock, ThinkingBlock

        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                ThinkingBlock(thinking="Let me think..."),
                TextBlock(text="Here's the answer."),
            ],
        )
        assert response.get_thinking_content() == "Let me think..."

    def test_internal_response_get_refusal(self):
        """Test InternalResponse.get_refusal method."""
        from llm_proxy.models import InternalResponse, RefusalBlock, TextBlock

        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                RefusalBlock(refusal="I cannot do that."),
                TextBlock(text="But I can help with other things."),
            ],
        )
        assert response.get_refusal() == "I cannot do that."


class TestConversationModelsInstantiation:
    """Test instantiation of conversation models."""

    def test_message_instantiation(self):
        """Test Message can be instantiated."""
        from llm_proxy.models import Message, TextBlock

        message = Message(role="user", content=[TextBlock(text="Hello!")])
        assert message.role == "user"
        assert message.text_content == "Hello!"

    def test_system_message_instantiation(self):
        """Test SystemMessage can be instantiated."""
        from llm_proxy.models import SystemMessage, TextBlock

        message = SystemMessage(
            role="system", content=[TextBlock(text="You are a helpful assistant.")]
        )
        assert message.role == "system"
        assert message.text_content == "You are a helpful assistant."

    def test_conversation_context_instantiation(self):
        """Test ConversationContext can be instantiated."""
        from llm_proxy.models import ConversationContext, Message, SystemMessage, TextBlock

        context = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="Be helpful.")],
            messages=[Message(role="user", content=[TextBlock(text="Hi!")])],
        )
        assert len(context.system_messages) == 1
        assert len(context.messages) == 1


class TestEmbeddingModelsInstantiation:
    """Test instantiation of embedding models."""

    def test_embedding_data_instantiation(self):
        """Test EmbeddingData can be instantiated."""
        from llm_proxy.models import EmbeddingData

        data = EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)
        assert data.index == 0
        assert data.object == "embedding"

    def test_internal_embedding_request_instantiation(self):
        """Test InternalEmbeddingRequest can be instantiated."""
        from llm_proxy.models import InternalEmbeddingRequest

        request = InternalEmbeddingRequest(model="text-embedding-3-small", input="Hello world")
        assert request.model == "text-embedding-3-small"
        assert request.request_type == "embedding"

    def test_internal_embedding_response_instantiation(self):
        """Test InternalEmbeddingResponse can be instantiated."""
        from llm_proxy.models import EmbeddingData, InternalEmbeddingResponse

        response = InternalEmbeddingResponse(
            model="text-embedding-3-small", data=[EmbeddingData(embedding=[0.1, 0.2], index=0)]
        )
        assert response.model == "text-embedding-3-small"
        assert len(response.data) == 1


class TestImageModelsInstantiation:
    """Test instantiation of image generation models."""

    def test_image_size_instantiation(self):
        """Test ImageSize can be instantiated."""
        from llm_proxy.models import ImageSize

        size = ImageSize(width=1024, height=1024)
        assert size.width == 1024
        assert size.height == 1024

    def test_image_size_parse(self):
        """Test ImageSize.parse class method."""
        from llm_proxy.models import ImageSize

        size = ImageSize.parse("1024x768")
        assert size.width == 1024
        assert size.height == 768

    def test_image_data_instantiation(self):
        """Test ImageData can be instantiated."""
        from llm_proxy.models import ImageData

        data = ImageData(url="https://example.com/image.png")
        assert data.url == "https://example.com/image.png"

    def test_image_usage_instantiation(self):
        """Test Usage can be instantiated for image responses."""
        from llm_proxy.models import Usage

        usage = Usage(input_tokens=100)
        assert usage.input_tokens == 100

    def test_internal_image_request_instantiation(self):
        """Test InternalImageRequest can be instantiated."""
        from llm_proxy.models import InternalImageRequest

        request = InternalImageRequest(model="dall-e-3", prompt="A beautiful sunset")
        assert request.model == "dall-e-3"
        assert request.prompt == "A beautiful sunset"
        assert request.request_type == "image_generation"

    def test_internal_image_response_instantiation(self):
        """Test InternalImageResponse can be instantiated."""
        from llm_proxy.models import ImageData, InternalImageResponse

        response = InternalImageResponse(
            created=1234567890, data=[ImageData(url="https://example.com/image.png")]
        )
        assert response.created == 1234567890


class TestParamModelsInstantiation:
    """Test instantiation of parameter models."""

    def test_common_params_instantiation(self):
        """Test CommonParams can be instantiated."""
        from llm_proxy.models import CommonParams

        params = CommonParams(temperature=0.7, max_tokens=1000)
        assert params.temperature == 0.7
        assert params.max_tokens == 1000

    def test_generation_params_instantiation(self):
        """Test GenerationParams can be instantiated."""
        from llm_proxy.models import GenerationParams

        params = GenerationParams(temperature=0.7, max_tokens=1000)
        assert params.temperature == 0.7
        assert params.max_tokens == 1000

    def test_openai_specific_params_instantiation(self):
        """Test OpenAISpecificParams can be instantiated."""
        from llm_proxy.models import OpenAISpecificParams

        params = OpenAISpecificParams(logprobs=True, top_logprobs=5)
        assert params.logprobs is True

    def test_anthropic_specific_params_instantiation(self):
        """Test AnthropicSpecificParams can be instantiated."""
        from llm_proxy.models import AnthropicSpecificParams

        params = AnthropicSpecificParams(top_k=50)
        assert params.top_k == 50

    def test_gemini_specific_params_instantiation(self):
        """Test GeminiSpecificParams can be instantiated."""
        from llm_proxy.models import GeminiSpecificParams

        params = GeminiSpecificParams(top_k=40)
        assert params.top_k == 40


class TestToolModelsInstantiation:
    """Test instantiation of tool models."""

    def test_function_tool_instantiation(self):
        """Test FunctionTool can be instantiated."""
        from llm_proxy.models import FunctionTool

        tool = FunctionTool(name="get_weather", parameters={"type": "object"})
        assert tool.name == "get_weather"

    def test_bash_tool_instantiation(self):
        """Test BashTool can be instantiated."""
        from llm_proxy.models.tools.anthropic_builtin import BashTool

        tool = BashTool()
        assert tool.name == "bash"

    def test_code_execution_tool_instantiation(self):
        """Test CodeExecutionTool can be instantiated."""
        from llm_proxy.models.tools.anthropic_builtin import CodeExecutionTool

        tool = CodeExecutionTool()
        assert tool.name == "code_execution"

    def test_web_search_tool_instantiation(self):
        """Test WebSearchTool can be instantiated."""
        from llm_proxy.models.tools.anthropic_builtin import WebSearchTool

        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_tool_choice_instantiation(self):
        """Test ToolChoice can be instantiated."""
        from llm_proxy.models import ToolChoice

        choice = ToolChoice(mode="auto")
        assert choice.mode == "auto"

    def test_tool_choice_function_instantiation(self):
        """Test ToolChoiceFunction can be instantiated."""
        from llm_proxy.models import ToolChoiceFunction

        choice = ToolChoiceFunction(name="get_weather")
        assert choice.name == "get_weather"


class TestSupportingTypesInstantiation:
    """Test instantiation of supporting types."""

    def test_usage_instantiation(self):
        """Test Usage can be instantiated."""
        from llm_proxy.models import Usage

        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_response_format_instantiation(self):
        """Test ResponseFormat can be instantiated."""
        from llm_proxy.models import ResponseFormat

        fmt = ResponseFormat(type="json_object")
        assert fmt.type == "json_object"

    def test_stream_options_instantiation(self):
        """Test StreamOptions can be instantiated."""
        from llm_proxy.models import StreamOptions

        opts = StreamOptions(include_usage=True)
        assert opts.include_usage is True

    def test_thinking_config_instantiation(self):
        """Test ThinkingConfig can be instantiated."""
        from llm_proxy.models import ThinkingConfig

        config = ThinkingConfig(type="enabled", budget_tokens=1000)
        assert config.type == "enabled"
        assert config.budget_tokens == 1000

    def test_image_source_instantiation(self):
        """Test ImageSource can be instantiated."""
        from llm_proxy.models import ImageSource

        source = ImageSource(type="base64", data="abc123", media_type="image/png")
        assert source.type == "base64"

    def test_audio_source_instantiation(self):
        """Test AudioSource can be instantiated."""
        from llm_proxy.models import AudioSource

        source = AudioSource(type="base64", data="abc123", media_type="audio/mp3")
        assert source.type == "base64"

    def test_document_source_instantiation(self):
        """Test DocumentSource can be instantiated."""
        from llm_proxy.models import DocumentSource

        source = DocumentSource(type="base64", data="abc123", media_type="application/pdf")
        assert source.type == "base64"

    def test_annotation_instantiation(self):
        """Test Annotation can be instantiated."""
        from llm_proxy.models import Annotation, UrlCitation

        citation = UrlCitation(
            url="https://example.com", title="Example", start_index=0, end_index=10
        )
        annotation = Annotation(type="url_citation", url_citation=citation)
        assert annotation.type == "url_citation"

    def test_token_logprob_instantiation(self):
        """Test TokenLogprob can be instantiated."""
        from llm_proxy.models import TokenLogprob

        logprob = TokenLogprob(token="hello", logprob=-0.5)
        assert logprob.token == "hello"

    def test_choice_logprobs_instantiation(self):
        """Test ChoiceLogprobs can be instantiated."""
        from llm_proxy.models import ChoiceLogprobs, TokenLogprob

        logprobs = ChoiceLogprobs(content=[TokenLogprob(token="hello", logprob=-0.5)])
        assert logprobs.content is not None

    def test_container_instantiation(self):
        """Test Container can be instantiated."""
        from llm_proxy.models import Container

        container = Container(id="container_123")
        assert container.id == "container_123"

    def test_cache_creation_instantiation(self):
        """Test CacheCreation can be instantiated."""
        from llm_proxy.models import CacheCreation

        cache = CacheCreation(ephemeral_1h_input_tokens=100, ephemeral_5m_input_tokens=50)
        assert cache.ephemeral_1h_input_tokens == 100

    def test_server_tool_usage_instantiation(self):
        """Test ServerToolUsage can be instantiated."""
        from llm_proxy.models import ServerToolUsage

        usage = ServerToolUsage(web_fetch_requests=5, web_search_requests=10)
        assert usage.web_fetch_requests == 5


class TestAllExport:
    """Test that __all__ contains all expected exports."""

    def test_all_contains_internal_models(self):
        """Test __all__ contains internal models."""
        from llm_proxy.models import __all__

        assert "InternalRequest" in __all__
        assert "InternalResponse" in __all__
        assert "RequestMetadata" in __all__

    def test_all_contains_content_blocks(self):
        """Test __all__ contains content blocks."""
        from llm_proxy.models import __all__

        assert "TextBlock" in __all__
        assert "ImageBlock" in __all__
        assert "AudioBlock" in __all__
        assert "ToolUseBlock" in __all__
        assert "ToolResultBlock" in __all__
        assert "ThinkingBlock" in __all__
        assert "ContentBlock" in __all__

    def test_all_contains_conversation_types(self):
        """Test __all__ contains conversation types."""
        from llm_proxy.models import __all__

        assert "Message" in __all__
        assert "SystemMessage" in __all__
        assert "ConversationContext" in __all__

    def test_all_count(self):
        """Test __all__ has expected number of exports."""
        from llm_proxy.models import __all__

        assert len(__all__) == 74
