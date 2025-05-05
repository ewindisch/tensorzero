package tests

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	client openai.Client
	ctx    context.Context
)

func TestMain(m *testing.M) {
	ctx = context.Background()
	client = openai.NewClient(
		option.WithBaseURL("http://127.0.0.1:3000/openai/v1"),
		option.WithAPIKey("donotuse"),
	)

	m.Run()
}

// systemAssistant builds a system‑role message with Tensorflow's assistant_name
func systemAssistant(name string) openai.ChatCompletionMessageParamUnion {
	n := openai.SystemMessage(name)
	n.OfSystem.WithExtraFields(
		map[string]any{
			"content": []any{
				map[string]any{
					"assistant_name": name,
				},
			},
		},
	)
	return n
}

func setEpisodeID(request *openai.ChatCompletionNewParams, episodeID string) {
	request.WithExtraFields(map[string]any{
		"tensorzero::episode_id": episodeID,
	})
}

func TestOpenAICompatibilityBasicInference(t *testing.T) {
	t.Run("it should perform basic inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
			Temperature: openai.Float(0.4),
		}
		setEpisodeID(req, episodeID.String())

		reqJson, err := req.MarshalJSON()
		require.NoError(t, err)
		t.Logf("%s", reqJson)

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err)

		t.Log(resp.RawJSON())

		var responseEpisodeID string
		json.Unmarshal([]byte(resp.JSON.ExtraFields["episode_id"].Raw()), &responseEpisodeID)

		assert.Equal(t, episodeID.String(), responseEpisodeID)
		assert.Equal(t, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.", resp.Choices[0].Message.Content)
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})

	t.Run("it should handle basic inference with invalid JSON field and throw proper validation errors", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		sysMsg := openai.SystemMessage("Alfred Pennyworth")
		sysMsg.OfSystem.WithExtraFields(
			map[string]any{
				"content": []any{
					map[string]any{
						"name_of_assistant": "Alfred Pennyworth",
					},
				},
			},
		)
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				sysMsg,
				openai.UserMessage("Hello"),
			},
			Temperature: openai.Float(0.4),
		}
		setEpisodeID(req, episodeID.String())

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "JSON Schema validation failed")
	})
}

func TestOpenAICompatibilityStreaming(t *testing.T) {

	t.Run("it should handle streaming inference", func(t *testing.T) {
		expectedText := []string{
			"Wally,",
			" the",
			" golden",
			" retriever,",
			" wagged",
			" his",
			" tail",
			" excitedly",
			" as",
			" he",
			" devoured",
			" a",
			" slice",
			" of",
			" cheese",
			" pizza.",
		}

		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
			Seed: openai.Int(69),
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)

		i := 0
		for stream.Next() {
			chunk := stream.Current()
			assert.Len(t, chunk.Choices, 1)

			if chunk.Choices[0].FinishReason != "" {
				assert.Equal(t, chunk.Choices[0].FinishReason, "stop")
				assert.Empty(t, chunk.Choices[0].Delta.Content)
				assert.Equal(t, chunk.Usage.PromptTokens, int64(10))
				assert.Equal(t, chunk.Usage.CompletionTokens, int64(16))
				assert.Equal(t, chunk.Usage.TotalTokens, int64(26))
				break
			}

			assert.Equal(t, len(chunk.Choices), 1)
			assert.Equal(t, chunk.Choices[0].Delta.Content, expectedText[i])
			assert.Empty(t, chunk.Choices[0].FinishReason)

			i++
		}
	})

	t.Run("it should handle streaming inference with non-existent function", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::does_not_exist",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		assert.Contains(t, stream.Err().Error(), "Unknown function: does_not_exist")
	})

	t.Run("it should handle streaming inference with missing function", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		assert.Contains(t, stream.Err().Error(), "cannot be empty")
	})

	t.Run("it should handle streaming inference with malformed function", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "chatgpt",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		assert.Contains(t, stream.Err().Error(), "`model` field must start with")
	})

	t.Run("it should handle streaming inference with missing model", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		assert.Contains(t, stream.Err().Error(), "missing field `model`")
	})

	t.Run("it should handle streaming inference with malformed input", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		sysMsg := openai.SystemMessage("Alfred Pennyworth")
		sysMsg.OfSystem.WithExtraFields(
			map[string]any{
				"content": []any{
					map[string]any{
						"name_of_assistant": "Alfred Pennyworth",
					},
				},
			},
		)

		req := &openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				sysMsg,
				openai.UserMessage("Hello"),
			},
			Model: "tensorzero::function_name::basic_test",
		}
		setEpisodeID(req, episodeID.String())

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		assert.Contains(t, stream.Err().Error(), "JSON Schema validation failed")
	})
}

func TestOpenAICompatibilityToolInference(t *testing.T) {

	t.Run("it should handle tool call inference", func(T *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::weather_helper",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hi I'm visiting Brooklyn from Brazil. What's the weather?"),
			},
			Temperature: openai.Float(0.4),
			TopP:        openai.Float(0.5),
		}
		setEpisodeID(req, episodeID.String())

		reqJson, err := req.MarshalJSON()
		require.NoError(t, err)
		t.Logf("%s", reqJson)

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err)

		t.Log(resp.RawJSON())

		var responseEpisodeID string
		json.Unmarshal([]byte(resp.JSON.ExtraFields["episode_id"].Raw()), &responseEpisodeID)

		assert.Equal(t, resp.Model, "tensorzero::function_name::weather_helper::variant_name::variant")

		// Check that content is empty and tool calls exist
		assert.Empty(t, resp.Choices[0].Message.Content)
		assert.NotEmpty(t, resp.Choices[0].Message.ToolCalls)

		// Verify tool calls
		toolCalls := resp.Choices[0].Message.ToolCalls
		assert.Len(t, toolCalls, 1)

		toolCall := toolCalls[0]
		assert.Equal(t, "function", string(toolCall.Type))
		assert.Equal(t, "get_temperature", toolCall.Function.Name)
		assert.Equal(t, `{"location":"Brooklyn","units":"celsius"}`, toolCall.Function.Arguments)

		// Verify usage metrics
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	})
}

func TestOpenAICompatibilityMalformedToolInference(t *testing.T) {
	t.Run("it should handle malformed tool call inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::weather_helper",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant("Alfred Pennyworth"),
				openai.UserMessage("Hi I'm visiting Brooklyn from Brazil. What's the weather?"),
			},
			PresencePenalty: openai.Float(0.5),
		}
		setEpisodeID(req, episodeID.String())

		req.WithExtraFields(map[string]any{
			"tensorzero::variant_name": "bad_tool",
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err)

		assert.Equal(t, "tensorzero::function_name::weather_helper::variant_name::bad_tool", resp.Model)
		assert.Empty(t, resp.Choices[0].Message.Content)
		assert.NotEmpty(t, resp.Choices[0].Message.ToolCalls)

		// Verify tool calls
		toolCalls := resp.Choices[0].Message.ToolCalls
		assert.Len(t, toolCalls, 1)

		toolCall := toolCalls[0]
		assert.Equal(t, "function", string(toolCall.Type))
		assert.Equal(t, "get_temperature", toolCall.Function.Name)
		assert.Equal(t, `{"location":"Brooklyn","units":"Celsius"}`, toolCall.Function.Arguments)

		// Verify usage metrics
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
	})
}
