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

func TestOpenAICompatibility(t *testing.T) {
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

	t.Run("it should handle basic json schema parsing and throw proper validation errors", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{
					{
						Text: "You are Alfred Pennyworth",
						Type: "system",
					},
				}),
				openai.UserMessage("Hello"),
			},
			Temperature: openai.Float(0.4),
		}
		setEpisodeID(req, episodeID.String())

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "JSON Schema validation failed")
	})

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
}
