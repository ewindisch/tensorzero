package tests

import (
	"context"
	//"encoding/base64"
	//"fmt"
	//"io/ioutil"
	//"path/filepath"
	//"strings"
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

func TestBasicInference(t *testing.T) {
	t.Run("with old model format", func(t *testing.T) {
		episodeID := uuid.New().String()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
			Temperature: openai.Float(0.4),
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID,
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err)

		assert.Equal(t, episodeID, resp.JSON.ExtraFields["tensorzero::episode_id"].Raw())
		assert.Equal(t, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.", resp.Choices[0].Message.Content)
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})
}
