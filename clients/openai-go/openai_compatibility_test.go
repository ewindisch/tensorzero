package tests

import (
	"context"
	//"encoding/base64"
	//fmt"
	//"io/ioutil"
	//"path/filepath"
	//"strings"
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	//"github.com/openai/openai-go/packages/resp"
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

/*type TensorZeroSystemMessage struct {
	Name string `json:"assistant_name"`
	JSON struct {
		Name resp.Field
	} `json:"-"`
}*/

// systemAssistant builds a system‑role message acceptable to the
// TensorZero proxy: { "role": "system", "assistant_name": "<name>" }.
func systemAssistant(t *testing.T, name string) openai.ChatCompletionMessageParamUnion {
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
	jsonBytes, err := n.MarshalJSON()
	require.NoError(t, err)
	t.Logf("%s", jsonBytes)
	return n
}

func TestOpenAICompatibility(t *testing.T) {
	t.Run("it should perform basic inference with old model format", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		req := &openai.ChatCompletionNewParams{
			Model: "tensorzero::function_name::basic_test",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemAssistant(t, "Alfred Pennyworth"),
				openai.UserMessage("Hello"),
			},
			Temperature: openai.Float(0.4),
			/*ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONObject: &openai.ResponseFormatJSONObjectParam{
					Type: "json_object",
				},
			},*/
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

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

	t.Run("it should perform basic inference", func(t *testing.T) {
		/*const messages: ChatCompletionMessageParam[] = [
		  {
			role: "system",
			content: [
			  {
				// @ts-expect-error - custom TensorZero property
				assistant_name: "Alfred Pennyworth",
			  },
			],
		  },
		  { role: "user", content: "Hello" },
		];

		const episodeId = uuidv7();
		const result = await client.chat.completions.create({
		  messages,
		  model: "tensorzero::function_name::basic_test",
		  temperature: 0.4,
		  // @ts-expect-error - custom TensorZero property
		  "tensorzero::episode_id": episodeId,
		});

		// @ts-expect-error - custom TensorZero property
		expect(result.episode_id).toBe(episodeId);

		expect(result.choices[0].message.content).toBe(
		  "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
		);

		const usage = result.usage;
		expect(usage?.prompt_tokens).toBe(10);
		expect(usage?.completion_tokens).toBe(10);
		expect(usage?.total_tokens).toBe(20);
		expect(result.choices[0].finish_reason).toBe("stop");*/
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
		req.WithExtraFields(map[string]interface{}{
			"tensorzero::episode_id": episodeID.String(),
			"response_schema": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "string",
				},
			},
		})

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "JSON Schema validation failed")
	})
}
