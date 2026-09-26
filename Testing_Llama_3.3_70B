#  Step 1: Install vLLM
# Step 2: Launch Local API Server
# Step 3: Send Request and Measure Time
{ time curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Llama-3.3-70B-Instruct/",
    "messages": [{"role": "user", "content": "What is an LLM?"}],
    "max_tokens": 256
  }' -o response.json; } 2> time_output.txt
# Step 4 Extract metrics and calculate TPS
PROMPT_TOKENS=$(jq '.usage.prompt_tokens' response.json)
OUTPUT_TOKENS=$(jq '.usage.completion_tokens' response.json)
TOTAL_TOKENS=$(jq '.usage.total_tokens' response.json)
TIME=$(grep real time_output.txt | awk '{print $2}' | sed 's/[ms]//g')

echo "Benchmark Results:"
echo "Prompt tokens: $PROMPT_TOKENS"
echo "Output tokens: $OUTPUT_TOKENS"
echo "Total tokens: $TOTAL_TOKENS"
echo "Time taken: ${TIME}s"
echo "TPS: $(echo "scale=2; $TOTAL_TOKENS / $TIME" | bc)"
