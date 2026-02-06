#!/bin/bash
# Pre-flight health check for vLLM and Ollama services
#
# Usage: ./scripts/health_check.sh [vllm_url] [ollama_url]

VLLM_URL="${1:-http://localhost:8000/v1}"
OLLAMA_URL="${2:-http://localhost:11434}"

echo "=================================="
echo "  Service Health Check"
echo "=================================="

# Check vLLM
echo -n "  vLLM ($VLLM_URL): "
VLLM_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$VLLM_URL/models" 2>/dev/null)
if [ "$VLLM_STATUS" = "200" ]; then
    MODELS=$(curl -s "$VLLM_URL/models" | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin).get('data',[])))" 2>/dev/null)
    echo "OK (models: $MODELS)"
else
    echo "FAILED (HTTP $VLLM_STATUS)"
fi

# Check Ollama
echo -n "  Ollama ($OLLAMA_URL): "
OLLAMA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$OLLAMA_URL/api/tags" 2>/dev/null)
if [ "$OLLAMA_STATUS" = "200" ]; then
    MODELS=$(curl -s "$OLLAMA_URL/api/tags" | python3 -c "import sys,json; print(', '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null)
    echo "OK (models: $MODELS)"
else
    echo "FAILED (HTTP $OLLAMA_STATUS)"
fi

echo "=================================="
