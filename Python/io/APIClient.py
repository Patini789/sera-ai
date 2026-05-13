"""
APIClient — Send prompts to model hosts and return responses.
Supports Gemini, OpenCode, and local LLM (LM Studio) backends.
"""
from openai import OpenAI
import requests
import numpy as np
import json

from .mcp_client import MCPClient
from ..core.logger import get_logger

logger = get_logger(__name__)


class APIClient:
    def __init__(self, embedding_url: str, embedding_model: str,
                       completion_url: str, completion_model: str,
                       gemini_key: str = None,
                       gemini_models: list[str] | None = None,
                       local_token: str = None,
                       disable_thinking: bool = False,
                       opencode_key: str = None,
                       opencode_models: list[str] = None) -> None:
        self.embed_url = embedding_url
        self.embed_model = embedding_model
        self.complete_url = completion_url
        self.complete_model = completion_model

        self.gemini_key = gemini_key
        # Ordered fallback list — first model with available quota wins
        self.gemini_models = gemini_models or ["gemma-4-26b"]
        self.gemini_model = self.gemini_models[0]  # Active model (may rotate)
        self.local_token = local_token
        self.disable_thinking = disable_thinking
        
        
        self.opencode_key = opencode_key
        self.opencode_models = opencode_models or ["big-pickle"]

        from urllib.parse import urlparse
        parsed = urlparse(completion_url)
        openai_base = f"{parsed.scheme}://{parsed.netloc}/v1"
        
        self.complete_client = OpenAI(
            base_url=openai_base, 
            api_key=local_token if local_token else "lm-studio"
        )

        self.session = requests.Session()

        # ── MCP Client (tools for Gemini function calling) ─────────
        self.mcp_client = MCPClient()
        try:
            self.mcp_client.connect_all()
        except Exception as e:
            logger.error(f"MCP initialization failed (tools won't be available): {e}")

    @staticmethod
    def _parse_inline_think_tags(text: str, currently_in_reasoning: bool):
        """Parse text that may contain inline <think>/<\/think> tags.

        Yields a sequence of:
            - ``"__THINK_OPEN__"``  when a ``<think>`` tag is encountered
            - ``"__THINK_CLOSE__"`` when a ``</think>`` tag is encountered
            - plain-text strings and proper ``<think>``/``</think>`` wrapper
              strings for downstream consumers

        This allows streaming backends to handle models that embed
        reasoning inside the regular ``content`` delta field instead of
        using a dedicated ``reasoning_content`` field.
        """
        import re
        # Fast path — no tags at all
        if '<think>' not in text and '</think>' not in text:
            if currently_in_reasoning:
                # We're inside a reasoning block opened by reasoning_content,
                # and now regular content arrived → close reasoning first.
                yield "\n</think>\n\n"
                yield "__THINK_CLOSE__"
            yield text
            return

        # Split around <think> and </think> tags, keeping them as tokens
        parts = re.split(r'(</?think>)', text)
        for part in parts:
            if not part:
                continue
            if part == '<think>':
                if not currently_in_reasoning:
                    yield "<think>\n"
                    yield "__THINK_OPEN__"
                    currently_in_reasoning = True
            elif part == '</think>':
                if currently_in_reasoning:
                    yield "\n</think>\n\n"
                    yield "__THINK_CLOSE__"
                    currently_in_reasoning = False
            else:
                yield part

    def gemini_complete(self, prompt: str, system_instr: str) -> str:
        """Call the Gemini/Gemma API with automatic model fallback and MCP tool support.

        Implements an agentic loop: if the model requests a functionCall,
        the corresponding MCP tool is executed and the result is sent back
        until the model produces a final text response.

        Tool calls are recorded and injected as ``<lmstudio_tools>`` tags
        so that downstream processors (training_data, conversation_engine,
        curador, etc.) can parse them with the same regex.
        """
        if not self.gemini_key:
            raise ValueError("No Gemini Key")

        # Try each model in fallback order
        last_error = None
        for model in self.gemini_models:
            try:
                result = self._gemini_complete_with_model(model, prompt, system_instr)
                self.gemini_model = model  # Remember the active model
                return result
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"⚠️ Gemini [{model}] quota exhausted (429), trying next model...")
                    last_error = e
                    continue
                raise  # Non-quota errors propagate immediately

        # All models exhausted
        raise ValueError(
            f"All Gemini models exhausted quota: {self.gemini_models}"
        ) from last_error

    def _gemini_complete_with_model(
        self, model: str, prompt: str, system_instr: str
    ) -> str:
        """Execute a single Gemini API call with the specified model."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_key,
        }

        # ── Build initial contents ────────────────────────────────
        # Gemma models don't support systemInstruction, so we prepend it
        if "gemma" in model.lower():
            final_prompt = f"{system_instr}\n\n{prompt}"
            contents = [{
                "role": "user",
                "parts": [{"text": final_prompt}],
            }]
        else:
            contents = [{
                "role": "user",
                "parts": [{"text": prompt}],
            }]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2000,
            },
        }

        # Gemini-family models support systemInstruction natively
        if "gemma" not in model.lower():
            payload["systemInstruction"] = {
                "parts": [{"text": system_instr}]
            }

        # Enable thinking for models that support it (only Gemini 2.5 family)
        # Gemma 4, Gemini 3, Gemini 2.0 do NOT support thinkingConfig
        if not self.disable_thinking and "2.5" in model:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 2048
            }

        # ── Attach MCP tools as Gemini functionDeclarations ───────
        gemini_tools = self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            payload["tools"] = [{
                "functionDeclarations": gemini_tools
            }]

        # ── Agentic loop (function calling) ───────────────────────
        extracted_tools: list[dict] = []
        max_rounds = 10  # Safety limit to prevent infinite loops

        for round_num in range(max_rounds):
            logger.info(f"Gemini [{model}] round {round_num + 1}...")
            resp = self.session.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code != 200:
                logger.error(f"Google Error: {resp.status_code} - {resp.text}")
            resp.raise_for_status()

            data = resp.json()
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])

            # DEBUG: Uncomment to see the raw API response structure
            # print("🔍 RAW GEMINI RESPONSE:\n", json.dumps(data, indent=2, ensure_ascii=False)[:2000])

            # Check if the model wants to call a function
            function_calls = [p for p in parts if "functionCall" in p]

            if function_calls:
                # ── Execute each function call via MCP ────────────
                # Append model's response to the conversation
                payload["contents"].append(candidate["content"])

                function_response_parts = []
                for fc_part in function_calls:
                    fc = fc_part["functionCall"]
                    fn_name = fc.get("name", "")
                    fn_args = fc.get("args", {})
                    fn_id = fc.get("id", "")

                    logger.info(f"🔧 Gemini requests tool: {fn_name}({fn_args})")

                    # Execute via MCP
                    result = self.mcp_client.call_tool(fn_name, fn_args)

                    # Record for <lmstudio_tools> injection
                    extracted_tools.append({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": str(result)[:500],  # Truncate for dataset
                    })

                    # Build the functionResponse part
                    fr_part = {
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"result": str(result)},
                        }
                    }
                    # Include id if the API provided one
                    if fn_id:
                        fr_part["functionResponse"]["id"] = fn_id

                    function_response_parts.append(fr_part)

                # Send function results back to the model
                payload["contents"].append({
                    "role": "user",
                    "parts": function_response_parts,
                })
                continue  # Next round

            # ── No function call → extract final text ─────────────
            # Gemini API marks thinking parts with "thought": true
            thinking_parts = []
            text_parts = []
            for p in parts:
                if "text" not in p:
                    continue
                if p.get("thought", False):
                    thinking_parts.append(p["text"])
                else:
                    text_parts.append(p["text"])

            final_text = ""

            # Wrap thinking in <think> tags for downstream compatibility
            thinking_content = "".join(thinking_parts).strip()
            if thinking_content and not self.disable_thinking:
                final_text += f"<think>\n{thinking_content}\n</think>\n\n"

            final_text += "".join(text_parts).strip()

            # Inject tool calls as <lmstudio_tools> for compatibility
            if extracted_tools:
                hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
                final_text += f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>"

            return final_text

        # If we exhausted the loop, return whatever text we got
        return "Error: Se alcanzó el límite de rondas de function calling."

    def gemini_complete_stream(self, prompt: str, system_instr: str, **kwargs):
        """Generator: yield each token from Gemini API with automatic model fallback and MCP tool support."""
        if not self.gemini_key:
            raise ValueError("No Gemini Key")

        # Try each model in fallback order
        last_error = None
        for model in self.gemini_models:
            try:
                gen = self._gemini_complete_with_model_stream(model, prompt, system_instr, **kwargs)
                try:
                    first_chunk = next(gen)
                    yield first_chunk
                except StopIteration:
                    return
                yield from gen
                return
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"⚠️ Gemini [{model}] quota exhausted (429), trying next model...")
                    last_error = e
                    continue
                raise

        raise ValueError(f"All Gemini models exhausted quota: {self.gemini_models}") from last_error

    def _gemini_complete_with_model_stream(
        self, model: str, prompt: str, system_instr: str, _contents=None, **kwargs
    ):
        """Generator: execute a single Gemini API stream call with the specified model."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_key,
        }

        if _contents is None:
            if "gemma" in model.lower():
                final_prompt = f"{system_instr}\n\n{prompt}"
                contents = [{
                    "role": "user",
                    "parts": [{"text": final_prompt}],
                }]
            else:
                contents = [{
                    "role": "user",
                    "parts": [{"text": prompt}],
                }]
        else:
            contents = _contents

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2000,
            },
        }

        if "gemma" not in model.lower():
            payload["systemInstruction"] = {
                "parts": [{"text": system_instr}]
            }

        if not self.disable_thinking and "2.5" in model:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 2048
            }

        gemini_tools = getattr(self, 'mcp_client', None) and self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            payload["tools"] = [{
                "functionDeclarations": gemini_tools
            }]

        logger.info(f"🚀 Sending to Gemini stream [{model}]")
        resp = self.session.post(url, headers=headers, json=payload, stream=True, timeout=60)
        resp.raise_for_status()

        in_reasoning = False
        assistant_parts = []
        function_calls = []

        for line in resp.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if not line_str.startswith('data:'):
                continue
            
            json_str = line_str[5:].strip()
            if not json_str:
                continue

            try:
                item = json.loads(json_str)
                candidates = item.get("candidates", [])
                if not candidates:
                    continue
                    
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue

                for p in parts:
                    if "functionCall" in p:
                        function_calls.append(p)
                        assistant_parts.append(p)
                    elif "text" in p:
                        text = p["text"]
                        if p.get("thought", False):
                            if not in_reasoning:
                                in_reasoning = True
                                yield "<think>\n"
                            yield text
                        else:
                            # Handle <think> tags embedded in content text
                            for chunk in self._parse_inline_think_tags(text, in_reasoning):
                                if chunk == "__THINK_OPEN__":
                                    in_reasoning = True
                                elif chunk == "__THINK_CLOSE__":
                                    in_reasoning = False
                                else:
                                    yield chunk
                        # Accumulate text parts just in case we need to add to contents
                        # (Normally we only add to contents if there are function calls)
                        assistant_parts.append(p)

            except json.JSONDecodeError:
                pass

        if in_reasoning:
            yield "\n</think>\n\n"

        if function_calls:
            # Reconstruct the assistant's message
            payload["contents"].append({
                "role": "model",
                "parts": assistant_parts
            })

            extracted_tools = []
            function_response_parts = []

            for fc_part in function_calls:
                fc = fc_part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                fn_id = fc.get("id", "")

                logger.info(f"🔧 Gemini stream requests tool: {fn_name}({fn_args})")
                result = self.mcp_client.call_tool(fn_name, fn_args)

                extracted_tools.append({
                    "type": "tool_call",
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": str(result)[:500],
                })

                fr_part = {
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": str(result)},
                    }
                }
                if fn_id:
                    fr_part["functionResponse"]["id"] = fn_id

                function_response_parts.append(fr_part)

            payload["contents"].append({
                "role": "user",
                "parts": function_response_parts,
            })

            hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
            yield f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>\n\n"

            # Recursive call
            yield from self._gemini_complete_with_model_stream(model, prompt, system_instr, _contents=payload["contents"], **kwargs)
        
    def embed(self, text: str):
        payload = {"input": text, "model": self.embed_model}
        headers = {"Content-Type": "application/json"}
        if self.local_token:
            headers["Authorization"] = f"Bearer {self.local_token}"

        try:
            resp = self.session.post(self.embed_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            emb = resp.json()["data"][0]["embedding"]
            return np.array(emb, dtype=np.float32)
        except requests.RequestException as e:
            logger.error(f"Error during embedding request: {e}")
            return np.zeros(1)

    def local_complete(self, messages: list, **kwargs) -> str:
        temperature = kwargs.get('temperature', 0.6)
        openai_url = self.complete_url.replace("/api/v1/chat", "/v1/chat/completions")

        # Convert Gemini tool format to OpenAI tool format
        openai_tools = []
        gemini_tools = getattr(self, 'mcp_client', None) and self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            for g_tool in gemini_tools:
                g = g_tool.copy()
                if "parameters" not in g or not g["parameters"]:
                    g["parameters"] = {"type": "object", "properties": {}}
                elif "type" not in g["parameters"]:
                    g["parameters"]["type"] = "object"
                openai_tools.append({"type": "function", "function": g})

        headers = {"Content-Type": "application/json"}
        if self.local_token:
            headers["Authorization"] = f"Bearer {self.local_token}"

        extracted_tools: list[dict] = []
        max_rounds = 10

        # We will loop for agentic behaviour
        for round_num in range(max_rounds):
            payload = {
                "model": self.complete_model,
                "messages": messages,
                "temperature": temperature
            }
            if openai_tools:
                payload["tools"] = openai_tools

            if not getattr(self, 'disable_thinking', False):
                payload["reasoning_effort"] = "high"

            try:
                resp = self.session.post(openai_url, headers=headers, json=payload, timeout=2400)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error(f"Error during completion request: {e}")
                if e.response is not None:
                    logger.debug(f"Server details: {e.response.text}")
                return "Local server error"

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                messages.append(message)
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_str = tc.get("function", {}).get("arguments", "{}")
                    fn_id = tc.get("id", "")

                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.info(f"🔧 Local requests tool: {fn_name}({fn_args})")
                    result = self.mcp_client.call_tool(fn_name, fn_args)

                    extracted_tools.append({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": str(result)[:500],
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": fn_id,
                        "content": str(result)
                    })
                continue # loop again

            # Extract final text
            final_text = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "")
            
            full_response = ""
            if reasoning and not getattr(self, 'disable_thinking', False):
                full_response += f"<think>\n{reasoning.strip()}\n</think>\n\n"
            full_response += final_text.strip()

            if extracted_tools:
                hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
                full_response += f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>"

            return full_response

        return "Error: Function calling round limit reached (Local)."

    def local_complete_stream(self, messages: list, **kwargs):
        """Generator: yield each token using LM Studio's OpenAI compatible endpoint with MCP."""
        temperature = kwargs.get('temperature', 0.6)
        self.disable_thinking = False

        openai_url = self.complete_url.replace("/api/v1/chat", "/v1/chat/completions")

        # Convert tools
        openai_tools = []
        gemini_tools = getattr(self, 'mcp_client', None) and self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            for g_tool in gemini_tools:
                g = g_tool.copy()
                if "parameters" not in g or not g["parameters"]:
                    g["parameters"] = {"type": "object", "properties": {}}
                elif "type" not in g["parameters"]:
                    g["parameters"]["type"] = "object"
                openai_tools.append({"type": "function", "function": g})

        payload = {
            "model": self.complete_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if openai_tools:
            payload["tools"] = openai_tools
        
        if not getattr(self, 'disable_thinking', False):
            payload["reasoning_effort"] = "high"
        
        headers = {"Content-Type": "application/json"}
        if self.local_token:
            headers["Authorization"] = f"Bearer {self.local_token}"

        try:
            logger.info(f"🚀 Sending to LM Studio (OpenAI format)")
            resp = self.session.post(openai_url, headers=headers, json=payload, stream=True, timeout=2400)
            resp.raise_for_status()

            in_reasoning = False
            tool_calls_buffer = {}
            assistant_content = ""

            for line in resp.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if not line_str.startswith('data:'):
                    continue
                
                json_str = line_str[5:].strip()
                if json_str == '[DONE]':
                    break

                try:
                    item = json.loads(json_str)
                    choices = item.get("choices", [{}])
                    if not choices:
                        continue
                        
                    delta = choices[0].get("delta", {})

                    # 1. Extract reasoning tokens
                    reasoning = delta.get("reasoning_content", "")
                    if reasoning and not getattr(self, 'disable_thinking', False):
                        if not in_reasoning:
                            in_reasoning = True
                            yield "<think>\n"
                        yield reasoning

                    # 2. Extract text content (handle inline <think> tags)
                    content = delta.get("content")
                    if content:
                        for chunk in self._parse_inline_think_tags(content, in_reasoning):
                            if chunk == "__THINK_OPEN__":
                                in_reasoning = True
                            elif chunk == "__THINK_CLOSE__":
                                in_reasoning = False
                            else:
                                if not chunk.startswith("<think>") and not chunk.startswith("\n</think>"):
                                    assistant_content += chunk
                                yield chunk

                    # 3. Accumulate tool calls
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            idx = tc.get("index")
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": ""
                                    }
                                }
                            args_chunk = tc.get("function", {}).get("arguments", "")
                            if args_chunk:
                                tool_calls_buffer[idx]["function"]["arguments"] += args_chunk

                except json.JSONDecodeError:
                    pass

            # Safety close if reasoning didn't end properly
            if in_reasoning:
                yield "\n</think>\n\n"

            # 4. Handle completed tool calls
            if tool_calls_buffer:
                formatted_tool_calls = list(tool_calls_buffer.values())
                
                # Append assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_content if assistant_content else "",
                    "tool_calls": formatted_tool_calls
                })

                extracted_tools = []
                for tc in formatted_tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args_str = tc["function"]["arguments"]
                    fn_id = tc["id"]
                    try:
                        fn_args = json.loads(fn_args_str)
                    except:
                        fn_args = {}
                        
                    logger.info(f"🔧 Local stream requests tool: {fn_name}({fn_args})")
                    result = self.mcp_client.call_tool(fn_name, fn_args)
                    
                    extracted_tools.append({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": str(result)[:500],
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": fn_id,
                        "content": str(result)
                    })

                hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
                yield f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>\n\n"
                
                # Recursive call to let the model generate the final response
                yield from self.local_complete_stream(messages, **kwargs)

        except Exception as e:
            logger.error(f"Error in local_complete_stream: {e}")
            yield f"Local connection error: {e}"

    def opencode_complete(self, prompt: str, system_instr: str) -> str:
        """Call the OpenCode (OpenAI protocol) API with automatic model fallback and MCP tool support."""
        if not getattr(self, 'opencode_key', None):
            raise ValueError("No OpenCode Key")

        # Try each model in fallback order
        last_error = None
        for model in self.opencode_models:
            try:
                result = self._opencode_complete_with_model(model, prompt, system_instr)
                return result
            except requests.HTTPError as e:
                # Catch 429 Too Many Requests
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"⚠️ OpenCode [{model}] quota exhausted (429), trying next model...")
                    last_error = e
                    continue
                raise  # Non-quota errors propagate immediately

        raise ValueError(f"All OpenCode models exhausted quota: {self.opencode_models}") from last_error

    def _opencode_complete_with_model(self, model: str, prompt: str, system_instr: str) -> str:
        """Execute a single OpenCode API call with the specified model and Agentic Loop."""
        url = "https://opencode.ai/zen/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.opencode_key}"
        }

        # Build standard OpenAI message format
        messages = [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": prompt}
        ]

        # Convert Gemini tool format to OpenAI tool format
        openai_tools = []
        gemini_tools = self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            for g_tool in gemini_tools:
                # Ensure parameters have the required format for OpenAI
                if "parameters" not in g_tool or not g_tool["parameters"]:
                    g_tool["parameters"] = {
                        "type": "object",
                        "properties": {}
                    }
                elif "type" not in g_tool["parameters"]:
                    g_tool["parameters"]["type"] = "object"
                
                openai_tools.append({
                    "type": "function",
                    "function": g_tool
                })

        extracted_tools: list[dict] = []
        max_rounds = 10

        for round_num in range(max_rounds):
            logger.info(f"OpenCode [{model}] round {round_num + 1}...")
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.9,
                "max_tokens": 2000
            }
            
            if openai_tools:
                payload["tools"] = openai_tools

            resp = self.session.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code != 200:
                logger.error(f"OpenCode Error: {resp.status_code} - {resp.text}")
            resp.raise_for_status()

            data = resp.json()

            logger.debug(f"DEBUG: RESPUESTA COMPLETA DE OPENCODE: {json.dumps(data, indent=2, ensure_ascii=False)}")

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            reasoning = message.get("reasoning", "")

            # Check for tool calls (OpenAI standard)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # OpenAI requires the tool request to be in the message history
                messages.append(message)

                # Execute each tool call
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_str = tc.get("function", {}).get("arguments", "{}")
                    fn_id = tc.get("id", "")

                    # Parse JSON arguments
                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    # Execute via MCP
                    result = self.mcp_client.call_tool(fn_name, fn_args)

                    # Record for dataset compatibility
                    extracted_tools.append({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": str(result)[:500], 
                    })

                    # OpenAI requires tool responses with the matching call ID
                    messages.append({
                        "role": "tool",
                        "tool_call_id": fn_id,
                        "content": str(result)
                    })

                continue

            # ── No function call → extract final text ─────────────
            final_text = message.get("content", "") or ""

            if reasoning:
                final_text = f"<think>\n{reasoning}\n</think>\n\n" + final_text

            # Inject tool calls for compatibility with UI/logs
            if extracted_tools:
                hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
                final_text += f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>"

            return final_text

        return "Error: Function calling round limit reached (OpenCode)."

    def opencode_complete_stream(self, prompt: str, system_instr: str, **kwargs):
        """Generator: yield each token from OpenCode (OpenAI protocol) API with automatic model fallback and MCP tool support."""
        if not getattr(self, 'opencode_key', None):
            raise ValueError("No OpenCode Key")

        # Try each model in fallback order
        last_error = None
        for model in self.opencode_models:
            try:
                gen = self._opencode_complete_with_model_stream(model, prompt, system_instr, **kwargs)
                try:
                    first_chunk = next(gen)
                    yield first_chunk
                except StopIteration:
                    # Generator finished without yielding anything
                    return
                yield from gen
                return
            except requests.HTTPError as e:
                # Catch 429 Too Many Requests
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"⚠️ OpenCode [{model}] quota exhausted (429), trying next model...")
                    last_error = e
                    continue
                raise  # Non-quota errors propagate immediately

        raise ValueError(f"All OpenCode models exhausted quota: {self.opencode_models}") from last_error

    def _opencode_complete_with_model_stream(self, model: str, prompt: str, system_instr: str, _messages=None, **kwargs):
        """Generator: execute a single OpenCode API stream call with the specified model and Agentic Loop."""
        url = "https://opencode.ai/zen/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.opencode_key}"
        }

        # Build standard OpenAI message format if _messages is not provided
        messages = _messages or [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": prompt}
        ]

        # Convert Gemini tool format to OpenAI tool format
        openai_tools = []
        gemini_tools = getattr(self, 'mcp_client', None) and self.mcp_client.get_gemini_function_declarations()
        if gemini_tools:
            for g_tool in gemini_tools:
                g = g_tool.copy()
                if "parameters" not in g or not g["parameters"]:
                    g["parameters"] = {"type": "object", "properties": {}}
                elif "type" not in g["parameters"]:
                    g["parameters"]["type"] = "object"
                
                openai_tools.append({
                    "type": "function",
                    "function": g
                })

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.9),
            "max_tokens": 2000,
            "stream": True
        }
        
        if openai_tools:
            payload["tools"] = openai_tools

        logger.info(f"🚀 Sending to OpenCode stream [{model}]")
        resp = self.session.post(url, headers=headers, json=payload, stream=True, timeout=60)
        resp.raise_for_status()

        in_reasoning = False
        tool_calls_buffer = {}
        assistant_content = ""

        for line in resp.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if not line_str.startswith('data:'):
                continue
            
            json_str = line_str[5:].strip()
            if json_str == '[DONE]':
                break

            try:
                item = json.loads(json_str)
                choices = item.get("choices", [{}])
                if not choices:
                    continue
                    
                delta = choices[0].get("delta", {})

                # OpenCode reasoning handling
                reasoning = delta.get("reasoning", delta.get("reasoning_content", ""))
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield "<think>\n"
                    yield reasoning

                # Text content (handle inline <think> tags)
                content = delta.get("content")
                if content:
                    for chunk in self._parse_inline_think_tags(content, in_reasoning):
                        if chunk == "__THINK_OPEN__":
                            in_reasoning = True
                        elif chunk == "__THINK_CLOSE__":
                            in_reasoning = False
                        else:
                            if not chunk.startswith("<think>") and not chunk.startswith("\n</think>"):
                                assistant_content += chunk
                            yield chunk

                # Tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index")
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": ""
                                }
                            }
                        args_chunk = tc.get("function", {}).get("arguments", "")
                        if args_chunk:
                            tool_calls_buffer[idx]["function"]["arguments"] += args_chunk

            except json.JSONDecodeError:
                pass

        # Safety close if reasoning didn't end properly
        if in_reasoning:
            yield "\n</think>\n\n"

        # Handle completed tool calls
        if tool_calls_buffer:
            formatted_tool_calls = list(tool_calls_buffer.values())
            
            messages.append({
                "role": "assistant",
                "content": assistant_content if assistant_content else "",
                "tool_calls": formatted_tool_calls
            })

            extracted_tools = []
            for tc in formatted_tool_calls:
                fn_name = tc["function"]["name"]
                fn_args_str = tc["function"]["arguments"]
                fn_id = tc["id"]
                try:
                    fn_args = json.loads(fn_args_str)
                except:
                    fn_args = {}
                    
                logger.info(f"🔧 OpenCode stream requests tool: {fn_name}({fn_args})")
                result = self.mcp_client.call_tool(fn_name, fn_args)
                
                extracted_tools.append({
                    "type": "tool_call",
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": str(result)[:500],
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": fn_id,
                    "content": str(result)
                })

            hidden_tools_json = json.dumps(extracted_tools, ensure_ascii=False)
            yield f"\n\n<lmstudio_tools>{hidden_tools_json}</lmstudio_tools>\n\n"
            
            # Recursive call
            yield from self._opencode_complete_with_model_stream(model, prompt, system_instr, _messages=messages, **kwargs)