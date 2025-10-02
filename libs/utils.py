import hashlib
import base64
import json

def uniq_hash(attrs):
    attrs_str = [f"{attr}" for attr in attrs]
    joined = ''.join(sorted(attrs_str))
    md5_hash = hashlib.md5(joined.encode('utf-8')).digest()
    base64_encoded = base64.urlsafe_b64encode(md5_hash).decode('utf-8')
    return base64_encoded.rstrip('=')


def get_api_client(model_id, inference_url=None, api_key=""):
    import os
    import boto3
    import anthropic
    from openai import OpenAI

    from . import parallel_openai, parallel_bedrock, parallel_claude

    if inference_url:
        client = OpenAI(
            api_key=api_key,
            base_url=inference_url,
        )
        method = lambda *args, **kwargs: parallel_openai.run_batch(*args, **kwargs, api_type="chat_completions")
    elif 'bedrock' in model_id.lower():
        client = boto3.client(service_name='bedrock-runtime')
        method = parallel_bedrock.run_batch
    elif 'claude' in model_id.lower():
        client = anthropic.Anthropic()
        method = parallel_claude.run_batch
    else:
        client = OpenAI(
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"),
        )
        method = lambda *args, **kwargs: parallel_openai.run_batch(*args, **kwargs, api_type="responses")

    return client, method


def get_api_settings(client,
                     temperature=1.0,
                     top_p=1.0,
                     max_tokens=256,
                     decoding_json_schema=None,
                     enable_thinking=False,
                     enable_web_search=False):

    import anthropic
    from openai import OpenAI

    if isinstance(client, OpenAI):  # **************************** OPENAI RESPONSE API **************************
        if "api.openai" in str(client.base_url):
            settings = {
                "max_output_tokens": max_tokens,
            }

            if enable_thinking:
                settings['reasoning'] = {"effort": "medium", "summary": "auto"}
            else:
                settings['temperature'] = temperature
                settings['top_p'] = top_p

            if enable_web_search:
                settings['tools'] = [{"type": "web_search"}]

            if decoding_json_schema:
                schema_obj = json.loads(decoding_json_schema)
                name = schema_obj.get('title', 'val_bench_json_schema')

                settings['text'] = {
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": schema_obj
                    }
                }
        else:      # **************************** OPENAI client compatible API ***********************************
            settings = {
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_tokens,
            }

            if decoding_json_schema:
                schema_obj = json.loads(decoding_json_schema)
                name = schema_obj.get('title', 'val_bench_json_schema')

                settings['response_format'] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema_obj
                    }
                }
    elif isinstance(client, anthropic.Anthropic):  # ************************** ANTHROPIC ************************
        if enable_thinking:
            settings = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 4096
                },
                "max_tokens": 4096 + max_tokens,
            }
        else:
            settings = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        if enable_web_search:
            settings["tools"]  = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2
                }
            ]
            settings['betas'] = ["interleaved-thinking-2025-05-14"]

    elif 'boto' in str(type(client)):  # ********************************* BEDROCK *******************************
        if enable_thinking:
            settings = {
                "inferenceConfig": {
                    "temperature": 1.0,
                    "topP": 1.0,
                    "maxTokens": 4096 + max_tokens,
                },
                "additionalModelRequestFields": {
                    "reasoning_config": {
                        "type": "enabled",
                        "budget_tokens": 4096
                    },
                }
            }
        else:
            settings = {
                "inferenceConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxTokens": max_tokens,
                }
            }
    else:
        raise TypeError(f"Unsupported API client: {type(client)}")

    return settings

