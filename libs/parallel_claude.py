from concurrent.futures import ThreadPoolExecutor

def _get_response(message):
    response_items = [getattr(content, 'text', "") for content in message.content]
    return "".join(response_items)

def _get_thinking(message):
    thinking_items = [getattr(content, 'thinking', "") for content in message.content]
    return "".join(thinking_items)

def _get_citations(message):
    return [{'title': citation.title, 'url': citation.url, 'text': citation.cited_text}
                 for content in message.content
                 for citation in (getattr(content, 'citations', []) or [])]


def run_batch(claude_client, model, system_prompt, input_messages_batch, settings,
              parallelism=10):
    if parallelism > 10:
        print(f"Forcing parallelism = 10 instead of {parallelism}")
        parallelism = min(parallelism, 10)

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        def process_item(item_with_seq):
            seq_id, inp = item_with_seq

            try:
                message = claude_client.beta.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=inp,
                    **settings
                )

                response_text = _get_response(message)
                reasoning_content = _get_thinking(message)
                citations = _get_citations(message)
            except Exception as e:
                print(f"Error: {e}")
                response_text = "No response"
                reasoning_content = ""
                citations = []

            return response_text, reasoning_content, citations
            
        enumerated_inputs = list(enumerate(input_messages_batch))
        results = list(executor.map(process_item, enumerated_inputs))

        return ([{'content': response_text, 'role': 'assistant'} for response_text, _, _ in results],
                [reasoning_content for _, reasoning_content, _ in results],
                [citation for _, _, citation in results])
