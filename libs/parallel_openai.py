from concurrent.futures import ThreadPoolExecutor

from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_output_message import ResponseOutputMessage

def _get_response(response):
    return response.output_text

def _get_thinking(response):
    return "\n".join([response.output[i].summary[j].text
                           for i in range(len(response.output))
                           if isinstance(response.output[i], ResponseReasoningItem)
                           for j in range(len(response.output[i].summary))])

def _get_citations(response):
    output_texts = [response.output[i].content[j]
                 for i in range(len(response.output))
                 if isinstance(response.output[i], ResponseOutputMessage)
                 for j in range(len(response.output[i].content))
                 ]

    citations = [{'url': citation.url, 'title': citation.title, 'text': output_text.text[citation.start_index: citation.end_index]}
                 for output_text in output_texts
                 for citation in getattr(output_text, 'annotations', [])
                 ]

    return citations


def run_batch(openai_client, model, system_prompt, input_messages_batch, settings,
              parallelism=128, api_type="chat_completions"):
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        def process_item(item_with_seq):
            seq_id, inp = item_with_seq

            if api_type == "responses":
                try:
                    response = openai_client.responses.create(
                        model=model,
                        instructions=system_prompt,
                        input=inp,
                        **settings
                    )

                    response_text = response.output_text
                    reasoning_content = _get_thinking(response).strip()
                    citations = _get_citations(response)
                except Exception as e:
                    print(f"Error: {e}")
                    response_text = "No response"
                    reasoning_content = ""
                    citations = []
            else:
                try:
                    completion = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}] + inp,
                        **settings
                    )

                    response_text = completion.choices[0].message.content
                    reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', "")
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