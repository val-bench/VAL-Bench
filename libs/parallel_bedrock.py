from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError

import time

def run_batch(bedrock_client, model_arn_id, system_prompt, input_messages_batch, settings, parallelism=10, num_retries=1):
    if parallelism > 10:
        print(f"Forcing parallelism = 10 instead of {parallelism}")
        parallelism = min(parallelism, 10)

    system_prompt_formatted = [{"text": system_prompt}]

    messages_formatted_batch = []
    for messages in input_messages_batch:
        messages_formatted = []
        for message in messages:
            messages_formatted.append({"role": message['role'], "content": [{'text': message['content']}] })
        messages_formatted_batch.append(messages_formatted)

    def process_item(item_with_seq):
        seq_id, inp = item_with_seq
        complete = False

        tries = 0

        while not complete and tries <= num_retries:
            tries += 1

            try:
                result = bedrock_client.converse(
                    modelId=model_arn_id,
                    system=system_prompt_formatted,
                    messages=inp,
                    **settings
                )

                complete = True
            except ClientError as e:
                print(f"Error processing item {seq_id}")
                print(e)
                print(f"Sleeping for 5 seconds and trying again...\n\n{inp}")

                if tries > num_retries:
                    print(f"Exhausted retries for item {seq_id}, returning default value")
                else:
                    time.sleep(5)
                    continue


        response_text = "No response"
        reasoning_content = ""
        try:
            for content in result['output']['message']['content']:
                if 'reasoningContent' in content:
                    reasoning_content = content['reasoningContent']['reasoningText']['text']
                elif 'text' in content:
                    response_text = content['text'].strip()
        except Exception as e:
            print(f"Error: {e}")

        return response_text, reasoning_content

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        enumerated_inputs = list(enumerate(messages_formatted_batch))
        results = list(executor.map(process_item, enumerated_inputs))

        return ([{'content': response_text, 'role': 'assistant'} for response_text, _ in results],
                [reasoning_content for _, reasoning_content in results],
                [[]] * len(results)
               )