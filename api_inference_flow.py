from metaflow import (FlowSpec, step, Parameter, current, pypi, retry)

S3_PATH = "local_data/api_inference/cache"


class ApiInferenceFlow(FlowSpec):
    dataset_path = Parameter('dataset_path', required=True,
                            help="HF Dataset ID or S3 dataset path")

    model_id = Parameter('model_id', required=True,
                         help="Model ID (e.g., gpt-4o, anthropic.claude-3-sonnet-20240229, etc.)")

    inference_url = Parameter('inference_url', required=False, default="",
                              help="Inference URL for Internal MC vllm instances")

    system_prompt = Parameter('system_prompt', default="", required=False,
                              help="System prompt. Set for chat completion")

    batch_size = Parameter('batch_size', default=128, required=False,
                           help="Batch size for batch inference")

    input_column = Parameter('input_column', required=True,
                             help="The column with conversation prefixes. If system_prompt is set,"
                                  "this is expected to be a JSON conversation")

    output_column = Parameter('output_column', required=True,
                             help="The output column with responses")

    temperature = Parameter('temperature', default=1.0, required=False,
                            help="Temperature for generation")

    top_p = Parameter('top_p', default=0.95, required=False,
                      help="Top P for generation")

    max_tokens = Parameter('max_tokens', default=512, required=False,
                           help="Max generated tokens for response")

    decoding_json_schema = Parameter('decoding_json_schema', required=False, default="",
                                     help="JSON schema for guided decoding")

    output_dataset_path = Parameter('output_dataset_path', required=True,
                           help="S3 or 'local' disk path or HF dataset id for output dataset")

    skip_cache = Parameter('skip_cache', required=False, default=False, type=bool,
                           help="Skip caching")

    enable_web_search = Parameter('enable_web_search', default=False, type=bool,
                      help="Whether to enable web search. Works for OpenAI and Claude APIs only.")

    enable_thinking = Parameter('enable_thinking', default=False, type=bool,
                                help="Must set to true if using a thinking only model. Works for all supported APIs.")


    @pypi(python="3.10", packages={"datasets": "4.0.0", "s3fs": "2025.3.0", "openai": "1.99.9", "anthropic": "0.67.0", "boto3": "1.37.3",
                                   "python-dotenv": "1.1.1" })
    @retry(times=0)
    @step
    def start(self):
        import os

        from dotenv import load_dotenv
        load_dotenv()

        from libs.load import load_from_path
        from libs.save import save_to_path
        from libs.utils import get_api_client, get_api_settings
        from libs.hf_dataset_caching import s3_cache
        from libs.utils import uniq_hash

        dataset = load_from_path(self.dataset_path, token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))

        unique_id_elements = [
            self.dataset_path,
            self.input_column,
            self.output_column,
            self.inference_url,
            self.model_id,
            self.system_prompt,
            self.temperature,
            self.top_p,
            self.max_tokens,
            self.decoding_json_schema,
            self.enable_thinking,
            self.enable_web_search,
        ]

        if self.skip_cache:
            unique_id_elements.append(f"{current.flow_name}-{current.run_id}")

        unique_id = uniq_hash(unique_id_elements)

        @s3_cache(S3_PATH)
        def run_inference(unique_id):
            client, run_method = get_api_client(model_id=self.model_id, inference_url=self.inference_url)
            api_settings = get_api_settings(client,
                                            temperature=self.temperature,
                                            top_p=self.top_p,
                                            max_tokens=self.max_tokens,
                                            decoding_json_schema=self.decoding_json_schema,
                                            enable_thinking=self.enable_thinking,
                                            enable_web_search=self.enable_web_search,
                                            )
            print(f"Using client: {type(client)}")
            print(f"Using settings:\n{api_settings}")

            responses, reasoning_contents, citations = run_method(client,
                       self.model_id,
                       self.system_prompt,
                       dataset[self.input_column],
                       api_settings,
                       self.batch_size)

            ds_response = dataset.add_column(self.output_column, responses)
            if any(reasoning_contents):
                ds_response = ds_response.add_column(f"{self.output_column}_reasoning", reasoning_contents)

            if any(citations):
                ds_response = ds_response.add_column(f"{self.output_column}_citations", citations)

            return ds_response

        responses_ds_path = run_inference(unique_id=unique_id)
        responses_ds = load_from_path(responses_ds_path)
        save_to_path(responses_ds, self.output_dataset_path, token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    ApiInferenceFlow()