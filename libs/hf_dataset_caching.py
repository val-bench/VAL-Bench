from functools import wraps
from typing import Callable, Any
from datasets import Dataset, DatasetDict, load_from_disk


def s3_cache(s3_path: str):
    """
    Decorator to cache HuggingFace datasets in S3.

    Args:
        s3_path (str): S3 path for caching (e.g., 's3://my-bucket/cache/')

    Usage:
        @s3_cache('s3://my-bucket/dataset-cache/')
        def create_dataset(self, data_source: str, unique_id: str) -> Dataset:
            # Your dataset creation logic here
            return dataset
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract unique_id from kwargs
            unique_id = kwargs.get('unique_id')
            if not unique_id:
                raise ValueError("Method must have 'unique_id' kwarg for S3 caching")

            # Create unique cache path
            method_name = func.__name__
            cache_key = f"{method_name}_{unique_id}"
            s3_cache_path = f"{s3_path.rstrip('/')}/{cache_key}"

            print(f"Checking S3 cache at: {s3_cache_path}")

            # Try to load from cache
            try:
                load_from_disk(s3_cache_path)
                print(f"Cache hit! Loaded dataset from {s3_cache_path}")
                return s3_cache_path
            except Exception:
                # Cache miss - execute the original function
                print(f"Cache miss. Executing {method_name}")
                result = func(*args, **kwargs)

                # Validate result is a HuggingFace dataset
                if not isinstance(result, (Dataset, DatasetDict)):
                    raise ValueError(f"Function {method_name} must return a Dataset or DatasetDict")

                # Save to cache
                result.save_to_disk(s3_cache_path)
                print(f"Dataset cached to {s3_cache_path}")

                return s3_cache_path

        return wrapper

    return decorator
