"""Wrapper around ForefrontAI APIs."""
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env


class ForefrontAI(LLM):
    """Wrapper around ForefrontAI large language models.

    To use, you should have the environment variable ``FOREFRONTAI_API_KEY``
    set with your API key.

    Example:
        .. code-block:: python

            from langchain.llms import ForefrontAI
            forefrontai = ForefrontAI(endpoint_url="")
    """

    endpoint_url: str = ""
    """Model name to use."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    length: int = 256
    """The maximum number of tokens to generate in the completion."""

    top_p: float = 1.0
    """Total probability mass of tokens to consider at each step."""

    top_k: int = 40
    """The number of highest probability vocabulary tokens to
    keep for top-k-filtering."""

    repetition_penalty: int = 1
    """Penalizes repeated tokens according to frequency."""

    forefrontai_api_key: Optional[str] = None

    base_url: Optional[str] = None
    """Base url to use, if None decides based on model name."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        forefrontai_api_key = get_from_dict_or_env(
            values, "forefrontai_api_key", "FOREFRONTAI_API_KEY"
        )
        values["forefrontai_api_key"] = forefrontai_api_key
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling ForefrontAI API."""
        return {
            "temperature": self.temperature,
            "length": self.length,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"endpoint_url": self.endpoint_url}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "forefrontai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to ForefrontAI's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = ForefrontAI("Tell me a joke.")
        """
        response = requests.post(
            url=self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.forefrontai_api_key}",
                "Content-Type": "application/json",
            },
            json={"text": prompt, **self._default_params},
        )
        response_json = response.json()
        text = response_json["result"][0]["completion"]
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
