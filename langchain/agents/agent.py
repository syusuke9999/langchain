"""入力を受け入れて、アクションとアクション入力を生成するチェーン"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, root_validator

from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException,
)
from langchain.tools.base import BaseTool
from langchain.utilities.asyncio import asyncio_timeout

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """基本エージェントクラス"""

    @property
    def return_values(self) -> List[str]:
        """エージェントの返値"""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """ 入力に基づいて、何をするかを決定します。

            引数:
                intermediate_steps: これまでにLLMが行ったステップと、
                    観察された内容
                callbacks: 実行するコールバック。
                **kwargs: ユーザーからの入力。

            戻り値:
                使用するツールを指定するアクション。
        """

    @abstractmethod
    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """入力に基づいて何をすべきかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが踏んだステップと、観測結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザーの入力。

                戻り値:
                    使用するツールを指定したアクション。
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """入力キーを返します。

                :meta private:
        """

    def return_stopped_response(
            self,
            early_stopping_method: str,
            intermediate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any,
    ) -> AgentFinish:
        """エージェントが最大反復回数により停止した際のレスポンスを返します。"""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        else:
            raise ValueError(
                f"Got unsupported early_stopping_method `{early_stopping_method}`"
            )

    @classmethod
    def from_llm_and_tools(
            cls,
            llm: BaseLanguageModel,
            tools: Sequence[BaseTool],
            callback_manager: Optional[BaseCallbackManager] = None,
            **kwargs: Any,
    ) -> BaseSingleActionAgent:
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """戻り値 エージェント種別の識別子"""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """エージェントの辞書表現を返します"""
        _dict = super().dict()
        _type = self._agent_type
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        else:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """エージェントを保存する。

                引数:
                    file_path: エージェントを保存するファイルへのパス。

                例:
                .. code-block:: python

                    # エージェント実行者と一緒に作業している場合
                    agent.agent.save(file_path="path/agent.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        return {}


class BaseMultiActionAgent(BaseModel):
    """基本エージェントクラス"""

    @property
    def return_values(self) -> List[str]:
        """エージェントの戻り値"""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """入力に基づいて何をするかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと観測値
                    callbacks: 実行するコールバック
                    **kwargs: ユーザー入力

                戻り値:
                    使用するツールを指定するアクション
        """

    @abstractmethod
    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """入力に基づいて何をするかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと、
                        観察結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザー入力。

                戻り値:
                    使用するツールを指定するアクション。
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
            self,
            early_stopping_method: str,
            intermediate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any,
    ) -> AgentFinish:
        """反復回数が最大に達してエージェントが停止した場合に、応答を返します"""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        else:
            raise ValueError(
                f"Got unsupported early_stopping_method `{early_stopping_method}`"
            )

    @property
    def _agent_type(self) -> str:
        """戻り値 エージェント種別の識別子"""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """エージェントの辞書的表現を返します"""
        _dict = super().dict()
        _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """エージェントを保存する。

        引数:
            file_path: エージェントを保存するファイルへのパス。

        例:
        .. code-block:: python

            # エージェント実行者と一緒に作業している場合
            agent.agent.save(file_path="path/agent.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        return {}


class AgentOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """テキストをエージェントの動作/終了にパースします"""


class LLMSingleActionAgent(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def dict(self, **kwargs: Any) -> Dict:
        """エージェントの辞書的表現を返します。"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """入力に基づいて、何をすべきかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと、
                        観察結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザー入力。

                戻り値:
                    使用するツールを指定するアクション。
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """入力に基づいて、何をすべきかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと、
                        観察結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザー入力。

                戻り値:
                    使用するツールを指定するアクション。
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


class Agent(BaseSingleActionAgent):
    """言語モデルを呼び出し、アクションを決定するクラス。

        これはLLMChainによって駆動されます。LLMChainのプロンプトには、
        エージェントが中間作業を行うための変数「agent_scratchpad」が含まれている必要があります。
    """

    llm_chain: LLMChain
    output_parser: AgentOutputParser
    allowed_tools: Optional[List[str]] = None

    def dict(self, **kwargs: Any) -> Dict:
        """エージェントの辞書的表現を返します"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

    @property
    def return_values(self) -> List[str]:
        return ["output"]

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this agent.")

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """エージェントの思考プロセスを継続させるためのスクラッチパッドを構築します"""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """入力に基づいて、何をすべきかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと、
                        観察結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザー入力。

                戻り値:
                    使用するツールを指定するアクション。
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """入力に基づいて何をするかを決定します。

                引数:
                    intermediate_steps: これまでにLLMが行ったステップと、
                        観察結果
                    callbacks: 実行するコールバック。
                    **kwargs: ユーザー入力。

                戻り値:
                    使用するツールを指定するアクション。
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    def get_full_inputs(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """中間ステップからLLMChainのフルインプットを作成します"""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    @property
    def input_keys(self) -> List[str]:
        """入力キーを返します。

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """プロンプトがフォーマットと一致していることを確認します"""
        prompt = values["llm_chain"].prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end."
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                raise ValueError(f"Got unexpected prompt type {type(prompt)}")
        return values

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """観測結果を追記するための接頭辞"""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """LLMコールに付け加える接頭辞"""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """このクラスのプロンプトを作成します"""

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """適切なツールが渡されていることを検証します"""
        pass

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """このクラスのデフォルトの出力パーサーを取得します"""

    @classmethod
    def from_llm_and_tools(
            cls,
            llm: BaseLanguageModel,
            tools: Sequence[BaseTool],
            callback_manager: Optional[BaseCallbackManager] = None,
            output_parser: Optional[AgentOutputParser] = None,
            **kwargs: Any,
    ) -> Agent:
        """LLMとツールからエージェントを作成します"""
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
            self,
            early_stopping_method: str,
            intermediate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any,
    ) -> AgentFinish:
        """反復回数が上限に達したことにより、エージェントが停止した場合の応答を返します"""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            # ジェネレートが最後のパスを決める
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # これまでのステップに加え、今度はLLMに最終的なプレッドを作るように指示します。
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # 最終的な答えを導き出そうとします
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # 抽出できるのであれば、正しいものを送ります
                return parsed_output
            else:
                # もし抽出できても、そのツールが最終的なツールでないのなら出力されたものを返します
                return AgentFinish({"output": full_output}, full_output)
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    name = "_Exception"
    description = "Exception tool"

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return query

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return query


class AgentExecutor(Chain):
    """ツールを使用するエージェントで構成されています"""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False

    @classmethod
    def from_agent_and_tools(
            cls,
            agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
            tools: Sequence[BaseTool],
            callback_manager: Optional[BaseCallbackManager] = None,
            **kwargs: Any,
    ) -> AgentExecutor:
        """エージェントやツールから作成します"""
        return cls(
            agent=agent, tools=tools, callback_manager=callback_manager, **kwargs
        )

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """ツールがエージェントと互換性があることを確認します"""
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """ツールがエージェントと互換性があることを確認します"""
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent, BaseMultiActionAgent):
            for tool in tools:
                if tool.return_direct:
                    raise ValueError(
                        "Tools that have `return_direct=True` are not allowed "
                        "in multi-action agents"
                    )
        return values

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - エージェント・エクゼキュータでは保存はサポートされていません。"""
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """基礎となるエージェントを保存します"""
        return self.agent.save(file_path)

    @property
    def input_keys(self) -> List[str]:
        """入力キーを返します。

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """単数形の出力キーを返します。

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """名前からツールを探します"""
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
                self.max_execution_time is not None
                and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """思考-行動-観察ループで1ステップ進む。

        エージェントが選択肢を作成し、それに基づいて行動する方法を制御するために、これをオーバーライドしてください。
        """
        try:
            # LLMをコールして、どうすればいいのか確認します。
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # それ以外の場合は、ツールを参照します
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # 次に、ツール入力でツールをコールして、観測結果を取得します
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """思考-行動-観察ループで1ステップを実行します。

        エージェントが選択肢を作成し、それに基づいて行動する方法を制御するために、これをオーバーライドしてください。
        """
        try:
            # LLMをコールして、何をすればいいのか確認します。
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # 選んだツールがフィニッシングツールであれば、終了して帰ります。
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output

        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # それ以外の場合は、ツールを探します
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # 次に、ツール入力でツールをコールして、観察結果を取得します
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await InvalidTool().arun(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            return agent_action, observation

        # asyncio.gatherを使用して、複数のtool.arun()呼び出しを同時に実行する
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """テキストを通し、エージェントの反応を見ることができます。"""
        # ツール名とツールの対応表を作成し、簡単に参照できるようにします
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # 各ツールと色とのマッピングを構築し、ログを取るために使用します
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # 反復回数と経過時間の追跡を始めましょう
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # これでエージェントループに入りました（何かを返すまで）
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # ツールがそのまま戻ってくるかどうかを確認
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """テキストを通し、エージェントの反応を見ることができます"""
        # ツール名とツールのマッピングを構築し、簡単に参照できるようにします。
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # 各ツールから色へのマッピングを構築し、ログを取る際に使用します。
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # 反復回数と経過時間の追跡を始めましょう
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # これでエージェントループに入りました（何かを返すまで）
        async with asyncio_timeout(self.max_execution_time):
            try:
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # ツールがそのまま戻ってくるかどうかを確認
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return, intermediate_steps, run_manager=run_manager
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )
            except TimeoutError:
                # asyncのタイムアウトで中断された場合、早期に停止します。
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """そのツールが戻りツールであるかどうかを確認します"""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # 無効なツールはマップに表示されないので、Falseを返します
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )
        return None
