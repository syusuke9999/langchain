from typing import Any, Dict, List, Set

from pydantic import validator

from langchain.schema import BaseMemory


class CombinedMemory(BaseMemory):
    """
    Class for combining multiple memories' data together.
    複数のメモリのデータを結合するためのクラス
    """

    memories: List[BaseMemory]
    """
    For tracking all the memories that should be accessed.
    アクセスされるべきすべてのメモリを追跡するため。
    """

    @validator("memories")
    def check_repeated_memory_variable(
        cls, value: List[BaseMemory]
    ) -> List[BaseMemory]:
        all_variables: Set[str] = set()
        for val in value:
            overlap = all_variables.intersection(val.memory_variables)
            if overlap:
                raise ValueError(
                    f"The same variables {overlap} are found in multiple"
                    "memory object, which is not allowed by CombinedMemory."
                )
            all_variables |= set(val.memory_variables)

        return value

    @property
    def memory_variables(self) -> List[str]:
        """
        All the memory variables that this instance provides.
        このインスタンスが提供するすべてのメモリ変数。
        """
        """
        Collected from the all the linked memories.
        リンクされたすべてのメモリから収集されたものです。
        """

        memory_variables = []

        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)

        return memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Load all vars from sub-memories.
        サブメモリから全てのvarをロードする。
        """
        memory_data: Dict[str, Any] = {}

        # Collect vars from all sub-memories
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            memory_data = {
                **memory_data,
                **data,
            }

        return memory_data

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context from this session for every memory.
        このセッションで得られたコンテキストを、すべてのメモリのために保存します。
        """
        # すべてのサブメモリーのコンテキストを保存する
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """
        Clear context from this session for every memory.
        このセッションのコンテキストを、すべてのメモリに対して消去します。
        """
        for memory in self.memories:
            memory.clear()
