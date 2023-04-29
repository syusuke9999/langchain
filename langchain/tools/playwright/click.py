from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    selector: str = Field(..., description="CSS selector for the element to click")


class ClickTool(BaseBrowserTool):
    name: str = "click_element"
    description: str = "Click on an element with the given CSS selector"
    args_schema: Type[BaseModel] = ClickToolInput

    def _run(self, selector: str) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        page.click(selector)
        return f"Clicked element '{selector}'"

    async def _arun(self, selector: str) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        await page.click(selector)
        return f"Clicked element '{selector}'"
