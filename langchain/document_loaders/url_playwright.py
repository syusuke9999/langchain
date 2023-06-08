"""Playwright を使用してページをロードし、その後「unstructured」を使用して HTML をロードするローダー。
"""
import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class PlaywrightURLLoader(BaseLoader):
    """Playwright を使用してページをロードし、構造化されていない HTML をロードするローダー。
        これは、JavaScript が必要なページのレンダリングに役立ちます。

        属性:
            urls (List[str]): ロードする URL のリスト。
            continue_on_failure (bool): True の場合、失敗時に他の URL のロードを続行します。
            headless (bool): True の場合、ブラウザはヘッドレスモードで実行されます。
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        headless: bool = True,
        remove_selectors: Optional[List[str]] = None,
    ):
        """Load a list of URLs using Playwright and unstructured."""
        try:
            import playwright  # noqa:F401
        except ImportError:
            raise ImportError(
                "playwright package not found, please install it with "
                "`pip install playwright`"
            )

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headless = headless
        self.remove_selectors = remove_selectors

    def load(self) -> List[Document]:
        """Playwright を使用して指定された URL をロードし、Document インスタンスを作成します。

                戻り値:
                    List[Document]: ロードされたコンテンツを持つ Document インスタンスのリスト。
        """
        from playwright.sync_api import sync_playwright
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            for url in self.urls:
                try:
                    page = browser.new_page()
                    page.goto(url)

                    for selector in self.remove_selectors or []:
                        elements = page.locator(selector).all()
                        for element in elements:
                            if element.is_visible():
                                element.evaluate("element => element.remove()")

                    page_source = page.content()
                    elements = partition_html(text=page_source)
                    text = "\n\n".join([str(el) for el in elements])
                    metadata = {"source": url}
                    docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(
                            f"Error fetching or processing {url}, exception: {e}"
                        )
                    else:
                        raise e
            browser.close()
        return docs
