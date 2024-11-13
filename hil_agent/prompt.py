from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


def get_planner_prompt_msgs(query: str) -> List[BaseMessage]:
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    chat_msgs = planner_prompt.invoke({"messages": [("user", query)]})
    return chat_msgs.messages
