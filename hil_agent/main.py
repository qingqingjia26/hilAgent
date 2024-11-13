import argparse
import traceback
import uuid
from enum import Enum
from typing import Annotated, Dict, List, Sequence, TypedDict, Union, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import BaseModel, Field

from hil_agent.prompt import get_planner_prompt_msgs
from hil_agent.tools import check_weather, run_cmd_on_host, write2file
from hil_agent.utils import agent_parser

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"  # 用于重置颜色


def colorful_str(color, text) -> str:
    return color + text + RESET


def colorful_print(color, text):
    print(colorful_str(color, text))


def colorful_input(color, text):
    return input(colorful_str(color, text))


class NodeType(str, Enum):
    """Enum of the Node Type."""

    PLAN = "plan"
    EXECUTE = "execute"


class NodeInfo:
    def __init__(self, node_type: NodeType, node_name: str):
        self.node_type = node_type
        self.node_name = node_name


Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]


def custom_add_messages(
    left: list[MessageLikeRepresentation], right: Messages
) -> Messages:
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    left += right
    return left


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], custom_add_messages]
    plan_messages: Annotated[Sequence[BaseMessage], custom_add_messages]
    custom_tools: Sequence[BaseTool]
    feedbacks: Dict[str, any]
    plan_dict: Dict[str, any]
    meta_dict: Dict[str, any]
    main_model: BaseChatModel
    plan_model: BaseChatModel
    origin_req: str


g_func2node_dict = {
    "custom_tool_func": NodeInfo(NodeType.EXECUTE, "tool"),
    "call_model": NodeInfo(NodeType.EXECUTE, "agent"),
    "human_feedback": NodeInfo(NodeType.EXECUTE, "human_feedback"),
    "should_continue": NodeInfo(NodeType.EXECUTE, "should_continue"),
    "plan_node_func": NodeInfo(NodeType.PLAN, "plan"),
    "plan_human_feedback_func": NodeInfo(NodeType.PLAN, "plan_human"),
    "plan_should_continue_func": NodeInfo(
        NodeType.PLAN, "plan_should_continue_func"
    ),
    "critic_node_func": NodeInfo(NodeType.PLAN, "critic"),
    "critic_human_feedback_func": NodeInfo(NodeType.PLAN, "critic_human"),
    "critic_should_continue_func": NodeInfo(
        NodeType.PLAN, "critic_should_continue_func"
    ),
}


def update_cur_node(func):
    def wrapper(state: AgentState) -> AgentState:
        meta_dict = state["meta_dict"]
        meta_dict["cur_node"] = g_func2node_dict[func.__name__]
        return func(state)

    return wrapper


def invoke_custom_tool(
    tools: Sequence[BaseTool], msg: AIMessage
) -> List[ToolMessage]:
    result = []
    tools_by_name = {tool.name: tool for tool in tools}
    for tool_called in msg.tool_calls:
        tool = tools_by_name[tool_called["name"]]
        observation = tool.invoke(tool_called["args"])
        result.append(
            ToolMessage(
                content=str(observation), tool_call_id=tool_called["id"]
            )
        )
    return result


@update_cur_node
def custom_tool_func(state: AgentState) -> AgentState:
    msgs = state["messages"]
    last_msg = msgs[-1]
    fbs = state["feedbacks"]
    if fbs["tool_feedback"] != "":
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            raise ValueError(
                "tool_feedback is not empty, but no tool calls found"
            )

        for tool_called in last_msg.tool_calls:
            tm = ToolMessage(
                content="Disagree with the parameters of the tool call",
                tool_call_id=tool_called["id"],
            )
            state["messages"] += [tm]
        state["messages"] += [HumanMessage(fbs["tool_feedback"])]
        fbs["tool_feedback"] = ""
        return
    last_msg = state["messages"][-1]
    return {"messages": invoke_custom_tool(state["custom_tools"], last_msg)}


@update_cur_node
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    main_model = state["main_model"]
    response = main_model.invoke(messages)
    return {"messages": [response]}


def parse_input(question: str) -> argparse.Namespace:
    while True:
        user_input = colorful_input(RED, f"{question}")
        args, help_info = agent_parser(user_input)
        if help_info != "":
            print(f"\n{help_info}")
            continue
        return args


g_feedbacks_dict = {
    "tool": {
        "key": "tool_feedback",
        "question": "\nDo you agree to use the tools mentioned above? If so, please enter 'y'. Otherwise, please enter 'no' along with your thoughts on the model, or type '-h' for help:",
    },
    "human": {
        "key": "human_feedback",
        "question": "If you believe the above results meet the expectations and want the agent to continue, please output 'y'. Otherwise, please enter 'no' along with your thoughts on the model, or type '-h' for help:",
    },
}


@update_cur_node
def human_feedback(state: AgentState) -> AgentState:
    fbs = state["feedbacks"]
    if fbs["skip_count"] > 0:
        fbs["skip_count"] -= 1
        return
    last_msg = state["messages"][-1]
    has_tool_calls = isinstance(last_msg, AIMessage) and last_msg.tool_calls
    qs_type = "tool" if has_tool_calls else "human"
    args = parse_input(g_feedbacks_dict[qs_type]["question"])
    if args.command == "y":
        if args.n != 0:
            fbs["skip_count"] = args.n
        return
    if args.command == "no":
        fbs[g_feedbacks_dict[qs_type]["key"]] = " ".join(args.input)
        return


@update_cur_node
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    plan_dict = state["plan_dict"]
    fbs = state["feedbacks"]

    has_tool_calls = (
        isinstance(last_message, AIMessage) and last_message.tool_calls
    )
    if has_tool_calls or fbs["tool_feedback"] != "":
        return "tool"
    if fbs["human_feedback"] != "":
        state["messages"] += [HumanMessage(fbs["human_feedback"])]
        fbs["human_feedback"] = ""
        return "agent"
    plan_idx = plan_dict["current_plan_idx"]
    if plan_idx < len(plan_dict["plans"]) - 1:
        plan_str = plan_dict["plans"][plan_idx + 1]
        state["messages"] += [HumanMessage(plan_str)]
        print(
            "================================== Plan Message ==================================\n"
        )
        print(f"Now process the step {plan_idx + 2} : {plan_str}")
        plan_dict["current_plan_idx"] += 1
        return "agent"
    return "critic"


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


@update_cur_node
def plan_node_func(state: AgentState):
    agent_msg = state["messages"]

    plan_model = state["plan_model"]
    plan_messages = state["plan_messages"]
    plan_dict = state["plan_dict"]

    chain = plan_model.with_structured_output(Plan)
    if len(agent_msg) != 0:
        rsp: Plan = chain.invoke(agent_msg + plan_messages)
    else:
        rsp: Plan = chain.invoke(plan_messages)
    steps = rsp.steps
    human_msg = f"LLM's plan is: {str(steps)}"

    plan_dict["plans"] = steps
    return {"plan_messages": HumanMessage(str(human_msg))}


@update_cur_node
def plan_human_feedback_func(state: AgentState):
    fbs = state["feedbacks"]
    if fbs["skip_count"] > 0:
        fbs["skip_count"] -= 1
        return
    question = "\nIf you agree with the plan, please input y. Otherwise, input no and your thoughts to the model, or -h for help:"
    args = parse_input(question)
    if args.command == "y":
        if args.n != 0:
            fbs["skip_count"] = args.n
        return
    if args.command == "no":
        fbs["plan_human_feedback"] = " ".join(args.input)
        return


@update_cur_node
def plan_should_continue_func(state: AgentState):
    fbs = state["feedbacks"]
    if fbs["plan_human_feedback"] != "":
        state["plan_messages"] += [HumanMessage(fbs["plan_human_feedback"])]
        fbs["plan_human_feedback"] = ""
        return "plan"
    plans = state["plan_dict"]["plans"]
    state["messages"] += [HumanMessage(plans[0])]
    print(
        "================================== Plan Message ==================================\n"
    )
    print(f"Now process the step 1 : {plans[0]}")
    return "agent"


class ShouldStop(BaseModel):
    """Check if the process should stop or not based on the given messages"""

    stop: bool = Field(description="Should stop the process or not")


@update_cur_node
def critic_node_func(state: AgentState):
    agent_msg = state["messages"]
    fbs = state["feedbacks"]
    origin_req = state["origin_req"]
    plan_model = state["plan_model"]
    critic_str = f"The original requirement is: {origin_req}. Based on the above chat record, has the most primitive requirement been fulfilled?"
    cricict_msg = HumanMessage(critic_str)
    chain = plan_model.with_structured_output(ShouldStop)
    rsp: ShouldStop = chain.invoke(agent_msg + [cricict_msg])
    fbs["critic_result"] = rsp.stop
    cricit_res = f"LLM critic result is: {rsp.stop}. LLM thinks the original requirement is {'completed' if rsp.stop else 'not completed'}"
    return {"plan_messages": HumanMessage(str(cricit_res))}


@update_cur_node
def critic_human_feedback_func(state: AgentState):
    fbs = state["feedbacks"]
    if fbs["skip_count"] > 0:
        fbs["skip_count"] -= 1
        return
    question = "\nIf you agree with the critic result, please input y. Otherwise, input no and your thoughts to the model, or -h for help:"
    args = parse_input(question)
    if args.command == "y":
        if args.n != 0:
            fbs["skip_count"] = args.n
        return
    if args.command == "no":
        fbs["critic_human_feedback"] = " ".join(args.input)
        return


@update_cur_node
def critic_should_continue_func(state: AgentState):
    fbs = state["feedbacks"]
    if fbs["critic_human_feedback"] != "":
        state["plan_messages"] += [HumanMessage(fbs["critic_human_feedback"])]
        fbs["critic_human_feedback"] = ""
        return "plan"
    if fbs["critic_result"] is False:
        return "plan"
    return "end"


def print_stream(graph: CompiledStateGraph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        meta_dict = s["meta_dict"]
        cur_node = meta_dict["cur_node"]
        if cur_node.node_type == NodeType.PLAN:
            msg = s["plan_messages"][-1]
            msg.pretty_print()
        else:
            msg = s["messages"][-1]
            msg.pretty_print()


def hil_pe_react_agent_func(
    model: ChatOpenAI,
    plan_model: ChatOpenAI,
    tools: Sequence[BaseTool],
    requirement: str,
    thread_id: str = "thread-1",
) -> AgentState:
    if not isinstance(requirement, str):
        raise ValueError("Requirement should be a string")
    tool_node = ToolNode(tools)
    # get the tool functions wrapped in a tool class from the ToolNode
    tool_classes = list(tool_node.tools_by_name.values())
    main_model = cast(BaseChatModel, model).bind_tools(
        tool_classes, strict=True
    )
    workflow = StateGraph(AgentState)
    workflow.set_entry_point("plan")
    workflow.add_node("agent", call_model)
    workflow.add_node("human", human_feedback)
    workflow.add_node("tool", custom_tool_func)
    workflow.add_node("plan", plan_node_func)
    workflow.add_node("plan_human", plan_human_feedback_func)
    workflow.add_node("critic", critic_node_func)
    workflow.add_node("critic_human", critic_human_feedback_func)

    workflow.add_edge("plan", "plan_human")
    workflow.add_edge("agent", "human")
    workflow.add_edge("tool", "agent")
    workflow.add_edge("critic", "critic_human")

    workflow.add_conditional_edges(
        "human",
        should_continue,
        {
            "tool": "tool",
            "agent": "agent",
            "critic": "critic",
        },
    )
    workflow.add_conditional_edges(
        "plan_human",
        plan_should_continue_func,
        {
            "plan": "plan",
            "agent": "agent",
            "plan": "plan",
        },
    )
    workflow.add_conditional_edges(
        "critic_human",
        critic_should_continue_func,
        {
            "plan": "plan",
            "end": END,
        },
    )
    graph: CompiledStateGraph = workflow.compile()
    # graph.get_graph().draw_png("hil_agent_graph.png")
    # raise ValueError("stop here")

    inputs = {
        "messages": [],
        "plan_messages": get_planner_prompt_msgs(requirement),
        "custom_tools": tools,
        "feedbacks": {
            "tool_feedback": "",
            "human_feedback": "",
            "plan_human_feedback": "",
            "critic_human_feedback": "",
            "skip_count": 0,
            "critic_result": True,
        },
        "plan_dict": {"plans": [], "current_plan_idx": 0},
        "main_model": main_model,
        "plan_model": plan_model,
        "origin_req": requirement,
        "meta_dict": {"cur_node": NodeInfo(NodeType.PLAN, "start node")},
    }
    config = {"configurable": {"thread_id": thread_id}}
    try:
        print_stream(graph, inputs, config)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False
    return True


def call_hil_pe_react_agent(
    model: ChatOpenAI, plan_model: ChatOpenAI, query: str
):
    tools = [
        run_cmd_on_host,
        write2file,
        check_weather,
        TavilySearchResults(max_results=3),
    ]
    return hil_pe_react_agent_func(
        model, plan_model, tools, query, thread_id=uuid.uuid4()
    )


def hil_react_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="what is the temperature of hometown of the mens 2024 Australia open winner today?",
        help="query to ask the agent",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to use",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="api_key to use",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        help="base_url to use",
    )
    args = parser.parse_args()
    if args.base_url is None or args.api_key is None or args.model is None:
        raise ValueError(
            "If llm is not provided, then api_key, base_url and model should be provided"
        )
    model = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    plan_model = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    call_hil_pe_react_agent(model, plan_model, args.query)


if __name__ == "__main__":
    hil_react_agent()
