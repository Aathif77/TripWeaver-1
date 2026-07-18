from langgraph.graph import END, START, StateGraph

from .entity import GraphState
from .nodes import (
    flight_node,
    general_qa_node,
    generate_response,
    hotel_node,
    route_after_extraction,
    router,
)
from .planner import planner_node


def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("router", router)
    builder.add_node("hotel_node", hotel_node)
    builder.add_node("flight_node", flight_node)
    builder.add_node("general_qa_node", general_qa_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_after_extraction,
        {
            "hotel": "hotel_node",
            "flight": "flight_node",
            "general": "general_qa_node",
            "planner": "planner_node",
        },
    )

    builder.add_edge("hotel_node", "generate_response")
    builder.add_edge("flight_node", "generate_response")
    builder.add_edge("general_qa_node", "generate_response")
    builder.add_edge("planner_node", "generate_response")
    builder.add_edge("generate_response", END)

    return builder


graph = build_graph().compile()