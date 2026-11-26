"""Tool input schemas, tool implementations, the tool registry, and registry helpers."""

from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ============================================================
# 1. Tool input schemas
# ============================================================

class ProductRequest(BaseModel):
    """Input schema for product-related requests."""

    product_name: str = Field(min_length=5)


class OrderRequest(BaseModel):
    """Input schema for order-related requests."""

    order_id: str = Field(min_length=5)


class WeatherRequest(BaseModel):
    """Input schema for weather-related requests."""

    place_name: str = Field(min_length=3)


# ============================================================
# 2. Tools
# ============================================================

@tool(args_schema=ProductRequest)
def get_product_price(product_name: str) -> str:
    """Give information about the product price."""
    return f"The price of {product_name} is $100."


@tool(args_schema=OrderRequest)
def get_order_status(order_id: str) -> str:
    """Check the current delivery status of an order."""
    if order_id == "DF1617":
        return f"Order {order_id} is currently out for delivery."

    raise ValueError(f"Order {order_id} was not found.")


@tool(args_schema=OrderRequest)
def cancel_order(order_id: str) -> str:
    """Cancel an order when cancellation is requested."""
    if order_id == "DF1617":
        return f"Order {order_id} is cancelled successfully."

    raise ValueError(f"Order {order_id} was not found.")


@tool(args_schema=WeatherRequest)
def get_weather_info(place_name: str) -> str:
    """Give information about the weather."""
    if place_name.capitalize() == "Mingora":
        return f"The weather in {place_name} is pleasant with clear skies."

    raise ValueError(f"Weather data not found for {place_name}.")


# ============================================================
# 3. Tool registry
# ============================================================

categories = {
    "product_information": {
        "tools": [
            {
                "tool": get_product_price,
                "description": "Checks the price of a specified product.",
                "requires_approval": False,
            }
        ],
        "description": "Handles product-related questions such as prices.",
    },
    "order_status": {
        "tools": [
            {
                "tool": get_order_status,
                "description": "Checks the current delivery status of an order.",
                "requires_approval": False,
            },
            {
                "tool": cancel_order,
                "description": "Cancels an order when cancellation is requested.",
                "requires_approval": True,
            },
        ],
        "description": "Handles questions about order tracking, delivery status, and cancellation.",
    },
    "weather_information": {
        "tools": [
            {
                "tool": get_weather_info,
                "description": "Checks the weather for a specified location.",
                "requires_approval": False,
            }
        ],
        "description": "Handles weather questions for a specified location.",
    },
}


# ============================================================
# 4. Registry helpers
# ============================================================

def validate_category(category: str) -> str:
    """Validate that a category exists in the registry."""
    if category not in categories:
        raise ValueError(f"Invalid category: {category}")
    return category


def get_category_description(category: str) -> str:
    """Return the description of a category."""
    validate_category(category)
    return categories[category]["description"]


def build_category_descriptions() -> str:
    """Build descriptions of all registered categories."""
    return "\n".join(
        f"- {name}: {metadata['description']}"
        for name, metadata in categories.items()
    )


def get_category_tools(category: str):
    """Return all tool metadata for a category."""
    validate_category(category)
    return categories[category]["tools"]


def get_tools(category: str):
    """Return executable tool objects for a category."""
    tool_metadata = get_category_tools(category)
    return [metadata["tool"] for metadata in tool_metadata]


def get_tool_names(category: str) -> List[str]:
    """Return tool names registered for a category."""
    return [tool_instance.name for tool_instance in get_tools(category)]


def build_tool_selection_prompt(category: str) -> str:
    """Build the capability-selection prompt."""
    tool_metadata = get_category_tools(category)

    lines = [f"Category: {category}", "", "Available capabilities:"]

    for metadata in tool_metadata:
        tool_instance = metadata["tool"]
        lines.append(f"\n{tool_instance.name}:")
        lines.append(metadata["description"])

    lines.append(
        "\nSelect ONLY the tool names that are necessary to answer the user's request."
    )

    return "\n".join(lines)


def validate_tool_selection(category: str, selected_tools: List[str]) -> List[str]:
    """Validate selected tools against the registry."""
    allowed_tools = set(get_tool_names(category))
    invalid_tools = [name for name in selected_tools if name not in allowed_tools]

    if invalid_tools:
        raise ValueError(
            f"Invalid tools selected for category '{category}': {invalid_tools}"
        )

    return selected_tools


def get_tools_by_names(category: str, tool_names: List[str]):
    """Resolve selected tool names to tool objects."""
    available_tools = get_tools(category)
    tool_map = {tool_instance.name: tool_instance for tool_instance in available_tools}
    validate_tool_selection(category, tool_names)
    return [tool_map[name] for name in tool_names]


def get_tool_metadata(category: str, tool_name: str) -> dict:
    """Return metadata for a specific registered tool."""
    validate_category(category)

    for metadata in get_category_tools(category):
        if metadata["tool"].name == tool_name:
            return metadata

    raise ValueError(
        f"Tool '{tool_name}' not found in category '{category}'."
    )


def check_approval_required(category: str, selected_tools: List[str]) -> bool:
    """Determine whether any selected tool requires human approval."""
    for tool_name in selected_tools:
        metadata = get_tool_metadata(category, tool_name)

        if metadata["requires_approval"]:
            return True

    return False
