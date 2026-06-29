from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
# MODEL = "gemma4:latest"
# MODEL = "qwen3:0.6b"
# gemma4:latest


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    products list: 1.laptop, 2.headphones, 3.keyboard, 4.monitor, 5.mouse, 6.webcam, 7.speaker, 8.printer, 9.tablet.
    """
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 249.99,
        "mouse": 39.99,
        "webcam": 79.95,
        "speaker": 119.90,
        "printer": 199.50,
        "tablet": 499.00,
    }
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: 1.bronze, 2.silver, 3.gold, 4.platinum."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23, "platinum": 30}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

@tool
def get_product_price_list(products: list[str]) -> dict[str, float]:
    """Look up the prices of a list of products in the catalog."""
    print(f"    >> Executing get_product_price_list(products={products})")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 249.99,
        "mouse": 39.99,
        "webcam": 79.95,
        "speaker": 119.90,
        "printer": 199.50,
        "tablet": 499.00,
    }
    return {product: prices.get(product, 0) for product in products}


@tool
def apply_discount_list(prices: dict[str, float], discount_tiers: dict[str, str]) -> dict[str, float]:
    """Apply discount tiers to a dictionary of prices and return the final prices."""
    print(
        f"    >> Executing apply_discount_list(prices={prices}, discount_tiers={discount_tiers})"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23, "platinum": 30}
    final_prices = {}
    for product, price in prices.items():
        discount_tier = discount_tiers.get(product)
        if discount_tier:
            discount = discount_percentages.get(discount_tier, 0)
            final_prices[product] = round(price * (1 - discount / 100), 2)
        else:
            final_prices[product] = price
    return final_prices
# --- Agent Loop ---


@traceable(name="agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount, get_product_price_list, apply_discount_list]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tools "
                "and a discount tools.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price or get_product_price_list first to get the real price.\n"
                "2. Only call apply_discount or apply_discount_list  AFTER you have received "
                "a price from get_product_price or get_product_price_list. Pass the exact price "
                "returned by get_product_price or get_product_price_list — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount or apply_discount_list tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
                
            )
        ),
        HumanMessage(content=question),
    ]
# "5. If the user asks for multiple products, you can call get_product_price_list and apply_discount_list to get prices and apply discounts in bulk."
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "Get the price of laptop, mouse and webcam after applying a bronze discount to the laptop, a silver discount to the mouse, and a gold discount to the webcam."
    )

    # Prompt examples:
    # What are the products available in the catalog?
    # What are the available discount tiers?
    # What is the price of a laptop after applying a platinum and gold discount?
    # What is the price of a laptop without discount?
    # Compare the price of a laptop and a tablet after applying a silver discount to both.
    # Compare the price of a laptop and a tablet after applying a platinum discount to the laptop and a gold discount to the tablet.
    # Get the price of laptop, mouse and webcam after applying a bronze discount to the laptop, a silver discount to the mouse, and a gold discount to the webcam.