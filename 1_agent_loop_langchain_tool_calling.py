from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
# MODEL = "qwen3:1.7b"
MODEL = "gemma4:latest" # This works as expected
# MODEL = "qwen3:0.6b"
# gemma4:latest

@tool
def get_product_price(product: str) -> float:
    """
    Retrieve the current price of a single product.

    Use this function whenever the user asks for the price of exactly one
    product, or when the price of a single product is required before applying
    a discount.

    If the user requests prices for multiple products, prefer using
    `get_multiple_product_prices` instead of calling this function repeatedly.

    Input:
        product:
            The name of the product.

            Example:
                "MacBook Air M4"

    Output:
        The current price of the product as a float.

        Example:
            999.0

    Notes:
        - Never guess or estimate a product's price.
        - Always use this tool to obtain the actual price.
        - Preserve the product name exactly as provided by the user whenever
          possible.
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
def get_product_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a single product price.

    Use this function only after obtaining the product's price from
    `get_product_price`.

    If discounts need to be applied to multiple products, prefer using
    `get_multiple_product_discounts` instead of calling this function
    repeatedly.

    Input:
        price:
            The product price returned by `get_product_price`.

        discount_tier:
            The discount tier to apply.

            Example:
                "GOLD"
                "SILVER"
                "BRONZE"

    Output:
        The discounted price as a float.

        Example:
            899.10

    Notes:
        - Never calculate discounts yourself.
        - Always pass the exact price returned by `get_product_price`.
        - Do not modify or round the input price before calling this tool.
        - If the discount tier is unknown, ask the user instead of assuming one.
    """
    print(
        f"    >> Executing get_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23, "platinum": 30}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@tool
def get_multiple_product_prices(products: list[str]) -> dict[str, float]:
    """
    Returns the current prices of multiple products.

    Use this function whenever the user asks:
    - "How much do these products cost?"
    - "Compare the prices of..."
    - "What's the price of A, B, and C?"
    - Any question that requires the prices of multiple products.

    Input:
        products: List of product names.

    Output:
        Dictionary where:
        - key = product name
        - value = product price as a float

    Example:
        get_multiple_product_prices([
            "laptop",
            "headphones"
        ])

        returns
        {
            "laptop": 999.0,
            "headphones": 1199.0
        }

    Always include every product requested by the user in the `products` list.
    Prefer this function over repeated single-product price lookups.
    """
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
def get_multiple_product_discounts(
    prices: dict[str, float], discount_tiers: dict[str, str]
) -> dict[str, float]:
    """
    Calculate the discounted price for multiple products based on their
    original prices and assigned discount tiers.

    Use this function whenever the user asks for:
    - Discounted prices for multiple products.
    - The final price after applying discounts.
    - Price comparisons after discounts.
    - The total cost of multiple discounted products.

    Input:
        prices:
            Dictionary mapping each product name to its original price.

            Example:
        {
            "laptop": 999.0,
            "headphones": 1199.0
        }

        discount_tiers:
            Dictionary mapping each product name to its discount tier.

            Example:
                {
                    "laptop": "gold",
                    "headphones": "silver"
                }

            Each product in `prices` should have a corresponding discount tier.

    Output:
        Dictionary mapping each product name to its final price after applying
        the corresponding discount.

        Example:
            {
                "laptop": 899.10,
                "headphones": 1139.05
            }

    Notes:
        - Pass all products in a single call instead of invoking the function
          multiple times.
        - Key and pair value in lower case.
        - Use the exact product names as the dictionary keys in both inputs.
        - The returned values are the final prices after discounts have been
          applied.
        - This function applies the discount associated with each product's
          discount tier; callers do not need to calculate discounts manually.
    """
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
    tools = [
        get_multiple_product_prices,
        get_multiple_product_discounts,
        get_product_price,
        get_product_discount
    ]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    with open('system_message.txt','r') as fobj:
        sys_msg=fobj.read()
    # print(sys_msg)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(sys_msg)
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


# "You are a helpful shopping assistant. "
# "You have access to a product catalog tools "
# "and a discount tools.\n\n"
# "STRICT RULES — you must follow these exactly:\n"
# "1. NEVER guess or assume any product price. "
# "You MUST call get_product_price or get_product_price_list first to get the real price.\n"
# "2. Only call apply_discount or apply_discount_list  AFTER you have received "
# "a price from get_product_price or get_product_price_list. Pass the exact price "
# "returned by get_product_price or get_product_price_list — do NOT pass a made-up number.\n"
# "3. NEVER calculate discounts yourself using math. "
# "Always use the apply_discount or apply_discount_list tool.\n"
# "4. If the user does not specify a discount tier, "
# "ask them which tier to use — do NOT assume one."