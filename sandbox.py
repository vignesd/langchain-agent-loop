from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from datetime import datetime
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
MODEL = "qwen3:0.6b"
# MODEL = "qwen3:1.7b"
# MODEL = "gemma4:latest"


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


def main(query, system_message):
    logging.info("Hello from agent loop sandbox!")
    llm = ChatOllama(temperature=0, model="qwen3:0.6b")
    tools = [get_multiple_product_prices, get_multiple_product_discounts]
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content=query),
                SystemMessage(content=system_message),
            ]
        }
    )
    logging.info(
        f"Total Messages: {len(response['messages'])}"
    )  # Log the total number of messages in the result
    logging.info(f"Agent result: \n{response['messages'][-1].content}")
    # print(f"\nRaw Data :\n{response['messages'][-1]}")
    for message in response["messages"]:
        # logging.info(f"Message: {message}")
        with open("response.txt", "a") as f:
            f.write(
                f"\n{datetime.now()}: Agent result:  \n{message}\n"
            )


if __name__ == "__main__":

    query = (
        "Get the price of laptop, mouse and webcam "
        "after applying a bronze discount to the laptop, a silver discount to the mouse, "
        "and a gold discount to the webcam."
    )
    system_message=None
    with open('System_Message.txt','r') as file:
        system_message=file.read()
    main(query,system_message)
