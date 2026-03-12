# Example: NVIDIA hosted model with AG2's OpenAI compatible client calling a function
# Needs an NVIDIA API key from build.nvidia.com
# pip install ag2[openai]

import math
import os
from pathlib import Path

from dotenv import load_dotenv

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.llm_config.config import LLMConfig
from pydantic import BaseModel

# NVIDIA_API_KEY stored in .env
load_dotenv(Path(__file__).parent / ".env")

# Create a Pydantic model for a structured output example
class OrderDetails(BaseModel):
    order_id: int
    item: str
    quantity: int
    price_per_item: float
    total_price: float

# Use OpenAI client as NVIDIA endpoints are OpenAI API compatible
llm_config = LLMConfig(
    {
        "api_type": "openai",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_API_KEY"],
    },
    temperature=1.0,
    top_p=0.95,
    response_format=OrderDetails
)

order_agent = ConversableAgent(
    name="order_agent",
    llm_config=llm_config,
    system_message="""There's one order in the system, here are the details:
    Order ID: 12345
    Item: Widget
    Quantity: 10
    Price per item: $2.50
    Total price: $25.00""",
)

response = order_agent.run(
    message="Can you get order 12345 details?",
    max_turns=1,
)

response.process()

output_formatted = OrderDetails.model_validate_json(response.summary)

print("Structured Output:")
print(output_formatted.model_dump_json(indent=4))

# SAMPLE OUTPUT:

# user (to order_agent):

# Can you get order 12345 details?

# --------------------------------------------------------------------------------

# >>>>>>>> USING AUTO REPLY...
# [autogen.oai.client: 03-13 07:42:34] {738} WARNING - Model nvidia/nemotron-3-super-120b-a12b is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
# order_agent (to user):

# {
#   "order_id": 12345,
#   "item": "Widget",
#   "quantity": 10,
#   "price_per_item": 2.50,
#   "total_price": 25.00
# }

# --------------------------------------------------------------------------------

# >>>>>>>> TERMINATING RUN (493d6f58-6833-4f1c-9c55-2c14bdf6546e): Maximum turns (1) reached
# Structured Output:
# {
#     "order_id": 12345,
#     "item": "Widget",
#     "quantity": 10,
#     "price_per_item": 2.5,
#     "total_price": 25.0
# }