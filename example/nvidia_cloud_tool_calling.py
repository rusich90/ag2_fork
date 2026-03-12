# Example: NVIDIA hosted model with AG2's OpenAI compatible client calling a function
# Needs an NVIDIA API key from build.nvidia.com
# pip install ag2[openai]

import math
import os
from pathlib import Path

from dotenv import load_dotenv

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.llm_config.config import LLMConfig

# NVIDIA_API_KEY stored in .env
load_dotenv(Path(__file__).parent / ".env")

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
)

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to the square root of n
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


math_agent = ConversableAgent(
    name="math_agent",
    llm_config=llm_config,
    system_message="You are a math assistant, always use your functions to answer questions.",
    functions=[is_prime],
)

response = math_agent.run(
    message="Is 37 a prime number?",
    max_turns=2,
)

response.process()

# SAMPLE OUTPUT:

# user (to math_agent):

# Is 37 a prime number?

# --------------------------------------------------------------------------------

# >>>>>>>> USING AUTO REPLY...
# [autogen.oai.client: 03-13 06:54:03] {738} WARNING - Model nvidia/nemotron-3-super-120b-a12b is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
# math_agent (to user):

# ***** Suggested tool call (chatcmpl-tool-b3c9cc34743294f3): is_prime *****
# Arguments: 
# {"n": 37}
# **************************************************************************

# --------------------------------------------------------------------------------

# >>>>>>>> EXECUTING FUNCTION is_prime...
# Call ID: chatcmpl-tool-b3c9cc34743294f3
# Input arguments: {'n': 37}

# >>>>>>>> EXECUTED FUNCTION is_prime...
# Call ID: chatcmpl-tool-b3c9cc34743294f3
# Input arguments: {'n': 37}
# Output:
# True
# user (to math_agent):

# ***** Response from calling tool (chatcmpl-tool-b3c9cc34743294f3) *****
# True
# ***********************************************************************

# --------------------------------------------------------------------------------

# >>>>>>>> USING AUTO REPLY...
# [autogen.oai.client: 03-13 06:54:09] {738} WARNING - Model nvidia/nemotron-3-super-120b-a12b is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
# math_agent (to user):

# Yes, 37 is a prime number.

# --------------------------------------------------------------------------------

# >>>>>>>> TERMINATING RUN (0fb1614a-98d3-413f-94c4-249b04d7cccc): Maximum turns (2) reached