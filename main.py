import os
from google import genai

# Initialize Gemini Client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"


def advocate_agent(product_name: str, product_details: str, priorities: str) -> str:
    """Advocate Agent: Builds a strong case for a specific product based on priorities."""
    prompt = f"""
    You are an expert product advocate representing '{product_name}'.
    
    Product Details:
    {product_details}
    
    User Priorities:
    {priorities}
    
    Build a compelling, factual case for why '{product_name}' is the best choice for the user. 
    Focus specifically on how it aligns with their priorities. Highlight key advantages, unique features, 
    and practical value. Keep your argument concise, structured, and persuasive.
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text


def critic_agent(product_name: str, product_details: str, priorities: str) -> str:
    """Critic Agent: Evaluates trade-offs, potential issues, and drawbacks."""
    prompt = f"""
    You are a critical consumer analyst evaluating '{product_name}'.
    
    Product Details:
    {product_details}
    
    User Priorities:
    {priorities}
    
    Provide an objective, critical assessment of '{product_name}'. Highlight potential drawbacks, 
    limitations, hidden trade-offs, or areas where it might fall short of meeting the user's priorities.
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text


def judge_agent(
    product_a: str, 
    argument_a: str, 
    critique_a: str, 
    product_b: str, 
    argument_b: str, 
    critique_b: str, 
    priorities: str
) -> str:
    """Judge Agent: Synthesizes arguments and renders an objective, grounded decision."""
    prompt = f"""
    You are an impartial, expert Purchasing Advisor. Your objective is to render a final grounded recommendation 
    between two products by evaluating multi-agent debates against strict user priorities.

    User Priorities:
    {priorities}

    --- PRODUCT A: {product_a} ---
    Advocate Argument:
    {argument_a}
    
    Critical Review:
    {critique_a}

    --- PRODUCT B: {product_b} ---
    Advocate Argument:
    {argument_b}
    
    Critical Review:
    {critique_b}

    --- EVALUATION INSTRUCTIONS ---
    Provide a well-structured final report in Markdown:
    1. **Executive Recommendation**: Declare the winning product clearly.
    2. **Priority Scorecard**: Compare both products on how well they meet each user priority.
    3. **Key Trade-Off Analysis**: Summarize what the user gains and sacrifices with each choice.
    4. **Final Justification**: Provide a clear, grounded explanation for why this recommendation best satisfies the user's criteria.
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text


def evaluate_purchase(
    product_a_name: str, 
    product_a_details: str, 
    product_b_name: str, 
    product_b_details: str, 
    priorities: str
) -> str:
    """Runs the full multi-agent debate and synthesis pipeline."""
    print(f"🤖 [1/3] Generating advocate & critique arguments for {product_a_name}...")
    arg_a = advocate_agent(product_a_name, product_a_details, priorities)
    crit_a = critic_agent(product_a_name, product_a_details, priorities)

    print(f"🤖 [2/3] Generating advocate & critique arguments for {product_b_name}...")
    arg_b = advocate_agent(product_b_name, product_b_details, priorities)
    crit_b = critic_agent(product_b_name, product_b_details, priorities)

    print("⚖️  [3/3] Judge Agent synthesizing arguments and rendering final decision...\n")
    decision = judge_agent(
        product_a_name, arg_a, crit_a,
        product_b_name, arg_b, crit_b,
        priorities
    )
    return decision


if __name__ == "__main__":
    prod_a = "MacBook Air M3 (16GB RAM, 512GB SSD)"
    prod_a_specs = "13.6-inch Liquid Retina, Apple M3 chip (8-core CPU, 10-core GPU), 18hr battery life, 2.7 lbs weight, fanless silent design."
    
    prod_b = "Dell XPS 13 Intel Core Ultra 7 (16GB RAM, 512GB SSD)"
    prod_b_specs = "13.4-inch FHD+ Display, Intel Core Ultra 7 155H, 13hr battery life, 2.6 lbs weight, Windows 11 Home, active cooling fan."
    
    user_priorities = "Long battery life, quiet operation, smooth video editing, and lightweight portability for frequent travel."

    print("=" * 60)
    print("      PURCHASE DECISION AGENT (PROJECT #37)      ")
    print("=" * 60)
    print(f"Comparing: {prod_a} VS {prod_b}")
    print(f"Priorities: {user_priorities}\n")

    result = evaluate_purchase(prod_a, prod_a_specs, prod_b, prod_b_specs, user_priorities)
    
    print("=" * 60)
    print("                 FINAL VERDICT                   ")
    print("=" * 60 + "\n")
    print(result)