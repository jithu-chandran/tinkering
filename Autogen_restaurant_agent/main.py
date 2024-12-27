
from typing import Dict, List, Tuple

# Importing necessary modules for conversational agents and additional utilities
from autogen import ConversableAgent
import numpy as np
import sys
import os

# Load environment variables from a .env file for secure configuration
from dotenv import load_dotenv
load_dotenv()

# Importing specific types for type annotations
from typing import Dict, List


# Class to manage memory for storing and retrieving reviews and scores
class AgentMemoryStore:
    
    def __init__(self):
        # Initialize lists to store reviews, food scores, and customer service scores
        # Initialize a counter to track review retrieval
        self.reviews_list = []
        self.get_counter = 0
        self.food_scores = []
        self.customer_service_scores = []

    def push_review_list(self, review_list: List[str]) -> None:
        # Store a list of reviews in the memory
        self.reviews_list = review_list

    def get_review(self) -> str:
        # Retrieve the next review from the list and increment the counter
        if self.get_counter < len(self.reviews_list):
            review = self.reviews_list[self.get_counter]
            self.get_counter += 1
            return review
        else:
            # Return "TERMINATE" when all reviews have been retrieved
            return "TERMINATE"
    
    def push_scores(self, food_score: float, customer_service_score: float) -> None:
        # Append the provided food and customer service scores to their respective lists
        self.food_scores.append(food_score)
        self.customer_service_scores.append(customer_service_score)
    
    def get_scores(self) -> Tuple[List[float], List[float]]:
        # Return the lists of food and customer service scores
        return (self.food_scores, self.customer_service_scores)


# Function to fetch restaurant reviews from a text file based on the restaurant name
def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # Open and read the restaurant data file
    review_doc = open("restaurant-data.txt", 'r').read()
    # Split the file contents into a list of reviews
    review_list = review_doc.splitlines()
    # Filter reviews that match the provided restaurant name
    tmp_restaurant_data_list = [review for review in review_list if restaurant_name.lower() in review.lower()]
    # Create a dictionary with the restaurant name as the key and the filtered reviews as the value
    tmp_restaurant_data_dict = {restaurant_name: tmp_restaurant_data_list}
    return tmp_restaurant_data_dict

# Function to calculate the overall score of a restaurant based on food and customer service scores
def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # Convert the scores to NumPy arrays for mathematical operations
    np_food_scores = np.array(food_scores)
    np_customer_service_scores = np.array(customer_service_scores)
    # Calculate the total number of scores
    N = len(food_scores)
    # Compute the overall score using a weighted formula
    overall_score = np.sum(np.sqrt(np_food_scores**2 * np_customer_service_scores) * 1/(N * np.sqrt(125))) * 10
    # Format the score to 3 decimal places
    overall_score = f"{overall_score:.3f}"
    return overall_score


# Function to generate a system message for the data-fetching agent
def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    # Provide instructions for extracting reviews based on the restaurant query
    prompt = """
You are an helpful AI assistant to fetch reviews related to restaurants.

TASK:
You are given user queries about specific restauratnts. You are supposed to identify the restaurant that the user\
is referring to and extract the reviews corresponding to that restaurant. Do not do anything else.

Matching with the restaurante name:
The restaurant name would always be one from the below list. Hence, use the name from the below list that is the closes match to \
what the user is referring to.
restaurant_name_list = List["Applebee's" 'Buffalo Wild Wings' 'Burger King' 'Cheesecake Factory'
 'Chick-fil-A' 'Chipotle' 'Cinnabon' 'Five Guys' 'IHOP' 'In-n-Out'
 'Krispy Kreme' "McDonald's" 'Olive Garden' 'Panda Express' 'Panera Bread'
 'Pret A Manger' 'Starbucks' 'Subway' 'Taco Bell' "Tim Horton's"]

**INSTRUCTION:
IF YOU HAVE ACHIEVED YOUR COMPLETE TASK THEN PROVIDE THE SOLUTION AND APPEND THE WORD "TERMINATE". \
IN THIS CASE DO NOT INCLUDE ANY OTHER ADDITIONAL TEXT. ALSO, DO NOT MISS OR IGNORE ANY OF THE REVIEWS. **

"""
    return prompt

# Function to generate a system message for the review analysis agent
def get_analysis_agent_prompt(restaurant_query: str) -> str:
    # Provide instructions for scoring reviews based on keywords
    prompt = """
You are an expert in analyzing a restaurant review and calculating food_score and customer_service_score for the review. \
You extract these two scores by looking for keywords in the review. Each review has keyword adjectives that correspond \
to the score that the restaurant should get for its `food_score` and `customer_service_score`. \
Here are the keywords the agent should look out for:

- Score 1/5 if the review has one of these adjectives: awful, horrible, or disgusting.
- Score 2/5 if the review has one of these adjectives: bad, unpleasant, or offensive.
- Score 3/5 if the review has one of these adjectives: average, uninspiring, or forgettable.
- Score 4/5 if the review has one of these adjectives: good, enjoyable, or satisfying.
- Score 5/5 if the review has one of these adjectives: awesome, incredible, or amazing.

Each review will have exactly only two of these keywords (adjective describing food and adjective describing customer service), \
and the score (N/5) is only determined through the above listed keywords. No other factors go into score extraction. \
To illustrate the concept of scoring better, here's an example review. 

Your response for each review must be strictly in the below JSON format.
{food_scores: int, customer_service_scores: int}

**STEPS TO FOLLOW:
1. Retrieve one review at a time.
2. Score each review to provide the food_score and customer_service_score.
3. Push these scores into the memory.
4. Repeat the above steps until you receive the message: TERMINATE.
5. Post Step 4, respond with the word TERMINTAE.

"""
    return prompt

# Function to generate a system message for the overall scoring agent
def overall_scoring_agent_prompt(restaurant_query: str) -> str:
    # Provide instructions for calculating the overall score
    prompt = """
You are an helpful AI asssitant. Your task is to calculate the overall score for a given restaurant.

**Steps to follow:
1. Get the list of food_score and customer_service_score from the memory.
2. Calculate the overall score.
3. Your response is the overall score in float format. ENSURE THAT THERE ARE # DECIMALS DISPLAYED, NOTHING LESS OR MORE.

**INSTRUCTION:
IF YOU HAVE ACHIEVED YOUR COMPLETE TASK THEN PROVIDE THE SOLUTION AND APPEND THE PHRASE **", TERMINATE"** in a new line. \
IN THIS CASE DO NOT INCLUDE ANY OTHER ADDITIONAL TEXT. ALSO, DO NOT MISS OR IGNORE ANY OF THE SCORES.
"""
    return prompt

# Functions for summarizing messages exchanged between agents
def analysis_agent_summary_method(sender: ConversableAgent, recipient: ConversableAgent, summary_args: dict):
    # Retrieve the second-to-last message exchanged between sender and recipient
    return recipient.chat_messages_for_summary(sender)[-2]["content"]

def data_agent_summary_method(sender: ConversableAgent, recipient: ConversableAgent, summary_args: dict):
    # Retrieve the second-to-last message exchanged between sender and recipient
    summary_message = recipient.chat_messages_for_summary(sender)[-2]['content']
    return summary_message


# Do not modify the signature of the "main" function.
def main(user_query: str):
    # Create an instance of AgentMemoryStore to manage review and score data
    agent_memory = AgentMemoryStore() # AGENT MEMORY INSTANCE 

    ####### WRAPPERS START ########

    # Wrapper to store a list of reviews in memory
    def push_review_list_wrapper(review_list: List[str]) -> str:
        return agent_memory.push_review_list(review_list)

    # Wrapper to retrieve the next review from memory
    def get_review_wrapper() -> str:
        return agent_memory.get_review()

    # Wrapper to store food and customer service scores in memory
    def push_scores_wrapper(food_score: float, customer_service_score: float) -> str:
        return agent_memory.push_scores(food_score, customer_service_score)

    # Wrapper to retrieve lists of scores from memory
    def get_scores_wrapper() -> Tuple[List[float], List[float]]:
        return agent_memory.get_scores()

    ####### WRAPPERS END ########
    
    entrypoint_agent_system_message = """
You are an helpful AI assistant. When given a user query about any restaurant, you work with other agents and \
    tools to provide a response and an overall score for the quality of the restaurant and a review summary for that restaurant."""

# TODO
    # example LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",
                                        is_termination_msg=lambda msg: (isinstance(msg.get("content"), str) 
                                                                        and "terminate" in msg["content"].lower()))
    
    # Register execution functions for the entrypoint agent
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="get_review")(get_review_wrapper)
    entrypoint_agent.register_for_execution(name="push_scores")(push_scores_wrapper)
    entrypoint_agent.register_for_execution(name="get_scores")(get_scores_wrapper)
    entrypoint_agent.register_for_execution(name="push_review_list")(push_review_list_wrapper)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

    # TODO
    # Create additional agents and their interactions
    data_fetch_agent = ConversableAgent("data_fetch_agent", 
                                        system_message=get_data_fetch_agent_prompt(user_query), 
                                        llm_config=llm_config,
                                        # max_consecutive_auto_reply=1,
                                        human_input_mode="NEVER")
    # Register data-fetch-related functions with the data fetch agent
    data_fetch_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    data_fetch_agent.register_for_llm(name="push_review_list",description="Stores the list of reviews of a specific restaurant into \
                                      the memory")(push_review_list_wrapper)

    analysis_agent = ConversableAgent("analysis_agent", 
                                            system_message=get_analysis_agent_prompt(user_query), 
                                            llm_config=llm_config,
                                            # max_consecutive_auto_reply=1,
                                            human_input_mode="NEVER")
    # Register review-scoring functions with the analysis agent
    analysis_agent.register_for_llm(name="get_review",description="Each call to this function retrieves one new review from the \
                                    list of reviews in the memory. Once all the reviews are retrieved once, subsequent call to the function \
                                    returns a message: \"NO MORE REVIEWS\".")(get_review_wrapper)
    
    analysis_agent.register_for_llm(name="push_scores",description="Pushes the food_score and customer_review_score \
                                    calculated for a single review into the memory")(push_scores_wrapper)

    
    overall_scoring_agent = ConversableAgent("overall_scoring_agent", 
                                            system_message=overall_scoring_agent_prompt(user_query), 
                                            llm_config=llm_config,
                                            # max_consecutive_auto_reply=1,
                                            human_input_mode="NEVER")
    # Register scoring functions with the overall scoring agent
    overall_scoring_agent.register_for_llm(name="calculate_overall_score", description="Calculates the overall score for a given restaurant")(calculate_overall_score)
    overall_scoring_agent.register_for_llm(name="get_scores",description="Gets the list of food_scores and customer_review_scores \
                                    calculated for all the reviews pertaining to the given restaurant")(get_scores_wrapper)


    # Initialize interactions between the agents and provide results
    result = entrypoint_agent.initiate_chats(
        
        [
            {
                "recipient": data_fetch_agent,
                "message": user_query,
                "max_turns": 4,
                "summary_method": data_agent_summary_method,
                "human_input_mode" : "NEVER"
            },

            {
                "recipient": analysis_agent,
                "message": user_query,
                "max_turns": None,
                "summary_method": analysis_agent_summary_method,
                "human_input_mode": "NEVER"
            },

            {
                "recipient": overall_scoring_agent,
                "message": user_query,
                "max_turns": 10,
                "summary_method": "last_msg",
            }                                                                         
        
        ]
    )
    
    
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
