from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    max_retries=2,
)

CONTINENT_RESPONSES = {
    "North America": "cold",
    "South America": "cloudy",
    "Europe": "rainy",
    "Asia": "humid",
    "Africa": "hot",
    "Oceania": "sunny",
}


@tool(description="Get the current weather for a given city")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    response = model.invoke(
        f"Which continent is {city} located in? Reply with only the continent name, which is one of {', '.join(CONTINENT_RESPONSES)}"
    )
    continent = response.text.strip().rstrip(".").title()

    weather = CONTINENT_RESPONSES.get(continent)
    if weather is None:
        return f"Not sure what the weather's like in {city}."
    return weather


def main() -> None:
    city = input("Which city do you want the weather for? ").strip()
    if not city:
        print("No city entered.")
        return
    
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful weather utility.",
    )

    template = "What's the weather in {city}? Include the population estimate(format 'POPULATION: POPULATION_NUMBER') and add a star emoji at the end if the city is the capital of its country"
    prompt = PromptTemplate.from_template(template)


    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=prompt.format(city=city)
                )
            ]
        }
    )
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()

