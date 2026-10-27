from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main() -> None:

    information = """

Satoru Iwata (Japanese: 岩田 聡; December 6, 1959 - July 11, 2015) was a Japanese businessman, video game programmer, and producer. Beginning in 2002, Iwata served as the fourth president of Nintendo, as well as the chief executive officer (CEO) of Nintendo of America from 2013 until his death in 2015. Iwata was a major contributor in broadening the appeal of video games by focusing on novel and entertaining software rather than top-of-the-line hardware.

Born in Sapporo, Iwata expressed interest in video games from an early age and created his first simple game while in high school. He majored in computer science at the Tokyo Institute of Technology. In 1980, he joined the game developer HAL Laboratory while attending the university. At HAL, he worked as a programmer and closely collaborated with Nintendo, coding his first commercial console game in 1983 and programming early Nintendo Entertainment System titles including Pinball, Golf, and Balloon Fight. Later games to which he contributed include EarthBound and numerous installments in the Kirby series. Following a financial downturn and near-bankruptcy, Iwata became the president of HAL in 1993 at the insistence of Nintendo president Hiroshi Yamauchi and restored the company's financial stability.
    """

    summary_template = """
    Given the information {information} about a person I want you to create 
    1. A short summary, featuring name, birth year(and death if they've passed away), occupation, and main thing they're known for
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        max_retries=2,
    )

    chain = summary_prompt_template | model
    response = chain.invoke(input={"information": information})
    print(response.text)


if __name__ == "__main__":
    main()

