import minsearch
from groq import Groq


def search(index: minsearch.Index, query: str, num_results: int = 10) -> list[dict[str, str]]:
    boost = {"question": 3.0, "section": 0.5}
    return index.search(
        query=query, filter_dict={"course": "data-engineering-zoomcamp"}, boost_dict=boost, num_results=num_results
    )


def build_prompt(query: str, search_results: list[dict[str, str]]) -> str:
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
    context = ""

    for doc in search_results:
        context = context + f'section: {doc["section"]}\nquestion: {doc["question"]}\nanswer: {doc["text"]}\n\n'

    return prompt_template.format(question=query, context=context).strip()


def llm(client: Groq, prompt: str) -> str | None:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


def rag(client: Groq, index: minsearch.Index, question: str) -> str | None:
    search_results = search(index, question, num_results=5)
    prompt = build_prompt(question, search_results)
    return llm(client, prompt)
