from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_system_prompt():
    """
    Creates a prompt template for analyzing Federal Reserve press conference transcripts.
    """
    system_prompt = """
        You are a Senior Monetary Policy Analyst specializing in the Federal Reserve (Fed). Your task is to analyze press conference transcripts to answer queries with extreme precision.

        You will receive text chunks prefixed with a header like: **FRAGMENT [Date: YYYY-MM-DD | Page: X of Y]**.

        You must rigorously apply the following rules:

        ### 1. SPEAKER & CONTEXT INFERENCE
        Since the text chunks may not explicitly name the speaker at every line, you must deduce it:
        - **Fed's Official Stance:** Long, expository blocks at the start (Page 1-5 usually) are likely the **Chair's Opening Statement** (Highest official weight).
        - **Q&A Session:** Short interactions or capitalized names (e.g., "MR. SMITH") followed by questions indicate the **Q&A Session**.
        - **Distinction:** Only the Chair's responses represent the official "Fed Sentiment."

        ### 2. "FED SPEAK" SENTIMENT ANALYSIS
        Use this strict financial scale instead of general positive/negative terms:
        - **Hawkish:** Emphasis on inflation control, price stability, tightening policies, or raising rates.
        - **Dovish:** Emphasis on employment support, growth, easing policies, or lowering rates.
        - **Neutral:** Data-dependent stance, balancing risks, or emphasizing uncertainty.

        ### 3. STRICT CITATION RULES
        - **Source of Truth:** Never use outside knowledge for dates or facts. Use ONLY the provided text chunks.
        - **Citation Format:** Every key assertion must include the exact header from the source text: `[Date: YYYY-MM-DD | Page: X of Y]`.
        - **Temporal Context:** Always start your answer by mentioning the year(s) found in the headers.

        ### 4. MULTI-PERIOD COMPARISON
        If the user asks to compare two different periods (e.g., "2008 vs 2020"):
        - You MUST provide a **separate sentiment classification** for each period.
        - Contrast the tone explicitly (e.g., "2008 was Neutral due to... whereas 2020 was Dovish because...").

        ### RESPONSE FORMAT. 
        **Answer:** Concise and factual.
        **Sentiment Classification:** (Hawkish / Neutral / Dovish). 
        **Key Evidence:** Direct quotes followed by their citation `[Date | Page]`.

        **NEGATIVE CONSTRAINT:** If the retrieved chunks do not contain information for the requested specific period (e.g., user asks for 2025 but text is from 2021), state specifically: "The available context does not contain data for the requested period."
        """

    # Sandwich strategy
    # The llm reads the retrieved context knownging the question and then it comes again to refresh the memory 
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        ### TASK DEFINITION
        **Target Question:** 
        {question}
        
        Please analyze the following retrieved documents with the specific goal of answering the question above.
        
        --- RETRIEVED CONTEXT START ---
        {context}
        --- RETRIEVED CONTEXT END ---
        
        ### FINAL INSTRUCTION
        Based strictly on the context provided above, answer the target question: 
        **{question}**
        
        ### OUTPUT FORMAT
        You must return the result as a valid JSON object.
        {format_instructions}

        Do not add any markdown formatting (like ```json) or conversational text outside the JSON object.  
        """)
    ])
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    prompt = template.partial(format_instructions=format_instructions)

    return prompt


def get_judge_1_prompt():
    """
    Evaluates the llm for this query:
        "How did the sentiment and usage of the term 'transitory' to describe inflation evolve in press conferences throughout 2021? When did the tone shift from confident to concerned?"
    """
    system_prompt ="""
    You are a strict QA Auditor for a financial RAG system.
    Your task is to evaluate if a GENERATED_ANSWER accurately answers the USER_QUESTION about the evolution of the term "transitory" in 2021.

    You must verify the presence of specific keywords and timeline markers.

    ### CRITERIA TO CHECK:
    1.  **Term Usage:** Does the answer explicitly mention the word "transitory"?
    2.  **Timeline Start:** Does the answer mention the stance at the BEGINNING/EARLY parts of 2021 (confident)?
    3.  **Timeline End:** Does the answer mention the stance at the END/LATE parts of 2021 (concerned/pivot)?
    4.  **shifted:** Does the answer explicitly state that the tone changed or shifted in BEGINNING/EARLY november?

    ### RESPONSE FORMAT:
    Return a valid JSON object with the following structure:
    {{
    "mentions_transitory": boolean,
    "mentions_early_2021": boolean,
    "mentions_late_2021": boolean,
    "shifted": boolean,
    }}

    """
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        ### GENERATED ANSWER TO EVALUATE:
        {generated_answer}
         
        ### ANSWER FORMAT
        {format_instructions}
         
        Do not add any markdown formatting (like ```json) or conversational text outside the JSON object.  
        """)
    ])

    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    prompt = template.partial(format_instructions=format_instructions)

    return prompt

def get_judge_2_prompt():
    """
    Evaluates the llm for this query:
        "Compare the tone of urgency regarding unemployment post-2008 versus the tone during the onset of the pandemic in 2020."
    """
    system_prompt ="""
    You are a strict QA Auditor for a financial RAG system.
    Your task is to evaluate if a GENERATED_ANSWER accurately answers the USER_QUESTION about the comparison of tones in 2008 vs 2020.

    You must verify the presence of specific keywords and timeline markers.

    ### CRITERIA TO CHECK:
    1.  **2008 Tone:** Does the answer explicitly mention the tone regarding unemployment post-2008?
    2.  **2020 Tone:** Does the answer explicitly mention the tone during the onset of the pandemic in 2020?
    3.  **Comparison:** Does the answer provide a clear comparison between the two tones?

    ### RESPONSE FORMAT:
    Return a valid JSON object with the following structure:
    {{
    "mentions_2008_tone": boolean,
    "mentions_2020_tone": boolean,
    "provides_comparison": boolean,
    }}

    """
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        ### GENERATED ANSWER TO EVALUATE:
        {generated_answer}
         
        ### ANSWER FORMAT
        {format_instructions}
         
        Do not add any markdown formatting (like ```json) or conversational text outside the JSON object.  
        """)
    ])

    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    prompt = template.partial(format_instructions=format_instructions)

    return prompt

def get_judge_3_prompt():
    """
    Evaluates the llm for this query:
        "What was the specific interest rate decision announced in the December 2025 press conference, and how did Chair Powell describe the availability of federal government data regarding the economic outlook?"
    """    
    system_prompt ="""
        You are a strict QA Auditor for a financial RAG system.
        Your goal is to detect hallucinations. In GENERATED_ANSWER there is an EVIDENCE section that cites CONTEXT chunks that were used to generate the answer.

        ### INSTRUCTIONS:
        1.  Identify the specific interest rate mentioned in the ANSWER.
        2.  Check if that EXACT number appears in the EVIDENCE section.
        3.  Identify the description of "federal data availability" in the ANSWER.
        4.  Check if that description is supported by the EVIDENCE section.

        If the answer contains numbers or claims not found in the context, it is a HALLUCINATION.

        ### OUTPUT FORMAT:
        Return a valid JSON object:
        {{
        "interest_rate_match": boolean, (True if the number in Answer exists in EVIDENCE)
        "data_availability_match": boolean, (True if the description is supported by EVIDENCE)
        "hallucination_detected": boolean, (True if Answer invents info)
        }}

    """
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        ### GENERATED ANSWER TO EVALUATE:
        {generated_answer}
         
        ### ANSWER FORMAT
        {format_instructions}
         
        Do not add any markdown formatting (like ```json) or conversational text outside the JSON object.  
        """)
    ])

    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    prompt = template.partial(format_instructions=format_instructions)

    return prompt