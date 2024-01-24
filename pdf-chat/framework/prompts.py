
from langchain.prompts import PromptTemplate, ChatPromptTemplate

SOB_UNI_AGR_ANALYZER_TEMPLATE = """

You are an expert in the field of the collective bargaining agreement analysis which is created by the retail company and the union.
You read the information below and answer the questions based on the information in a clear, crisp and concise form. 

If the information is not found in the knowledgebase provided in the context to answer the question, say "Not specified".
Answer the question in the same language of the question.
Answer **must** be based only on the known context:

{context}

DO NOT PREFIX YOUR ANSWER WITH ANYTHING. JUST ANSWER THE QUESTION.

Question: {question}

Answer:
"""

SOB_UNI_AGR_ANALYZER_PROMPT = ChatPromptTemplate.from_template(
    SOB_UNI_AGR_ANALYZER_TEMPLATE)


SOB_UNI_AGR_ANALYZER_TEMPLATE_v2 = """

You are an expert in the field of the collective bargaining agreement analysis which is created by the retail company and the union.
You read the information below and answer the questions based on the information in a clear, crisp and concise form. 

If the information is not found in the knowledgebase provided in the context to answer the question, say "Not specified".
Answer the question in the same language of the question.
Answer **must** be based only on the known context:

{context}

DO NOT PREFIX YOUR ANSWER WITH ANYTHING. JUST ANSWER THE QUESTION.

Consider this bargaining agreement is applicable for various category of workers such as regular, part time, temporary, seasonal, etc.
You should extract the information for the category of worker provided below and then answer the question.

Category of worker: {persona}

Question: {question}

Answer:
"""

SOB_UNI_AGR_ANALYZER_PROMPT_v2 = ChatPromptTemplate.from_template(
    SOB_UNI_AGR_ANALYZER_TEMPLATE_v2)


SOB_UNI_AGR_ANALYZER_FR_TEMPLATE = """
Vous êtes un expert dans le domaine de l’analyse des conventions collectives qui est créée par l’entreprise de vente au détail et le syndicat.
Vous lisez les informations ci-dessous et répondez aux questions basées sur les informations sous une forme claire, nette et concise. 

Si l’information ne se trouve pas dans la base de connaissances fournie dans le contexte pour répondre à la question, dites « Non spécifié ».
Répondez à la question dans la même langue que celle de la question.
La réponse **doit** être basée uniquement sur le contexte connu :

{context}

NE FAITES PAS PRÉCÉDER VOTRE RÉPONSE DE QUOI QUE CE SOIT. IL SUFFIT DE RÉPONDRE À LA QUESTION.

Considérez que cette convention collective s’applique à diverses catégories tels que les salariés réguliers, salariés partiels, salariés Semaine réduite, etc.
Vous devez tenir compte de la catégorie indiquée ci-dessous lorsque vous répondez à la question, le cas échéant. Et si ce n’est pas le cas, considérez la question pour la catégorie des travailleurs réguliers.

Catégorie de travailleur : {persona}

Question : {question}

Répondre:

"""

SOB_UNI_AGR_ANALYZER_PROMPT_FR = ChatPromptTemplate.from_template(
    SOB_UNI_AGR_ANALYZER_FR_TEMPLATE)
