import streamlit as st8
from task import *

# Streamlit main code

# Streamlit Constants
height = 700
title = "Bonus Task"
icon = ":robot:"

# Set page title and icon
st8.set_page_config(page_title=title, page_icon=icon)

# Initialize conversation history
if "conversation" not in st8.session_state:
    st8.session_state.conversation = []

# Initialize user questions history
if "user_questions_arr" not in st8.session_state:
    st8.session_state.user_questions_arr = []

user_questions_arr = []

st8.write("For each question you ask, you will get 6 responses (3 RAG, 3 Prompt)")
# Chat container
messages = st8.container(border=True)

def writeAsAssistant(text):
    st8.session_state.conversation.append({
        "assistant": text,
    })

# writeAsAssistant("For each question you ask, you will get 6 responses (3 RAG, 3 Prompt)")
# writeAsAssistant("After first 3 questions, you will get evaluations of the RAG")

def standard_response(query):

    writeAsAssistant("Link of document: https://arxiv.org/pdf/2503.18968.pdf")
    writeAsAssistant("RAG 1: Basic RAG with Document Stuffing")
    writeAsAssistant(basic_rag(query))

    writeAsAssistant("RAG 2: Contextual Compression RAG")
    writeAsAssistant(compression_rag(query))

    writeAsAssistant("RAG 3: Hybrid search RAG (Dense + Sparse retrieval)")
    writeAsAssistant(hybrid_rag(query))

    writeAsAssistant("Prompt 1: Zero Shot Prompt")
    writeAsAssistant(zero_shot_prompt(query))

    writeAsAssistant("Prompt 2: Chain of Thought Prompt")
    writeAsAssistant(chain_of_thought_prompt(query))

    writeAsAssistant("Prompt 3: Role Prompt")
    writeAsAssistant(role_prompt("an expert and prominent figure of the field related to the task",query))





def evaluation():
    writeAsAssistant("Comparing RAG Architectures")
    comparison_results = compare_rag_architectures(st8.session_state.user_questions_arr)

    # Display comparative analysis
    writeAsAssistant("\nComparative Analysis of RAG Architectures:\n")
    for analysis in comparison_results["comparative_analysis"]:
        writeAsAssistant(f"Query: {analysis['query']}")
        writeAsAssistant(f"Analysis: {analysis['comparison']}")
    

    overall_recommendation = second_llm_eval.invoke(recommendation_prompt).content
    writeAsAssistant("\nOverall Recommendation for RAG Architecture Selection:\n")
    writeAsAssistant(overall_recommendation)




# Call backend for response
def generate_message(user_input):
    
    try:
        st8.session_state.conversation.append({"user": user_input})

        # writeAsAssistant("testing")

        # Check the length of the array
        if len(st8.session_state.user_questions_arr) < 3:
            st8.session_state.user_questions_arr.append(user_input)
            standard_response(user_input)
            # writeAsAssistant(st8.session_state.user_questions_arr)

        else:
            writeAsAssistant("limit of 3 reached, lastest input will be ignored.")
            # writeAsAssistant(st8.session_state.user_questions_arr)
            # writeAsAssistant("RAGs will be evaluated based on the above 3 questions")
            evaluation()
            st8.session_state.user_questions_arr.clear()
        
        # Iterate over conversation history
        for entry in st8.session_state.conversation:
            if 'user' in entry and entry['user']:
                messages.chat_message("user").write(entry['user'])

            if 'assistant' in entry and entry['assistant']:
                messages.chat_message("assistant").write(entry['assistant'])
    
    except Exception as e:
        return "Exception occurred here 2: " + str(e)



# for users to enter questions, this will call the API in backend
if prompt := st8.chat_input("Enter your question", key="prompt"):
    generate_message(prompt)
