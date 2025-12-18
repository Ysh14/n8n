import os
import json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_models import ChatOllama # <-- MODIFIED
from langchain_core.pydantic_v1 import BaseModel, Field

# Ensure the following imports are correct for your local setup
from iac_tool_wrappers import validate_terraform_code
from rag_retriever import iac_retriever

# 1. Define the State (No Change)
class AgentState(TypedDict):
    user_prompt: str
    code: str
    validation_output: str
    iterations: int
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# 2. Define the Agent Tool and LLM (MODIFIED FOR OLLAMA)
engineer_agent_tools = [validate_terraform_code]

# Initialize Ollama LLM with Llama 3 model
# Ensure OLLAMA_BASE_URL is set as environment variable
# NOTE: Llama 3's tool-calling/function-calling capability is leveraged here.
llm = ChatOllama(model="llama3", temperature=0) # <-- MODIFIED

# IMPORTANT: Ollama tool-calling often requires the `bind_tools` function 
# to wrap the tool-calling logic inside the graph. 
# We'll rely on the LangGraph/LangChain orchestration to manage tool-calling 
# capabilities built into the ChatOllama interface.

# 3. Define the Nodes (Functions remain the same, logic relies on the agent prompt)

def generate_iac(state: AgentState) -> AgentState:
    """Node: Generates or regenerates IaC code using RAG context."""
    print("--- GENERATING/REGENERATING CODE ---")
    
    # 1. Get RAG context
    context = iac_retriever.get_context(state['user_prompt'])
    
    # 2. System Prompt (Key for self-correction and RAG grounding)
    system_prompt = f"""
    You are an expert IaC Engineer using the Llama 3 model. Your goal is to generate robust, clean, and valid 
    Terraform HCL code to fulfill the user's request.
    
    RULES:
    1. **USE RAG CONTEXT:** Refer to the provided documentation context for resource schemas and required arguments.
    2. **USE TOOL:** You MUST call the `validate_terraform_code` tool with the generated Terraform code.
    3. **SELF-CORRECT:** If the validation tool returns 'FAIL', analyze the 'details' error message and rewrite the code 
       in the next attempt. You are debugging your own code.

    --- RAG CONTEXT (Max 4 chunks) ---
    {context}
    ----------------------------------
    """
    
    # Build the message history for the LLM
    # Append the last tool execution result if this is a correction iteration
    if state['iterations'] > 0:
        # Pass the validation output as a human message so the model can see the failure
        correction_message = f"VALIDATION FAILED. Please review the error and CORRECT the code. \nError details: {state['validation_output']}"
        state['messages'].append(HumanMessage(content=correction_message))
        
    state['messages'].insert(0, SystemMessage(content=system_prompt))
    
    # 3. Invoke the LLM with the tool
    # NOTE: We bind the tools for each call, as required by some LLM integrations
    response = llm.bind_tools(engineer_agent_tools).invoke(state['messages'])
    state['messages'].append(response)
    
    # 4. Extract code from the tool call arguments
    # LangChain handles parsing the structured tool call from the Llama 3 response
    try:
        tool_call = response.tool_calls[0]
        state['code'] = tool_call['args']['code']
    except (IndexError, KeyError):
        # Handle cases where the model fails to call the tool correctly
        state['code'] = "ERROR: Model failed to generate a structured tool call. Check system prompt compliance."
        state['validation_output'] = json.dumps({"status": "FAIL", "reason": "TOOL_CALL_ERROR", "details": state['code']})
        
    return state

def run_validation_tool(state: AgentState) -> AgentState:
    """Node: Executes the deterministic IaC validation tool."""
    print("--- RUNNING VALIDATION TOOL ---")
    
    # Execute the wrapped tool function from iac_tool_wrappers.py
    tool_output = validate_terraform_code.invoke({"code": state['code']})
    
    state['validation_output'] = tool_output
    state['iterations'] += 1
    
    # Add tool message to history for the next iteration
    # NOTE: This message helps the LLM see the explicit result of its action
    state['messages'].append(HumanMessage(content=f"Tool Output:\n{tool_output}"))
    
    return state

# 4. Define the Conditional Edge (Router) - (No Change)
def route_to_next_step(state: AgentState) -> str:
    """Router: Decides if the loop should continue or end."""
    validation_result = json.loads(state['validation_output'])
    
    if validation_result.get("status") == "PASS":
        return "end"
    
    if state['iterations'] >= 3: # Max 3 attempts
        print("--- MAX ITERATIONS REACHED. MANUAL REVIEW REQUIRED. ---")
        return "end"
        
    return "generate_iac"

# 5. Build and Compile the LangGraph (No Change)
workflow = StateGraph(AgentState)
workflow.add_node("generate_iac", generate_iac)
workflow.add_node("run_validation_tool", run_validation_tool)
workflow.add_edge("generate_iac", "run_validation_tool")
workflow.add_conditional_edges(
    "run_validation_tool",
    route_to_next_step,
    {
        "end": END,
        "generate_iac": "generate_iac"
    }
)
workflow.set_entry_point("generate_iac")
app = workflow.compile()

# 6. Execution Example
if __name__ == "__main__":
    initial_state = {
        # This prompt requires the LLM to use the RAG context to find the 
        # correct resource type and required arguments.
        "user_prompt": "Create an AWS S3 bucket named 'my-test-bucket-1234' with versioning enabled.",
        "code": "",
        "validation_output": "",
        "iterations": 0,
        "messages": [HumanMessage(content="Create an AWS S3 bucket named 'my-test-bucket-1234' with versioning enabled.")],
    }
    
    print("--- STARTING PoC EXECUTION ---")
    
    # Ensure you have the `terraform` CLI installed and available in your shell
    final_state = app.invoke(initial_state)

    print("\n\n####################################")
    print("### FINAL VALIDATED IAC CODE ###")
    print("####################################")
    print(final_state['code'])