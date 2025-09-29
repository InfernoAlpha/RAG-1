from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage
from typing import TypedDict,Annotated,Literal
from langchain_core.prompts import PromptTemplate
from torch import load
import sqlite3
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sih_tool import predict_crop_yield
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

class llm_state(TypedDict):
    emotion:str
    messages:Annotated[list[BaseMessage],add_messages]

class sih_tool_scheme(BaseModel):
    Region:Literal['North','South','East','West'] = Field(..., description="which part or region of india")
    Soil_Type:Literal['Sandy','Clay','Loam','Silt','Chalky','Peaty'] = Field(..., description="Soil type category")
    Crop:Literal['Maize','Rice','Wheat','Barley','Cotton','Soybean'] = Field(..., description="Crop type category")
    Rainfall_mm: float = Field(..., description="Rainfall in millimeters")
    Temperature_Celsius: float = Field(..., description="Temperature in Celsius")
    Fertilizer_Used: bool = Field(..., description="Whether fertilizer was used")
    Irrigation_Used: bool = Field(..., description="Whether irrigation was used")
    Weather_Condition:Literal['Sunny','Rainy','Cloudy'] = Field(..., description="Weather condition")

search_tool = DuckDuckGoSearchRun(region = "in-en")

@tool
def yield_pred(data:sih_tool_scheme):
    """
    Predict crop yield in tons per hectare given environmental and farming conditions.
    """
    print(data)
    y_pred = predict_crop_yield(data.model_dump())
    print(y_pred)
    return {'predicted_yield_tons_per_hectare':y_pred}
    
    
"""def cond(state:llm_state):
    msg = state["messages"][-1]
    if msg.name == "mental_health_advice":
        return "end"
    else:
        return "continue"
"""

tools  = [search_tool,yield_pred]
tool_node = ToolNode(tools)

def llm_node(state:llm_state)->llm_state:
    #model = ChatOpenAI()
    model = ChatOllama(model="gpt-oss:20b",temperature=0.5)
    #model = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)
    model = model.bind_tools(tools)
    response = model.invoke(state['messages'])
    return {"messages":[response]}

connection = sqlite3.connect(database='chat_history.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(llm_state)
graph.add_node("llm",llm_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"llm")
graph.add_conditional_edges("llm",tools_condition)
graph.add_edge("tools","llm")
#graph.add_edge('llm',END)

agent = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

