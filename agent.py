from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pandas as pd
import json
import os
import shap
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
import base64
import io

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

class AgentState(TypedDict):
    
    file_path: str        # uploaded CSV
    goal: str             # user's natural language goal
    problem_type: str     # classification / regression / clustering
    df_info: str          # string summary of the dataframe
    cleaned_data: str     # path to cleaned CSV after preprocessing
    features: list        # selected feature columns
    target: str           # target column name
    model_results: dict   # results from all trained models
    best_model: str       # name of winning model
    shap_summary: str     # SHAP explanation text
    report: str           # final report
    error: str            # any error caught for self-correction
    retry_count: int      # how many times self-correction has run
    shap_plot: str

def understand_problem(state: AgentState):

    data = pd.read_csv(state["file_path"])

    df_info = f"Shape: {data.shape}\nColumns: {data.dtypes}\nNulls:\n{data.isnull().sum()}\nSample:\n{data.head()}" 

    prompt = f"""
    You are a data science expert. 
    Dataset summary: {df_info}
    User goal: {state['goal']}

    Analyze this and return ONLY a JSON object with exactly these three keys:
        - "problem_type": one of "classification", "regression", or "clustering"
        - "target": the name of the target column as a string
        - "features": a list of feature column names

        Return JSON only. No explanation. No markdown. No extra text.
        """
    try:
        response = llm.invoke(prompt).content

        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response.strip())

        return {
            "df_info": df_info,
            "problem_type": result['problem_type'],
            "target": result['target'], 
            "features": result['features']
            }
    except Exception as e:
        return {"error": str(e), "df_info": df_info}

def clean_data(state: AgentState):

    raw_data = pd.read_csv(state["file_path"])

    prompt = f"""You are a Data Science Specialist
        You have a database whose information is stored in this {state['df_info']}
        based on this information of the database find out which cleaning steps are needed:
            -Which columns to drop
            -How to handle null
            -Which column needs encoding 
        Return ONLY a JSON with:
            - "columns_to_drop": list of column names to drop
            - "columns_to_encode": list of categorical columns needing encoding
            - "fill_null_strategy": either "mean", "median", or "drop"
        Return JSON only. No explanation. No markdown. No extra text.
        """
    
    try: 
        response = llm.invoke(prompt).content

        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response.strip())

        #Droping columns
        raw_data = raw_data.drop(columns = result["columns_to_drop"], errors = "ignore")

        #Handling Nulls
        if result["fill_null_strategy"] == "drop":
            raw_data = raw_data.dropna()
        elif result["fill_null_strategy"] == "mean":
            raw_data = raw_data.fillna(raw_data.mean(numeric_only=True))
        elif result["fill_null_strategy"] == "median":
            raw_data = raw_data.fillna(raw_data.median(numeric_only=True))

        #Encoding column
        raw_data = pd.get_dummies(raw_data, columns = result["columns_to_encode"])
    
        cleaned_path = os.path.join(os.path.dirname(state["file_path"]), "cleaned_data.csv")
        raw_data.to_csv(cleaned_path, index=False)
        return {"cleaned_data": cleaned_path}

    except Exception as e:
        return {"error": str(e)}
        
def feature_engineering(state: AgentState):
    
    cleaned_data = pd.read_csv(state["cleaned_data"])

    prompt = f"""
        You are an Data Science Expert.
        You have these informations about an dataset
            -{state["df_info"]}
            -{state["target"]}
            -{state["features"]}
        By these information about the dataset are there any 
        new usefull features can be created from the existing columns:
        Return JSON list of new features to create as pandas expressions
        Return ONLY a JSON with:
            - "new_features": a list of objects, each with "name" and "expression" keys
        Return JSON only. No explanation. No markdown. No extra text.
    """

    try:
        response = llm.invoke(prompt).content

        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response.strip())

        new_features = result.get("new_features", [])

        for feature in new_features:
            try:
                cleaned_data[feature["name"]] = eval(feature["expression"].replace("df", "cleaned_data"))
            except:
                pass

        updated_features = state["features"] + [f["name"] for f in new_features]

        cleaned_path = os.path.join(os.path.dirname(state["file_path"]), "cleaned_data.csv")
        cleaned_data.to_csv(cleaned_path, index=False)
        return {"features": updated_features, "cleaned_data": cleaned_path}

    except Exception as e:
            return {"error": str(e)}

def train_models(state: AgentState):

    cleaned_data = pd.read_csv(state["cleaned_data"])

    try:
        available_features = [f for f in state["features"] if f in cleaned_data.columns]
        X = cleaned_data[available_features]
        y = cleaned_data[state["target"]]

        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state = 42
        )

   

        if state["problem_type"] == "classification":
            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier()
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train,y_train)
                preds = model.predict(X_test)
                results[name] = accuracy_score(y_test, preds)

        elif state["problem_type"] == "regression":
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegression": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor()
            }
            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = np.sqrt(mean_squared_error(y_test, preds))

        return {"model_results": results}

    except Exception as e:
        return {"error": str(e)}

def select_best_model(state: AgentState):

    try:
        results = state["model_results"]
        if state["problem_type"] == "classification":
            model = max(results, key=results.get)

        elif state["problem_type"] == "regression":
            model = min(results, key=results.get)

        return {"best_model": model}
    
    except Exception as e:
        return {"error": str(e)}

def explain_with_shap(state: AgentState):

    try:
        data = pd.read_csv(state["cleaned_data"])
        available_features = [f for f in state["features"] if f in data.columns]
        X = data[available_features]
        y = data[state["target"]]

        model_map = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "LinearRegression": LinearRegression(),
        "RandomForestRegression": RandomForestRegressor(),
        }
        best_model = model_map[state["best_model"]]
        best_model.fit(X,y)

        explainer = shap.Explainer(best_model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show = False)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        shap_plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        shap_importance = pd.DataFrame({
            "feature": available_features,
            "importance": abs(shap_values.values).mean(axis = 0)
        }).sort_values("importance", ascending=False).to_string()

        prompt = f"""
            You are an Data Science Specialist.
            You have the shap values regarding an trained model {shap_importance}
            Based on these values:
                Explain the Top features in Plain English.
        """
        response = llm.invoke(prompt).content

        return {"shap_summary": response, "shap_plot": shap_plot_base64}

    except Exception as e:
        return {"error": str(e)}

def generate_report(state: AgentState):

    try:
        prompt = f"""
            You are an Data Science Expert.
            You have access to these following documents:
                -User's Goal- {state['goal']}
                -Problem Type- {state['problem_type']}
                -Target- {state['target']}
                -Best model for this- {state['best_model']}
                -The model Results- {state['model_results']}
                -Shap summary- {state['shap_summary']}
            You have to create an clean structured report based on these information as resource
            Report should be like that a non-Technical person can understand.
            Report should include:
                -What problem was solved
                -What the best model was and its score
                -What the key features were (from SHAP)
                -What the results mean in plain English
            Write in a friendly, clear tone. Use sections and bullet points where helpful.
            """

        response = llm.invoke(prompt).content

        return {"report": response}

    except Exception as e:
        return {"error": str(e)}

def self_correct(state: AgentState):

    try:
        prompt = f"""
            You are an Debugging Expert.
            This error occured in data analysis pipeline: {state['error']}
            The user's Goal was: {state['goal']}
            Suggest a Brief fix. BE CONCISE
        """

        response = llm.invoke(prompt).content

        return{
            "error": "",
            "retry_count": state["retry_count"] + 1
            }
    
    except Exception as e:
        return {"error": str(e)}

def should_self_correct(state: AgentState):

    if state.get("error") and state.get("retry_count", 0) < 3:
        return "self_correct"
    elif state.get("error") and state.get("retry_count", 0) >= 3:
        return "continue"
    return "continue"

graph = StateGraph(AgentState)

graph.add_node("understand_problem", understand_problem)
graph.add_conditional_edges("understand_problem", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "clean_data"
})

graph.add_node("clean_data", clean_data)
graph.add_conditional_edges("clean_data", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "feature_engineering"
})

graph.add_node("feature_engineering", feature_engineering)
graph.add_conditional_edges("feature_engineering", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "train_models"
})

graph.add_node("train_models", train_models)
graph.add_conditional_edges("train_models", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "select_best_model"
})

graph.add_node("select_best_model", select_best_model)
graph.add_conditional_edges("select_best_model", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "explain_with_shap"
})

graph.add_node("explain_with_shap", explain_with_shap)
graph.add_conditional_edges("explain_with_shap", should_self_correct, {
    "self_correct": "self_correct",
    "continue": "generate_report"
})

graph.add_node("generate_report", generate_report)
graph.add_conditional_edges("generate_report", should_self_correct, {
    "self_correct": "self_correct",
    "continue": END
})

graph.add_node("self_correct", self_correct)
graph.add_edge("self_correct", "understand_problem")
graph.set_entry_point("understand_problem")
app = graph.compile()


if __name__ == "__main__":
    file_path = r"D:/Auto ML/Titanic-Dataset.csv"
    goal = input("Enter your analysis goal: ")
    
    result = app.invoke({
        "file_path": file_path,
        "goal": goal,
        "problem_type": "",
        "df_info": "",
        "cleaned_data": "",
        "features": [],
        "target": "",
        "model_results": {},
        "best_model": "",
        "shap_summary": "",
        "report": "",
        "error": "",
        "retry_count": 0
    })
    
    print("\n===== ANALYSIS REPORT =====")
    print(result["report"])
    print("\nBest Model:", result["best_model"])
    print("Model Results:", result["model_results"])