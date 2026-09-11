from fastmcp import FastMCP
from agent import app

mcp = FastMCP("AutoML Agent")

@mcp.tool()
def analyse_dataset(csv_content: str, goal: str) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
        
    result = app.invoke({
        "file_path": temp_path,
        "goal":goal,
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
        "retry_count": 0,
        "shap_plot": ""
    })

    return f"""
    ANALYSIS COMPLETED

    Best Model: {result["best_model"]}
    Model_Results: {result["model_results"]}

    SHAP PLOT: {result["shap_plot"]}
    
    SHAP Summary: {result["shap_summary"]}

    Full Report: {result["report"]}
    """

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)