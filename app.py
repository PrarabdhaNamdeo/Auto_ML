import streamlit as st
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import base64
from PIL import Image
import io

async def run_analysis(csv_content: str, goal: str):
    transport = StreamableHttpTransport("https://web-production-3340c.up.railway.app/mcp")
    async with Client(transport) as client:
        result = await client.call_tool("analyse_dataset", {
            "csv_content": csv_content,
            "goal": goal
        })
        return result

# Session state initialization
if "report" not in st.session_state:
    st.session_state["report"] = ""
if "stage" not in st.session_state:
    st.session_state["stage"] = "upload"

st.title("AutoML") 

if st.session_state["stage"] == "upload":
    uploaded_file = st.file_uploader("Upload your CSV here ", type="csv")
    goal = st.text_input("Enter Your Goal: ")
    
    if st.button("Analyse Dataset"):
        if uploaded_file and goal:
            csv_string = uploaded_file.read().decode("utf-8")
            with st.spinner("Evaluating and generating result..."):
                result = asyncio.run(run_analysis(csv_string, goal))
                st.session_state["report"] = result
                st.session_state["stage"] = "results"

if st.session_state["stage"] == "results":
    report_text = st.session_state["report"].content[0].text

    if "SHAP PLOT:" in report_text:
        parts = report_text.split("SHAP PLOT:")
        before_plot = parts[0].strip()
        after_plot = parts[1].split("SHAP Summary:")[1] if "SHAP Summary:" in parts[1] else parts[1]
        shap_summary = parts[1].split("SHAP Summary:")[1].split("Full Report:")[0] if "SHAP Summary:" in parts[1] else ""
        full_report = parts[1].split("Full Report:")[1] if "Full Report:" in parts[1] else ""
        plot_base64 = parts[1].split("SHAP Summary:")[0].strip()

        # Show model results
        st.markdown(before_plot)
        
        # Show SHAP plot
        image_data = base64.b64decode(plot_base64)
        image = Image.open(io.BytesIO(image_data))
        st.subheader("SHAP Feature Importance Plot")
        st.image(image)

        # Show SHAP summary
        st.subheader("SHAP Summary")
        st.markdown(shap_summary)

        # Show full report
        st.subheader("Full Analysis Report")
        st.markdown(full_report)

    if st.button("Analyse Another"):
        st.session_state["stage"] = "upload"
        st.session_state["report"] = ""
        st.rerun()