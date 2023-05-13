import streamlit as st
import pandas as pd
import openai
import base64
import io
import sys
import xlsxwriter
import time
import requests
import re

st.title("Q&AI A from Excel or Google Sheets file")

# Sidebar for user to enter API key and choose model
api_key = st.sidebar.text_input("Enter your OpenAI API key")
model = st.sidebar.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'])
temperature = st.sidebar.slider("Set temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

openai.api_key = api_key

# Option to upload an Excel file or enter a Google Sheets URL
option = st.selectbox("Choose an option", ["Upload an Excel file", "Enter a Google Sheets URL"])
if option == "Upload an Excel file":
    uploaded_file = st.file_uploader("", type="xlsx")
    if uploaded_file is not None:
        if st.button("Commencer"):  # Only start the process when the start button is pressed
            data = pd.read_excel(uploaded_file)

elif option == "Enter a Google Sheets URL":
    google_sheets_url = st.text_input("")
    if google_sheets_url != "":
        if st.button("Commencer"):
            # Function to extract sheet id from URL
            def extract_sheet_id(url):
                match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
                if match:
                    sheet_id = match.group(1)
                else:
                    raise ValueError("URL invalide")
                return sheet_id

            # Function to get sheet values
            def get_sheet_values(sheet_url):
                sheet_id = extract_sheet_id(sheet_url)
                sheet_gid = "0"
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}"
                response = requests.get(export_url)

                for encoding in ('utf-8', 'iso-8859-1', 'cp1252'):
                    try:
                        data = pd.read_csv(io.StringIO(response.text), encoding=encoding)
                        return data
                    except UnicodeDecodeError:
                        continue

                # If none of the encodings work, raise an error
                raise ValueError("Aucun encodage n'a fonctionné")


            data = get_sheet_values(google_sheets_url)

            # Create an empty DataFrame to store the Q&A
            qna_data = pd.DataFrame(columns=["Question", "Response"])  # Remove the instruction column

            def CustomChatGPT(instruction, question):
                # Initialize messages for each question
                messages = []

                # Treat all instructions as system role
                messages.append({"role": "system", "content": instruction})

                # Treat all questions as user role
                messages.append({"role": "user", "content": question})

                # Retry logic
                for _ in range(5):  # Try up to 5 times
                    try:
                        response = openai.ChatCompletion.create(
                            model=model,
                            temperature=temperature,
                            messages=messages
                        )
                        return response["choices"][0]["message"]["content"]
                    except openai.error.RateLimitError:
                        st.write("Rate limit error encountered. Retrying after 5 seconds...")
                        time.sleep(5)  # Wait for 5 seconds before retrying

            # Initialize lists to store questions and responses
            questions = []
            responses = []

            for _, row in data.iterrows():
                instruction = row[0]  # Assuming instruction is in the first column
                question = row[1]  # Assuming question is in the second column

                # Convert the encoding of the question
                question = question.encode('iso-8859-1').decode('utf-8')

                response = CustomChatGPT(instruction, question)
                st.write(f"Response: {response}")

                # Add question and response to the lists
                questions.append(question)
                responses.append(response)

            # Create a DataFrame from the lists
            qna_data = pd.DataFrame({"Question": questions, "Response": responses})

            # Display the Q&A DataFrame
            st.write(qna_data)

            # Function to download data as a csv file
            def create_download_link_csv(df, filename):
                df.to_csv('temporary.csv', index=False)
                with open('temporary.csv', 'r', encoding='utf-8') as file:
                    csv = file.read()
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'


            # Function to download data as an excel file
            def create_download_link_excel(df, filename):
                stream = io.BytesIO()
                df.to_excel(stream, index=False, sheet_name='Sheet1')  
                stream.seek(0)
                excel_data = stream.read()
                b64 = base64.b64encode(excel_data).decode()
                return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'


            # Create and display download links as soon as DataFrame is ready
            if not qna_data.empty:
                download_link_csv = create_download_link_csv(qna_data, 'QA.csv')
                download_link_excel = create_download_link_excel(qna_data, 'QA.xlsx')

                # Create a two column layout for the download links
                col1, col2 = st.columns(2)
                
                # Assign a download link to each column
                with col1:
                    st.markdown(download_link_csv, unsafe_allow_html=True)
                with col2:
                    st.markdown(download_link_excel, unsafe_allow_html=True)
