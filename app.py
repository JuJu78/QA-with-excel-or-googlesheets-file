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

with st.expander("Fonctionnement de l'application"):
    st.markdown("""Cette application permet de générer des réponses fournies par l'un des modèles d'IA générative d'OpenAI (GPT-3.5 turbo ou GPT-4) à partir d'instructions et de questions posées dans un fichier Excel ou une feuille Google Sheets.\
    \n\n**Instructions** :\
    \n\n1. **Créer un fichier Excel ou une feuille Google Sheets avec deux colonnes** (la feuille Google Sheets doit être partagée à tous les utilisateurs qui ont le lien pour que cela fonctionne):\
    \n\n\t **Important** : Le nom des entêtes de colonne importe peu. Mais il faut qu'elles soient présentes car l'IA ne commence à répondre aux questions qu'à partir de la deuxième ligne. Ne posez pas de questions à l'IA dès la 1ère ligne, il n'y répondra pas !\
    \n\n\t- **La première colonne** contient les instructions à donner à l'IA.\
    \n\n\t\t- Par exemple : "Tu es une intelligence artificielle spécialisée dans le SEO. Tu réponds de manière précise et pédagogique aux questions posées par l'utilisateur".\
    \n\n\t- **La deuxième colonne** contient les questions à poser à l'IA.\
    \n\n2. **Uploadez le fichier Excel ou la feuille Google Sheets**.\
    \n\n3. **Cliquer sur le bouton "Commencer"**.\
    \n\n4. **Attendre que l'IA génère les réponses**.\
    \n\n5. **Lorsque l'IA a fini de parcourir le fichier de générer ses réponses, un dataframe s'affiche contenant les questions et les réponses**.\
    \n\n6. **Télécharger le fichier Excel ou CSV contenant les questions et réponses de l'IA**.\
    """)

# Sidebar for user to enter API key and choose model
api_key = st.sidebar.text_input("Enter your OpenAI API key")
model = st.sidebar.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'])
temperature = st.sidebar.slider("Set temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

openai.api_key = api_key

# Option to upload an Excel file or enter a Google Sheets URL
option = st.selectbox("Choose an option", ["Upload an Excel file", "Enter a Google Sheets URL"])

start_processing = False  # Initialize start_processing as False
if option == "Upload an Excel file":
    uploaded_file = st.file_uploader("Excel File", type="xlsx")
    if uploaded_file is not None:
        if st.button("Commencer"):  # Only start the process when the start button is pressed
            data = pd.read_excel(uploaded_file)
            start_processing = True  # Set start_processing to True

# ...
elif option == "Enter a Google Sheets URL":
    google_sheets_url = st.text_input("Google Sheets URL")
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

                if response.status_code != 200:
                    st.write(f"Erreur lors de la récupération des données de la feuille Google Sheets : {response.status_code}")
                    return pd.DataFrame()

                for encoding in ('utf-8', 'iso-8859-1', 'cp1252'):
                    try:
                        data = pd.read_csv(io.StringIO(response.text), encoding=encoding)
                        return data
                    except UnicodeDecodeError:
                        continue

                # If none of the encodings work, raise an error
                raise ValueError("Aucun encodage n'a fonctionné")

            data = get_sheet_values(google_sheets_url)
            if data.empty:
                st.write("Aucune donnée récupérée depuis la feuille Google Sheets")
            else:
                start_processing = True  # Set start_processing to True

            data = get_sheet_values(google_sheets_url)
            start_processing = True  # Set start_processing to True

# Create an empty DataFrame to store the Q&A
qna_data = pd.DataFrame(columns=["Questions", "Responses"])  # Remove the instruction column

class CustomChatGPT:
    def __init__(self, model):
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def chat(self, instruction, question):
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
                    model=self.model,
                    temperature=temperature,
                    messages=messages
                )
                self.total_prompt_tokens += response['usage']['prompt_tokens']
                self.total_completion_tokens += response['usage']['completion_tokens']
                self.total_tokens += response['usage']['total_tokens']
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError:
                st.write("Rate limit error encountered. Retrying after 5 seconds...")
                time.sleep(5)  # Wait for 5 seconds before retrying

    def get_total_cost(self):
        # Cost per 1000 tokens
        if self.model == 'gpt-4':
            cost_per_1000_prompt = 0.03
            cost_per_1000_completion = 0.06
        elif self.model == 'gpt-3.5-turbo':
            cost_per_1000_prompt = 0.002
            cost_per_1000_completion = 0.002

        # Calculate the total cost
        total_cost = (self.total_prompt_tokens / 1000) * cost_per_1000_prompt + (self.total_completion_tokens / 1000) * cost_per_1000_completion
        return total_cost


# Initialize lists to store questions and responses
questions = []
responses = []

if start_processing:
    chatbot = CustomChatGPT(model)

    for _, row in data.iterrows():
        instruction = row[0]  # Assuming instruction is in the first column
        question = row[1]  # Assuming question is in the second column
        response = chatbot.chat(instruction, question)  # Use chatbot to call the chat method
        st.markdown(f"**Question**:{question}", unsafe_allow_html=True)
        st.markdown(f"**Response**: {response}", unsafe_allow_html=True)

        question = question.encode('utf-8', errors='ignore').decode('utf-8') # Convert the encoding of the question
        response = response.encode('utf-8', errors='ignore').decode('utf-8') # Convert the encoding of the response

        # Append the question and response to their respective lists
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
    download_link_csv = create_download_link_csv(qna_data, 'QnA.csv')
    download_link_excel = create_download_link_excel(qna_data, 'QnA.xlsx')

    # Create a two column layout for the download links
    col1, col2 = st.columns(2)
    
    # Assign a download link to each column
    with col1:
        st.markdown(download_link_csv, unsafe_allow_html=True)
    with col2:
        st.markdown(download_link_excel, unsafe_allow_html=True)

    # display the total cost of the process
    st.markdown(f"**Model used**: {chatbot.model}<br><br>**Total cost**: {chatbot.get_total_cost()} dollars", unsafe_allow_html=True)

    # display the number of used tokens
    st.markdown(f"**Total prompt tokens used**: {chatbot.total_prompt_tokens}", unsafe_allow_html=True)
    st.markdown(f"**Total completion tokens used**: {chatbot.total_completion_tokens}", unsafe_allow_html=True)
    st.markdown(f"**Total tokens used**: {chatbot.total_tokens}", unsafe_allow_html=True)

