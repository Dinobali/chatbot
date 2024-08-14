# chatbot
The chatbot is configured to use OpenAI embeddings for high-quality text processing and Ollama models for generating responses.


<img width="1792" alt="Screenshot 2024-08-14 at 22 27 19" src="https://github.com/user-attachments/assets/c25259a3-5754-424e-836d-840dbd04347d">



### How to Install and Run the Chatbot Application

#### Prerequisites

1. **Ollama Installed**:
   - Ensure that you have **Ollama** installed on your machine.
   - Download a model like **llama3.1** or any other preferred model.

2. **Python 3 and pip**:
   - You need to have **Python 3** and **pip** installed on your machine.

#### Installation Steps

1. **Clone the Repository**:
   - Open your terminal and run the following command to clone the repository:
     ```bash
     git clone https://github.com/Dinobali/chatbot.git
     ```
   - Navigate into the `chatbot` directory:
     ```bash
     cd chatbot
     ```

2. **Set Up a Virtual Environment**:
   - Create a virtual environment using Python:
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     - On **Linux/MacOS**:
       ```bash
       source venv/bin/activate
       ```
     - On **Windows**:
       ```bash
       venv\Scripts\activate
       ```

3. **Install Dependencies**:
   - Install the required dependencies using pip:
     ```bash
     pip install -r requirements.txt
     ```

4. **Configure API Key**:
   - Open the `constants.py` file to enter your OpenAI API key:
     ```bash
     nano constants.py
     ```
   - Enter your OpenAI key and save the file by pressing `Ctrl + S`, then exit by pressing `Ctrl + X`.

5. **Run the Application**:
   - Start the application by running:
     ```bash
     python3 app.py --host 0.0.0.0
     ```
   - If everything is set up correctly, you should see a message similar to the one below:
     ```plaintext
     INFO - WARNING: This is a development server. Do not use it in a production deployment.
     * Running on all addresses (0.0.0.0)
     * Running on http://127.0.0.1:5005
     * Running on http://192.168.178.122:5005
     ```

---

Once the server is running, you can interact with the chatbot through your web browser by navigating to the displayed IP address (e.g., `http://127.0.0.1:5005`).

If you run into any issues or need further assistance, feel free to ask!
