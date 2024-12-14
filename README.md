Automatic Generation of Visualizations and Infographics using Large Language Models (LLMs)

📖 Project Summary
This project aims to develop a platform that makes data analysis and visualization accessible to non-technical users. The platform features a Flask backend and a frontend built with HTML, JavaScript, and CSS. Users can upload datasets (Excel, CSV), ask questions in natural language, and receive automatically generated visualizations (e.g., charts, graphs) and insights using Large Language Models (LLMs).

The system offers an interactive experience, allowing users to select visualization options and modify them dynamically. Guardrails ensure the accuracy and best practices in data representation.

🚀 Features
Dataset Upload: Upload CSV or Excel files.
Natural Language Queries: Ask questions about your data in plain English.
Automated Visualizations: Generate visualizations like charts, plots, and graphs based on your queries.
Dynamic Modifications: Adjust chart types, axis labels, and other elements in real-time.
Downloadable Visuals: Export visualizations in high-quality formats (PNG, SVG, PDF).
Guardrails: Ensures accurate and appropriate data visualization practices.

🛠️ Technologies Used
Frontend
HTML / CSS (Framework: Bootstrap / Tailwind CSS)
JavaScript
Backend
Flask (Python)
LangChain Agents / LIDA for visualization generation
Pandas for data parsing and handling

⚙️ Installation

Clone the Repository:
git clone https://github.com/cerencelik25/ChatBot.git
cd ChatBot

Create a Virtual Environment:
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Run the Application:
flask run

Access the Platform: Open your browser and navigate to:
http://127.0.0.1:5000/

📝 Usage

Upload Dataset:
Go to the "Upload" section and upload CSV or Excel files.
Supported formats: .csv, .xlsx.

Ask a Question:
Enter a question about the dataset in the input box.

Example:
Show me the sales trend over the last year.

Generate Visualizations:
The system will generate appropriate visualizations based on your query.
Options to modify chart types, adjust labels, and download the visuals.

🔄 Workflow and Architecture

File Upload Pipeline:
Upload and validate datasets (CSV, Excel).
Parse and extract metadata (columns, data types).
Store data temporarily in the backend for processing.

Visualization Pipeline:
Users select visualization methods (LangChain Agents, LIDA).
The system generates and executes visualization code dynamically.
Visualizations can be modified in real-time (chart type, axis labels).

Frontend Interface:
File Upload Section: Upload datasets with real-time feedback.
Query Input Section: Input natural language queries.
Visualization Display: Interactive area to view and modify charts.
Download Options: Export visualizations in multiple formats.

📊 Example Visualizations
  Type	               Description
Line Chart	      Shows trends over time
Bar Chart	      Compares categories
Scatter Plot	   Displays data distribution
Pie Chart	      Represents proportions

🔧 Future Enhancements
Enhanced Visualization Options: Support for more complex charts.
Authentication: User login for saving and managing datasets.
AI Insights: Advanced AI-driven insights and recommendations.

🤝 Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

Fork the Project
Create a Feature Branch:
git checkout -b feature/NewFeature

Commit Changes:
git commit -m "Add some NewFeature"

Push to Branch:
git push origin feature/NewFeature

Open a Pull Request

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

📧 Contact
For feedback or issues, please contact:

Ceren Celik
Email: ceren06pasa@gmail.com


