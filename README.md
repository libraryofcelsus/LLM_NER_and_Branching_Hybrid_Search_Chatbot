# NER-and-Hybrid-Search-Ai-Chatbot
An example of Named-entity Recognition and relation mapping using an LLM and Vector Database.  A Hybrid Search Chatbot to utilize extracted relations.

The Hybrid Search will search both Source Entities and Target Entities, resulting in a better search ability than traditional RAG.  This approach allows relevant memories that may not have the same implicit semantic meaning to be returned.  The method used here is a simple version, but a continued nested approach could be used at the expense of additional context length.

Originally made as a solution for my Aetherius Ai Assistant Project, however I could never get it to work reliably with smaller LLMs.

Main Ai Assistant Project: https://github.com/libraryofcelsus/Aetherius_AI_Assistant

Vector Database: https://github.com/qdrant/qdrant

### Changelog

- 2/19 Changed PDF reading to use Tesseract

- 2/18 Added CSV output

## Installation

1. Install Python 3.10.6, Make sure you add it to PATH: https://www.python.org/downloads/release/python-3106/
2. Install Git: https://git-scm.com/ (Git can be skipped by downloading the repo as a zip file under the green code button)
3. Install tesseract for OCR: https://github.com/UB-Mannheim/tesseract/wiki Once installed, copy the "Tesseract-OCR" folder from Program Files to the Main Project Folder.  Alternativly you can also install it directly to a folder named "Tesseract-OCR" in the project folder on initial install.
4. If using Qdrant Cloud copy their Api key and Url to their respective key in the settings.json.  Qdrant Cloud: https://qdrant.to/cloud
5. To use a local Qdrant server, first install Docker: https://www.docker.com/
6. Now run: docker pull qdrant/qdrant:v1.5.1 in CMD
7. Next run: docker run -p 6333:6333 qdrant/qdrant:v1.5.1
8. Once the local Qdrant server is running, it should be auto detected by the chatbot.
(See: https://docs.docker.com/desktop/backup-and-restore/ for how to make backups.)
9. Open CMD as Admin
10. Run git clone: **git clone https://github.com/libraryofcelsus/NER-and-Hybrid-Search-Ai-Chatbot.git**
11. Navigate to Project folder: cd PATH_TO_CHATBOT_INSTALL
12. Create a virtual environment: python -m venv venv
13. Activate the environment: .\venv\scripts\activate
14. Install the required packages: pip install -r requirements.txt
15. Edit settings in "settings.json"
16. Run "Hybrid_Search_Example.py" to use the chatbot. Run "Extract_Relation_From_File.py" to extract relations from the upload folder.

------

Join my Discord for help or to get more in-depth information!

Discord Server: https://discord.gg/pb5zcNa7zE

------

My Ai development is self-funded by my day job, consider donating if you find it useful!  

<a href='https://ko-fi.com/libraryofcelsus' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
