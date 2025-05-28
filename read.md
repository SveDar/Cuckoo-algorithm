✅ Step 1: Install Python (if not already)
You need Python 3.7+.
Check version:

bash
Copy
Edit
python --version
If not installed, download from: https://www.python.org/downloads/

✅ Step 2: Create and Activate a Virtual Environment
🔹 Windows (CMD or PowerShell):

bash
Copy
Edit
python -m venv cuckoo_env
cuckoo_env\Scripts\activate
🔹 macOS/Linux:

bash
Copy
Edit
python3 -m venv cuckoo_env
source cuckoo_env/bin/activate
✅ Step 3: Install Required Libraries
Once the virtual environment is activated, install the packages:

bash
Copy
Edit
pip install numpy matplotlib pandas
✅ Step 4: Save the Script
Create a file named cuckoo_search_experiments.py
Paste the full script from earlier into it.

✅ Step 5: Run the Script
In the terminal (while your virtual environment is activated):

bash
Copy
Edit
python cuckoo_search_experiments.py
You will see:

Output printed for each function.

A file named cuckoo_search_results.csv.

A folder named plots/ with .png graphs inside.

🧯 Optional Cleanup
To deactivate the environment when you're done:

bash
Copy
Edit
deactivate
