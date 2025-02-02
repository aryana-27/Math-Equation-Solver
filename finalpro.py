import cv2
import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
from pix2tex.cli import LatexOCR  # text -> latex -> sympy
from latex2sympy2 import latex2sympy  # text -> latex -> sympy
import sympy as sp
import re  # regex for eliminating portion of string using regex
import os  # for gemini key
import google.generativeai as genai  # gemini api key
import ssl  # for avoiding ssl error
import urllib.request  # for avoiding ssl error
import certifi  # for avoiding ssl error
import sqlite3
from streamlit import session_state

# **********************************************

# Set the valid Google API key here
os.environ['GOOGLE_API_KEY'] = "YOUR API KEU"

# Initialize the GenerativeModel with the API key
model = genai.GenerativeModel("models/gemini-1.5-flash")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up SSL context to avoid SSL certification errors
context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))

# Fetch a URL (Example to show handling of urllib errors)
url = 'https://www.google.com'
try:
    with opener.open(url) as response:
        html = response.read()
except urllib.error.URLError as e:
    st.error(f"Failed to fetch the page: {e.reason}")

# Tesseract OCR path setup and configuration
pytesseract.pytesseract.tesseract_cmd = '/opt/local/bin/tesseract'
custom_config = r'--oem 3 --psm 6'

# image processing section - 

def preprocess_image(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)  # converting to grayscale
    _, binary_image = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # binarization for sharp images
    return binary_image

def extract_text(img):
    text = pytesseract.image_to_string(img, config=custom_config)  # pytesseract for text extraction
    return text

def extract_latex(image):
    l = LatexOCR()
    latex_expr = l(image)
    return latex_expr

#removing specificic string from latex code using regex  

def remove_textstyle(latex_code):

    clean_latex_expr = re.sub(r'\\textstyle', '', latex_code)  # using regex to eliminate \textstyle
    return clean_latex_expr

# conversion to latex -> sympy  

def latex_to_sympy(latex_code):
    cleaned_latex = remove_textstyle(latex_code)
    s_expr = latex2sympy(cleaned_latex)
    if isinstance(s_expr, list):
        return s_expr
    elif isinstance(s_expr, sp.Basic):
        return [s_expr]
    else:
        raise ValueError("Invalid SymPy expression format")

# solving for log questions 

def solve_log_equation(s_expr):
    x = sp.symbols('x')
    eq = s_expr if isinstance(s_expr, sp.Equality) else sp.Eq(s_expr, 0)
    lhs, rhs = eq.lhs, eq.rhs

    if lhs.has(sp.log):
        lhs = sp.logcombine(lhs, force=True)
        log_arg = lhs.args[0]
        log_base = lhs.args[1] if len(lhs.args) == 2 else sp.E

        if isinstance(log_base, sp.Symbol):
            log_base = sp.exp(1)

        exponential_eq = sp.Eq(log_base**log_arg, log_base**rhs)
        solution = sp.solve(exponential_eq, x)
        return solution
    else:
        solution = sp.solve(eq, x)
        return solution

# solving for linear equation in one variable 

def solve_lineq(s_expr):
    x = sp.symbols('x')
    solution = sp.solve(s_expr, x)
    return solution

# handling database for user details and chat history 

def database_connect():
    conn = sqlite3.connect('user.db', timeout=10.0)
    return conn

def database_tables():
    conn = database_connect()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users(first_name TEXT,last_name TEXT,age INTEGER,username TEXT,password TEXT)''')
    # c.execute('''CREATE TABLE IF NOT EXISTS chat_history(id INTEGER PRIMARY KEY,username TEXT,question TEXT)''')
    conn.commit()
    conn.close()

database_tables()

# functions for chat histroy - 

# def get_chat_history(): # to fetch chat history
#     conn = database_connect()
#     c = conn.cursor()
#     c.execute("SELECT question FROM chat_history WHERE username = ?",(st.session_state.username[0],))
#     chat_history = c.fetchall()
#     conn.close()
#     return chat_history

# def add_chat(question , username): # adding in chat history
#     conn = database_connect()
#     c = conn.cursor()
#     try:
#         with conn:
#             c.execute("INSERT INTO chat_history (username , question) VALUES (?,?)", (username[0],question))
#             # c.execute("INSERT INTO chat_history (username , question) VALUES (?,?)", (username[0],question))
#     except sqlite3.Error as e:
#         st.error(f"SQLite error: {e}")
#     finally:
#         conn.close()

# def chat_history():
#     try:
#         with st.sidebar.header("Chat History"):
#             conn = database_connect()
#             c = conn.cursor()
#             c.execute("SELECT question FROM chat_history WHERE username = ?",(st.session_state.username[0],))
#             chat_history = c.fetchall()
#             conn.close()
#             his = get_chat_history()
#             for i in his:
#                 st.sidebar.text(i[0])
#     except sqlite3.Error as e:
#         st.error(f"Error fetching chat history: {e}")

#USER AUTHENTICATION LOGIN/SIGNUP      

def signup_user(first_name, last_name, age, username, password):
    conn = database_connect()
    c = conn.cursor()
    try:
        with conn:
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (first_name, last_name, age, username, password))
    except sqlite3.Error as e:
        st.error(f"SQLite error: {e}")
    finally:
        conn.close()

    
def login_user(username, password):
    conn = database_connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# GENERATING STEP BY STEP METHOD FOR QUESTIONS USER UPLOADS

def step_by_step(latex_expr):
    try:
        cleaned_latex = remove_textstyle(latex_expr)
        prompt = f"Show all steps to solve {latex_expr}"
        res = model.generate_content(prompt)
        solution_text = res.text

        # for storing steps
        steps = []
        curr_step = ""

        for i in solution_text.splitlines():
            if i.startswith("Step"):
                if curr_step:
                    steps.append(curr_step)
                curr_step = i
            else:
                curr_step += i + "\n"
        if curr_step:
            steps.append(curr_step)

        return steps
    except Exception as e:
        return [f"Error fetching steps: {e}"]

# MAIN BLOCK STARTS FROM HERE - 

def main():
    st.title("Math Equation Solver!")
    st.header("Hey! What do you want to learn today?")
    choice = st.sidebar.selectbox('Login/Signup', ['Login', 'Signup'])

    if choice == 'Login':
        login_page()
    else:
        signup_page()

# MATHEMATIC HANDLING OF DIFFERENT METHODS - 

def mathparser():
    st.subheader("Math Equation Solver Parser")
    upload_image = st.file_uploader("Upload an image", type=["jpg", "png"])
    if upload_image is not None:
        try:
            image = Image.open(upload_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            processed_image = preprocess_image(image)

            if cv2.Laplacian(processed_image, cv2.CV_64F).var() < 200:
                st.error("Image is too blurry! Try uploading a clear image!")
                return

            text = extract_text(processed_image)
            # st.subheader("Extracted Text:")
            # st.text_area("Text", text, height=200, key='extracted_text')
           

            latex_expr = extract_latex(image)
            # st.subheader("Extracted LaTeX:")
            # st.text_area("LaTeX", latex_expr, height=100, key='latex_expr')
         
           # HANDLING DIFFERENT FUNCTIONS  - 

            try:
                s_expr_list = latex_to_sympy(latex_expr)
                for s_expr in s_expr_list:
                    st.subheader("SymPy Expression:")
                    st.write(s_expr)

                    if r'\int' in latex_expr: #integrals
                        x = sp.symbols('x')
                        integration = sp.integrate(s_expr, x)
                        st.subheader("Solution step by step:")
                        st.write(f"Integral expression: {integration}")
                        st.write(f"Simplified solution: {integration.simplify()}")

                    elif r'\log' in latex_expr: # logarithmic 
                        s_expr = latex2sympy(latex_expr)[0]
                        solution = solve_log_equation(s_expr)
                        if solution:
                            st.subheader("Solution for x:")
                            for sol in solution:
                                st.write(f"x = {sol}")
                        else:
                            st.error("No solution found.")

                    elif r'\sqrt' in latex_expr: # square roots
                        inside = s_expr.args[0]
                        constant = s_expr.args[1]
                        square_eq = inside**2 - constant**2
                        solutions = sp.solve(square_eq, inside)
                        st.subheader("Solutions for x:")
                        for s in solutions:
                            st.write(f"x = {s}")

                    elif r'\sum' in latex_expr: # summation
                        sum_res = s_expr.doit()
                        st.subheader("Solution (Summation):")
                        st.write(sum_res)

                    elif isinstance(s_expr, sp.Basic) and s_expr.is_Equality:
                        solution = solve_lineq(s_expr)
                        if solution:
                            st.subheader("Solution for x:")
                            st.write(solution)
                        else:
                            st.error("No solution found.")

                    elif r'\\frac{d}{dx}' in latex_expr: # differentiation
                        x = sp.symbols('x')
                        differentiation = sp.diff(s_expr, x)
                        st.subheader("Solution (differentiation):")
                        st.write(f"Integral expression: {differentiation}")
                        st.write(f"Simplified solution: {differentiation.simplify()}")
                
                if latex_expr:
                    steps = step_by_step(latex_expr)
                    for step in steps:
                        st.write(step)

                # steps = step_by_step(latex_expr)
                # st.subheader("Step-by-Step Solution:")
                # for step in steps:
                #     st.text(step)
                    
                    # add_chat(s_expr , st.session_state.username)

            except Exception as e:
                st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
    
# USER AUTHENTICATION LOGIN/SIGNUP      

def login_page():
    st.subheader("Login")
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = [username]
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def signup_page():
    st.subheader("Signup")
    first_name = st.text_input('First-Name')
    last_name = st.text_input('Last-Name')
    age = st.number_input('Age', min_value=0)
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Signup'):
        signup_user(first_name, last_name, age, username, password)
        st.session_state.logged_in = True
        st.experimental_rerun()

if __name__ == '__main__':
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = []

    if st.session_state.logged_in:
        mathparser()
        # chat_history()
    else:
        main()

