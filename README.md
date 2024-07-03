# Math Equation Solver

Welcome to the Math Equation Solver, a Streamlit application that allows users to upload images of math equations, extract the text from image, and solve the equations. The app supports various mathematical operations such as differentiation, integration, logarithms, square roots, summation, quadratic equations , cubic equations , linear equations.

-  Features

- User Authentication: Signup and login functionality.
- Image Upload: Upload images of math equations in `jpg` or `png` format.
- Text Extraction: Use Tesseract OCR to extract text from the uploaded images.
- Math Equation Solving: Solve equations for differentiation, integration, logarithms, square roots, and summation, quadratic equations , cubic equations , linear equations.
- Step-by-Step Solutions: Generate step-by-step solutions for the equations using the Gemini API.

# Installation

- Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Tesseract OCR
- OpenCV
- Streamlit
- PIL (Pillow)
- numpy
- pytesseract
- pix2tex
- latex2sympy2
- sympy
- re
- os
- google-generativeai
- ssl
- urllib
- certifi
- sqlite3

# Installing Dependencies

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/math-equation-solver.git
    cd math-equation-solver
    ```

2. **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install Tesseract OCR:**

    - For Windows:
      Download the Tesseract installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and run it.
    - For macOS:
      ```bash
      brew install tesseract
      ```
    - For Linux:
      ```bash
      sudo apt-get install tesseract-ocr
      ```

4. **Set up the Google Gemini API:**

    Replace the placeholder in the code with your actual Google API key.

    ```python
    os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"
    ```

5. **Run the application:**

    ```bash
    streamlit run finalpro.py
    ```

# Usage

1. Sign Up or Log In:

   - Use the sidebar to navigate to the Signup or Login page.
   - Fill in the necessary details and click the Signup/Login button.

2. Upload an Image:

   - Once logged in, you will be redirected to the Math Equation Solver page.
   - Upload an image containing the math equation you want to solve.

3. Processing the Image:

   - The app will preprocess the image and extract the text using Tesseract OCR.

4. Solving the Equation:

   - The app handles differentiation, integration, logarithms, square roots, summation, quadratic equations , cubic equations , linear equations.
   - Step-by-step solutions are generated using the Gemini API.

## Project Structure

- `finalpro.py`: The main application file containing all the Streamlit code.
- `requirements.txt`: List of all required Python packages.



