import os
import pytesseract
import cv2
import pandas as pd
from PIL import Image

def preprocess_image(image_path):
    """ Convert image to grayscale and apply thresholding for better OCR """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, gray)
    return processed_image_path

def format_text_as_csv(text):
    """ Converts extracted text into a structured CSV format """
    lines = text.strip().split("\n")

    # Filtering out empty lines
    structured_lines = [line.strip() for line in lines if line.strip()]

    # Creating CSV-like rows (splitting by spaces or special characters)
    formatted_data = []
    for line in structured_lines:
        row = line.split()  # Splitting by spaces
        formatted_data.append(row)

    return formatted_data

def extract_text_to_csv(image_filename, output_csv_filename, lang="eng+ben"):
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, image_filename)
        output_csv_path = os.path.join(script_dir, output_csv_filename)

        # Preprocess image for better text recognition
        processed_image_path = preprocess_image(image_path)

        # Open the processed image
        image = Image.open(processed_image_path)

        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(image, lang=lang)

        # Format extracted text into CSV-like structure
        formatted_data = format_text_as_csv(extracted_text)

        # Convert formatted text into a DataFrame
        df = pd.DataFrame(formatted_data)

        # Save structured text into a CSV file
        df.to_csv(output_csv_path, index=False, header=False)

        print(f"Text extracted and saved in structured format to: {output_csv_path}")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Input image file (must be in the same directory as the script)
    image_filename = "sample-photo.jpeg"  # Change as needed
    output_csv_filename = "formatted_extracted_text.csv"

    # Run the OCR extraction
    extract_text_to_csv(image_filename, output_csv_filename)