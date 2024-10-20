import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import BarChart
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from datetime import datetime

# Set the page layout to wide to utilize full screen width
st.set_page_config(layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: sky-blue; font-size: 50px; font-family: "Arial Black", sans-serif;'>
    Parkinson's Disease Detection - Multi-Model
    </h1>
    """,
    unsafe_allow_html=True
)

# Initialize report_filename in session state if not already done
if 'report_filename' not in st.session_state:
    st.session_state.report_filename = None

# Load all models
spiral_model = load_model('Models/parkinson_disease_detection_model(93%).h5')
mri_model = load_model('Models/parkinson_disease_detection_model(MRI).h5')
wave_model = load_model('Models/parkinson_disease_detection_model(wave).h5')

# Streamlit app title and description
st.markdown(
    """
    <style>
    input {
        width: 100px;  /* Adjust the width of the input boxes */
        margin-bottom: 10px; /* Spacing between input boxes */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Collect user information with custom titles and smaller input fields
st.subheader("*User Information*")  # Making title more impressive
user_name = st.text_input("*Name:*", max_chars=30, placeholder="Enter your name", help="Your full name", key="name")
user_age = st.number_input("*Age:*", min_value=0, max_value=120, step=1, help="Your age", key="age", format="%d")
user_email = st.text_input("*Email:*", placeholder="Enter your email", help="Your email address", key="email")




st.write("""Upload three images, and the models will predict whether each image indicates signs of Parkinson's Disease or a Healthy condition.""")


col1, col2, col3 = st.columns([1, 1, 1])  # Each column takes up 33% of the width

with col1:
    uploaded_spiral = st.file_uploader("Upload an image for Spiral Model (128x128)", type=["png", "jpg", "jpeg"], key="spiral")
with col2:
    uploaded_mri = st.file_uploader("Upload an image for MRI Model (128x128)", type=["png", "jpg", "jpeg"], key="mri")
with col3:
    uploaded_wave = st.file_uploader("Upload an image for Wave Model (128x128)", type=["png", "jpg", "jpeg"], key="wave")

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array):
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    return confidence

def display_prediction(uploaded_file, model, model_name, col):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col:
            st.image(image, caption=f'Uploaded Image for {model_name}', use_column_width=True)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = preprocess_image(image)

        with st.spinner(f'Analyzing image for {model_name}...'):
            confidence = predict_image(model, img_array)

        with col:
            if confidence > 0.5:
                st.markdown(f"<h3 style='color: red;'>{model_name} predicts Parkinson's Disease with {confidence * 100:.2f}% confidence.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: green;'>{model_name} predicts Healthy with {(1 - confidence) * 100:.2f}% confidence.</h3>", unsafe_allow_html=True)
        return confidence
    


# Initialize confidence variables
spiral_confidence = None
mri_confidence = None
wave_confidence = None

# Display predictions for each model
if uploaded_spiral:
    spiral_confidence = display_prediction(uploaded_spiral, spiral_model, "Spiral Model", col1)
if uploaded_mri:
    mri_confidence = display_prediction(uploaded_mri, mri_model, "MRI Model", col2)
if uploaded_wave:
    wave_confidence = display_prediction(uploaded_wave, wave_model, "Wave Model", col3)

# Logic for final prediction
if all(conf is not None for conf in [spiral_confidence, mri_confidence, wave_confidence]):
    if (spiral_confidence <= 0.5 and mri_confidence <= 0.5 and wave_confidence <= 0.5):
        final_prediction = "Healthy"
        final_confidence = (1 - (spiral_confidence + wave_confidence + mri_confidence) / 3) * 100
    elif (spiral_confidence > 0.5 and mri_confidence > 0.5 and wave_confidence > 0.5):
        final_prediction = "Parkinson's Disease"
        final_confidence = (spiral_confidence + wave_confidence + mri_confidence) / 3 * 100
    else:
        final_prediction = "Likely Parkinson's Disease"
        final_confidence = 0.00

    if final_confidence is not None:
        st.markdown(f"<h3 style='color: {'red' if final_prediction == 'Parkinson\'s Disease' else 'green' if final_prediction == 'Healthy' else 'yellow'};'>Model predicts {final_prediction} ,need to consult doctor.</h3>", unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    import matplotlib.pyplot as plt

    if all(conf is not None for conf in [spiral_confidence, mri_confidence, wave_confidence]):
        model_names = ['Spiral', 'MRI', 'Wave']
        confidences = [spiral_confidence, mri_confidence, wave_confidence]
        
        # Set up a smaller figure size
        fig, ax = plt.subplots(figsize=(2, 2))  # Adjust the figure size to make it even smaller

        # Create the bar chart
        ax.bar(model_names, confidences, color=['red', 'green', 'blue'])

        # Reduce font sizes for labels and title
        ax.set_ylabel('Confidence Level', fontsize=8)
        ax.set_title('Model Confidence Levels', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)  # Small tick labels

        # Adjust the layout to make the chart compact
        plt.tight_layout()

        # Display the bar chart using Streamlit
        st.pyplot(fig)

# Function to create a PDF report
def create_report(filename, user_name, final_prediction, final_confidence):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom Styles
        custom_style = ParagraphStyle(
            'Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.black,
            spaceAfter=12
        )
        
        report_elements = []

        # Title
        title = Paragraph("Parkinson's Disease Detection Report", styles['Title'])
        report_elements.append(title)
        report_elements.append(Spacer(1, 12))

        # User Information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info = Paragraph(f"Generated for: {user_name}", styles['Normal'])
        report_elements.append(user_info)
        report_elements.append(Paragraph(f"Report Timestamp: {timestamp}", styles['Normal']))
        report_elements.append(Spacer(1, 12))

        # Prediction Results
        report_elements.append(Paragraph(f"<b>Final Prediction:</b> {final_prediction}", custom_style))
        report_elements.append(Paragraph(f"<b>Confidence:</b> {final_confidence:.2f}%", custom_style))
        report_elements.append(Spacer(1, 12))

        # Medical Advice with Bullet Points
        advice_title = Paragraph("<b>Medical Advice</b>", styles['Heading2'])
        report_elements.append(advice_title)
        
        advice_items = [
            "Consult your healthcare provider for personalized treatment.",
            "Engage in regular physical activity.",
            "Maintain a balanced diet rich in antioxidants.",
            "Stay socially active to support mental health.",
            "Consider joining support groups for shared experiences."
        ]
        
        advice_bullets = [Paragraph(f'• {item}', styles['Normal']) for item in advice_items]
        advice_list = ListFlowable(advice_bullets, bulletType='bullet', spaceAfter=10)
        report_elements.append(advice_list)
        report_elements.append(Spacer(1, 12))




        # Create PDF
        doc.build(report_elements)
        return True
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return False

# Function to send email with the report attached
def send_email(report_filename, user_email):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # Common port for TLS
    smtp_user = "tanmayzade87@gmail.com"  # Your email address
    smtp_password = "rszrxettdtfimnsj"  # App-specific password

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = user_email
    msg['Subject'] = "Your Parkinson's Disease Detection Report"

    with open(report_filename, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype='pdf')
        attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_filename))
        msg.attach(attach)

    body = f"Dear {user_name},\n\nPlease find attached your Parkinson's Disease detection report.\n\nBest regards,\nYour Healthcare Team"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# Generate and send report when button is clicked
if st.button("Generate Report"):
    if final_prediction and final_confidence is not None:  # Ensure final_confidence can be 0
        report_filename = f"Parkinsons_Disease_Report_{user_name}.pdf"
        if create_report(report_filename, user_email, final_prediction, final_confidence):
            st.session_state.report_filename = report_filename  # Save filename in session state
            st.success(f"Report generated: {report_filename}")

# Button to download report
if st.session_state.report_filename:
    with open(st.session_state.report_filename, "rb") as file:
        st.download_button(
            label="Download Report",
            data=file,
            file_name=os.path.basename(st.session_state.report_filename),
            mime="application/pdf"
        )

# Button to send email with the report
if st.session_state.report_filename:
    if st.button("Send Email"):
        send_email(st.session_state.report_filename, user_email)


st.markdown("[Schedule a Telemedicine Appointment](https://example.com/schedule)")






feedback = st.text_area("We value your feedback. Please share your thoughts:")
if st.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.success("Feedback submitted!")


# Initialize FAQ visibility in session state if not already done
if 'faq_visible' not in st.session_state:
    st.session_state.faq_visible = False  # Start with FAQs hidden

# Sample FAQ data
faqs = [
    {
        "question": "What is Parkinson's disease?",
        "answer": "Parkinson's disease is a progressive neurological disorder that affects movement."
    },
    {
        "question": "What are the symptoms?",
        "answer": "Common symptoms include tremors, stiffness, and difficulty with balance and coordination."
    },
    {
        "question": "How is it diagnosed?",
        "answer": "Diagnosis is usually based on medical history and neurological examination."
    },
    {
        "question": "What treatments are available?",
        "answer": "Treatments can include medications, physical therapy, and in some cases, surgery."
    },
        {
        "question": "What is Parkinson's disease?",
        "answer": "Parkinson's disease is a progressive neurological disorder that affects movement."
    },
    {
        "question": "What are the symptoms?",
        "answer": "Common symptoms include tremors, stiffness, and difficulty with balance and coordination."
    },
    {
        "question": "How is it diagnosed?",
        "answer": "Diagnosis is usually based on medical history and neurological examination."
    },
    {
        "question": "What treatments are available?",
        "answer": "Treatments can include medications, physical therapy, and in some cases, surgery."
    },
    {
        "question": "What causes Parkinson's disease?",
        "answer": "Parkinson's disease is believed to result from a combination of genetic and environmental factors. The exact cause is not fully understood, but it involves the degeneration of dopamine-producing neurons in the brain."
    },
    {
        "question": "Is Parkinson's disease hereditary?",
        "answer": "While most cases of Parkinson's disease are not directly inherited, genetic factors can increase the risk. Having a family member with the disease may slightly elevate your risk, but the majority of cases are sporadic."
    },
    {
        "question": "Can Parkinson's disease be cured?",
        "answer": "Currently, there is no cure for Parkinson's disease. However, various treatments and therapies can help manage symptoms and improve quality of life."
    },
    {
        "question": "What lifestyle changes can help manage symptoms?",
        "answer": "Regular exercise, a balanced diet, adequate sleep, and stress management techniques can significantly help in managing symptoms. Engaging in social activities and maintaining a strong support network are also beneficial."
    },
    {
        "question": "Are there any clinical trials available for Parkinson's disease?",
        "answer": "Yes, many clinical trials are ongoing to explore new treatments and therapies for Parkinson's disease. Patients interested in participating should consult with their healthcare provider for information on current trials and eligibility."
    },
]

# Button to toggle FAQ visibility
if st.button("Toggle FAQs"):
    st.session_state.faq_visible = not st.session_state.faq_visible  # Toggle visibility

# Display FAQs if visible
if st.session_state.faq_visible:
    st.write("### Frequently Asked Questions")
    for faq in faqs:
        st.write(f"*{faq['question']}*")
        st.write(f"{faq['answer']}")






st.write("*Helpful Resources:*")
st.markdown("[Parkinson's Disease Foundation](https://www.pdf.org)")
st.markdown("[National Parkinson's Foundation](https://www.parkinson.org)")



# Parkinson's Awareness Section
st.subheader("Parkinson's Disease Awareness")


# Create two columns
col1, col2 = st.columns(2)

# Display images in the center by using both columns
with col1:
    st.image("https://imgs.search.brave.com/j94_4aRGXRAzvhEWY0BXJl2AnXyfm-0JlQYCzDAZLqg/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly93d3cu/cGFya2luc29uLm9y/Zy9zaXRlcy9kZWZh/dWx0L2ZpbGVzL3N0/eWxlcy80MDBweF93/aWRlL3B1YmxpYy9p/bWFnZXMvVGFrZTZG/b3JQRC0xLnBuZz9p/dG9rPVV3RV9mS1Y0", 
              caption="Parkinson's Disease Awareness Month", 
              use_column_width=True)

with col2:
    st.image("https://imgs.search.brave.com/rOLcvpMq7nGNVvvyQJzECrhMhagCMtBXT8NAPHvF7Xo/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly93d3cu/aG9tZXdhdGNoY2Fy/ZWdpdmVycy5jb20v/c3ViLzQ2NTUzL2lt/YWdlcy9wYXJrc2lu/c29ucy5wbmc", 
              caption="Parkinson's Signs and Symptoms", 
              use_column_width=True)