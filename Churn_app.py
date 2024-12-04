import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = pickle.load(open('Churn_Predictor.pkl', 'rb'))

# Initialize the scaler (must match the one used during training)
scaler = StandardScaler()

# App title
st.title("Customer Churn Prediction using Logistic Regression")

# User inputs
st.header("Enter Customer Details")
# Input fields for user data
gender = st.selectbox("Select Gender", options=['Female', 'Male'])
SeniorCitizen = st.selectbox("Are you a Senior Citizen?", options=['Yes', 'No'])
Partner = st.selectbox("Do you have a partner?", options=['Yes', 'No'])
Dependents = st.selectbox("Are you dependent on others?", options=['Yes', 'No'])
PhoneService = st.selectbox("Do you have phone service?", options=['Yes', 'No'])
MultipleLines = st.selectbox("Do you have multiple lines?", options=['No', 'Yes', 'No phone service'])
InternetService = st.selectbox("Do you have internet service?", options=['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Does your package include online security?", options=['Yes', 'No'])
DeviceProtection = st.selectbox("Does your package include device protection?", options=['Yes', 'No'])
TechSupport = st.selectbox("Do you have access to tech support?", options=['Yes', 'No'])
StreamingTV = st.selectbox("Do you have TV streaming services?", options=['Yes', 'No'])
StreamingMovies = st.selectbox("Do you have movie streaming services?", options=['Yes', 'No'])
Contract = st.selectbox("What is your contract type?", options=['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Do you use paperless billing?", options=['Yes', 'No'])
PaymentMethod = st.selectbox(
    "What is your payment method?",
    options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
)
tenure = st.number_input("Enter your tenure (in months):", min_value=0, max_value=100, value=1)
MonthlyCharges = st.number_input("Enter your monthly charges:", min_value=0.0, step=0.01)
TotalCharges = st.number_input("Enter your total charges:", min_value=0.0, step=0.01)

# Prediction button
if st.button("Predict Churn"):
    # Create a DataFrame from the user input
    user_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })

    # Encode categorical variables
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    for col in categorical_columns:
        user_data[col] = user_data[col].astype('category').cat.codes

    # Scale numerical variables
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    user_data[numerical_columns] = scaler.fit_transform(user_data[numerical_columns])

    # Predict churn
    prediction = model.predict(user_data)

    # Tips for Churn Prevention
    churn_tips_data = {
    "Tips for Churn Prevention": [
        "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
    ]
}

    # Tips for Customer Retention (Not Churning)
    retention_tips_data = {
    "Tips for Customer Retention": [
        "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
    ]
}

    # Display the result
    if prediction== 1:
        st.error("The customer is likely to churn.")
        st.write("Here are 10 tips for Churn Prevention:")
        st.dataframe(churn_tips_data, height=800,width=800)
    else:
        st.success("The customer is not likely to churn.")
        st.write("Here are 10 tips for Customer Retention (Not Churning):")
        st.dataframe(retention_tips_data, height=800,width=800)

