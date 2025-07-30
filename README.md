# Employee Salary Prediction Model 💼

A machine learning application that predicts whether an individual's income is **≤$50K** or **>$50K** based on various demographic and work-related attributes.

## 🚀 Features

- **Interactive Streamlit Web App**: User-friendly interface for income prediction
- **Machine Learning Pipeline**: Includes preprocessing, class imbalance handling, and model training
- **Model Performance Visualization**: Confusion matrix, ROC curves, and detailed metrics
- **Categorical Feature Insights**: Explore how different categories impact income predictions
- **Real-time Predictions**: Input employee details and get instant predictions

## 📊 Dataset

This project uses the [Adult Census Income Dataset](https://archive.ics.uci.uci.edu/ml/datasets/Adult) from the UCI Machine Learning Repository, which contains demographic data from the 1994 Census.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## 📁 Project Structure

```
├── streamlit_app.py                          # Main Streamlit application
├── Employee.ipynb                            # Jupyter notebook for model development
├── NewEmployee.ipynb                         # Additional analysis notebook
├── best_income_prediction_pipeline.joblib   # Trained ML pipeline
├── income_label_encoder.joblib              # Label encoder for target variable
├── adult 3.csv                              # Dataset
├── requirements.txt                         # Python dependencies
└── README.md                               # Project documentation
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Jeswanth-009/EmployeeSalaryPredictionModel.git
cd EmployeeSalaryPredictionModel
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 4. Open in Browser
The app will automatically open in your default browser at `http://localhost:8501`

## 🔧 Usage

1. **Input Employee Details**: Use the sidebar to input various employee attributes such as:
   - Age
   - Work class
   - Education level
   - Marital status
   - Occupation
   - And more...

2. **Get Prediction**: Click the "✨ Predict Income ✨" button to get the prediction

3. **Explore Model Performance**: View detailed metrics, confusion matrix, and ROC curves

4. **Analyze Feature Impact**: Explore how different categorical features impact income predictions

## 📈 Model Performance

The model achieves strong performance with:
- **Balanced Class Weights**: Handles class imbalance effectively
- **Comprehensive Preprocessing**: One-hot encoding for categorical features, MinMaxScaler for numerical features
- **Robust Classification**: Uses RandomForestClassifier or GradientBoostingClassifier for optimal results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Jeswanth**
- GitHub: [@Jeswanth-009](https://github.com/Jeswanth-009)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Census Income Dataset
- Streamlit team for the amazing framework
- Scikit-learn developers for the machine learning tools

---

⭐ If you found this project helpful, please give it a star!
