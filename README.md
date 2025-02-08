# Dietary-Improvements-Suggestions-EDA

This project was developed during the first semester of my Master's in Computer Science and Engineering as part of the Data Intensive Computing course. It was a collaborative effort by a team of four members, each contributing significantly to different aspects of the project.

**Objective:** Analyze users' calorie intake to assess their diet and suggest improvements.

**Team Members:**

| Student Name           | Student UB Number |
|------------------------|-------------------|
| Karthik Sharma Madugula | 50611293          |
| Santosh Kota           | 50593968          |
| Harshita Sherla        | 50593920          |
| Riya Agarwal           | 50609491          |


Project Phase 2:

Karthik Sharma Madugula:

Hypothesis 1: The interaction between macronutrients and specific micronutrients  significantly influences the overall health benefits of food items, as measured by a health score that considers caloric content, nutritional density, and micronutrient adequacy.
Hypothesis 2: Predictive Modeling of Dietary Patterns Based on Nutrient Profiles
Dietary patterns can be classified into distinct types (e.g., Mediterranean, vegetarian, high-protein) based on the nutrient profiles of foods consumed, where nutrient diversity and ratios are key indicators of dietary adherence.

The Code, Explanation and Analysis associated with each of these two questions is located in file 50611293_DIC_Phase2.ipynb
The report can be found as part of DIC_project-phase-2-report.pdf

Santosh Kota:

Hypothesis -1: Foods rich in fatty acids, antioxidants, B-vitamins, and amino acids can improve cognitive performance, particularly when these nutrients are combined in specific ratios.
Hypothesis -2: Physical exercise frequency is linked to higher protein intake among age group , with BMR also influencing protein consumption

The Code, Explanation and Analysis associated with each of these two questions is located in file 50593968_DIC_Project_Phase2.ipynb
The report can be found as part of DIC_project-phase-2-report.pdf

Harshita Sherla:

Hypothesis 1: “Age, weight, physical exercise level, and daily meal frequency collectively influence BMR.”
Hypothesis 2: “People with higher BMR consume more protein."

The Code, Explanation and Analysis associated with each of these two questions is located in file 50593920_DIC_Phase2.ipynb
The report can be found as part of DIC_project-phase-2-report.pdf

Riya Agarwal

Hypothesis 1: Users with higher physical activity levels have distinct macronutrient needs compared to those with low physical activity levels.
Hypothesis 2: Users cluster into distinct dietary patterns based on macronutrient ratios (carbs, proteins, fats).
Hypothesis 3: Age and gender together influence the likelihood of a user exceeding daily recommended calorie intake.

The Code, Explanation and Analysis associated with each of these three questions is located in file 50609491_DIC_Phase2.ipynb
The report can be found as part of DIC_project-phase-2-report.pdf

Project Phase 3:

Karthik Sharma Madugula:

Hypothesis : The interaction between macronutrients and specific micronutrients significantly influences the overall health benefits of food items, as measured by a health score that considers caloric content, nutritional density, and micronutrient adequacy.

The Code is located in file app.py in lines 228-385
The Explanation and Analysis associated with the hypothesis can be found as part of DIC_project-phase-3-report.pdf

Riya Agarwal

Hypothesis 1 : Physical exercise frequency is linked to higher protein intake among age group , with BMR also influencing protein consumption
Hypothesis 2 : Food Comparison Tool (Takes different food as input and give nutrients present in the food, so that user can choose accordingly on what to eat.

The Code is located in file app.py in lines 1-226 for Hypothesis 1 and in lines 389-422 for Hypothesis 2 
The Explanation and Analysis associated with the hypothesis can be found as part of DIC_project-phase-3-report.pdf

Santosh Kota:

Hypothesis : Predictive Modeling of Dietary Patterns Based on Nutrient Profiles Hypothesis: Dietary patterns can be classified into distinct types (e.g., Mediterranean, vegetarian, high-protein) based on the nutrient profiles of foods consumed, where nutrient diversity and ratios are key indicators of dietary adherence.

The Code is located in file app.py in lines 424-610
The Explanation and Analysis associated with the hypothesis can be found as part of DIC_project-phase-3-report.pdf

Harshita Sherla:

Hypothesis : Categorizing foods into weight-gain and weight-loss categories based on calorie content and suggest foods that align with individual dietary goals (e.g., achieving a target calorie intake). 

The Code is located in file app.py in lines 612-723
The Explanation and Analysis associated with the hypothesis can be found as part of DIC_project-phase-3-report.pdf

user_nutritional_data.csv and nutrients.csv are toy datasets.

The "app" folder contains the application code in file app.py. It contains the required datasets, user_nutritional_data.csv and nutrients.csv and also Nutri-ML.png file for appliaction.
The “exp” folder contains the final version of the previous python notebook code and final reports of Phase 1 and Phase 2 that has experimental results and explanations associated with each hypothesis executed previously.

Instructions to build the app from source code:

```markdown
## How to Run the Project

### Prerequisites
- **Python 3.7+** must be installed on your system.  
  Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)

- The required packages (e.g., Streamlit, Pandas, NumPy) must be installed globally or in the user space.

### Installing Dependencies
Open a terminal (Command Prompt on Windows or Terminal on macOS/Linux) and run:
```bash
pip install streamlit pandas numpy
```

If you need other packages that your application depends on, install them similarly:
```bash
pip install <other_package_name>
```

*Note: On some systems, you may need to use `pip3` instead of `pip`. On macOS/Linux, if you encounter permission issues, add `--user` at the end of the command.*

### Project Setup
1. Ensure that your datasets (e.g., `user_daily_nutritional_intake.csv` and `nutrients.csv`) are placed in the app directory as referenced by the code.

### Running the Application
From the project directory, run:
```bash
streamlit run app.py
```

### Accessing the Application
After running the command, Streamlit will print a local URL in the terminal, typically:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

Open your web browser and go to [http://localhost:8501](http://localhost:8501). You should see and be able to interact with your Streamlit application.

### Troubleshooting

- **Command not found**: If `streamlit` is not recognized, verify that Python and Pip are correctly installed and added to your system's PATH.

