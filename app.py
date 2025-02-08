import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

conn = sqlite3.connect('nutrition.db')

nutrients_df = pd.read_csv('nutrients.csv')  

user_nutritional_data = pd.read_csv('user_nutritional_data.csv')  
st.image("./Nutri-ML.png", use_container_width=True)

st.header("Dietary Intake and Analysis")
dietary_food_choices = st.multiselect("Select Foods Consumed for Analysis", nutrients_df["name"].unique())

st.sidebar.title(f"**User Information**")
age = st.sidebar.number_input("Age", min_value=15, max_value=75, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=122, max_value=188, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=35, max_value=150, value=70)
exercise_levels = {"None": 1.2, "Light": 1.375, "Moderate": 1.55, "Heavy": 1.725, "Very Heavy": 1.9}
exercise = st.sidebar.selectbox("Exercise Level", list(exercise_levels.keys()))

bmr = 10 * weight + 6.25 * height - 5 * age + 5  


calorie_needs = bmr * exercise_levels[exercise]

st.write(f"Your Basal Metabolic Rate (BMR) is: **{bmr:.2f} kcal/day**")
st.write(f"Your estimated daily calorie needs: **{calorie_needs:.2f} kcal/day**")
height_in_meters = height / 100  # Converting height to meters
bmi = weight / (height_in_meters ** 2)

st.write(f"Your Body Mass Index (BMI) is: **{bmi:.2f}**")

if bmi < 18.5:
    st.write("Your BMI indicates you are underweight. Consider increasing your calorie intake and consulting a nutritionist.")
elif 18.5 <= bmi < 24.9:
    st.write("Your BMI is within the healthy weight range. Keep up the good work!")
elif 25 <= bmi < 29.9:
    st.write("Your BMI indicates you are overweight. Consider a balanced diet and regular physical activity.")
else:
    st.write("Your BMI indicates obesity. It may be beneficial to consult a healthcare professional for advice on weight management.")



if calorie_needs < 2000:
    st.write("Suggested Exercise Level: **Moderate or High**")
elif calorie_needs >= 2000 and calorie_needs <= 2500:
    st.write("Suggested Exercise Level: **Moderate**")
else:
    st.write("Suggested Exercise Level: **Low or Moderate**")

# Initialize nutrient analysis dictionary with keys matching your dataset
total_nutrients_analysis = {"Calories": 0, "Carbohydrate (g)": 0, "Protein (g)": 0, "Fat (g)": 0}

# Iterate over the selected food items and calculate total nutrients
for _, row in nutrients_df[nutrients_df["name"].isin(dietary_food_choices)].iterrows():
    try:
        total_nutrients_analysis["Calories"] += row.get("Calories", 0)
        total_nutrients_analysis["Carbohydrate (g)"] += row.get("Carbohydrate (g)", 0)
        total_nutrients_analysis["Protein (g)"] += row.get("Protein (g)", 0)
        total_nutrients_analysis["Fat (g)"] += row.get("Fat (g)", 0)
    except KeyError as e:
        st.error(f"Column missing in the dataset: {e}")

# Display total nutrient intake
st.subheader("Total Nutrient Intake from Dietary Analysis")
st.write(f"Calories: **{total_nutrients_analysis['Calories']:.2f} kcal**")
st.write(f"Carbohydrates: **{total_nutrients_analysis['Carbohydrate (g)'] / 1000:.2f} g**")
st.write(f"Proteins: **{total_nutrients_analysis['Protein (g)']:.2f} g**")
st.write(f"Fats: **{total_nutrients_analysis['Fat (g)']:.2f} g**")

if total_nutrients_analysis["Calories"] > calorie_needs:
    st.write("Your calorie intake exceeds your daily needs. Consider increasing your exercise level.")
elif total_nutrients_analysis["Calories"] < calorie_needs:
    st.write("Your calorie intake is below your daily needs. Consider a more moderate or high exercise level.")
else:
    st.write("Your calorie intake matches your daily needs. A moderate exercise level should be appropriate.")

#age and exerise

st.header("Hypothesis Analysis: Physical Exercise and Protein Intake")

# Segment users into age groups
age_groups = pd.cut(
    [age],  # User's age
    bins=[0, 18, 30, 45, 60, 75],  # Age group ranges
    labels=["Under 18", "18-30", "30-45", "45-60", "60+"]
)
st.write(f"Your age group: **{age_groups[0]}**")

# Analyze protein intake relative to exercise level and BMR
exercise_factor = exercise_levels[exercise]  # Get exercise factor for BMR calculation
adjusted_bmr = bmr * exercise_factor  # Adjust BMR based on exercise level
st.write(f"Your exercise-adjusted BMR: **{adjusted_bmr:.2f} kcal**")

# Fetch protein intake from user-selected foods
protein_intake = total_nutrients_analysis.get("Protein (g)", 0)

# Display the relationship between BMR, exercise level, and protein intake
st.subheader("Protein Intake Analysis")
if protein_intake > 0:
    st.write(f"Your protein intake: **{protein_intake:.2f} g**")

    # Plot the relationship between exercise levels and protein intake
    exercise_levels_df = pd.DataFrame({
        "Exercise Level": list(exercise_levels.keys()),
        "Factor": list(exercise_levels.values())
    })
    exercise_levels_df["Predicted Protein Intake (g)"] = exercise_levels_df["Factor"] * (protein_intake / exercise_factor)


    # Display insights
    if protein_intake < adjusted_bmr * 0.1:
        st.warning("Your protein intake is below recommended levels for your activity level.")
        st.write("Consider incorporating more high-protein foods into your diet.")
    elif protein_intake > adjusted_bmr * 0.2:
        st.success("Your protein intake is well-aligned with your activity level and BMR!")
    else:
        st.info("Your protein intake is within an acceptable range but could be optimized.")
else:
    st.write("No protein intake data available for analysis. Please select dietary items.")

# Insights for Age Groups
st.subheader("Protein Intake Recommendations by Age Group")
age_group_recommendations = {
    "Under 18": "Protein is crucial for growth and development. Ensure adequate intake of lean meats, beans, and dairy.",
    "18-30": "Focus on maintaining muscle mass and overall health with consistent protein consumption.",
    "30-45": "Include a mix of plant-based and lean protein sources to support active lifestyles.",
    "45-60": "Prioritize protein to prevent muscle loss and maintain energy levels.",
    "60+": "Higher protein intake is recommended to combat age-related muscle loss and support overall health."
}
st.write(age_group_recommendations.get(age_groups[0], "No specific recommendation."))

# Overall interpretation

st.subheader("Dietary Recommendations")


if 'name' in nutrients_df.columns:
    # Extract the numeric columns for health score calculation
    numeric_columns = [
        "Protein (g)", "Fiber (g)", "Calcium (mg)", "Iron, Fe (mg)", "Vitamin C (mg)",
        "Cholesterol (mg)", "Saturated Fats (g)"
    ]
    health_data = nutrients_df[nutrients_df["name"].isin(dietary_food_choices)][numeric_columns].copy()

    # Handle missing data (drop NaN values)
    health_data = health_data.dropna(subset=numeric_columns)

    # Apply health score calculation to each selected food
    if health_data.empty:
        st.warning(
            "No valid data available for the selected foods. Please select foods with valid nutritional information."
        )
    else:
        def calculate_health_score(row):
            score = (
                    row["Protein (g)"] * 2 +
                    row["Fiber (g)"] * 1.5 +
                    row["Calcium (mg)"] * 0.1 +
                    row["Iron, Fe (mg)"] * 0.1 +
                    row["Vitamin C (mg)"] * 0.2 -
                    row["Cholesterol (mg)"] * 0.05 -
                    row["Saturated Fats (g)"] * 0.2
            )
            return score

        health_data["Health_Score"] = health_data.apply(calculate_health_score, axis=1)

        # Select features for Random Forest model (with interactions)
        health_data["Protein_x_Calcium"] = health_data["Protein (g)"] * health_data["Calcium (mg)"]
        health_data["Fiber_x_Iron"] = health_data["Fiber (g)"] * health_data["Iron, Fe (mg)"]

        # Define X and y for regression
        X_health = health_data[[
            "Protein (g)", "Fiber (g)", "Calcium (mg)", "Iron, Fe (mg)", "Vitamin C (mg)",
            "Cholesterol (mg)", "Saturated Fats (g)", "Protein_x_Calcium", "Fiber_x_Iron"
        ]]
        y_health = health_data["Health_Score"]

        # Initialize predicted_score to None to handle cases without predictions
        predicted_score = None

        # Check if there are enough samples to perform train-test split
        if len(X_health) < 2:
            st.warning("Not enough data to train the model. Please select more foods for analysis.")
        else:
            # Train and test RandomForestRegressor
            X_train, X_test, y_train, y_test = train_test_split(X_health, y_health, test_size=0.2, random_state=42)
            model_health = RandomForestRegressor(random_state=42)
            model_health.fit(X_train, y_train)

            # Predict health scores
            predicted_health_score = model_health.predict(X_test)

            # Ensure predictions are not empty
            if len(predicted_health_score) > 0:
                predicted_score = predicted_health_score[0]  # Taking the first predicted value
                st.subheader("Predicted Health Score")
                st.write(f"Your predicted health score is: **{predicted_score:.2f}**")
            else:
                st.warning("No predictions could be made due to insufficient test data.")

        # Provide feedback only if a score was predicted
        if predicted_score is not None:
            # Interpret health score and provide dataset-driven suggestions
            st.subheader("Health Recommendations Based on Your Diet")

            # Suggestions based on missing nutrients
            total_nutrients_analysis = {
                "Protein (g)": health_data["Protein (g)"].sum(),
                "Fiber (g)": health_data["Fiber (g)"].sum(),
                "Calcium (mg)": health_data["Calcium (mg)"].sum(),
                "Iron, Fe (mg)": health_data["Iron, Fe (mg)"].sum(),
                "Vitamin C (mg)": health_data["Vitamin C (mg)"].sum(),
                "Cholesterol (mg)": health_data["Cholesterol (mg)"].sum(),
                "Saturated Fats (g)": health_data["Saturated Fats (g)"].sum()
            }

            # Provide feedback based on predicted health score
            if predicted_score > 75:
                st.success("You have a very healthy diet!")
                st.write("Keep up the great work! Continue focusing on a balanced diet with a variety of nutrients.")
            elif predicted_score > 50:
                st.info("You have a relatively healthy diet, but there's room for improvement.")
                st.write("Consider adding more fruits and vegetables for added vitamins and minerals.")
            elif predicted_score > 25:
                st.warning("Your diet could use some improvement. Consider adding more nutritious foods.")
                st.write("Try including more fiber-rich foods like whole grains, legumes, and vegetables.")
                st.write(
                    "Increasing your protein intake from lean sources like chicken, fish, and plant-based proteins could be beneficial.")
                st.write(
                    "Additionally, reducing your intake of saturated fats and cholesterol could help improve your overall health.")
            else:
                st.error("Your diet appears to be quite unhealthy. Consider seeking professional advice.")
                st.write("Focus on cutting back on processed foods, sugary snacks, and fried foods.")
                st.write("Try increasing your intake of fresh vegetables, fruits, lean proteins, and whole grains.")
                st.write(
                    "Consulting a nutritionist or dietitian could be a good step towards creating a balanced and healthy eating plan.")

        # Dataset-driven Suggestions for the user based on their selected foods
            if total_nutrients_analysis["Protein (g)"] < 50:
                st.warning(
                    "Your protein intake appears to be low. Consider incorporating more high-protein foods like beans, lentils, chicken, or fish.")
                high_protein_foods = nutrients_df.sort_values(by="Protein (g)", ascending=False).head(5)
                st.write("Recommended High-Protein Foods:")
                st.table(high_protein_foods[["name", "Protein (g)"]])

            if total_nutrients_analysis["Fiber (g)"] < 25:
                st.warning(
                    "Your fiber intake seems low. To improve digestion and overall health, try adding more fiber-rich foods such as whole grains, oats, and vegetables.")
                high_fiber_foods = nutrients_df.sort_values(by="Fiber (g)", ascending=False).head(5)
                st.write("Recommended High-Fiber Foods:")
                st.table(high_fiber_foods[["name", "Fiber (g)"]])

            if total_nutrients_analysis["Calcium (mg)"] < 800:
                st.warning(
                    "Your calcium intake is below the recommended level. Consider adding more calcium-rich foods like dairy products, leafy greens, or fortified plant-based milk.")
                high_calcium_foods = nutrients_df.sort_values(by="Calcium (mg)", ascending=False).head(5)
                st.write("Recommended High-Calcium Foods:")
                st.table(high_calcium_foods[["name", "Calcium (mg)"]])

            if total_nutrients_analysis["Iron, Fe (mg)"] < 10:
                st.warning(
                    "Your iron intake appears to be low. You can increase your intake by consuming more iron-rich foods like spinach, red meat, beans, and fortified cereals.")
                high_iron_foods = nutrients_df.sort_values(by="Iron, Fe (mg)", ascending=False).head(5)
                st.write("Recommended High-Iron Foods:")
                st.table(high_iron_foods[["name", "Iron, Fe (mg)"]])

            if total_nutrients_analysis["Vitamin C (mg)"] < 60:
                st.warning(
                    "Your vitamin C intake seems low. Add more citrus fruits, strawberries, bell peppers, and leafy greens to your diet.")
                high_vitamin_c_foods = nutrients_df.sort_values(by="Vitamin C (mg)", ascending=False).head(5)
                st.write("Recommended High-Vitamin C Foods:")
                st.table(high_vitamin_c_foods[["name", "Vitamin C (mg)"]])

            if total_nutrients_analysis["Cholesterol (mg)"] > 300:
                st.warning(
                    "Your cholesterol intake is high. To lower cholesterol levels, try reducing consumption of animal-based fats and processed foods.")
                low_cholesterol_foods = nutrients_df[nutrients_df["Cholesterol (mg)"] < 20].head(5)
                st.write("Recommended Low-Cholesterol Foods:")
                st.table(low_cholesterol_foods[["name", "Cholesterol (mg)"]])

            if total_nutrients_analysis["Saturated Fats (g)"] > 20:
                st.warning(
                    "Your intake of saturated fats is high. Reducing saturated fats from red meat and dairy can benefit heart health.")
                low_saturated_fat_foods = nutrients_df[nutrients_df["Saturated Fats (g)"] < 1].head(5)
                st.write("Recommended Low-Saturated Fat Foods:")
                st.table(low_saturated_fat_foods[["name", "Saturated Fats (g)"]])


# Part 3: Food Comparison Tool
# Function for comparing nutritional values of selected food items
# Food Comparison Tool function
def food_comparison_tool():
    st.header("Food Comparison Tool")
    food_items = nutrients_df['name'].tolist()

    # Allow user to select multiple food items for comparison
    selected_foods_for_comparison = st.multiselect("Select food items for comparison:", food_items)

    if selected_foods_for_comparison:
        comparison_data = nutrients_df[nutrients_df["name"].isin(selected_foods_for_comparison)]
        nutrient_columns = [col for col in comparison_data.columns if col not in ['name', 'Food Group','ID']]
        # Allow user to choose which nutrients to compare
        nutrients_for_comparison = st.multiselect("Select nutrients to compare:",
                                                  nutrient_columns)

        # Filter data for selected nutrients
        if nutrients_for_comparison:
            comparison_data_filtered = comparison_data[["name"] + nutrients_for_comparison]

            # Display a table comparing the selected nutrients for the chosen foods
            st.write(comparison_data_filtered)

            # Generate a bar chart for comparison
            comparison_data_filtered.set_index("name").plot(kind="bar", figsize=(10, 6))
            plt.title("Nutrient Comparison")
            plt.ylabel("Amount")
            plt.xlabel("Food Items")
            st.pyplot()  # Display the plot
        else:
            st.write("No nutrients selected for comparison.")
    else:
        st.write("No food items selected for comparison.")

# Run the food comparison tool
food_comparison_tool()

#--------------------------------KNN Model------------------------------------------------

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.impute import SimpleImputer

st.header("Diet Pattern Classification")
st.write("Identify the category your dietary patterns fall into and receive diet suggestions based on the nutrients you are lacking.")
st.subheader("Categories")
dietary_categories = {
    'High-Protein': 'Diet rich in protein, lower in carbohydrates and fats.',
    'Mediterranean': 'High in fiber and healthy fats, moderate protein, lower in sugar.',
    'Vegetarian': 'Plant-based diet, low in fat, high in carbohydrates.',
    'Standard': 'Balanced diet without specific health focus.'
}

# Creating a dataframe from category dictionary
df_categories = pd.DataFrame.from_dict(dietary_categories, orient='index', columns=['Description'])
st.table(df_categories)
# feature selection for classification
features = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)', 
            'Fiber (g)', 'Sugars (g)', 'Calcium (mg)', 
            'Iron, Fe (mg)', 'Potassium, K (mg)']
imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'most_frequent' as well
nutrients_df[features] = imputer.fit_transform(nutrients_df[features])
X = nutrients_df[features]


# Craeting labels based on nutrient ratios from the gievn datset
def create_label(row):
    protein_ratio = row['Protein (g)'] / (row['Carbohydrate (g)'] + row['Fat (g)']) if (row['Carbohydrate (g)'] + row['Fat (g)']) != 0 else 0
    fiber_ratio = row['Fiber (g)'] / row['Carbohydrate (g)'] if row['Carbohydrate (g)'] > 0 else 0

    if protein_ratio > 0.5:
        return 'High-Protein'
    elif fiber_ratio > 0.1 and row['Sugars (g)'] < 10:
        return 'Mediterranean'
    elif row['Fat (g)'] < 5 and row['Carbohydrate (g)'] > 20:
        return 'Vegetarian'
    else:
        return 'Standard'
def suggest_best_alternative(selected_foods, predicted_pattern):
    best_alternative = None
    best_food_name = ""
    vegetarian_food_groups = ['Fruits', 'Vegetables', 'Grains and Pasta', 'Dairy and Egg Products', 'Sweets', 'Beverages', 'Breakfast Cereals', 'Spices and Herbs', 'Fats and Oils', 'Nuts and Seeds', 'Baby Foods']
    for user_input_food in selected_foods:
        selected_food = nutrients_df[nutrients_df['name'] == user_input_food]
        
        if selected_food.empty:
            continue
        
        selected_features = selected_food[features].values.reshape(1, -1)
        scaled_features = scaler.transform(selected_features)
        
        distances, indices = knn.kneighbors(scaled_features)
        
        recommended_foods = nutrients_df.iloc[indices[0]]
        recommended_foods = recommended_foods[recommended_foods['name'] != user_input_food]
        

        if predicted_pattern == 'Vegetarian':
            recommended_foods = recommended_foods[recommended_foods['Food Group'].isin(vegetarian_food_groups)]
        if recommended_foods.empty:
            continue
        
        better_alternatives = recommended_foods[
            (recommended_foods['Calcium (mg)'] > selected_food['Calcium (mg)'].values[0]) |
            (recommended_foods['Iron, Fe (mg)'] > selected_food['Iron, Fe (mg)'].values[0]) |
            (recommended_foods['Potassium, K (mg)'] > selected_food['Potassium, K (mg)'].values[0])
        ]
        
        if not better_alternatives.empty:
            top_alternative_for_this_food = better_alternatives[['name', 'Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)', 'Calcium (mg)', 'Iron, Fe (mg)', 'Potassium, K (mg)']].head(1)
            
            if best_alternative is None or top_alternative_for_this_food['Calories'].values[0] < best_alternative['Calories'].values[0]:
                best_alternative = top_alternative_for_this_food
                best_food_name = user_input_food
    
    return best_alternative, best_food_name

def get_recommendations(predicted_pattern, nutrients_df):
    if predicted_pattern == 'Vegetarian':
        nutrient = 'Fat (g)'
        vegetarian_food_groups = ['Fruits', 'Vegetables', 'Grains and Pasta', 'Dairy and Egg Products', 'Sweets', 'Beverages', 'Breakfast Cereals', 'Spices and Herbs', 'Fats and Oils', 'Nuts and Seeds', 'Baby Foods']
        nutrients_df = nutrients_df[nutrients_df['Food Group'].isin(vegetarian_food_groups)]
    elif predicted_pattern == 'High-Protein':
        nutrient = 'Carbohydrate (g)'
    elif predicted_pattern == 'Mediterranean':
        nutrient = 'Sugars (g)'
    else:  # Standard diet
        nutrient = 'Protein (g)'
    
    food_groups = nutrients_df['Food Group'].unique()
    recommendations = pd.DataFrame()
    
    for group in food_groups:
        group_df = nutrients_df[nutrients_df['Food Group'] == group]
        top_5 = group_df.nlargest(5, nutrient)[['name', nutrient, 'Calories', 'Food Group']]
        recommendations = pd.concat([recommendations,top_5])
    
    return recommendations, nutrient



y = X.apply(create_label, axis=1)

# Splitting the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model tarining
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


if dietary_food_choices:
    selected_food_data = nutrients_df[nutrients_df['name'].isin(dietary_food_choices)]
    total_nutrients = selected_food_data[features].sum()
    table_info = total_nutrients.to_frame(name='Total')
    tot_cal = total_nutrients["Calories"]
    
    st.subheader("Your Nutrient Profile")
    st.write(table_info)
    
    user_nutrients_scaled = scaler.transform([total_nutrients])
    predicted_pattern = knn.predict(user_nutrients_scaled)[0]
    st.success(f"Predicted Dietary Pattern: **{predicted_pattern}**")

    st.subheader("Nutrient Profile Comparison")
    nutrient_means = X.groupby(y).mean()
    comparison_data = pd.concat([nutrient_means, total_nutrients.to_frame().T])
    comparison_data.index = list(nutrient_means.index) + ['Your Diet']
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(comparison_data, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Nutrient Profile Comparison')
    st.pyplot(plt)
#displaying results only if the mentioned calorie is less than max calorie 
    if tot_cal < calorie_needs:
        st.subheader("Food Recommendations Based on Your Dietary Pattern")
        recommendations, nutrient = get_recommendations(predicted_pattern, nutrients_df)
        #removed food groups consisting of eggs and meat if DP is vegetarian
        if predicted_pattern == 'Vegetarian':
            st.info("As your diet pattern is predicted to be Vegetarian, we've excluded meat-based food groups from the recommendations.")
        
        food_groups = ['All'] + list(recommendations['Food Group'].unique())
        selected_group = st.selectbox("Select Food Group:", food_groups)
        
        if selected_group != 'All':
            filtered_recommendations = recommendations[recommendations['Food Group'] == selected_group]
        else:
            filtered_recommendations = recommendations
        
        filtered_recommendations = filtered_recommendations.reset_index(drop=True)
        filtered_recommendations.index += 1
        
        st.write(f"Top foods to boost your {nutrient} intake:")
        st.dataframe(filtered_recommendations)

#new  - micronutrients feature
        top_alternative, original_food_name = suggest_best_alternative(dietary_food_choices,predicted_pattern)
        
        if top_alternative is None:
            st.write("No better alternatives found based on micronutrient content.")
        else:
            st.write(f"To get required micro-nutrients take this instead of ***'{original_food_name}':***")
            st.dataframe(top_alternative)
    else:
        st.warning("Your calorie intake exceeds your daily needs. Consider reducing your portion sizes for food recommendation.")
else:
    st.warning("Please select some foods to predict your dietary pattern.")


#----------------------------------------------------------------new model------------------------

import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    # One-hot encode the 'Food Group' column
    encoder = OneHotEncoder(drop='first')
    food_group_encoded = encoder.fit_transform(data[['Food Group']]).toarray()
    
    # Combine encoded data with other features
    X = np.hstack([data[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)', 'Sugars (g)']].values, food_group_encoded])
    
    # Create target variable: 1 for weight gain, 0 for weight loss based on calories
    y = (data['Calories'] > 250).astype(int).values  # Adjusted threshold for better classification
    
    return X, y, encoder

# Build and train the HistGradientBoostingClassifier model
def build_and_train_model(X, y):
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    return model

# Function to suggest foods to meet calorie goals
def suggest_foods_to_meet_goal(data, remaining_calories):
    # Suggesting foods that can help meet the remaining calorie goal without exceeding it
    suggested_foods = data[data['Calories'] <= remaining_calories]
    sorted_suggested_foods = suggested_foods[['name', 'Calories']].sort_values(by='Calories', ascending=False).reset_index(drop=True)
    sorted_suggested_foods.index += 1  # Start index from 1 instead of 0
    return sorted_suggested_foods


def main():
    st.header("Nutrition Recommendation System")
    
    # Load data
    file_path = "nutrients.csv" 
    data = load_data(file_path)

    X, y, encoder = preprocess_data(data)

    model = build_and_train_model(X, y)

    st.write("This model helps determine whether your selected foods contribute to weight gain or loss and suggests foods to meet your dietary requirements.")
    
    selected_foods = st.multiselect("Select Foods", data['name'].tolist())
    
    target_calories = st.number_input("Enter your target calorie intake:", min_value=0, value=2000)
    
    if st.button("Run Prediction"):
        if selected_foods:
            selected_data = data[data['name'].isin(selected_foods)]
            
            if not selected_data.empty:
                selected_features = np.hstack([selected_data[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)', 'Sugars (g)']].values,
                                                              encoder.transform(selected_data[['Food Group']]).toarray()])

                # Predict weight gain or loss for selected foods
                predictions = model.predict(selected_features)
                avg_prediction = np.mean(predictions)

                # Calculating total calories of selected foods and remaining calories to reach goal
                total_calories_selected = selected_data['Calories'].sum()
                st.write(f"Your current calorie intake is: **{total_calories_selected:.2f}g** calories.")
                remaining_calories = target_calories - total_calories_selected

                if avg_prediction > 0.5:
                    prediction_result = "weight gain"
                else:
                    prediction_result = "weight loss"

                # Checking if user's goal calorie intake matches the prediction result and provide feedback
                if (target_calories>total_calories_selected and avg_prediction > 0.5) or (target_calories<total_calories_selected and avg_prediction <= 0.5):
                    st.success(f"You have chosen foods that contribute to your goal of {prediction_result}!")
                else:
                    st.warning(f"The selected foods are likely to contribute to {prediction_result}.")

                if remaining_calories > 0:
                    st.write(f"You need an additional **{remaining_calories:.2f}g** calories to reach your target.")
                    suggested_foods = suggest_foods_to_meet_goal(data, remaining_calories)
                    
                    if not suggested_foods.empty:
                        st.write("You can consider these higher calorie foods to meet your calorie goal:")
                        st.dataframe(suggested_foods[['name', 'Calories']])
                    else:
                        st.write("No additional foods available to meet your calorie goal.")
                elif remaining_calories < 0:
                    st.write(f"You have exceeded your daily calorie goal by **{-remaining_calories:.2f}g** calories.")
                    # Suggest lower-calorie foods for weight loss
                    weight_loss_threshold = abs(remaining_calories)
                    weight_loss_foods = data[data['Calories'] <= weight_loss_threshold]
                    sorted_weight_loss_foods = weight_loss_foods[['name', 'Calories']].sort_values(by='Calories', ascending=False).reset_index(drop=True)
                    sorted_weight_loss_foods.index += 1
                    
                    st.write("You can consider these lower-calorie foods for weight loss:")
                    st.dataframe(sorted_weight_loss_foods[['name', 'Calories']])
                    
            else:
                st.write("No valid foods selected.")
        else:
            st.write("Please select at least one food.")

if __name__ == "__main__":
    main()