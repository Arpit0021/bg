import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
import plotly.graph_objects as go
from fpdf import FPDF
from collections import Counter
import matplotlib.pyplot as plt

# --- API Key Setup ---
GEMINI_API_KEY = "AIzaSyDhMC_PEi-3ueM7a6jVc1qZhxTQSfQd7ZU"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini Prompt & Response ---
def get_full_food_analysis(food_item, qty_g):
    model = genai.GenerativeModel("gemini-2.0-flash")
    convo = model.start_chat()
    prompt = f"""
    Give a detailed, and well-structured nutritional breakdown of {qty_g}g of {food_item}:
    - Calories, protein, carbs, fat, sugar, fiber, cholesterol, sodium
    - Glycemic Index (out of 100) for this quantity
    - Provide detailed and engaging text about its nutritional properties
    - Is this food healthy or not, with a conclusion
    - Suggest 2-3 ways to make it healthier
    - Ensure the response is well-organized with big headings and bullet points for clarity
    - You as a certified nutritionist
    """
    convo.send_message(prompt)
    return convo.last.text

# --- Extract Nutrients ---
def extract_nutrients(text):
    nutrients = Counter()
    pattern = r"(Calories|Protein|Fat|Carbohydrates|Sugar|Fiber|Sodium|Cholesterol)[^\d]*(\d+\.?\d*)"
    for nut, val in re.findall(pattern, text, re.IGNORECASE):
        key = nut.capitalize()
        nutrients[key] += float(val)
    return dict(nutrients)

# --- Extract Glycemic Index ---
def extract_glycemic_index(text):
    m = re.search(r"Glycemic Index[^\d]*(\d+\.?\d*)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None

# --- Plot Donut-Gauge for GI ---
def plot_gi_donut(gi):
    rem = max(0, 100 - gi)
    fig = go.Figure(go.Pie(
        labels=["GI", "Remaining"],
        values=[gi, rem],
        hole=0.6, sort=False, direction="clockwise",
        marker_colors=["#FF7F0E", "#E5ECF6"], textinfo="none"))
    fig.update_layout(
        showlegend=False,
        annotations=[{"text": f"<b>{gi:.0f}/100</b>",
                      "font": {"size": 36}, "showarrow": False,
                      "x": 0.5, "y": 0.5}],
        margin=dict(l=20, r=20, t=40, b=20),
        title="🍚 Glycemic Index")
    return fig

# --- Nutritional Goal Tracker ---
def track_nutritional_goals(nutrients, goals, daily_intake):
    tracker = {}
    for nutrient in nutrients:
        if nutrient in goals:
            daily_intake[nutrient] = daily_intake.get(nutrient, 0) + nutrients[nutrient]
            tracker[nutrient] = {
                "Current": daily_intake[nutrient],
                "Goal": goals[nutrient],
                "Status": "Meets goal" if daily_intake[nutrient] >= goals[nutrient] else "Below goal"
            }
    return tracker, daily_intake

# --- Plot Nutritional Goals Progress ---
def plot_nutritional_progress(daily_intake, goals):
    labels = list(daily_intake.keys())
    current = list(daily_intake.values())
    target = [goals.get(label, 0) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Ensure 'Current Intake' is blue and 'Goal' is red
    ax.bar(labels, current, label="Current Intake", color="blue", alpha=0.7)
    ax.bar(labels, target, label="Goal", color="red", alpha=0.3)
    ax.set_ylabel('Amount (g or kcal)')
    ax.set_title('Nutritional Progress vs Goal')
    ax.legend()

    st.pyplot(fig)

# --- Personalized Meal Suggestions ---
def suggest_personalized_meal(goal):
    suggestions = {
        "Calories": [("Apple", 52), ("Banana", 96), ("Chicken Breast", 165)],
        "Protein": [("Eggs", 6), ("Tofu", 8), ("Salmon", 22)],
        "Carbohydrates": [("Rice", 45), ("Sweet Potato", 20), ("Oats", 27)],
        "Fat": [("Avocado", 15), ("Olives", 11), ("Almonds", 14)],
    }

    meals = {}
    for nutrient in goal:
        if nutrient in suggestions:
            meals[nutrient] = suggestions[nutrient]

    return meals

# --- Compare Foods ---
def compare_foods(food1, food2, qty1, qty2):
    rep1 = get_full_food_analysis(food1, qty1)
    rep2 = get_full_food_analysis(food2, qty2)

    nuts1 = extract_nutrients(rep1)
    nuts2 = extract_nutrients(rep2)

    return nuts1, nuts2, rep1, rep2

# --- Save Report ---
def save_report(food, text):
    path = f"{food}_report.txt"
    with open(path, "w") as f: f.write(text)
    return path

# --- Streamlit UI ---
st.set_page_config("🍱 Food Analyzer Gemini", layout="wide")
st.title("🥗 Advanced Food Analyzer")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🍲 Single Food Analysis", "⚖️ Compare Foods", "📝 Save / BMI Advice", "🎯 Nutritional Tracker"]
)

# Tab 1: Single Analysis + GI
with tab1:
    food = st.text_input("Enter a food item:")
    qty = st.number_input("Quantity (g):", min_value=1, value=100)
    if st.button("Analyze"):
        if not food:
            st.warning("Please enter a food item.")
        else:
            st.info("Analyzing with Gemini…")
            rep = get_full_food_analysis(food, qty)
            nuts = extract_nutrients(rep)
            gi = extract_glycemic_index(rep)

            st.subheader("📋 Gemini's Detailed Analysis")
            st.write(rep)

            if gi is not None:
                st.subheader("🍚 Glycemic Index")
                st.plotly_chart(plot_gi_donut(gi), use_container_width=True)

# Tab 2: Comparison (table only, no chart)
with tab2:
    c1, c2 = st.columns(2)
    f1 = c1.text_input("First food:")
    q1 = c1.number_input("Qty (g):", min_value=1, value=100, key="c1")
    f2 = c2.text_input("Second food:")
    q2 = c2.number_input("Qty (g):", min_value=1, value=100, key="c2")

    if st.button("Compare"):
        if f1 and f2:
            st.info("Comparing…")
            n1, n2, r1, r2 = compare_foods(f1, f2, q1, q2)
            all_keys = sorted(set(n1) | set(n2))

            comparison_data = {
                "Nutrient": all_keys,
                f"{f1.title()} ({q1}g)": [n1.get(k, 0) for k in all_keys],
                f"{f2.title()} ({q2}g)": [n2.get(k, 0) for k in all_keys],
            }
            df_comparison = pd.DataFrame(comparison_data)
            st.subheader("📊 Nutritional Comparison Table")
            st.dataframe(df_comparison)

            with st.expander(f"{f1.title()} Report"):
                st.write(r1)
            with st.expander(f"{f2.title()} Report"):
                st.write(r2)
        else:
            st.warning("Enter both foods.")

# Tab 3: Save / BMI
with tab3:
    st.subheader("📝 Save Report")
    sf = st.text_input("Food to save report:")
    sq = st.number_input("Qty (g):", min_value=1, value=100, key="save")
    if st.button("Save Report"):
        if sf:
            rp = get_full_food_analysis(sf, sq)
            p = save_report(sf, rp)
            st.success("Saved.")
            with open(p) as f: st.download_button("📄 Download", data=f, file_name=p)
        else:
            st.warning("Enter a food.")

    st.subheader("🧮 BMI Advice")
    w = st.number_input("Weight (kg)", min_value=30.0)
    h = st.number_input("Height (cm)", min_value=100.0)
    if st.button("Get BMI Advice"):
        bmi = w / ((h / 100) ** 2)
        st.write(f"Your BMI: {bmi:.1f}")
        if bmi < 18.5:
            st.warning("Underweight: Increase protein & calories.")
        elif bmi < 25:
            st.success("Normal: Keep balanced diet.")
        elif bmi < 30:
            st.warning("Overweight: Reduce sugars/fats, increase fiber.")
        else:
            st.error("Obese: Consult nutritionist & lower calorie density.")

# Tab 4: Nutritional Goal Tracker (Multiple Foods)
with tab4:
    st.subheader("🎯 Set Your Nutritional Targets")
    targets = {
        "Calories": st.number_input("Daily Calorie Goal (kcal)", min_value=1000, value=2000),
        "Protein": st.number_input("Daily Protein Goal (g)", min_value=50, value=100),
        "Carbohydrates": st.number_input("Daily Carb Goal (g)", min_value=100, value=250),
        "Fat": st.number_input("Daily Fat Goal (g)", min_value=20, value=70),
    }

    if 'daily_intake' not in st.session_state:
        st.session_state['daily_intake'] = {}

    st.subheader("🍎 Track Your Food Intake")
    food_col, qty_col, add_col = st.columns([3, 2, 1])
    with food_col:
        food_item = st.text_input("Food Item", key="food_input")
    with qty_col:
        quantity = st.number_input("Quantity (g)", min_value=1, value=100, key="qty_input")
    with add_col:
        if st.button("Add Food", key="add_button"):
            if food_item:
                analysis_result = get_full_food_analysis(food_item, quantity)
                nutrients = extract_nutrients(analysis_result)
                tracker, st.session_state['daily_intake'] = track_nutritional_goals(nutrients, targets, st.session_state['daily_intake'])
                st.success(f"{quantity}g of {food_item} added to tracker.")
            else:
                st.warning("Please enter a food item.")

    if st.session_state['daily_intake']:
        st.subheader("📊 Daily Nutritional Progress")
        goal_summary = pd.DataFrame.from_dict(
            {k: v for k, v in st.session_state['daily_intake'].items() if k in targets},
            orient='index',
            columns=['Current']
        )
        goal_summary['Goal'] = goal_summary.index.map(targets)
        goal_summary['Status'] = goal_summary.apply(lambda row: "Meets goal" if row['Current'] >= row['Goal'] else "Below goal", axis=1)
        st.dataframe(goal_summary)
        plot_nutritional_progress(st.session_state['daily_intake'], targets)

    if st.button("Clear Tracker"):
        st.session_state['daily_intake'] = {}
        st.info("Nutritional tracker cleared.")

