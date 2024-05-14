from tkinter import *
from customtkinter import *
from main import manual_testing
from tkinter import ttk

set_default_color_theme("dark-blue")
set_appearance_mode("dark")
app = CTk()
app.title("Fake News Detection")
app.geometry("700x700")
app.resizable(False, False)

def update_text_area_height(event):
    current_height = text_area['height']
    current_text_length = len(text_area.get('1.0', 'end-1c'))
    if current_text_length > 60:
        new_height = min(current_text_length // 60 * 10 + 100, 200)
        if new_height > current_height:
            text_area.configure(height=new_height)

def classify_news():
    news = text_area.get('1.0', 'end-1c')
    lr_pred, dt_pred, gb_pred, rf_pred = manual_testing(news)
    
    # Generate report
    report = f"LR Prediction: {lr_pred}\nDT Prediction: {dt_pred}\nGB Prediction: {gb_pred}\nRF Prediction: {rf_pred}"
    print(report)    
    # Generate overall summary
    overall_summary = generate_overall_summary(lr_pred, dt_pred, gb_pred, rf_pred)
    overall_summary_label.configure(text=overall_summary)
    
    # Convert predictions to binary values for progress bar (0 for fake, 1 for real)
    lr_pred_binary = 1 if lr_pred else 0
    dt_pred_binary = 1 if dt_pred else 0
    gb_pred_binary = 1 if gb_pred else 0
    rf_pred_binary = 1 if rf_pred else 0

    # Calculate total correct predictions
    total_correct = lr_pred_binary + dt_pred_binary + gb_pred_binary + rf_pred_binary

    # Calculate progress bar value (percentage of correct predictions)
    progress_value = total_correct * 25  # Each prediction contributes 25% to the progress

    # Update progress bar value
    slider.configure(value=progress_value)

    news = text_area.get('1.0', 'end-1c')
    
    # Determine the background color based on predictions
    if 100 > progress_value > 50:
        text_area_result.configure(fg_color="white")  # Between 100 - 50
    elif progress_value < 50:
        text_area_result.configure(fg_color="red")   # Less than 50
    elif progress_value == 100:
        text_area_result.configure(fg_color="green")   # Natural
    
    # Update news text area
    text_area_result.delete(1.0, END)
    text_area_result.insert(END, news)


def generate_overall_summary(lr_pred, dt_pred, gb_pred, rf_pred):
    true_count = sum([lr_pred, dt_pred, gb_pred, rf_pred])
    false_count = 4 - true_count
    overall_summary = f"Overall Summary:\n"
    overall_summary += f"Total True Predictions: {true_count}\n"
    overall_summary += f"Total False Predictions: {false_count}\n"
    overall_summary += f"Percentage of True Predictions: {true_count / 4 * 100}%\n"
    overall_summary += f"Percentage of False Predictions: {false_count / 4 * 100}%\n"
    return overall_summary

label_for_text_area = CTkLabel(app, text="Enter the news:")
label_for_text_area.place(x=100, y=10)

text_area = CTkTextbox(app, width=500, height=100)
text_area.pack(pady=40)
text_area.bind("<Key>", update_text_area_height)

button_classify = CTkButton(app, text="Classify", width=150, height=50, command=classify_news)
button_classify.pack()
button_classify.configure(font=("Helvetica", 14))  # Set font size to 14

slider = ttk.Progressbar(app, length=500, style='success.Striped.Horizontal.TProgressbar', mode='determinate')
slider.pack(pady=20)

overall_summary_label = CTkLabel(app, text="", wraplength=500, justify="left")
overall_summary_label.pack(pady=10)

text_area_result = CTkTextbox(app, width=500, height=100)
text_area_result.pack(pady=10)

app.mainloop()
