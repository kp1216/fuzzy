import os
import re
import gradio as gr
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()  # expects GOOGLE_API_KEY in .env
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ─── Core function: single structured prompt ────────────────────────────────────
def match_invoice(image_text, sms_text):
    # Since now we are only receiving text, no PIL image opening
    prompt = f"""
You are a financial verification assistant.
Input: 1) An invoice text, 2) A transaction SMS text.

Task:

From the invoice, extract:

Total Amount

Due Date (Due Date only, not the invoice date)

Customer Name ("Bill To" or the customer the bill is made for)

Invoice Number (if any)

From the SMS, extract:

Credited amount

Transaction date

Payer name

Any invoice number, transaction ID, or UPI ID

Compare the invoice vs. SMS values and award points based on EXACT criteria:
• 0 points: Nothing matches
• 1 point: ONLY the total amount on invoice matches credited amount in SMS
• 2 points: Amount matches AND (due date matches transaction date OR customer name matches payer name)
• 3 points: Amount matches AND due date matches AND customer name matches
• 4 points: Amount matches AND due date matches AND customer name matches AND any additional identifier matches (UPI ID, invoice number, etc.)

IMPORTANT: Be strict with matching criteria. Only award the exact points based on what actually matches.

Output format:
EXTRACTED FROM INVOICE:

Amount: [amount]

Due Date: [date]

Customer Name: [name]

Invoice Number: [number if any]

EXTRACTED FROM SMS:

Credited Amount: [amount]

Transaction Date: [date]

Payer Name: [name]

Additional IDs: [any IDs]

MATCHING ANALYSIS:

Amount Match: [Yes/No - explain]

Date Match: [Yes/No - explain]

Name Match: [Yes/No - explain]

Additional Match: [Yes/No - explain if applicable]

FINAL SCORE: [0/1/2/3/4]

EXPLANATION: [Brief explanation of why this score was awarded]
"""

    # Send invoice text and sms text to the model
    resp = model.generate_content([prompt, image_text, sms_text])
    text = resp.text.strip()

    # More robust score extraction - look for "FINAL SCORE:" pattern
    score_match = re.search(r"FINAL SCORE:\s*([0-4])", text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
    else:
        # Fallback: look for score at the end of response
        lines = text.split('\n')
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in ['score', 'final', 'result']):
                digit_match = re.search(r"[0-4]", line)
                if digit_match:
                    score = int(digit_match.group())
                    break
        else:
            score = 0  # Default if no score found

    return score, text

def process_and_display(invoice_text, sms_text):
    """Wrapper function to return both score and explanation"""
    if not invoice_text.strip() or not sms_text.strip():
        return 0, "Please provide both an invoice text and SMS text."

    try:
        score, explanation = match_invoice(invoice_text, sms_text)
        return score, explanation
    except Exception as e:
        return 0, f"Error processing: {str(e)}"

# ─── Gradio Interface ───────────────────────────────────────────────────────────
iface = gr.Interface(
    fn=process_and_display,
    inputs=[
        gr.Textbox(lines=8, label="Invoice Text", placeholder="Paste the invoice text here..."),
        gr.Textbox(lines=4, label="Transaction SMS", placeholder="Paste your transaction SMS here...")
    ],
    outputs=[
        gr.Number(label="Match Score (0–4)"),
        gr.Textbox(lines=15, label="Detailed Analysis", show_copy_button=True)
    ],
    title="Invoice–SMS Matcher (Enhanced)",
    description=(
        "Paste the invoice text and transaction SMS to get a match score (0–4):\n"
        "• 0: No matches\n"
        "• 1: Amount only\n"
        "• 2: Amount + (Date OR Name)\n"
        "• 3: Amount + Date + Name\n"
        "• 4: Amount + Date + Name + Additional ID"
    ),
    examples=[
        # Add example pairs here if you have sample data
    ]
)

if __name__ == "__main__":
    iface.launch()
