import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure UTF-8 output on Windows consoles for Devanagari/Indian scripts
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load MuRIL Tokenizer and Model
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))

model_candidates = [
    os.path.join(project_root, "models", "trained_muril_sarcasm"),
    os.path.join(project_root, "models", "500kdataTrained_muril_sarcasm"),
    os.path.join(base_dir, "trained_muril_sarcasm"),
    "trained_muril_sarcasm",
    "Mayurkhalse/muril-sarcasm-detector"
]
model_name = next((p for p in model_candidates if os.path.exists(p)), "Mayurkhalse/muril-sarcasm-detector")
id2label = {0: "Not Sarcastic", 1: "Sarcastic"}

print(f"Loading tokenizer and model from: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# 2. Input texts (supports English, Hindi, Hinglish, Bengali, Tamil, etc.)
texts = [
    # Sarcastic (English)
    "Oh brilliant! Another meeting that could have been an email.",
    "Wow, you did an amazing job breaking everything. Truly a genius!",
    "Great, my laptop crashed right before the deadline. Just perfect!",
    "I absolutely love waiting three hours for a five-minute appointment.",
    "Fantastic weather! Exactly what I wanted for my outdoor event.",
    "Oh sure, because ignoring the problem always makes it disappear.",
    "What a wonderful idea, let's make a simple task unnecessarily complicated.",
    "Congratulations on being late again. Your consistency is impressive.",
    "Amazing customer service! They hung up on me before I could speak.",
    "Oh, that's a brilliant plan. What could possibly go wrong?",
    "I just adore it when people interrupt me to explain what I already know.",
    "Wonderful, another software update that fixes nothing and breaks everything.",
    "Thanks for the helpful advice that absolutely nobody asked for.",
    "Oh fantastic, the printer stopped working exactly when I need it.",
    "You're so organized that you managed to lose the only important document.",
    "I love how you take credit for work you didn't even do.",
    "What a genius move, deleting the only backup file.",
    "Perfect timing! The power goes out just as I start my presentation.",
    "Oh yes, blaming everyone else is definitely a sign of great leadership.",
    "Brilliant strategy! Let's ignore all the evidence and hope for the best.",
    "I really appreciate being informed about the deadline after it has passed.",
    "Wow, another genius who thinks confidence is the same as competence.",
    "Lovely, my favorite thing is paying extra for hidden charges.",
    "Oh great, the traffic is moving slower than a sleeping turtle.",
    "Fantastic explanation! I somehow understand even less now.",
    "I enjoy group projects where I do all the work and everyone gets credit.",
    "What a refreshing surprise, the website is down again.",
    "Sure, because shouting louder automatically makes your argument correct.",
    "Amazing! You managed to make the easiest question the hardest one.",
    "Oh wonderful, another password reset for the password I just reset.",
    "Your ability to avoid responsibility deserves an award.",
    "Great job leaving the kitchen mess for someone else to clean.",
    "I love paying for a premium service that never works.",
    "Oh yes, let's schedule the meeting during everyone's lunch break. Genius!",
    "What a thoughtful gift, something I specifically said I didn't want.",
    "Brilliant work! You fixed the typo by deleting the entire paragraph.",
    "I was hoping for a quiet evening, so naturally the neighbors started drilling.",
    "Oh, I can see you've put absolutely no effort into this. Impressive.",
    "Wonderful decision! Let's change the plan five minutes before launch.",
    "Fantastic, my phone battery died while I was using the map.",
    "Clearly, the best way to solve a disagreement is to refuse to listen.",
    "What an incredible achievement, arriving at the wrong location confidently.",
    "Oh great, another app asking for permission to access everything.",
    "I love when people ask for honest feedback and then get offended.",
    "Perfect! The food is cold, the service is slow, and the bill is huge.",
    "Wow, you read one article and now you're an expert on everything.",
    "Excellent teamwork! Nobody communicated, and now everything is delayed.",
    "Oh lovely, the Wi-Fi stopped working during my online exam.",
    "That was such a helpful suggestion. It solved absolutely nothing.",
    "Amazing planning! You forgot the one thing the entire event depends on.",

    # Sarcastic (Hinglish)
    "Arey wah! 3 ghante late aakar bolte ho 'I am on time'!",
    "Wah bhai, kya dimaag hai! Password bhoolkar password reset karna bhi bhool gaye.",
    "Kya baat hai! Assignment deadline ke baad submit karke marks chahiye full.",
    "Wah! Puri raat reels dekhkar bolte ho ki neend nahi aayi.",
    "Bahut badhiya! Internet band hote hi tumhari duniya khatam ho gayi.",
    "Arey genius, code delete karke bug fix kar diya!",
    "Kya planning hai! Exam kal hai aur syllabus aaj shuru kiya.",
    "Wah yaar, bina padhe exam dene ka confidence toh dekho!",
    "Bahut khoob! Galti khud ki aur blame poori team ko.",
    "Kya service hai! Order kiya pizza aur mila sirf disappointment.",
    "Arey wah, meeting mein late aakar sabko time management sikha rahe ho.",
    "Kya talent hai! Ek simple kaam ko bhi complicated bana diya.",
    "Wah bhai! Phone charge karna bhool gaye aur battery ko blame kar rahe ho.",
    "Amazing yaar, bina backup ke saari files delete kar di!",
    "Kya baat hai! Khud kuch kaam nahi kiya aur credit poora le liya.",
    "Wah! Alarm band karke phir bolte ho ki alarm baja hi nahi.",
    "Bahut sahi! Traffic mein late hue aur ghar se nikle hi late the.",
    "Arey genius, galat answer confidently dekar sabko confuse kar diya.",
    "Wah bhai, diet kal se shuru hogi aur aaj teesra burger kha rahe ho.",
    "Kya kamaal hai! Wi-Fi ka password galat daalke router ko gaali de rahe ho.",
    "Bahut badhiya! Project last minute par start karke laptop ko blame kar diya.",
    "Wah yaar, bina instructions padhe setup kharab kar diya!",
    "Kya dedication hai! Class mein so kar bolte ho lecture boring tha.",
    "Arey wah! Khud message seen karke reply nahi kiya aur ab attitude bol rahe ho.",
    "Kya smartness hai! Garam chai peeke bolte ho chai bahut garam hai.",
    "Wah bhai, paise kharch karke broke hone par surprise ho rahe ho.",
    "Bahut khoob! Ek hi galti baar-baar karke experience gain kar rahe ho.",
    "Kya idea hai! Baarish mein chhatri ghar par chhodkar weather ko blame karo.",
    "Wah! Phone haath mein lekar poore ghar mein phone dhoondh rahe ho.",
    "Arey genius, file save kiye bina application band kar di!",
    "Kya baat hai! Khud late reply karke doosron se instant reply expect karte ho.",
    "Wah yaar, bina test kiye code production mein daal diya!",
    "Bahut sahi! Notes nahi banaye aur ab topper se notes maang rahe ho.",
    "Kya planning hai! Trip book kar li aur leave lena bhool gaye.",
    "Wah bhai, bina context samjhe advice dene aa gaye!",
    "Arey wah! Khud loud music bajakar neighbours ko noise ke liye complain karte ho.",
    "Kya efficiency hai! Do minute ka kaam do ghante mein complete kiya.",
    "Bahut badhiya! Galat train mein baithkar railway ko blame kar rahe ho.",
    "Wah! Charger ghar par bhoolkar battery ko responsible bol rahe ho.",
    "Kya confidence hai! Google par ek cheez padhkar doctor ban gaye.",
    "Arey genius, sabko reminder bhejkar khud event bhool gaye!",
    "Wah bhai, khud instructions nahi padhe aur system ko useless bol rahe ho.",
    "Kya kamaal hai! Online class attend karke bhi attendance miss kar di.",
    "Bahut khoob! Ek din workout karke fitness transformation expect kar rahe ho.",
    "Wah yaar, deadline extend karwane ke baad bhi kaam complete nahi hua!",
    "Kya talent hai! Remote haath mein hai aur TV ka remote dhoondh rahe ho.",
    "Arey wah! Bina data check kiye conclusion nikaal diya.",
    "Bahut badhiya! Khud galat direction dekar bolte ho map hi galat hai.",
    "Kya genius move hai! Important email draft ki aur send karna bhool gaye.",
    "Wah bhai, har baar late aakar traffic ko hi responsible batate ho!",

    # Not Sarcastic (English)
    "The package arrived safely today and was packed well.",
    "I enjoyed the movie because the story was interesting.",
    "The meeting starts at ten o'clock tomorrow morning.",
    "She completed the assignment before the deadline.",
    "My laptop is working properly after the repair.",
    "The weather is pleasant and suitable for a walk.",
    "I appreciate your help with the project.",
    "The restaurant served fresh food and friendly service.",
    "He arrived early to prepare for the presentation.",
    "Please send me the document when you have time.",
    "The team worked together and finished the task successfully.",
    "I need to charge my phone before leaving home.",
    "The train arrived at the station on schedule.",
    "She explained the concept clearly during the lecture.",
    "I have planned my study schedule for the upcoming exam.",
    "The software update improved the application's performance.",
    "We enjoyed spending time with our friends yesterday.",
    "The customer support team resolved my issue.",
    "I will attend the workshop on Saturday.",
    "The book contains useful information about machine learning.",
    "He apologized for arriving late to the meeting.",
    "I finished cleaning my room this afternoon.",
    "The internet connection is stable now.",
    "Please let me know if you need any assistance.",
    "The presentation includes the latest project results.",

    # Not Sarcastic (Hinglish)
    "Aaj maine apna assignment deadline se pehle complete kar liya.",
    "Kal hum sab friends ke saath movie dekhne gaye the.",
    "Mera laptop repair hone ke baad achhe se kaam kar raha hai.",
    "Meeting subah das baje start hogi.",
    "Maine exam ke liye apna study schedule bana liya hai.",
    "Aaj weather walk karne ke liye kaafi achha hai.",
    "Tumhari help se mera project complete ho gaya, thank you.",
    "Maine phone charge par laga diya hai.",
    "Train time par station par pahunch gayi thi.",
    "Teacher ne topic ko bahut clearly explain kiya.",
    "Mujhe kal office ke liye jaldi nikalna hai.",
    "Restaurant ka khana fresh aur tasty tha.",
    "Maine saare important documents ek folder mein save kar diye.",
    "Team ne milkar task time par complete kiya.",
    "Internet connection ab stable ho gaya hai.",
    "Main Saturday ko workshop attend karne wala hoon.",
    "Usne meeting se pehle presentation prepare kar li thi.",
    "Aaj maine apne room ki cleaning complete kar li.",
    "Customer support ne meri problem solve kar di.",
    "Maine kal ke liye train ticket book kar li hai.",
    "Mera dost mujhe exam ki preparation mein help kar raha hai.",
    "Naya software update install ho gaya hai.",
    "Mujhe is topic ke baare mein aur information chahiye.",
    "Hum Sunday ko family ke saath lunch karenge.",
    "Maine apne laptop ki files ka backup le liya hai."
]
# 3. Tokenize inputs
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# 4. Perform Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

# 5. Display predictions
for i, text in enumerate(texts):
    prob_sarcastic = probabilities[i, 1].item()
    prob_not_sarcastic = probabilities[i, 0].item()
    pred_label = "Sarcastic" if prob_sarcastic > 0.5 else "Not Sarcastic"
    confidence = max(prob_sarcastic, prob_not_sarcastic)

    print(f"\nText: {text}")
    print(f"Prediction: {pred_label} (Confidence: {confidence:.2%})")
    print(f"Probabilities -> Not Sarcastic: {prob_not_sarcastic:.4f}, Sarcastic: {prob_sarcastic:.4f}")
