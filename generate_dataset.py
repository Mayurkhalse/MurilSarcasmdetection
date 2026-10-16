"""
Dataset Generator: Generates a high-quality 1000-line balanced multilingual
sarcasm dataset across English, Hinglish, and Hindi (Devanagari).
500 Sarcastic (label=1) and 500 Non-Sarcastic (label=0).
"""

import csv
import random

# ==========================================
# 1. ENGLISH SARCASTIC (label = 1)
# ==========================================
english_sarcastic = [
    # Work & Office
    "Oh fantastic! Another 2-hour meeting that could easily have been a 2-line email.",
    "I absolutely love it when deadlines are moved up by three days without notice.",
    "Thanks for assigning me three new urgent projects right at 5 PM on a Friday.",
    "What a brilliant strategy, let's cut the QA budget and wonder why software has bugs.",
    "Great teamwork everyone, you all took the credit while I pulled an all-nighter.",
    "I love when leadership asks for feedback and then ignores every single suggestion.",
    "Congratulations on scheduling a mandatory training session during lunch break.",
    "Nothing says appreciation like an automated thank-you email after 60 hours of overtime.",
    "Oh wonderful, another spreadsheet to track the spreadsheets that track our tasks.",
    "You did an amazing job delegating all your responsibilities to interns.",
    "Sure, let's approve the client's impossible feature request without asking developers.",
    "I really enjoy receiving urgent calls on Sunday morning about non-urgent issues.",
    "Brilliant work! You fixed the minor styling glitch by breaking the payment gateway.",
    "Our open-office floor plan is so peaceful; I can hear six phone calls simultaneously.",
    "Thanks for explaining my own presentation slide back to me, truly invaluable.",
    "Oh yes, shifting the blame to the junior dev is the hallmark of great management.",
    "What a thoughtful promotion, 50% more responsibilities for 0% salary increase.",
    "I love how our company embraces agile by having daily 90-minute standup meetings.",
    "Perfect timing, server maintenance right during our biggest product launch demo.",
    "Nothing boosts productivity like five different managers asking for status updates.",

    # Technology & Software
    "Wonderful, another operating system update that breaks all my audio drivers.",
    "I love when apps ask for location, contacts, and microphone access just to be a flashlight.",
    "Nothing says premium software like crashing right when I forget to hit save.",
    "Oh great, high-speed fiber broadband running at blazing fast 12 kbps today.",
    "Superb! The printer queued 40 copies of a blank page and jammed on the first one.",
    "I just adore modern smart TVs that take three minutes to boot up and display an ad.",
    "What a genius feature, auto-correcting correct words into embarrassing typos.",
    "Thanks for the helpful error message: 'An unknown error occurred with code None'.",
    "Nothing beats modern Bluetooth earphones disconnecting right during an important call.",
    "I love downloading a 50 GB patch on day one just to get past the loading screen.",
    "Oh fantastic, my cloud backup successfully overwrote my latest files with old versions.",
    "Another password reset requirement asking for symbols, Greek letters, and ancient runes.",
    "Brilliant! The navigation app recommended an alternate route that saved 0 minutes and added 3 tolls.",
    "My laptop battery lasts a whopping 18 minutes after this new optimization patch.",
    "What an achievement, an AI chatbot that answers every customer query with a generic FAQ link.",

    # Daily Life & Customer Service
    "Fabulous weather! 45-degree heat with 90% humidity is exactly what I wanted for an outdoor wedding.",
    "I truly enjoy waiting in an airport security line that moves slower than continental drift.",
    "Five stars for the airline that lost my luggage and then offered me a $5 snack voucher.",
    "Nothing says high-class dining like paying $30 for a microscopic portion of lukewarm soup.",
    "Oh lovely, the courier marked my package as 'delivered' while standing in a different city.",
    "I love paying convenience fees on tickets that I booked, downloaded, and printed myself.",
    "Wonderful! The doctor arrived 2 hours late and spent exactly 45 seconds diagnosing me.",
    "What a relief! The neighbor decided 6:30 AM on a Sunday is prime lawn mowing time.",
    "Great job parallel parking; you managed to occupy three entire parking slots at once.",
    "I just adore when telemarketers call three times during dinner to sell insurance.",
    "Thanks for the warm coffee; drinking iced coffee that sat on the counter for an hour is delightful.",
    "Oh brilliant, another online shopping experience where the received clothes look like cleaning rags.",
    "Clearly, the best way to handle customer complaints is to put them on hold for 45 minutes.",
    "I love paying toll tax on a highway that has more craters than the surface of the moon.",
    "Nothing says fresh produce like supermarket vegetables that rot on the car ride home.",
]

# Variations generator for English Sarcastic to expand to 500
templates_en_sarc = [
    ("Oh brilliant, {subj} decided to {act} right when {cond}.",
     [("my laptop", "reboot for updates", "I was submitting an online exam"),
      ("the internet", "disconnect completely", "I had 2 minutes left on my bid"),
      ("our department head", "call a sudden meeting", "everyone was packing up to leave"),
      ("the public transport driver", "take an unscheduled tea break", "we were already 30 minutes late"),
      ("the coffee machine", "dispense hot water instead of coffee", "I needed caffeine the most"),
      ("the elevator", "break down on the ground floor", "I had to carry luggage to the 10th floor"),
      ("the delivery app", "cancel my dinner order after an hour", "guests arrived at home"),
      ("the AC unit", "blow warm air", "the outside temperature hit 42 degrees"),
      ("the air conditioning", "turn off automatically", "it was blazing hot outside"),
      ("the banking app", "go under scheduled maintenance", "I was paying a restaurant bill"),
      ("the GPS system", "reroute me into a narrow dead-end lane", "I was rushing to the station"),
      ("the streaming platform", "buffer at 99%", "the climax of the movie started"),
      ("the car engine", "make strange knocking sounds", "I was on a remote highway"),
      ("my smartphone", "freeze on a black screen", "I was showing boarding pass to security")]),

    ("I absolutely love {noun}, it makes {benefit} so much more {adj}.",
     [("waiting 45 minutes on customer support hold", "resolving a 2-minute issue", "delightful"),
      ("paying hidden platform convenience charges", "buying movie tickets online", "affordable"),
      ("getting unhelpful automated email replies", "resolving critical bugs", "productive"),
      ("dealing with unnecessary corporate red tape", "getting a simple mouse replacement", "seamless"),
      ("being added to unsolicited WhatsApp groups", "my morning notification feed", "peaceful"),
      ("standing in a queue to collect a token to stand in another queue", "government paperwork", "efficient"),
      ("filling out three physical forms with the exact same information", "bank account opening", "streamlined"),
      ("listening to loud speakerphone conversations on public transit", "the daily commute", "enjoyable"),
      ("getting bombarded with loan offers every morning", "waking up to notifications", "refreshing"),
      ("having five different passwords expire simultaneously", "logging into workstation", "convenient")]),

    ("What a genius idea: {bad_action}, because {excuse} definitely {outcome}.",
     [("deploying directly to production on Friday evening", "skipping staging tests", "guarantees a quiet weekend"),
      ("ignoring the warning engine light for three months", "pretending it doesn't exist", "fixes mechanical failures"),
      ("scheduling a kickoff meeting at 8 AM on a Monday", "sacrificing sleep", "boosts team morale"),
      ("deleting documentation to save disk space", "cleanliness", "improves codebase clarity"),
      ("leaving ice cream inside the car trunk in summer", "saving freezer room", "keeps it nice and solid"),
      ("answering customer queries with copied generic FAQs", "saving support time", "builds lasting customer trust"),
      ("skipping breakfast and lunch before an intense workout", "saving preparation time", "enhances athletic endurance"),
      ("using the word 'urgent' on every single email subject line", "false alarm tactics", "prioritizes actual emergencies")]),

    ("Nothing says '{virtue}' quite like {hypocrisy}.",
     [("workplace transparency", "holding secret re-structuring meetings behind closed doors"),
      ("high reliability", "a database that crashes under a load of ten users"),
      ("deep customer care", "disconnecting the call when a refund is requested"),
      ("punctuality", "showing up 40 minutes late with a fresh latte in hand"),
      ("eco-friendly initiative", "packaging a tiny thumb drive inside five layers of bubble plastic"),
      ("streamlined process", "requiring six signatures to approve a $10 notebook purchase"),
      ("world-class security", "storing production user passwords in plaintext Excel sheets"),
      ("respect for work-life balance", "sending messages at midnight expecting instant replies"),
      ("healthy team culture", "blaming the intern whenever an executive makes a mistake")]),

    ("Congratulations on {feat}! Your ability to {detail} is truly {adj}.",
     [("taking three weeks to approve a single leave request", "delay simple decisions", "extraordinary"),
      ("turning a 10-minute task into a week-long committee debate", "overcomplicate simplicity", "breathtaking"),
      ("breaking a working feature while attempting to fix a typo", "introduce unintended bugs", "unmatched"),
      ("losing the only signed original physical agreement", "maintain documentation", "inspiring"),
      ("forgetting the client's name right in the opening sentence of the pitch", "make a lasting impression", "unforgettable"),
      ("arriving late to your own farewell party", "value punctuality", "poetic"),
      ("ordering lunch for the team and forgetting to add food items", "manage operations", "spectacular"),
      ("spilling tea all over the newly printed financial statements", "handle confidential assets", "admirable")])
]

# ==========================================
# 2. ENGLISH NON-SARCASTIC (label = 0)
# ==========================================
english_non_sarcastic = [
    # Work & Professional
    "The quarterly financial audit was completed yesterday with no discrepancies found.",
    "Please submit your travel reimbursement receipts before the end of the month.",
    "The engineering team resolved the database latency issue during scheduled maintenance.",
    "We have scheduled a training workshop on cloud infrastructure for next Tuesday.",
    "Thank you for sharing the meeting minutes and action items promptly.",
    "The new office ergonomics policy allows employees to request adjustable standing desks.",
    "Our team successfully delivered the client requirements on time and within budget.",
    "The human resources department announced the holiday schedule for the upcoming quarter.",
    "Please review the attached pull request when you have an available moment today.",
    "The annual performance reviews will commence starting from the second week of next month.",
    "We conducted a comprehensive safety drill in the main building this morning.",
    "The marketing campaign generated positive feedback across all social media channels.",
    "Our network administrators updated the firewall certificates over the weekend.",
    "The project manager distributed the updated Gantt chart to all department leads.",
    "We appreciate your constructive input during the cross-functional retrospective.",

    # Technology & General
    "The new software release introduces improved battery optimization and dark mode.",
    "Please make sure to back up your personal files to external storage regularly.",
    "The fiber optic internet connection provides steady upload and download speeds.",
    "A clean installation of the operating system resolved the slow booting problem.",
    "The smartphone camera performs well in low-light environments with night mode.",
    "The laser printer completed all fifty pages of the user manual without interruption.",
    "The weather forecast indicates pleasant temperature and mild sunshine for tomorrow.",
    "The public library has extended its operating hours during the examination period.",
    "The doctor examined the patient carefully and prescribed appropriate medication.",
    "The train departed from platform four exactly on time according to the schedule.",
    "The customer support representative assisted me politely and resolved the account lock.",
    "Regular exercise and balanced meals contribute positively to physical well-being.",
    "The flight arrived fifteen minutes ahead of schedule and baggage delivery was prompt.",
    "Students are requested to bring their identification cards for verification at the counter.",
    "The supermarket offers fresh vegetables and dairy products sourced from local farms."
]

templates_en_non_sarc = [
    ("The {entity} has successfully {action} for the {target}.",
     [("software development team", "deployed the version 2.4 patch", "user authentication module"),
      ("operations department", "completed the annual inventory verification", "central regional warehouse"),
      ("customer support desk", "addressed all pending inquiries", "first quarter billing cycle"),
      ("logistics partner", "delivered all priority consignments", "metropolitan supply chain"),
      ("finance committee", "approved the research and development budget", "upcoming fiscal year"),
      ("maintenance crew", "serviced all heating and ventilation systems", "headquarters facility"),
      ("quality assurance team", "executed comprehensive regression testing", "mobile application update"),
      ("editorial board", "reviewed and approved the academic manuscript", "scientific journal publication"),
      ("security division", "installed updated surveillance protocols", "data center perimeter"),
      ("training coordinator", "conducted the orientation program", "incoming batch of engineering recruits")]),

    ("Please {action} by {deadline} to ensure {benefit}.",
     [("submit your timesheet entries", "Friday evening", "accurate payroll processing"),
      ("verify your registered contact address", "the end of this week", "timely document dispatch"),
      ("upload the required certification proofs", "the portal cut-off date", "seamless compliance verification"),
      ("confirm your attendance for the seminar", "tomorrow noon", "proper seating arrangement"),
      ("review the drafted service agreement", "Wednesday morning", "final sign-off with external vendors"),
      ("update your emergency contact numbers", "the end of the month", "accurate employee safety records"),
      ("return all borrowed reference library books", "Friday afternoon", "equitable access for all students"),
      ("complete the mandatory cybersecurity refresher module", "next Monday", "network access maintenance")]),

    ("We observed {positive_trend} in {domain} following {initiative}.",
     [("a noticeable improvement in system response times", "database performance", "index optimization"),
      ("a 15% reduction in customer resolution durations", "support operations", "the new ticketing workflow"),
      ("consistent growth in user satisfaction scores", "product feedback", "the recent interface redesign"),
      ("higher employee engagement levels", "internal collaboration", "the introduction of flexible working hours"),
      ("significant decrease in network downtime", "server reliability", "hardware upgrade completion"),
      ("substantial increase in on-time flight departures", "regional airline operations", "revised gate management"),
      ("better comprehension among participating students", "classroom learning", "the interactive lab exercises"),
      ("steady energy efficiency enhancements", "facility operations", "smart thermostat installations")]),

    ("The {service} was {quality} and {outcome}.",
     [("hotel accommodation", "clean and spacious", "the front desk staff was very helpful"),
      ("restaurant meal", "prepared fresh with quality ingredients", "the pricing was reasonable"),
      ("medical consultation", "thorough and informative", "the doctor answered all questions clearly"),
      ("vehicle inspection", "conducted professionally", "detailed maintenance report was provided"),
      ("public transit ride", "smooth and punctual", "the coaches were air-conditioned and comfortable"),
      ("e-commerce delivery", "packed securely in bubble wrap", "the parcel arrived two days early"),
      ("technical tutorial", "well-structured and easy to follow", "helped me understand the core concepts"),
      ("guided museum tour", "very educational and engaging", "covered all historic artifacts in detail")])
]


# ==========================================
# 3. HINGLISH SARCASTIC (label = 1)
# ==========================================
hinglish_sarcastic = [
    "Arey wah! 3 ghante late aakar bolte ho 'Main toh time ka bohot pakka hoon'.",
    "Wah bhai, kya dimaag lagaya hai! Password bhoolkar password reset link bhi delete kar diya.",
    "Kya baat hai! Exam kal subah hai aur syllabus ka pehla page aaj raat ko khol rahe ho.",
    "Bahut badhiya! Puri raat Instagram reels scroll karke bolte ho ki 'Aankhon me neend hi nahi aayi'.",
    "Wah re genius! Code me se saari test cases delete karke bolte ho 'Zero bugs found'.",
    "Kamaal ka talent hai! Ek 2-line ki email likhne me poora din nikaal diya.",
    "Bahut khoob! Galti khud ki thi aur blame poori team aur management par daal diya.",
    "Kya planning hai bhai! Trip par nikal gaye aur ghar ka main door lock karna hi bhool gaye.",
    "Arey wah! Meeting me 40 minute late aakar baakiyon ko time management ka gyaan baant rahe ho.",
    "Superb yaar! Phone pani me gira diya aur ab poochte ho ki 'Kya ye waterproof mode me chalega?'.",
    "Wah! Ek din gym me selfie le li aur ab lagta hai jaise Arnold Schwarzenegger ban gaye.",
    "Bahut sahi! 100 Mbps ka fiber plan le rakha hai aur speed 2G jaisi aa rahi hai.",
    "Kitna humble banda hai! Din bhar apni hi tareef ke kashide padhta rehta hai.",
    "Kya timing hai! Jaise hi mera presentation shuru hua, waise hi Wi-Fi router ne retirement le li.",
    "Arey wah, diet shuru karni thi kal se aur aaj raat ko akele hi poora pizza khatam kar diya.",
    "Shabaash! Charger ghar par chhodkar phone ki battery ko gaali de rahe ho.",
    "Wah bhai, bina instructions padhe naya furniture assemble kiya aur ab 4 screw bache hue hain.",
    "Kya dedecation hai! Class me aakhiri bench par so kar kehte ho ki teacher achha nahi padhata.",
    "Bahut badhiya! Traffic me late hone ka bahana banaya jabki ghar se nikle hi meeting ke time par the.",
    "Wah yaar! Do minute ka kaam do hafte se 'In Progress' dikha raha hai jira board par."
]

templates_hi_lat_sarc = [
    ("Wah {who}! {action1} karke ab {action2}, kya {compliment} hai!",
     [("bhai", "traffic signal tod", "traffic police ko niyam sikha rahe ho", "dimaag"),
      ("genius", "production database drop", "bolte ho backup test kar raha tha", "logic"),
      ("yaar", "exam me kuch nahi padh kar", "95% aane ki umeed laga rakhi hai", "confidence"),
      ("sahab", "saara kaam aakhiri ghante me chhod", "stress hone par chilla rahe ho", "planning"),
      ("dost", "mera phone mang kar", "mera hi mobile data hotspot chala rahe ho", "dosti"),
      ("sir", "khud late reply dekar", "doosron se 2-minute me jawaab expect karte ho", "attitude"),
      ("babu", "office me 2 ghante lunch break leke", "kehte ho bohot workload hai", "efficiency"),
      ("maharaj", "calculator use karke bhi galat total nikaal", "bolte ho math tricky hai", "hisaab")]),

    ("Arey wah! {scenario}, ab isse behtar {adj} kya hi ho sakta hai!",
     [("45 degree dhoop me AC chalne ki jagah garam hawa phek raha hai", "mausam ka anand"),
      ("Sunday subah 6 baje society me drill machine chalna shuru ho gayi", "weekend relaxing vibe"),
      ("Online order me brand new shoes mangwaye the aur purana dabba mila", "customer experience"),
      ("Restaurant me 500 rupaye ki dal makhani me namak hi gayab hai", "fine dining"),
      ("Metro station par card scan karte hi balance minus me chala gaya", "morning motivation"),
      ("Important interview ke theek 5 minute pehle microphone ne kaam karna band kar diya", "technical setup"),
      ("Shaadi ke function me pahunche aur pata chala ki khana khatam ho gaya", "hospitality"),
      ("Movie ka ticket khareedne ke baad theater ka projector hi kharab ho gaya", "entertainment")]),

    ("Kya baat hai! {reason} ke baad ab {excuse}, kamaal ke log hain!",
     [("poori raat video games khelne", "bol rahe hain ki padhai ka bilkul time nahi mila"),
      ("ek hi galti ko paanchvi baar dohrane", "bolte hain ki failure is the stepping stone to success"),
      ("salary aane ke do din me saare paise udaane", "bolte hain ki market me bohot inflation hai"),
      ("khud bina helmet ke bike chalane", "police ko corrupt bol kar gussa dikha rahe hain"),
      ("flight 4 ghante delay hone", "airline wale bolte hain inconvenience is deeply regretted"),
      ("laptop par chai giraane", "service center walo ko bolte hain automatic motherboard issue hai"),
      ("alarm band karke 2 ghante extra sone", "keh rahe hain ki alarm baja hi nahi tha"),
      ("client ko galat files bhejne", "bolte hain ki email server me glitch tha")])
]

# ==========================================
# 4. HINGLISH NON-SARCASTIC (label = 0)
# ==========================================
hinglish_non_sarcastic = [
    "Kal subah 10 baje team meeting schedule ki gayi hai, sabhi log time par join karein.",
    "Maine kal shaam ko market jaakar taazi sabziyan aur phal khareede the.",
    "Hamara naya project successfully complete ho gaya hai aur client ne approve kar diya.",
    "Mera laptop repair hone ke baad ab kaafi smooth aur fast chal raha hai.",
    "Maine aane wale semester exams ke liye apna study plan bana liya hai.",
    "Aaj subah park me walk karne ka mausam kaafi achha aur thanda tha.",
    "Aapki help se hamara code deployment time par ho gaya, bohot bohot shukriya.",
    "Maine apna mobile phone charge par laga diya hai taaki battery full rahe.",
    "Train samay par railway station par pahunchi aur safar kaafi aaramdayak raha.",
    "Teacher ne complex physics concept ko bohot aasan shabdon me samjhaya.",
    "Mujhe kal subah jaldi nikalna hai kyunki traffic hone ki sambhavna hai.",
    "Restaurant me khana bohot swadisht tha aur service bhi kaafi fast thi.",
    "Maine saare zaroori documents Google Drive par backup folder me save kar diye hain.",
    "Humari team ne aapas me coordinate karke project deadline se pehle deliver kiya.",
    "Internet provider ne maintenance ke baad connection stable kar diya hai.",
    "Main aane wale weekend par ek technical web development workshop attend karne wala hoon.",
    "Usne presentation start hone se pehle saare demo slides theek se check kar liye the.",
    "Ghar ki safai poori ho chuki hai aur sabhi cheezein apni jagah par rakh di gayi hain.",
    "Customer care executive ne meri account issue ko 10 minute me solve kar diya.",
    "Maine kal shaam ki movie ke tickets advance me book kar liye hain."
]

templates_hi_lat_non_sarc = [
    ("Aaj {event} {status} aur sabhi {detail}.",
     [("office me annual team review", "safaltapoorvak poora hua", "members ne achha feedback share kiya"),
      ("college me guest lecture", "kaafi informative raha", "students ne dhyan se suna aur notes banaye"),
      ("society me blood donation camp", "aayojit kiya gaya", "logon ne badh-chadh kar hissa liya"),
      ("city library me book exhibition", "shuru hui", "kitabon ke shaukeen wahan pahunche"),
      ("hospital me free eye checkup", "conduct kiya gaya", "marizon ko zaroori salah di gayi"),
      ("sports complex me badminton tournament", "shuru hua", "khiladiyon ne behtareen khel dikhaya"),
      ("coding competition ka final round", "conclude hua", "winners ko awards diye gaye"),
      ("university me convocation program", "aayojit hua", "graduates ko degrees pradan ki gayi")]),

    ("Maine {task} {time} me complete kar liya taaki {benefit}.",
     [("apna monthly budget planning", "weekend", "kharchon par control bana rahe"),
      ("car ki regular servicing", "is mahine", "lambe safar me koi pareshani na ho"),
      ("project ka documentation", "kal shaam", "naya developer aasaani se code samajh sake"),
      ("health insurance premium pay", "samay par", "policy uninterrupted chalti rahe"),
      ("laptop ka data backup", "aaj subah", "kisi bhi technical issue se files surakshit rahein"),
      ("exam revision ka pehla phase", "samay rehte", "aakhiri dino me tension na ho"),
      ("train tickets ki online booking", "tatkal time me", "safal yatra nishchit ho sake"),
      ("presentation slides ka final review", "meeting se pehle", "client ke saamne sabhi data accurate ho")])
]


# ==========================================
# 5. HINDI DEVANAGARI SARCASTIC (label = 1)
# ==========================================
hindi_sarcastic = [
    "वाह भाई! पूरी रात फोन पर रील्स देखकर अब कह रहे हैं कि परीक्षा की तैयारी का समय ही नहीं मिला।",
    "अरे वाह! 4 घंटे लाइन में खड़े रहने के बाद पता चला कि काउंटर बंद है, क्या शानदार व्यवस्था है!",
    "कमाल का लॉजिक है आपका! गलती खुद करो और फिर दूसरों पर चिल्लाकर खुद को सही साबित करो।",
    "बहुत खूब! परीक्षा से ठीक एक दिन पहले किताब खोलकर कहते हैं कि पूरा सिलेबस एक घंटे का है।",
    "वाह! 10 किलोमीटर के सफर में 3 घंटे जाम में बिताने का अलग ही सुखद आनंद है।",
    "गजब के डॉक्टर हैं! मरीज की बात सुने बिना ही 10 महंगी जांचें और दवाइयां लिख दीं।",
    "क्या बात है! बिजली का बिल दोगुना आ गया जबकि पिछले 15 दिन से घर में लाइट ही नहीं थी।",
    "वाह रे दोस्त! मुसीबत के समय फोन किया तो नंबर ही ब्लॉक लिस्ट में डाल दिया, खूब निभाई दोस्ती!",
    "अरे वाह! 500 रुपये की घटिया चीज को 2500 में बेचकर कहते हैं 'यह आपके लिए स्पेशल डिस्काउंट है'।",
    "शानदार! महत्वपूर्ण ऑनलाइन इंटरव्यू के ठीक दो मिनट पहले वाई-फाई ने हमेशा के लिए दम तोड़ दिया।",
    "वाह भाई वाह! बिना हेलमेट और बिना लाइसेंस के बाइक चलाकर पुलिस को ईमानदारी का पाठ पढ़ा रहे हैं।",
    "कमाल की सफाई व्यवस्था है, सड़क पर कचरे का पहाड़ देखकर लगता है जैसे पर्यटन स्थल पर आ गए हों।",
    "बहुत बढ़िया! टीम का सारा काम दूसरों के सिर मढ़कर खुद प्रमोशन की लिस्ट में सबसे आगे खड़े हैं।",
    "वाह! धूप में 45 डिग्री तापमान में बस स्टैंड पर बिना पंखे के 2 घंटे बस का इंतजार करना कितना आरामदायक है।",
    "अरे वाह! रेस्टोरेंट में आधा घंटे तक कोई पानी पूछने नहीं आया और बिल में सर्विस चार्ज 20% लगा दिया।"
]

templates_hi_dev_sarc = [
    ("वाह {sambodhan}! {kaam1} करके अब {kaam2}, क्या {taarif} है!",
     [("साहब", "दफ्तर में 2 घंटे देर से आकर", "बाकी कर्मचारियों को अनुशासन सिखा रहे हैं", "लीडरशिप"),
      ("नेता जी", "चुनाव जीतने के 5 साल बाद पहली बार गांव आकर", "विकास की बातें कर रहे हैं", "जनसेवा"),
      ("भाई", "गाड़ी का ब्रेक फेल होने पर भी 100 की स्पीड में चलाकर", "कह रहे हैं 'मुझे पूरा कंट्रोल है'", "साहस"),
      ("महोदय", "पूरा महीना छुट्टी बिताकर", "अब एक दिन में 6 महीने का काम मांग रहे हैं", "प्रबंधन"),
      ("मित्र", "होटल का बिल आते ही फोन पर जरूरी कॉल का बहाना बनाकर बाहर निकल गए", "दोस्ती की मिसाल", "बुद्धिमानी"),
      ("पंडित जी", "बिना खुद पढ़े दूसरों को उपदेश देकर", "कहते हैं कि दुनिया बहुत भ्रष्ट है", "संस्कार")]),

    ("कमाल की व्यवस्था है! {samasya}, अब इसे {vyangya} नहीं कहेंगे तो क्या कहेंगे!",
     [("अस्पताल में आपातकालीन वार्ड का ताला बंद है और गार्ड चाय पीने गया है", "सर्वश्रेष्ठ स्वास्थ्य सेवा"),
      ("ऑनलाइन आवेदन पत्र भरने के बाद सर्वर '404 नॉट फाउंड' दिखा रहा है", "डिजिटल इंडिया की गति"),
      ("सड़क बनते ही अगले दिन पाइपलाइन बिछाने के लिए दोबारा खोद दी गई", "स्मार्ट सिटी का नियोजन"),
      ("बारिश की पहली ही बूंद में पूरे शहर की बिजली गुल हो गई", "आधुनिक पावर ग्रिड"),
      ("परीक्षा का प्रश्नपत्र परीक्षा शुरू होने से 2 घंटे पहले सोशल मीडिया पर वायरल हो गया", "परीक्षा की शुचिता"),
      ("ट्रेन 12 घंटे लेट होने के बाद उद्घोषणा होती है कि असुविधा के लिए खेद है", "भारतीय रेल की समयबद्धता")])
]

# ==========================================
# 6. HINDI DEVANAGARI NON-SARCASTIC (label = 0)
# ==========================================
hindi_non_sarcastic = [
    "आज सुबह मौसम बहुत सुहावना था और हल्की ठंडी हवा के साथ बारिश हुई।",
    "पुस्तकालय में विद्यार्थियों के अध्ययन के लिए सभी आवश्यक संदर्भ पुस्तकें उपलब्ध कराई गई हैं।",
    "कार्यालय में सभी कर्मचारियों ने समयबद्ध तरीके से वार्षिक लेखा परीक्षा का कार्य पूरा किया।",
    "चिकित्सक ने मरीज के स्वास्थ्य की गहन जांच की और आवश्यक दवाइयों के साथ परामर्श दिया।",
    "नई मेट्रो लाइन शुरू होने से शहर के नागरिकों को यातायात में काफी सुविधा और समय की बचत हो रही है।",
    "डिजिटल सेवा केंद्र के माध्यम से नागरिकों को आवश्यक प्रमाण पत्र सरलता से प्राप्त हो रहे हैं।",
    "दोनों पक्षों ने आपसी बातचीत के माध्यम से विवाद का शांतिपूर्ण और न्यायसंगत समाधान निकाल लिया।",
    "नियमित अध्ययन और अनुशासन के बल पर छात्र ने राष्ट्रीय स्तर की प्रतियोगिता में सफलता पाई।",
    "नगर निगम ने आगामी वर्षा ऋतु को ध्यान में रखते हुए नालियों की सफाई का विशेष अभियान शुरू किया है।",
    "कृषि वैज्ञानिकों ने किसानों को जैविक खाद और उन्नत बीजों के प्रयोग से संबंधित प्रशिक्षण दिया।",
    "पर्यावरण संरक्षण के लिए विद्यालय के छात्रों ने परिसर में 50 नए पौधे लगाए।",
    "रेलवे प्रशासन ने त्योहारों के अवसर पर यात्रियों की सुविधा हेतु विशेष ट्रेनों का संचालन किया है।",
    "बैंक ने वरिष्ठ नागरिकों के लिए घर बैठे डिजिटल जीवन प्रमाण पत्र जमा करने की सुविधा शुरू की है।",
    "स्वयंसेवी संस्था ने जरूरतमंद परिवारों को सर्दी से बचाव हेतु गर्म वस्त्र वितरित किए।",
    "शहरी विकास प्राधिकरण ने आवासीय क्षेत्र में नए सामुदायिक पार्क का निर्माण कार्य पूरा किया।"
]

templates_hi_dev_non_sarc = [
    ("{sanstha} ने {karya} किया जिससे {labh} हुआ।",
     [("स्वास्थ्य विभाग", "टीकाकरण अभियान का सफलतापूर्वक आयोजन", "हजारों बच्चों को सुरक्षा कवच प्राप्त"),
      ("शिक्षा बोर्ड", "वार्षिक परीक्षा परिणाम समय पर घोषित", "विद्यार्थी आगे की कक्षाओं में समय पर प्रवेश ले सके"),
      ("ऊर्जा निगम", "सौर ऊर्जा संयंत्र का विस्तार", "क्षेत्र में निर्बाध बिजली आपूर्ति सुनिश्चित"),
      ("ग्रामीण विकास बैंक", "कम ब्याज दर पर किसानों को ऋण वितरित", "कृषि उपकरणों की खरीद में सहायता"),
      ("जिला प्रशासन", "जल संरक्षण हेतु नए तालाबों का निर्माण", "भूजल स्तर में उल्लेखनीय सुधार"),
      ("वन विभाग", "अवैध कटाई पर सख्त निगरानी व्यवस्था स्थापित", "जंगलों के क्षेत्रफल में वृद्धि")]),

    ("आज {karyakram} में {shrota} ने {anubhav} प्राप्त किया।",
     [("विज्ञान संगोष्ठी", "प्रतिभागियों", "आधुनिक तकनीक एवं कृत्रिम बुद्धिमत्ता का ज्ञान"),
      ("साहित्यिक परिचर्चा", "विद्वानों", "प्राचीन पांडुलिपियों के संरक्षण पर विचार-विमर्श"),
      ("योग एवं ध्यान शिविर", "नागरिकों", "मानसिक शांति और शारीरिक स्फूर्ति का अनुभव"),
      ("उद्यमिता कार्यशाला", "युवा उद्यमियों", "व्यवसाय विस्तार और निवेश प्राप्त करने का मार्गदर्शन"),
      ("शास्त्रीय संगीत सम्मेलन", "श्रोताओं", "मधुर रागों और संगीत की समृद्ध परंपरा का आनंद"),
      ("कृषि मेला", "किसानों", "नई तकनीकों और आधुनिक कृषि यंत्रों की जानकारी")])
]


import csv
import random
import re


def expand_templates(base_items, templates, target_count):
    """Systematically expands and combines items to reach the target count."""
    items = list(base_items)
    # Generate items from templates
    for tmpl, variations in templates:
        cleaned_tmpl = re.sub(r"\{[a-zA-Z0-9_]+\}", "{}", tmpl)
        for var in variations:
            if isinstance(var, tuple):
                items.append(cleaned_tmpl.format(*var))
            else:
                items.append(cleaned_tmpl.format(var))
    
    # If still need more to reach target, generate contextual variations
    result = list(items)
    idx = 0
    while len(result) < target_count:
        base = items[idx % len(items)]
        # Add slight natural contextual prefixes if repeating
        prefixes = [
            "In fact, ", "Honestly, ", "Well, ", "Truth be told, ", 
            "Actually, ", "Just saying, ", "Real talk: ", "Seriously, "
        ]
        hi_prefixes = [
            "सच कहूं तो, ", "देखा जाए तो, ", "असल में, ", "हैरानी की बात है कि, ",
            "साफ बात है कि, ", "वाकई, ", "ध्यान देने वाली बात है कि, "
        ]
        prefix = random.choice(prefixes if any(c.isascii() for c in base[:5]) else hi_prefixes)
        new_item = f"{prefix}{base[0].lower()}{base[1:]}"
        if new_item not in result:
            result.append(new_item)
        idx += 1
        
    return result[:target_count]


def generate_dataset():
    # 500 Sarcastic
    # 180 English, 170 Hinglish, 150 Hindi
    en_sarc = expand_templates(english_sarcastic, templates_en_sarc, 180)
    hi_lat_sarc = expand_templates(hinglish_sarcastic, templates_hi_lat_sarc, 170)
    hi_dev_sarc = expand_templates(hindi_sarcastic, templates_hi_dev_sarc, 150)
    
    all_sarcastic = en_sarc + hi_lat_sarc + hi_dev_sarc  # Exactly 500
    
    # 500 Non-Sarcastic
    # 180 English, 170 Hinglish, 150 Hindi
    en_non_sarc = expand_templates(english_non_sarcastic, templates_en_non_sarc, 180)
    hi_lat_non_sarc = expand_templates(hinglish_non_sarcastic, templates_hi_lat_non_sarc, 170)
    hi_dev_non_sarc = expand_templates(hindi_non_sarcastic, templates_hi_dev_non_sarc, 150)
    
    all_non_sarcastic = en_non_sarc + hi_lat_non_sarc + hi_dev_non_sarc  # Exactly 500
    
    dataset = []
    for text in all_sarcastic:
        dataset.append((text.strip(), 1))
    for text in all_non_sarcastic:
        dataset.append((text.strip(), 0))
        
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(dataset)
    
    # Write to CSV
    output_files = ["sarcasm_1000_dataset.csv", "sarcasm_train_dataset.csv"]
    for filename in output_files:
        with open(filename, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text", "label"])
            for text, label in dataset:
                writer.writerow([text, label])
                
    print(f"Generated {len(dataset)} balanced rows across: {output_files}")
    sarc_count = sum(1 for _, l in dataset if l == 1)
    non_sarc_count = sum(1 for _, l in dataset if l == 0)
    print(f"Sarcastic (1): {sarc_count}, Non-Sarcastic (0): {non_sarc_count}")


if __name__ == "__main__":
    generate_dataset()
