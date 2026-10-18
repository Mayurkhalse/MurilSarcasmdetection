import os
import csv
import random
import pandas as pd

def generate_1000_sarcasm_dataset(output_path: str = "random_synthetic_sarcasm_1000.csv"):
    random.seed(2026)

    # =========================================================================
    # 1. SARCASTIC DATA GENERATION (Target: 500 items, Label = 1)
    # =========================================================================
    sarcastic_sentences = set()

    # Curated standalone sarcastic sentences
    curated_sarcastic = [
        "Oh fantastic, another mandatory team bonding workshop on Saturday morning.",
        "I just adore when the fire alarm goes off during a cold shower.",
        "Nothing says luxury dining like a lukewarm burger delivered 90 minutes late.",
        "Great job scheduling our server maintenance right during Black Friday peak hours.",
        "I love how my laptop fan sounds like a commercial Boeing 747 taking off.",
        "Thanks for the warm beer, really hits the spot on a freezing winter evening.",
        "What a brilliant move to replace our entire technical support team with a broken chatbot.",
        "I truly enjoy when my flight is delayed four hours so I miss my connecting flight.",
        "Oh wonderful, another software update that moved every single button I use daily.",
        "Congratulations on fixing one CSS margin by completely disabling user checkout.",
        "I love paying a hefty convenience fee to print my own concert tickets at home.",
        "Superb parallel parking job, taking up three parking slots at the crowded mall.",
        "Nothing screams productivity like an hour-long meeting discussing our next meeting.",
        "Oh great, high-speed fiber internet giving me a blazing 500 bytes per second.",
        "I love waking up to twenty spam phone calls about car insurance I never asked for.",
        "What a masterclass in management, blaming junior developers for executive delays.",
        "Brilliant idea to host an outdoor corporate retreat in 42-degree summer heat.",
        "I just love having to enter an OTP, a password, a fingerprint, and a captcha to view my bus pass.",
        "Thanks for explaining my own research paper back to me in the simplest possible terms.",
        "Oh delightful, the train is only eighty minutes behind schedule today.",
        "I really appreciate when the waiter disappears for forty minutes after bringing the bill.",
        "Nothing like a flat tire in the pouring rain to make a Monday morning complete.",
        "What an honor to receive a generic virtual thank-you badge after working sixty hours overtime.",
        "Oh perfect, my wireless headphones died thirty seconds into my run.",
        "I love how the weather forecast promised clear sunny skies right before a thunderstorm hit.",
        "Brilliant strategy, cutting the customer support budget to boost customer satisfaction scores.",
        "I just love receiving promotional emails from websites I visited once five years ago.",
        "Nothing says reliable public transit like buses that only show up when three come together.",
        "Oh look, another password policy requiring hieroglyphs and the name of my first pet.",
        "What a treat, watching 3 unskippable thirty-second ads before a five-second video clip."
    ]
    sarcastic_sentences.update(curated_sarcastic)

    # Combinatorial Generators for Sarcastic Sentences
    sarc_templates = [
        # Work & Management
        ("Oh {0}, our {1} decided to {2} right when {3}.", [
            ("brilliant", "department head", "schedule an emergency meeting", "we were all walking out the door"),
            ("fantastic", "project manager", "add five major deliverables", "the sprint was already finished"),
            ("wonderful", "team lead", "reassign all my tasks", "I was about to submit them"),
            ("superb", "director", "call an all-hands call", "lunch was just delivered"),
            ("delightful", "client", "request an entire redesign", "we were scheduled to deploy to production"),
            ("stellar", "tech lead", "refactor the entire codebase", "we were preparing for a live demo"),
            ("marvelous", "scrum master", "schedule a two-hour retrospective", "we had hard release deadlines"),
            ("magnificent", "manager", "demand a status report", "the server was literally on fire"),
            ("splendid", "supervisor", "assign another urgent priority", "my shift ended five minutes ago"),
            ("charming", "VP", "announce weekend overtime", "everyone had holiday travel plans")
        ]),

        # Tech & Software Failures
        ("I absolutely love how {0} {1}, it makes {2} so {3}.", [
            ("this new operating system", "crashes whenever I open two browser tabs", "multitasking", "seamless"),
            ("this navigation app", "directs me straight into a massive traffic jam", "commuting", "relaxing"),
            ("my smartphone battery", "drops from 80% to 1% in four minutes", "staying connected", "convenient"),
            ("this smart television", "takes five minutes to open the settings menu", "entertainment", "effortless"),
            ("this cloud backup tool", "quietly deleted my newest presentation files", "working remotely", "reliable"),
            ("this antivirus software", "slows down my workstation to a complete crawl", "productivity", "enjoyable"),
            ("this wireless mouse", "randomly disconnects every three minutes", "gaming", "thrilling"),
            ("this food delivery app", "cancels my order after making me wait two hours", "ordering dinner", "stress-free"),
            ("this streaming app", "buffers continuously during the final minute of the match", "watching sports", "immersive"),
            ("this banking application", "locks my debit card whenever I try to buy groceries", "paying bills", "flawless")
        ]),

        # Customer Service & Retail Irony
        ("Nothing says '{0}' quite like {1}.", [
            ("customers come first", "leaving people on hold for seventy minutes with elevator music"),
            ("world-class customer care", "disconnecting the support call right as I explain my issue"),
            ("zero hidden fees", "charging a twenty-dollar processing fee for an online ticket"),
            ("fast and reliable shipping", "taking three weeks to deliver a standard envelope across town"),
            ("exceptional dining experience", "serving cold soup in a dirty bowl after an hour-long wait"),
            ("unmatched hotel comfort", "a mattress that feels exactly like a slab of solid concrete"),
            ("eco-friendly packaging", "wrapping a tiny plastic pen inside three layers of cardboard and bubble wrap"),
            ("premium airline service", "charging fifty dollars for a small carry-on bag and a lukewarm cup of water"),
            ("dedicated healthcare support", "making patients wait four hours past their appointment time"),
            ("guaranteed customer satisfaction", "having a strict no-refunds and no-returns policy on damaged goods")
        ]),

        # Everyday Disasters with Fake Enthusiasm
        ("What a {0} day to {1}, especially with {2}!", [
            ("glorious", "go for a pleasant morning jog", "heavy construction drilling right outside my window"),
            ("perfect", "drive to an important job interview", "a surprise puncture on the busiest highway"),
            ("delightful", "host an outdoor barbecue", "swarms of aggressive mosquitoes and sudden downpours"),
            ("splendid", "study for final semester exams", "the building power going out for ten straight hours"),
            ("marvelous", "wear brand new white sneakers", "stepping directly into an ankle-deep puddle of mud"),
            ("lovely", "catch up on much-needed sleep", "the upstairs neighbors moving furniture at 3 AM"),
            ("wonderful", "enjoy a peaceful quiet evening", "endless loud car alarms echoing through the alley"),
            ("fantastic", "attend an outdoor wedding ceremony", "blistering 43-degree humidity with no shade"),
            ("great", "cook a nice homemade dinner", "finding out the stove burner stopped working mid-meal"),
            ("stellar", "enjoy my morning coffee", "spilling it straight onto my freshly ironed shirt")
        ]),

        # Rhetorical Irony
        ("Because who doesn't want to {0} right after {1}?", [
            ("fill out a forty-page feedback questionnaire", "getting denied an insurance claim"),
            ("listen to automated elevator music for another hour", "being disconnected twice"),
            ("deal with six layers of bureaucratic paperwork", "trying to get a replacement identity card"),
            ("pay twenty dollars for airport parking", "waiting in a security queue for three hours"),
            ("clean up water leaking from the ceiling", "getting home from a sixteen-hour double shift"),
            ("re-enter all personal information twenty times", "the government portal crashed again"),
            ("argue with an automated bot that repeats the same FAQ", "spending thirty minutes troubleshooting"),
            ("stand in a crowded train with broken air conditioning", "working all day in summer heat"),
            ("pay extra baggage fees", "the airline already lost two of my suitcases"),
            ("re-iron an entire suit", "the airline crammed it into an overhead bin")
        ]),

        # Mock Praise for Incompetence
        ("Kudos to {0} for {1}, truly {2}!", [
            ("the local telecom operator", "providing zero cell signal inside the city center", "a technological marvel"),
            ("the web developer", "making every button on the landing page completely unclickable", "visionary UX design"),
            ("the municipal corporation", "digging up the newly paved road three days after finishing it", "outstanding planning"),
            ("the building maintenance team", "shutting off the water supply with zero prior notice", "impeccable coordination"),
            ("the rideshare driver", "driving in the opposite direction for twenty minutes", "flawless navigation"),
            ("the restaurant chef", "substituting fresh garlic with burning charcoal", "culinary innovation"),
            ("the courier company", "tossing my fragile glass package directly over a six-foot gate", "gentle handling"),
            ("the online retailer", "shipping one left shoe and one completely different sneaker", "exceptional packing"),
            ("the audio technician", "blasting ear-piercing feedback through the microphones", "unforgettable sound design"),
            ("the airline crew", "announcing a departure delay five minutes after scheduled takeoff", "top-tier punctuality")
        ]),

        # Classic Sarcastic Expressions
        ("Oh sure, {0}, that will definitely {1}.", [
            ("let's cut the software testing phase by half", "ensure a bug-free production release"),
            ("let's ignore the strange knocking sound under the car hood", "make the engine run like new"),
            ("let's add ten more managers to the team", "double our coding productivity"),
            ("let's skip reading the manual entirely", "make assembling this complex wardrobe easy"),
            ("let's drink four energy drinks at midnight", "help me wake up fresh and well-rested at 6 AM"),
            ("let's leave the windows wide open during a sandstorm", "keep the living room pristine and clean"),
            ("let's postpone studying until the morning of the exam", "guarantee a top rank in the class"),
            ("let's keep opening fifty Chrome tabs on 4GB RAM", "make the browser lightning fast"),
            ("let's schedule a three-hour meeting without any agenda", "solve all organizational bottlenecks"),
            ("let's buy cheap knockoff phone chargers online", "safeguard our expensive battery life")
        ])
    ]

    for template, var_list in sarc_templates:
        for vars_tuple in var_list:
            sarcastic_sentences.add(template.format(*vars_tuple))

    # Expand systematically to exactly 500 unique sarcastic items
    prefix_variations = [
        "Oh wow, ", "Truly magnificent: ", "How wonderful! ", "Honestly, ", "What a shocker: ",
        "Brilliant, absolutely brilliant: ", "Just great: ", "My absolute favorite part of the day: ",
        "Could this be any better? ", "Color me surprised: ", "Top-tier experience: ",
        "Pure genius at work: ", "I am so thrilled that ", "What a pleasant surprise: ",
        "Unbelievable, ", "Nothing short of genius: ", "Gotta love it when ",
        "Such a joy: ", "Remarkable, ", "Hands down the highlight of the week: "
    ]
    
    base_sarc = list(sarcastic_sentences)
    for pref in prefix_variations:
        for base in base_sarc:
            new_sent = f"{pref}{base[0].lower()}{base[1:]}"
            sarcastic_sentences.add(new_sent)
            if len(sarcastic_sentences) >= 500:
                break
        if len(sarcastic_sentences) >= 500:
            break

    sarcastic_list = list(sarcastic_sentences)[:500]

    # =========================================================================
    # 2. NON-SARCASTIC DATA GENERATION (Target: 500 items, Label = 0)
    # =========================================================================
    non_sarcastic_sentences = set()

    # Curated standalone non-sarcastic sentences
    curated_non_sarcastic = [
        "The software engineering team resolved the high database latency issue on time.",
        "Please remember to submit your monthly expense receipts before next Monday.",
        "The morning weather forecast indicates clear skies and light winds throughout the day.",
        "The airline staff promptly assisted all passengers with their boarding luggage.",
        "We are pleased to announce our latest product update with improved security features.",
        "The coffee shop offers a quiet study environment with fast and reliable Wi-Fi.",
        "The annual company financial audit was successfully completed without any discrepancies.",
        "The new electric bus fleet reduced commuter travel times across the metropolitan area.",
        "Our customer support team is available twenty-four hours a day to resolve inquiries.",
        "The university campus library will remain open until midnight during exam week.",
        "The doctor carefully reviewed the blood test results and prescribed appropriate medication.",
        "The hotel room was spotless, quiet, and equipped with all essential amenities.",
        "All attendees received a copy of the meeting minutes and action items via email.",
        "The train arrived exactly on schedule at platform number four this morning.",
        "Our team achieved all key performance milestones for the third financial quarter.",
        "The municipal department completed the road resurfacing project ahead of schedule.",
        "The online payment gateway processed the credit card transaction securely and instantly.",
        "The conference keynote speaker delivered an inspiring lecture on renewable energy.",
        "Regular cardiovascular exercise and balanced nutrition contribute to long-term health.",
        "The supermarket stocks fresh organic fruits and vegetables from local farms every morning.",
        "The mobile application update introduces automatic dark mode and bug fixes.",
        "All new employees must complete the online cybersecurity safety training module.",
        "The restaurant served delicious authentic pasta with polite and attentive service.",
        "The city museum offers free admission to students and seniors on the first Sunday of each month.",
        "The construction crew wore hard hats and followed all safety protocols on site.",
        "The package was carefully delivered to the front door within the promised delivery window.",
        "The research team published their findings in a reputable peer-reviewed scientific journal.",
        "The laptop battery provides up to nine hours of continuous video playback on a full charge.",
        "Please keep your seatbelts fastened while the aircraft taxiing to the gate.",
        "The company offers comprehensive health insurance and retirement planning benefits."
    ]
    non_sarcastic_sentences.update(curated_non_sarcastic)

    # Combinatorial Generators for Non-Sarcastic Sentences
    non_sarc_templates = [
        # Professional & Business Operations
        ("The {0} department successfully {1} for {2}.", [
            ("finance", "completed the annual budgetary forecast", "the upcoming fiscal year"),
            ("human resources", "onboarded twenty new software engineers", "the expansion program"),
            ("marketing", "launched the international digital campaign", "the new mobile product"),
            ("operations", "optimized the warehouse logistics network", "faster retail delivery"),
            ("quality assurance", "executed comprehensive automated test suites", "the web platform"),
            ("legal", "reviewed and finalized the commercial vendor contracts", "the regional office"),
            ("cybersecurity", "conducted an internal vulnerability assessment", "enterprise data systems"),
            ("customer success", "resolved over five hundred customer tickets", "the quarterly evaluation"),
            ("research", "analyzed user behavioral metrics", "the application redesign"),
            ("procurement", "negotiated volume discount agreements", "hardware supplies")
        ]),

        # Technology, Science & Infrastructure
        ("The new {0} provides {1} which improves {2}.", [
            ("database cluster", "automated real-time failover support", "overall platform uptime"),
            ("cloud computing infrastructure", "dynamic auto-scaling capabilities", "resource efficiency"),
            ("machine learning pipeline", "faster batch inference throughput", "data processing speeds"),
            ("caching architecture", "instant memory lookups", "page load latency for mobile users"),
            ("encryption protocol", "end-to-end data security", "user privacy across the network"),
            ("content delivery network", "distributed regional edge nodes", "video streaming performance"),
            ("continuous integration pipeline", "automated unit and integration testing", "release cycle stability"),
            ("battery management system", "balanced cell thermal regulation", "long-term hardware lifespan"),
            ("fiber optic link", "high-bandwidth low-latency connectivity", "cross-datacenter transfers"),
            ("microservices framework", "modular service isolation", "fault tolerance during traffic spikes")
        ]),

        # Everyday Factual Statements & Public Information
        ("Passengers traveling on {0} should note that {1}.", [
            ("the central express train", "services run every fifteen minutes during peak commute hours"),
            ("the cross-town bus route", "new electric buses have been introduced to reduce emissions"),
            ("the airport shuttle service", "pick-up points are located directly outside Terminal 2"),
            ("the regional ferry line", "schedules may be adjusted in accordance with tidal conditions"),
            ("the light rail transit", "contactless payment cards are accepted at all turnstile gates"),
            ("the suburban railway line", "platform renovations will take place on weekends only"),
            ("the city metro system", "trains are equipped with air conditioning and digital display maps"),
            ("the high-speed rail corridor", "complimentary high-speed Wi-Fi is available in all carriages"),
            ("the intercity coach service", "seat reservations must be confirmed twenty-four hours in advance"),
            ("the local feeder buses", "connections to main subway terminals are guaranteed during morning hours")
        ]),

        # Genuine Reviews & Customer Experiences
        ("I had a {0} experience with {1}; the staff was {2} and {3}.", [
            ("very pleasant", "the hotel front desk team", "extremely courteous", "our room was ready early"),
            ("wonderful", "the local dental clinic", "gentle and professional", "the procedure went smoothly"),
            ("positive", "the auto repair workshop", "honest and transparent", "the pricing was fair"),
            ("great", "the neighborhood bakery", "welcoming and cheerful", "the bread was freshly baked"),
            ("delightful", "the customer support helpline", "patient and attentive", "my billing issue was fixed promptly"),
            ("satisfying", "the dry cleaning service", "efficient and thorough", "all garments were ready on time"),
            ("refreshing", "the community bookstore", "knowledgeable and friendly", "they recommended fantastic books"),
            ("reassuring", "the veterinary hospital", "caring and thorough", "our pet recovered very quickly"),
            ("seamless", "the car rental agency", "prompt and organized", "vehicle pickup took under five minutes"),
            ("pleasant", "the optometry clinic", "punctual and polite", "the eye examination was comprehensive")
        ]),

        # Education & Academics
        ("Students enrolled in {0} must complete {1} by {2}.", [
            ("Advanced Data Structures", "the weekly programming assignments", "Friday evening"),
            ("Introduction to Macroeconomics", "the midterm chapter readings", "the end of the week"),
            ("Modern World History", "the analytical term essay", "the beginning of November"),
            ("Organic Chemistry Laboratory", "the safety protocol evaluation", "prior to the first lab session"),
            ("Artificial Intelligence Principles", "the peer-reviewed project proposal", "mid-semester"),
            ("Linear Algebra", "the problem set exercises", "next Wednesday morning"),
            ("Environmental Science", "the field observation report", "the upcoming seminar"),
            ("Business Ethics and Governance", "the case study presentation", "next Tuesday afternoon"),
            ("Cognitive Psychology", "the research experiment survey", "the scheduled deadline"),
            ("Software Engineering Practice", "the team code review milestone", "the sprint conclusion")
        ]),

        # Weather & Nature Observations
        ("Today's forecast calls for {0} in {1} with temperatures around {2}.", [
            ("partly cloudy skies and occasional gentle breezes", "the northern valley", "22 degrees Celsius"),
            ("bright sunny weather and clear visibility", "the coastal plain", "26 degrees Celsius"),
            ("intermittent light rain showers in the afternoon", "the metropolitan area", "19 degrees Celsius"),
            ("cool autumn winds and crisp morning air", "the highland foothills", "15 degrees Celsius"),
            ("warm sunshine and calm afternoon winds", "the southern peninsula", "28 degrees Celsius"),
            ("overcast conditions with moderate humidity", "the eastern district", "24 degrees Celsius"),
            ("dry pleasant conditions suitable for hiking", "the western mountain trail", "18 degrees Celsius"),
            ("a mild evening breeze following an afternoon shower", "the central harbor", "21 degrees Celsius"),
            ("clear night skies offering excellent stargazing", "the desert observatory", "14 degrees Celsius"),
            ("fresh seasonal sunshine and gentle westerly breezes", "the lakeside suburbs", "23 degrees Celsius")
        ])
    ]

    for template, var_list in non_sarc_templates:
        for vars_tuple in var_list:
            non_sarcastic_sentences.add(template.format(*vars_tuple))

    # Expand systematically to exactly 500 unique non-sarcastic items
    non_sarc_prefixes = [
        "In addition, ", "According to the official report, ", "Furthermore, ",
        "For your information, ", "It was noted that ", "As confirmed by the team, ",
        "Official records confirm that ", "The latest bulletin indicates that ",
        "It is widely recognized that ", "The announcement stated that ",
        "On regular weekdays, ", "During normal business operations, ",
        "As scheduled, ", "The published guidelines state that ",
        "Please be advised that ", "It is reported that ", "By standard procedure, ",
        "As part of the plan, ", "In the latest update, ", "Community members noted that "
    ]

    base_non_sarc = list(non_sarcastic_sentences)
    for pref in non_sarc_prefixes:
        for base in base_non_sarc:
            new_sent = f"{pref}{base[0].lower()}{base[1:]}"
            non_sarcastic_sentences.add(new_sent)
            if len(non_sarcastic_sentences) >= 500:
                break
        if len(non_sarcastic_sentences) >= 500:
            break

    non_sarcastic_list = list(non_sarcastic_sentences)[:500]

    # Combine, format with label, and shuffle
    dataset = []
    for s in sarcastic_list:
        dataset.append({"text": s, "sarcasm_label": 1})
    for ns in non_sarcastic_list:
        dataset.append({"text": ns, "sarcasm_label": 0})

    random.shuffle(dataset)

    # Save to CSV
    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully generated {len(df)} samples into: {output_path}")
    print(f"Class counts:\n{df['sarcasm_label'].value_counts()}")
    return output_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    out = os.path.join(project_root, "datasets", "random_synthetic_sarcasm_1000.csv")
    generate_1000_sarcasm_dataset(out)
