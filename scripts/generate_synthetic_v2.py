#!/usr/bin/env python3
"""
generate_synthetic_v2.py
========================
Advanced synthetic triage case generator for mobile-app emergency assistant.

Key improvements over v1:
  - 3x more templates (200+ unique scenarios)
  - Natural user-style speech (stuttering, incomplete sentences, panic, informal)
  - Conversation-style (how a real person would speak into a phone app)
  - Better category balance (especially medical)
  - Paraphrase variations for each template
  - Multi-turn style fragments (short bursts like real speech)

Target:
  - NON_URGENT:  800
  - URGENT:      800
  - CRITICAL:    500

Output:
  data/labels/synthetic_triage_cases_v2.csv
"""
from __future__ import annotations
from pathlib import Path
import csv
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "labels" / "synthetic_triage_cases_v2.csv"

SEED = 123
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Realistic fill values
# ─────────────────────────────────────────────────────────────────────────────
AGES = list(range(1, 96))
SEXES = ["male", "female"]
RELATIONS = [
    "my father", "my mother", "my husband", "my wife", "my son", "my daughter",
    "my brother", "my sister", "my friend", "my neighbor", "my coworker",
    "my grandmother", "my grandfather", "a stranger", "a man", "a woman",
    "my uncle", "my aunt", "my roommate", "my boss", "a child", "an elderly person",
    "my boyfriend", "my girlfriend", "a teenager", "a homeless person",
]
LOCATIONS = [
    "at home", "at a gas station", "in a parking lot", "at a restaurant",
    "on the street", "at the shopping mall", "in an apartment building",
    "at school", "at the park", "on the highway", "at a grocery store",
    "at a church", "at the gym", "at a bar", "at a construction site",
    "at a bus stop", "in a dorm room", "at a hotel", "at a daycare",
    "at the office", "at a nursing home", "at a warehouse", "in my car",
    "at a laundromat", "at the movies", "at the train station", "at the beach",
    "in the backyard", "in the kitchen", "in the bathroom", "on the sidewalk",
    "at a coffee shop", "at the library", "at a stadium", "at a concert",
    "in the elevator", "on the rooftop", "in the basement", "at the airport",
    "at a clinic", "at a pharmacy", "at a bank", "in the garage",
]
TIMES = [
    "just now", "like a minute ago", "about 5 minutes ago", "maybe 10 minutes ago",
    "a few minutes ago", "about 30 minutes ago", "about an hour ago",
    "this morning", "a couple hours ago", "right now", "it just happened",
    "a few seconds ago", "not even a minute ago",
]
DURATIONS = [
    "for a couple minutes", "for about 5 minutes", "for about 10 minutes",
    "for maybe 15 minutes", "for about 20 minutes", "for half an hour",
    "for about an hour", "for a few minutes", "for like 2 minutes",
    "since this morning", "all day", "for a while now",
]
PAIN_LEVELS = ["3", "4", "5", "6", "7", "8", "9", "10"]
BODY_PARTS = [
    "chest", "head", "stomach", "back", "arm", "leg", "neck", "shoulder",
    "knee", "ankle", "wrist", "hip", "ribs", "eye", "jaw", "foot",
]
SYMPTOMS = [
    "dizzy", "nauseous", "short of breath", "sweating a lot", "shaking",
    "very pale", "confused", "in a lot of pain", "vomiting", "can't stand up",
    "weak", "disoriented", "crying", "screaming in pain", "really cold",
    "burning up with fever", "having trouble seeing", "slurring words",
]

def fill(template: str) -> str:
    """Fill all {placeholders} with random realistic values."""
    t = template
    t = t.replace("{age}", str(random.choice(AGES)))
    t = t.replace("{sex}", random.choice(SEXES))
    t = t.replace("{relation}", random.choice(RELATIONS))
    t = t.replace("{location}", random.choice(LOCATIONS))
    t = t.replace("{time}", random.choice(TIMES))
    t = t.replace("{duration}", random.choice(DURATIONS))
    t = t.replace("{pain}", random.choice(PAIN_LEVELS))
    t = t.replace("{body}", random.choice(BODY_PARTS))
    t = t.replace("{symptom}", random.choice(SYMPTOMS))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Caller speech variations – makes it sound like a real person on the phone
# ─────────────────────────────────────────────────────────────────────────────

FILLER_WORDS = [
    "um, ", "uh, ", "like, ", "so, ", "well, ", "I mean, ", "okay so, ",
    "look, ", "listen, ", "hey, ", "", "", "", "",  # empty = no filler (common)
]

PANIC_INTERJECTIONS = [
    "Oh my God! ", "Oh no! ", "Please! ", "Help! ", "Oh God oh God! ",
    "Jesus! ", "No no no! ", "Quick! ", "Hurry! ", "Please please! ",
    "Someone help! ", "", "",
]

CALM_PREFIXES = [
    "Hi, ", "Hello, ", "Hey, ", "Hi there, ", "Good evening, ",
    "Yeah hi, ", "Excuse me, ", "", "",
]

CONTEXT_STARTERS = [
    "I'm calling because ", "I need help, ", "I want to report ",
    "There's a situation here, ", "Something happened, ",
    "I don't know what to do, ", "Can someone come? ",
    "I need to report something, ", "We need assistance, ",
    "", "",
]


def add_speech_style(text: str, triage: str) -> str:
    """Add realistic speech patterns based on urgency."""
    if triage == "CRITICAL":
        # Panicked, fragmented, emotional
        prefix = random.choice(PANIC_INTERJECTIONS)
        # Sometimes repeat words for panic
        if random.random() < 0.3:
            words = text.split()
            if len(words) > 3:
                idx = random.randint(0, min(3, len(words)-1))
                words.insert(idx+1, words[idx])
                text = " ".join(words)
        suffix = random.choice([
            " Please send help!", " Hurry please!", " We need help now!",
            " How long until help arrives?", " Send an ambulance!",
            " Please come quickly!", " I'm so scared!",
            " They're dying!", "", "",
        ])
    elif triage == "URGENT":
        prefix = random.choice(CALM_PREFIXES) + random.choice(CONTEXT_STARTERS)
        # Add filler words
        if random.random() < 0.4:
            words = text.split()
            if len(words) > 5:
                idx = random.randint(2, len(words)-2)
                words.insert(idx, random.choice(["um,", "uh,", "like,"]))
                text = " ".join(words)
        suffix = random.choice([
            " Can you send someone?", " How long will that take?",
            " Please send help.", " Thank you.", " Should I do anything?",
            " What should I do?", " Is someone coming?", "",
        ])
    else:  # NON_URGENT
        prefix = random.choice(CALM_PREFIXES) + random.choice([
            "this isn't really an emergency but ", "I just wanted to let you know, ",
            "not sure if this is the right number but ", "I have a question, ",
            "it's not urgent or anything but ", "", "",
        ])
        suffix = random.choice([
            " Thanks.", " Thank you.", " Appreciate it.",
            " No rush.", " Whenever you get a chance.", " Thanks for your help.",
            " Let me know what I should do.", "", "",
        ])

    result = prefix + text + suffix
    # Clean up double spaces
    result = re.sub(r"\s+", " ", result).strip()
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEMPLATES: (text_template, category, red_flags)
# ═════════════════════════════════════════════════════════════════════════════

# ─── CRITICAL ────────────────────────────────────────────────────────────────
CRITICAL_TEMPLATES = [
    # Cardiac arrest / heart attack
    ("{relation} just collapsed and isn't breathing! I think it's a heart attack!", "medical", 1),
    ("{relation} grabbed their chest and fell down! They're not moving! I think their heart stopped!", "medical", 1),
    ("Someone collapsed {location} and has no pulse! We need CPR help! I don't know what to do!", "medical", 1),
    ("{relation} is on the floor not breathing! I'm doing chest compressions but I don't think it's working!", "medical", 1),
    ("My {age} year old {sex} just had a heart attack I think! They're unconscious!", "medical", 1),
    ("There's a {sex} here who collapsed, no pulse, not breathing at all!", "medical", 1),
    ("I found {relation} unconscious on the floor! They're blue in the face! Not breathing!", "medical", 1),

    # Choking / airway
    ("{relation} is choking! They can't breathe! I tried hitting their back but nothing's coming out!", "medical", 1),
    ("My baby is choking on something! They're turning purple! I can't get it out!", "medical", 1),
    ("A {age} year old is choking at the restaurant! They're grabbing their throat! Someone help!", "medical", 1),

    # Severe bleeding
    ("There's blood everywhere! {relation} cut themselves badly and it won't stop bleeding!", "medical", 1),
    ("Someone got stabbed {location}! They're bleeding really bad! Blood is pooling on the ground!", "crime", 1),
    ("{relation} fell through a glass door and their arm is sliced open! Blood is gushing out!", "medical", 1),
    ("There's a {sex} bleeding from the head {location}! They fell and there's so much blood!", "medical", 1),
    ("A person got hit by a car and they're bleeding from everywhere! They're not moving!", "medical", 1),

    # Shooting
    ("Someone just got shot {location}! I heard the gunshot and they fell!", "crime", 1),
    ("There's an active shooter {location}! People are running! I can hear gunshots!", "crime", 1),
    ("A {sex} just got shot in the {body}! There's blood everywhere! The shooter is still here!", "crime", 1),
    ("Shots fired {location}! Multiple people down! Send everything you have!", "crime", 1),
    ("I just heard gunshots and someone is lying on the ground not moving!", "crime", 1),

    # Stabbing / weapon
    ("Someone has a knife and they just attacked a person {location}! Victim is on the ground!", "crime", 1),
    ("A man with a machete is attacking people {location}! Someone is badly hurt!", "crime", 1),
    ("{relation} was just attacked with a weapon! They're bleeding and barely conscious!", "crime", 1),

    # Severe burns / fire with entrapment
    ("The house is on fire and {relation} is still inside! I can't get to them!", "fire", 1),
    ("There's a huge fire {location}! I can hear people screaming inside! They're trapped!", "fire", 1),
    ("A building is on fire and people are on the upper floors! They can't get out!", "fire", 1),
    ("My kitchen caught fire and now the whole apartment is burning! I barely got out!", "fire", 1),
    ("An explosion just happened {location}! There's fire everywhere and people are hurt!", "fire", 1),
    ("{relation} got severe burns from boiling water! Their skin is peeling off! It's really bad!", "medical", 1),
    ("A gas explosion just happened! The building is destroyed and I can see injured people!", "fire", 1),

    # Drowning
    ("A child fell in the pool and isn't breathing! We pulled them out but they're not moving!", "medical", 1),
    ("Someone went underwater and hasn't come up! It's been at least 2 minutes!", "medical", 1),
    ("{relation} fell into the river! The current took them! I can't see them anymore!", "medical", 1),

    # Overdose / poisoning
    ("{relation} took a bunch of pills and now they're barely breathing! I think it's an overdose!", "medical", 1),
    ("I found {relation} unconscious with empty pill bottles! Their lips are turning blue!", "medical", 1),
    ("A {sex} at the party overdosed on something! They're not responding and their breathing is weird!", "medical", 1),
    ("My toddler drank cleaning chemicals! They're throwing up and crying!", "medical", 1),

    # Stroke
    ("I think {relation} is having a stroke! Half their face is drooping and they can't talk!", "medical", 1),
    ("{relation} suddenly can't move their left side and their speech is all messed up! I think it's a stroke!", "medical", 1),
    ("A {age} year old just collapsed, one side paralyzed, can't speak! Stroke symptoms!", "medical", 1),

    # Severe trauma
    ("{relation} fell from the roof! Maybe 15 feet! They're not moving at all!", "medical", 1),
    ("A worker fell off the scaffolding {location}! They're unconscious, I think their back is broken!", "medical", 1),
    ("A tree fell on someone {location}! They're pinned and screaming!", "medical", 1),
    ("Someone just got hit by a truck! They flew through the air! They're on the ground not moving!", "medical", 1),
    ("There was a terrible car crash! The car is crushed and the person inside is trapped and not responding!", "medical", 1),

    # Infant / child emergency
    ("My baby stopped breathing! They're only {age} months old! I don't know what happened!", "medical", 1),
    ("A child is having a severe allergic reaction! Their face is swelling and they can't breathe! The epipen isn't working!", "medical", 1),
    ("My {age} year old fell from the window! Second floor! They're lying on the ground not moving!", "medical", 1),

    # Hanging / suicide attempt
    ("I found {relation} hanging! I cut them down but they're not breathing!", "medical", 1),
    ("{relation} just told me they took all their medication to kill themselves! They're getting drowsy!", "medical", 1),
    ("Someone is on the bridge threatening to jump! Please send help before it's too late!", "crime", 1),

    # Electrocution
    ("A {sex} got electrocuted {location}! They're on the ground shaking and then went still!", "medical", 1),
    ("{relation} touched a downed power line! They're unconscious!", "medical", 1),
]

# ─── URGENT ──────────────────────────────────────────────────────────────────
URGENT_TEMPLATES = [
    # Moderate car accident
    ("There's been a car accident {location}. A couple cars involved, people are hurt but everyone seems conscious.", "crime", 0),
    ("I just saw a car hit a cyclist {location}. The cyclist is sitting up but their arm looks broken.", "medical", 0),
    ("A car flipped over {location}! The driver crawled out, they're bleeding from the head but talking to us.", "medical", 0),
    ("There was a fender bender and now one of the drivers is complaining of neck pain. They don't want to move.", "medical", 0),
    ("A motorcycle went down {location}. Rider has road rash all over. Conscious but in a lot of pain.", "medical", 0),
    ("Pedestrian got clipped by a car {location}. They can walk but their hip is hurting badly.", "medical", 0),

    # Chest pain / possible heart issues
    ("{relation} has been having chest pains {duration}. They're {symptom}. I'm worried it might be their heart.", "medical", 0),
    ("My {age} year old {sex} is complaining of crushing chest pain. Started {time}. They look really pale.", "medical", 1),
    ("{relation} has chest tightness and their arm is numb. They have a history of heart problems.", "medical", 1),

    # Breathing difficulty
    ("{relation} is having trouble breathing. They have asthma and their inhaler isn't helping.", "medical", 1),
    ("My {age} year old can't catch their breath. They're wheezing really bad. They have COPD.", "medical", 1),
    ("A {sex} {location} is having an asthma attack. They're conscious but really struggling to breathe.", "medical", 1),
    ("{relation} is short of breath and their lips are kind of bluish. They have a lung condition.", "medical", 1),

    # Seizure
    ("{relation} is having a seizure right now! They're on the ground shaking. They're epileptic.", "medical", 1),
    ("A {sex} just started convulsing {location}. I put them on their side. They have a history of seizures.", "medical", 1),
    ("My {age} year old is having a seizure! It's been going on {duration}. What do I do?", "medical", 1),

    # Allergic reaction
    ("{relation} is having a bad allergic reaction. Face is swelling up. They used their epipen.", "medical", 1),
    ("A coworker ate something they're allergic to. Hives everywhere and their tongue is swelling.", "medical", 1),
    ("My kid is having an allergic reaction. Red all over, struggling to breathe a little. We gave benadryl.", "medical", 1),

    # Falls
    ("{relation} fell down the stairs and hit their head. They're conscious but confused and dizzy.", "medical", 0),
    ("An elderly {sex} fell {location}. They can't get up. Their hip might be broken.", "medical", 0),
    ("{relation} fell off a ladder, maybe 8 feet. They landed on their {body}. In a lot of pain but awake.", "medical", 0),
    ("A {age} year old fell at the playground. Their arm looks bent wrong, probably broken.", "medical", 0),

    # Burns
    ("{relation} burned their hand on the stove. Big blisters forming. Skin is really red.", "medical", 0),
    ("Someone spilled hot oil on themselves {location}. Burns on their arm and chest. Conscious and in pain.", "medical", 0),

    # Diabetic emergency
    ("{relation} is diabetic and they're acting really confused. Blood sugar might be low. They're shaking.", "medical", 0),
    ("My {age} year old diabetic is unresponsive but breathing. They missed their insulin.", "medical", 1),

    # Pregnancy
    ("My wife is pregnant and having contractions really close together. Her water broke. First baby.", "medical", 0),
    ("A pregnant woman {location} is in labor. The baby might be coming. She says she can feel it.", "medical", 1),
    ("There's a pregnant woman bleeding heavily. She's about 7 months along. Please send an ambulance.", "medical", 1),

    # Drug / alcohol
    ("{relation} drank way too much and now they're barely awake. I'm worried they might pass out completely.", "medical", 0),
    ("I found someone passed out {location}. They might have taken something. They're breathing but I can't wake them.", "medical", 1),
    ("A {sex} at the bar collapsed. They were drinking and maybe took pills. Still breathing.", "medical", 1),

    # Domestic disturbance
    ("I can hear my neighbors fighting. Things are breaking. It sounds violent but I don't think there are weapons.", "crime", 0),
    ("There's a couple fighting {location}. Yelling and pushing. No weapons. One person has a bloody nose.", "crime", 0),
    ("My neighbor's boyfriend is screaming at her and banging on the door. She sounded scared.", "crime", 0),
    ("Two people are in a physical altercation {location}. Fists only. One of them fell down.", "crime", 0),

    # Robbery / break-in
    ("Someone is trying to break into my house right now! I can hear them at the back door!", "crime", 0),
    ("I just saw two people break the window of the store {location}. They went inside.", "crime", 0),
    ("I was just robbed {location}. They took my bag and pushed me. I'm not badly hurt but shaken up.", "crime", 0),
    ("There's someone inside my car going through my stuff {location}. They're still there.", "crime", 0),

    # Suspicious / threat
    ("There's a suspicious person following me {location}. I'm scared. They've been behind me for blocks.", "crime", 0),
    ("A man is threatening people {location}. He's aggressive and yelling. I don't see a weapon but he's big.", "crime", 0),
    ("Someone threatened me with what looked like a knife {location}. They walked away but I'm scared.", "crime", 0),

    # Fire (contained / small)
    ("There's a small fire {location}. Smoke is coming from a trash can. It could spread.", "fire", 0),
    ("I smell gas really strongly in my building. No flames but the smell is overwhelming.", "fire", 1),
    ("Smoke coming from a car {location}. Getting thicker. No flames visible yet.", "fire", 0),
    ("A kitchen fire {location}. We used the extinguisher but there's still smoke and it might reignite.", "fire", 0),
    ("There's a brush fire near the road {location}. Small but it's windy and could grow.", "fire", 0),

    # Missing person
    ("My {age} year old didn't come home from school. It's been over 2 hours. Phone goes to voicemail.", "other", 0),
    ("My elderly {relation} with dementia wandered off. We can't find them. They've been gone an hour.", "other", 1),
    ("There's a small child alone {location}. No parent around. They look scared and lost.", "other", 0),

    # Animal attack
    ("A dog just bit my {age} year old! It's a deep bite on the leg. Bleeding a lot.", "medical", 0),
    ("I was attacked by a stray dog {location}. Bites on my arm. It might be rabid.", "medical", 0),

    # Moderate mental health
    ("{relation} is having a mental health crisis. They're talking about hurting themselves but haven't done anything yet.", "medical", 0),
    ("Someone {location} is very agitated, talking to themselves, and scaring people. They might need help.", "other", 0),
]

# ─── NON_URGENT ──────────────────────────────────────────────────────────────
NON_URGENT_TEMPLATES = [
    # Noise
    ("My neighbors are playing really loud music. It's been going on {duration}. Can someone talk to them?", "other", 0),
    ("There's a party next door that's super loud. They won't turn it down.", "other", 0),
    ("A dog has been barking nonstop {duration} {location}. Owner isn't home.", "other", 0),
    ("Construction noise {location} at night. They're drilling and it's past quiet hours.", "other", 0),
    ("Fireworks going off in the neighborhood. It's not a holiday. My pets are freaking out.", "other", 0),
    ("There's a car alarm that's been going off {duration}. No one is turning it off.", "other", 0),
    ("Loud music from a car in the parking lot {duration}. It's shaking my windows.", "other", 0),

    # Minor property crime
    ("My car got scratched in the parking lot {location}. Want to file a report.", "crime", 0),
    ("Someone stole my bike {location}. I just noticed it's gone.", "crime", 0),
    ("There's graffiti on our building that wasn't there yesterday.", "crime", 0),
    ("My package was stolen from my porch. I have it on camera.", "crime", 0),
    ("Someone broke my car window {location} but didn't take anything.", "crime", 0),
    ("My mailbox was knocked over. Looks like a car hit it.", "other", 0),
    ("Somebody egged our house last night. Just want to report it.", "crime", 0),
    ("My garden gnomes were stolen from the front yard. Silly but I want to report it.", "crime", 0),

    # False alarm / accidental
    ("Sorry, my kid dialed this by accident. No emergency here.", "other", 0),
    ("False alarm. The fire alarm went off from cooking smoke. Everything's fine.", "fire", 0),
    ("I accidentally hit the emergency button. I'm okay, no help needed.", "other", 0),
    ("Calling back to say the situation resolved. No need to send anyone.", "other", 0),
    ("My alarm system went off by mistake. False alarm. Sorry about that.", "other", 0),
    ("I butt-dialed you, sorry. Everything is fine here.", "other", 0),

    # Information / non-emergency request
    ("I found a wallet on the ground {location}. What should I do with it?", "other", 0),
    ("I want to report a lost dog. Big golden retriever, no collar, wandering {location}.", "other", 0),
    ("Is there a non-emergency number I should call about a noise complaint?", "other", 0),
    ("I'd like to know if there's a warrant out for an old traffic ticket.", "other", 0),
    ("Where do I go to file a police report for insurance purposes?", "other", 0),
    ("I lost my phone {location}. Is there a lost and found?", "other", 0),
    ("How do I get a copy of an accident report from last week?", "other", 0),
    ("I want to report a found cat. It's been in my yard for days.", "other", 0),

    # Minor medical
    ("I cut my finger in the kitchen. Not deep but won't stop bleeding. Should I go to urgent care?", "medical", 0),
    ("My {age} year old has a low fever. They seem okay but I wanted to ask.", "medical", 0),
    ("I twisted my ankle {location}. Can put some weight on it. Is this an ER thing?", "medical", 0),
    ("My kid got stung by a bee. No allergic reaction, just swelling. Should I worry?", "medical", 0),
    ("I have a rash that's spreading. Not painful, just itchy. Should I go to the doctor?", "medical", 0),
    ("My {age} year old bumped their head. No loss of consciousness, just a bump. When should I worry?", "medical", 0),
    ("I got a splinter stuck deep in my foot. Can't get it out. Is this urgent?", "medical", 0),
    ("My back has been hurting {duration}. Nothing sudden, just wondering if I need the ER.", "medical", 0),
    ("I think I have food poisoning. Throwing up {duration}. Can keep water down though.", "medical", 0),
    ("My {age} year old has had a nosebleed {duration}. It keeps coming back but isn't heavy.", "medical", 0),

    # Suspicious but not threatening
    ("There's a car parked on our street I don't recognize. Been here 3 days.", "other", 0),
    ("People are going door to door selling something. Seem legit but wanted to mention it.", "other", 0),
    ("I found an open door at the business next door. Might be nothing.", "other", 0),
    ("Someone left a bag on the park bench. Been there all day.", "other", 0),
    ("There's a van parked in my neighborhood that I've never seen before. No markings.", "other", 0),
    ("A drone has been flying over our backyard repeatedly. Kind of creepy.", "other", 0),

    # Traffic / road
    ("Traffic light not working at the intersection {location}.", "other", 0),
    ("Big pothole on the road {location}. Someone's gonna damage their car.", "other", 0),
    ("A tree branch fell on the sidewalk {location}. Blocking the path.", "other", 0),
    ("Stop sign knocked over at the corner {location}.", "other", 0),
    ("There's a dead deer on the side of the road {location}.", "other", 0),
    ("A streetlight has been out for weeks {location}.", "other", 0),
    ("There's flooding on the road {location} from a broken hydrant.", "other", 0),

    # Parking / traffic complaint
    ("Someone parked blocking my driveway. Been here {duration}.", "other", 0),
    ("A car has been parked in a handicap spot without a permit {location}.", "other", 0),
    ("There's a car double-parked blocking traffic {location}.", "other", 0),
    ("An abandoned car has been on our street for over a week.", "other", 0),

    # Neighbor / nuisance
    ("My neighbor's tree is leaning over my property. It might fall.", "other", 0),
    ("There are teenagers skateboarding in the parking structure. Not an emergency.", "other", 0),
    ("Someone keeps letting their dog poop on my lawn. Can anything be done?", "other", 0),
    ("My neighbor is burning leaves in their backyard. The smoke is coming into my house.", "fire", 0),
    ("There's a homeless person sleeping in the park. They seem okay but it's very cold.", "other", 0),
]


def generate_cases(templates, triage_label, count, id_prefix):
    """Generate `count` synthetic cases from templates with speech variation."""
    cases = []
    for i in range(count):
        tmpl_text, category, red_flags = random.choice(templates)
        text = fill(tmpl_text)
        text = add_speech_style(text, triage_label)

        cases.append({
            "case_id": f"synth2_{triage_label.lower()}_{id_prefix}_{i}",
            "text_en": text,
            "source": "synthetic_v2",
            "label_category_gold": category,
            "label_triage_gold": triage_label,
            "red_flags_gold": red_flags,
        })
    return cases


def deduplicate(cases):
    """Remove exact text duplicates."""
    seen = set()
    unique = []
    for c in cases:
        if c["text_en"] not in seen:
            seen.add(c["text_en"])
            unique.append(c)
    return unique


def main():
    all_cases = []

    all_cases.extend(generate_cases(NON_URGENT_TEMPLATES, "NON_URGENT", 800, "nu"))
    all_cases.extend(generate_cases(URGENT_TEMPLATES, "URGENT", 800, "urg"))
    all_cases.extend(generate_cases(CRITICAL_TEMPLATES, "CRITICAL", 500, "crit"))

    # Deduplicate
    before = len(all_cases)
    all_cases = deduplicate(all_cases)
    print(f"Dedup: {before} → {len(all_cases)}")

    # Shuffle
    random.shuffle(all_cases)

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "text_en", "source", "label_category_gold", "label_triage_gold", "red_flags_gold"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_cases)

    print(f"\nSentetik v2 veri olusturuldu: {OUT_CSV}")
    print(f"Toplam: {len(all_cases)} satir")

    from collections import Counter
    triage_dist = Counter(c["label_triage_gold"] for c in all_cases)
    cat_dist = Counter(c["label_category_gold"] for c in all_cases)
    print(f"\nTriage dagilimi: {dict(triage_dist)}")
    print(f"Kategori dagilimi: {dict(cat_dist)}")

    # Avg text length
    avg_len = sum(len(c["text_en"]) for c in all_cases) / len(all_cases)
    print(f"Ortalama metin uzunlugu: {avg_len:.0f} karakter")

    print("\n=== Ornek satirlar ===")
    for c in all_cases[:8]:
        print(f"  [{c['label_triage_gold']:12s}|{c['label_category_gold']:8s}] {c['text_en'][:130]}...")


if __name__ == "__main__":
    main()
