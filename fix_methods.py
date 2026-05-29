"""
fix_methods.py — Doda metode treninga za vse trike v tricks_lookup.json
Zazenite: python fix_methods.py
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "model", "tricks_lookup.json")

METHODS = {
    "Sit": [
        {"method_name": "Lure metoda", "best_for": "Vse pasme, začetniki",
         "steps": "1. Drži priboljšek pred nosom psa\n2. Počasi premakni roke nazaj čez glavo\n3. Pes bo naravno sedel\n4. Takoj ko sede: 'Yes!' + nagrada\n5. Dodaj besedo 'Sit' preden začneš\n6. Postopno zmanjšuj lure"},
        {"method_name": "Capture metoda", "best_for": "Inteligentne pasme",
         "steps": "1. Čakaj da pes sede sam\n2. Takoj: 'Yes!' + nagrada\n3. Ponovi 10-20x\n4. Dodaj 'Sit' tik preden pričakuješ vedenje\n5. Vadi dokler ni zanesljiv"},
    ],
    "Down": [
        {"method_name": "Lure iz Sit", "best_for": "Večina pasem",
         "steps": "1. Začni s psom v Sit\n2. Drži priboljšek pred nosom\n3. Počasi premakni roke navzdol do tal\n4. Premakni malo naprej\n5. Ko se komolci dotaknejo tal: 'Yes!' + nagrada\n6. Postopno zmanjšuj lure"},
        {"method_name": "Shaping metoda", "best_for": "Inteligentne, motivirane pasme",
         "steps": "1. Začni s psom v Sit\n2. Nagradi vsak gib navzdol (naklon glave, pokrčitev)\n3. Postopno zahtevaj več\n4. Nagradi samo bližje Down položaju\n5. Dodaj ukaz ko je vedenje zanesljivo"},
    ],
    "Stay": [
        {"method_name": "Duration metoda", "best_for": "Vse pasme",
         "steps": "1. Pes v Sit ali Down\n2. Reci 'Stay' z dlanjo proti psu\n3. Počakaj 2 sekundi, nato nagradi\n4. Postopno povečuj: 5, 10, 20 sekund\n5. Dodaj razdaljo šele ko je trajanje zanesljivo\n6. Vadi release besedo ('OK' ali 'Free')"},
    ],
    "Come": [
        {"method_name": "Recall metoda", "best_for": "Vse pasme",
         "steps": "1. Začni na kratki razdalji (1 meter)\n2. Pokleči, odpri roke, veselo reci 'Come!'\n3. Ko pride: velika nagrada + pohvala\n4. Nikoli ne kaznuj po recall\n5. Postopno povečuj razdaljo\n6. Vadi z dolgim povodcem"},
        {"method_name": "Igra recall", "best_for": "Pasme z visoko igrivostjo",
         "steps": "1. Pokaži igračo\n2. Teci stran od psa medtem ko kličeš 'Come!'\n3. Ko pride, igraj se z njim 10 sekund\n4. Menjaj hrano in igro kot nagrado"},
    ],
    "Leave It": [
        {"method_name": "Dve roki metoda", "best_for": "Vse pasme",
         "steps": "1. Priboljšek v zaprto pest\n2. Pes bo lizal, grizal pest - ignoriraj\n3. Ko odneha: 'Yes!' + nagrada iz DRUGE roke\n4. Ponovi, dodaj 'Leave it'\n5. Postopno prehodi na tla, nato sprehod"},
        {"method_name": "Napredna Leave It", "best_for": "Pasme z visoko trenabilnostjo",
         "steps": "1. Priboljšek na tla, pokrij z nogo\n2. Ko pes odneha od noge: nagradi\n3. Postopno odkrivaj priboljšek\n4. Vadi z različnimi predmeti\n5. Dodaj vadi med sprehodom"},
    ],
    "Drop It": [
        {"method_name": "Trade metoda", "best_for": "Vse pasme",
         "steps": "1. Ko ima pes predmet v gobcu\n2. Ponudi visoko vreden priboljšek pred nosom\n3. Ko spusti: 'Yes!' + nagrada\n4. Vrni predmet - tako ne bo bojeval\n5. Dodaj 'Drop it' preden ponudiš menjavo"},
        {"method_name": "Two toy metoda", "best_for": "Pasme ki radi igrajo fetch",
         "steps": "1. Igraj se z igračo 1\n2. Pokaži igračo 2 in reci 'Drop it'\n3. Ko spusti igračo 1: vrzi igračo 2\n4. Menjajte igrači\n5. Postopno dodajaj zamudo pred metom"},
    ],
    "Watch Me": [
        {"method_name": "Lure na oči", "best_for": "Vse pasme",
         "steps": "1. Priboljšek pri nosu psa\n2. Počasi premakni prst do svojih oči\n3. Ko vzpostavi očesni kontakt: 'Yes!' + nagrada\n4. Začni z 1 sekundo, podaljšuj\n5. Dodaj 'Watch me' preden premakneš prst"},
    ],
    "Touch": [
        {"method_name": "Target stick metoda", "best_for": "Vse pasme, odlično za začetek",
         "steps": "1. Iztegni dlan pred psov nos (10 cm)\n2. Pes bo ovonhal - takoj: 'Yes!' + nagrada\n3. Ponovi 10x\n4. Premakni dlan v različne položaje\n5. Dodaj 'Touch' preden iztegneš dlan\n6. Postopno povečuj razdaljo"},
    ],
    "Heel": [
        {"method_name": "Lure metoda", "best_for": "Hrana motivirane pasme",
         "steps": "1. Priboljšek pri levi nogi (višina psovega nosa)\n2. Naredi korak - če pes sledi: nagradi\n3. Postopno dodajaj korake\n4. Dodaj 'Heel' ukaz\n5. Vadi zavoje in spremembe tempa"},
        {"method_name": "Penalty yards", "best_for": "Psi ki vlečejo",
         "steps": "1. Hodi naprej\n2. Ko pes vleče: takoj se ustavi\n3. Vrni se 5 korakov nazaj\n4. Začni znova\n5. Ponavljaj - pes se nauči da vlečenje = nazaj\n6. Nagradi hojo ob nogi"},
    ],
    "Wait": [
        {"method_name": "Door metoda", "best_for": "Vse pasme",
         "steps": "1. Pri vratih: 'Wait'\n2. Odpri vrata malo - če pes gre mimo, zapri\n3. Ponavljaj dokler ne čaka\n4. Nagradi čakanje, nato 'OK' za sprostitev\n5. Povečuj trajanje in distrakcije"},
    ],
    "No": [
        {"method_name": "Redirect metoda", "best_for": "Vse pasme",
         "steps": "1. Ko pes dela neželjeno vedenje: 'No' (mirno)\n2. Takoj preusmeri na željeno vedenje\n3. Nagradi željeno vedenje\n4. Ne ponavljaj 'No' večkrat\n5. Budi dosledni vsi v družini"},
    ],
    "Stand": [
        {"method_name": "Lure metoda", "best_for": "Vse pasme",
         "steps": "1. Pes v Sit\n2. Priboljšek pred nosom, premakni ravno naprej\n3. Ko vstane: 'Yes!' + nagrada\n4. Dodaj 'Stand' preden premakneš lure\n5. Vadi iz Sit in Down"},
    ],
    "Place/Go to Bed": [
        {"method_name": "Shaping na podlogo", "best_for": "Vse pasme",
         "steps": "1. Postavi podlogo/blazino\n2. Nagradi vsak korak proti podlogi\n3. Nato nagradi samo ko stopi na podlogo\n4. Nato nagradi samo Down na podlogi\n5. Dodaj 'Place' ukaz\n6. Postopno povečuj trajanje"},
    ],
    "Settle": [
        {"method_name": "Capture metoda", "best_for": "Vse pasme",
         "steps": "1. Ko se pes ulegel sam: 'Yes!' + nagrada\n2. Ponavljaj\n3. Dodaj 'Settle' preden se uleže\n4. Vadi v mirnem okolju\n5. Postopno vadi ob distrakcijah"},
    ],
    "Shake/Paw": [
        {"method_name": "Lure tačko", "best_for": "Vse pasme",
         "steps": "1. Pes v Sit\n2. Priboljšek v zaprto pest pred tačko\n3. Pes bo skušal dobiti - ko dvigne tačko: 'Yes!'\n4. Odpri pest in nagradi\n5. Dodaj 'Shake' ko tačka pride v tvojo roko"},
        {"method_name": "Capture metoda", "best_for": "Pasme ki pogosto dvigajo tačke",
         "steps": "1. Čakaj da pes sam dvigne tačko\n2. Takoj: 'Yes!' + nagrada\n3. Dodaj 'Shake' preden dvigne\n4. Prosi za tačko z iztegnjeno roko"},
    ],
    "High Five": [
        {"method_name": "Iz Shake", "best_for": "Psi ki znajo Shake",
         "steps": "1. Začni s Shake\n2. Postopno premikaj dlan višje\n3. Ko tačka doseže višino roke: 'Yes!'\n4. Dodaj 'High five' ukaz\n5. Vadi obe tački"},
    ],
    "Roll Over": [
        {"method_name": "Lure metoda", "best_for": "Večina pasem",
         "steps": "1. Začni z Down\n2. Priboljšek pri nosu, premakni v lok do hrbta\n3. Pes bo sledil in se zavrtel\n4. 'Yes!' ko se popolnoma zavrne\n5. Dodaj 'Roll over'\n6. Vadi počasi - nikoli na trdi podlagi"},
    ],
    "Play Dead": [
        {"method_name": "Lure na bok", "best_for": "Večina pasem",
         "steps": "1. Začni z Down\n2. Priboljšek pri nosu, lure na bok\n3. Ko leži na boku: 'Yes!' + nagrada\n4. Dodaj dramatičen 'Bang!'\n5. Zahtevaj daljše 'mrtve' ležanje\n6. Dodaj vstajanje na 'Živiš!'"},
    ],
    "Spin": [
        {"method_name": "Lure krog", "best_for": "Vse pasme",
         "steps": "1. Priboljšek pri nosu\n2. Počasi naredi krog (v smeri urinega kazalca)\n3. Ko naredi cel krog: 'Yes!' + nagrada\n4. Dodaj 'Spin'\n5. Nauči obe smeri ('Spin' in 'Twist')"},
    ],
    "Crawl": [
        {"method_name": "Lure pod nogo", "best_for": "Strpne pasme",
         "steps": "1. Pes v Down\n2. Sede na tla, iztegni nogo\n3. Lure priboljšek pod nogo\n4. Pes mora plaziti pod nogo\n5. Postopno dviguj nogo\n6. Dodaj 'Crawl'"},
    ],
    "Speak": [
        {"method_name": "Capture lajanje", "best_for": "Pasme ki radi lajajo",
         "steps": "1. Počakaj na naravno lajanje\n2. Takoj: 'Yes!' + nagrada\n3. Dodaj 'Speak' preden zahlaja\n4. Nauči 'Tiho' hkrati\n5. Nikoli ne uči Speak brez Tiho"},
        {"method_name": "Trigger metoda", "best_for": "Psi ki lajajo na zvonec",
         "steps": "1. Pozvoni zvonec\n2. Ko zahlaja: 'Yes!' + nagrada\n3. Dodaj 'Speak' preden pozvoniš\n4. Postopno zmanjšuj trigger"},
    ],
    "Quiet": [
        {"method_name": "Tiho metoda", "best_for": "Vse pasme",
         "steps": "1. Najprej nauči Speak\n2. Ko lajai: 'Tiho' (mirno, enkrat)\n3. Počakaj sekundo tišine\n4. Takoj: 'Yes!' + nagrada\n5. Postopno podaljšuj tišino\n6. Nikoli ne kričite - misli da lajate skupaj"},
    ],
    "Back Up": [
        {"method_name": "Hodimo nazaj", "best_for": "Vse pasme",
         "steps": "1. Stoj pred psom\n2. Počasi hodi proti njemu\n3. Pes bo natural stopil nazaj\n4. 'Yes!' + nagrada\n5. Dodaj 'Back' ukaz\n6. Postopno dodaj gesture z roko"},
    ],
    "Fetch": [
        {"method_name": "Two toy metoda", "best_for": "Pasme z nagоnom za prinašanje",
         "steps": "1. Igraj se z igračo 1\n2. Vrzi igračo 1\n3. Ko jo prinese, pokaži igračo 2\n4. Ko spusti igračo 1: vrzi igračo 2\n5. Postopno dodaj 'Fetch' ukaz\n6. Vadi prinašanje v roke"},
        {"method_name": "Shaping fetch", "best_for": "Pasme brez naravnega nagona",
         "steps": "1. Nagradi gledanje igrače\n2. Nagradi hoditi do igrače\n3. Nagradi dotik z nosom\n4. Nagradi vzeti v gobec\n5. Nagradi hoditi z igračo\n6. Nagradi prinesti v roke"},
    ],
    "Give": [
        {"method_name": "Trade metoda", "best_for": "Vse pasme",
         "steps": "1. Ko ima predmet v gobcu\n2. Ponudi priboljšek\n3. Ko spusti: 'Yes!' + nagrada\n4. Dodaj 'Give' ukaz\n5. Vadi z različnimi predmeti"},
    ],
    "Weave Through Legs": [
        {"method_name": "Lure med nogami", "best_for": "Srednje in male pasme",
         "steps": "1. Stoj z razmaknjenimi nogami\n2. Lure med nogami (zig-zag)\n3. 'Yes!' ko naredi en prehod\n4. Postopno poveži prehode\n5. Dodaj 'Weave' ukaz\n6. Hodi naprej med vajenjem"},
    ],
    "Take a Bow": [
        {"method_name": "Lure poklon", "best_for": "Vse pasme",
         "steps": "1. Pes v Stand\n2. Lure navzdol med sprednjimi nogami\n3. Ko sprednji del gre dol (zadek gor): 'Yes!'\n4. Dodaj 'Bow'\n5. Zahtevaj daljši poklon"},
    ],
    "Jump Through Hoop": [
        {"method_name": "Postopno dviganje", "best_for": "Energične pasme",
         "steps": "1. Drži obroč na tleh\n2. Lure skozi obroč: 'Yes!'\n3. Postopno dviguj obroč\n4. Dodaj 'Hoop' ukaz\n5. Vadi brez lure"},
    ],
    "Balance Treat on Nose": [
        {"method_name": "Stay + leave it", "best_for": "Pasme z dobrim Stay",
         "steps": "1. Pes mora znati Stay in Leave it\n2. Daj priboljšek na nos\n3. 'Stay' - sekundo, nato 'OK'\n4. Pes bo vrgel in ujel priboljšek\n5. Postopno povečuj čas\n6. Vadi z lahkimi predmeti najprej"},
    ],
    "Find It (Scent Work)": [
        {"method_name": "Hide and seek", "best_for": "Vse pasme, posebej hrti in lovske",
         "steps": "1. Pokaži priboljšek, vrzi pred psa\n2. 'Find it!' ko gre iskat\n3. Postopno skrivaj pod kozarcem\n4. Nato v sobi\n5. Nato zunaj\n6. Dodaj specifičen vonj (nogavica)"},
    ],
    "Ring Bell": [
        {"method_name": "Touch na zvonec", "best_for": "Psi ki znajo Touch",
         "steps": "1. Namesti zvonec na vrata\n2. Lure nos do zvonca\n3. Ko se dotakne in zazvoní: 'Yes!' + takoj ven\n4. Dodaj 'Bell' ukaz\n5. Vadi pred vsakim izhodom"},
    ],
    "Tidy Up Toys": [
        {"method_name": "Fetch v škatlo", "best_for": "Psi ki znajo Fetch in Drop",
         "steps": "1. Postavi škatlo za igrače\n2. Drži igračo nad škatlo\n3. 'Drop it' ko je nad škatlo\n4. Postopno pes sam prinese igračo\n5. Dodaj 'Pospravi' ukaz\n6. Vadi z več igračami"},
    ],
    "Skateboard": [
        {"method_name": "Postopno navajanje", "best_for": "Drzne, energične pasme",
         "steps": "1. Nagrajuj samo bližanje deski\n2. Nagrajuj dotik deske\n3. Nagrajuj stopanje na desko\n4. Potisni desko malo - nagradi\n5. Postopno povečuj razdaljo\n6. Vadi na ravni površini"},
    ],
}

# Naloži obstoječe trike
with open(PATH, encoding="utf-8") as f:
    tricks = json.load(f)

# Dodaj metode
updated = 0
for trick in tricks:
    name = trick["trick_name"]
    if name in METHODS:
        trick["methods"] = METHODS[name]
        updated += 1
    elif not trick.get("methods"):
        trick["methods"] = []

# Shrani
with open(PATH, "w", encoding="utf-8") as f:
    json.dump(tricks, f, ensure_ascii=False, indent=2)

print(f"Posodobljeno: {updated} trikov z metodami")
print("Preverba:")
for t in tricks[:15]:
    print(f"  {t['trick_name']:30s} {len(t.get('methods',[]))} metod")