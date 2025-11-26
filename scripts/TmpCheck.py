import json
from pathlib import Path
names = [
"Intel Corp. v. Advanced Micro Devices, Inc.",
"ZF Automotive US, Inc. v. Luxshare, Ltd.",
"AlixPartners, LLP v. Fund for Prot. of Inv. Rights",
"In re Application of Asta Medica, S.A.",
"In re Schlich",
"In re Porsche Automobil Holding SE",
"Nat'l Broadcasting Co. v. Bear Stearns & Co.",
"Republic of Kazakhstan v. Biedermann Int'l",
"Abdul Latif Jameel Transp. Co. v. FedEx Corp.",
"Servotronics, Inc. v. Boeing Co.",
"Servotronics, Inc. v. Rolls-Royce PLC",
"Consorcio Ecuatoriano de Telecom. S.A. v. JAS Forwarding (USA), Inc.",
"Euromepa S.A. v. R. Esmerian, Inc.",
"Brandi-Dohrn v. IKB Deutsche Industriebank AG",
"Mees v. Buiter",
"Certain Funds, Accounts &/or Inv. Vehicles v. KPMG, LLP",
"In re del Valle Ruiz",
"In re Guo",
"In re Bayer AG",
"In re Application of Gianoli Aldunate",
"In re Metallgesellschaft AG",
"In re Letter of Request from Supreme Ct. of Hong Kong",
"In re Clerici",
"Sergeeva v. Tripleton Int'l Ltd.",
"Furstenberg Finance SAS v. Litai Assets LLC",
"Republic of Ecuador v. Hinchee",
"Heraeus Kulzer GmbH v. Biomet, Inc.",
"In re Application of Malev Hungarian Airlines",
"In re Application of Esses",
"Four Pillars Enter. Co. v. Avery Dennison Corp.",
"In re Letters Rogatory from Tokyo Dist. Court",
"In re Premises Located at 840 140th Ave. NE, Bellevue, Wash.",
"In re CPC Patent Techs. Pty Ltd.",
"Khrapunov v. Prosyankin",
"In re Republic of Ecuador",
"Andover Healthcare, Inc. v. 3M Co.",
"Kiobel by Samkalden v. Cravath, Swaine & Moore LLP",
"In re Biomet Orthopaedics Switzerland GmbH",
"In re Accent Delight Int'l Ltd.",
"Schmitz v. Bernstein Liebhard LLP",
"In re Sarrio, S.A.",
"In re Edelman",
"In re Roz Trading Ltd.",
"In re Application of Babcock Borsig AG",
"In re Chevron Corp.",
"In re Application of Peruvian Sporting Goods S.A.C.",
"In re Hand Held Prods., Inc.",
"Daedalus Prime LLC v. MediaTek, Inc.",
"Amazon.com, Inc. v. Nokia Corp.",
"In re Netgear, Inc."
]

cases_dir = Path('data/case_law/1782_discovery')
entries = []
for path in cases_dir.glob('*.json'):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    name = (data.get('caseNameFull') or data.get('caseName') or path.stem or '').lower()
    entries.append((name, path.name))

for target in names:
    tl = target.lower()
    prefix = tl.split(' v')[0]
    hits = [fname for name, fname in entries if prefix in name]
    print(f"{target} -> {hits[:3] if hits else 'MISSING'}")
