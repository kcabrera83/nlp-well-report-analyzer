"""Generate synthetic well report text data with structured fields."""

import random
import json

WELL_NAMES = [
    "Pemex-A1", "Pemex-B3", "Pemex-C7", "Pemex-D12", "Pemex-E5",
    "Cantarell-X1", "Cantarell-X4", "Cantarell-X9", "Cantarell-X15",
    "Ku-Maloob-A2", "Ku-Maloob-B6", "Ku-Maloob-C11",
    "Zama-1", "Zama-3", "Zama-8", "Ixachi-1A", "Ixachi-2B",
    "Litoral-7", "Litoral-14", "Altamira-21", "Burgos-N3", "Burgos-S5",
    "Chicontepec-1", "Chicontepec-9", "Tampico-Misantla-4",
    "Veracruz-11", "Macuspana-6", "Musaceo-2", "Cordon-Campo-1",
    "Sierra-Blanca-3", "La-Cruz-8", "Jujo-10", "El-Angel-15",
]

FORMATIONS = [
    "Ku-Nu", "Nito", "Arenoso", "Lime Creek", "Edwards",
    "Eocene", "Cretaceous", "Jurassic", "Turonian", "Cenomanian",
    "Cantarell Formation", "Ku-Maloob Formation", "Shila Formation",
    "Tuxpan Platform", "Golden Lane", "Sureste Basin",
    "Burgos Basin", "Veracruz Basin", "Chiapas Fold Belt",
    "Mesozoic Carbonate", "Tertiary Sandstone", "Volcanic Sequence",
    "La Luna Formation", "Misoa Formation", "Lagunillas Formation",
]

EQUIPMENT = [
    "drill bit PDC", "drill bit tricone", "casing 7 inch", "casing 9 5/8 inch",
    "cementing unit", "mud pump", "BOP stack", "blowout preventer",
    "production tubing 3 1/2 inch", "ESP pump", "rod pump",
    "hydraulic fracturing equipment", "coiled tubing unit",
    "wireline logging tool", "MWD tool", "LWD tool",
    "packer assembly", "production liner", "perforating gun",
    "acidizing equipment", "sand control screens", "gravel pack assembly",
    "wellhead assembly", "Christmas tree", "choke manifold",
    "separator", "test separator", "flowback equipment",
]

ISSUES = [
    "lost circulation zones encountered at depth", "casing shoe failure during pressure test",
    "stuck pipe incident requiring sidetrack", "lost returns during drilling",
    "lost circulation material pumped", "wellbore instability in shale section",
    "kick detected during drilling operations", "lost time due to equipment failure",
    "cement channeling observed behind casing", "differential sticking observed",
    "lost circulation material lost down hole", "severe mud losses encountered",
    "formation damage near wellbore", "scale buildup in production tubing",
    "sand production issues", "corrosion in casing detected",
    "casing collapse in lower section", "annular pressure buildup",
    "gas breakthrough in production zone", "water encroachment noted",
    "paraffin deposition in tubing", "emulsion blockage in formation",
    "high H2S concentration detected", "asphaltene precipitation",
]

RECOMMENDATIONS = [
    "recommend sidetrack to avoid lost zone", "recommend running additional casing string",
    "recommend cement squeeze remediation", "recommend acid stimulation treatment",
    "recommend gravel pack completion", "recommend hydraulic fracturing treatment",
    "recommend workover to replace ESP", "recommend rod pump conversion",
    "recommend artificial lift optimization", "recommend well integrity test",
    "recommend reservoir pressure maintenance", "recommend water injection support",
    "recommend periodic well testing", "recommend production optimization review",
    "recommend corrosion inhibitor treatment", "recommend sand control measures",
    "recommend P&A due to mechanical failure", "recommend recompletion in upper zone",
    "recommend drilling ahead with larger mud weight", "recommend casing patch repair",
]

REPORT_TEMPLATES = {
    "drilling": [
        "Drilling report for well {well}. Total depth reached {depth} feet in the {formation} formation. "
        "Drilling fluid weight was {mud_weight} ppg. {issues}. "
        "Wellhead pressure recorded at {whp} psi. Drilling rate averaged {rop} ft/hr.",

        "Well {well} drilled to TD of {depth} ft through {formation}. "
        "Casing set at {casing_depth} feet, {casing_size} inch. {issues}. "
        "Mud system: {mud_type}. Total drilling time: {time} days.",

        "Drilling operations at {well} completed. Final depth {depth} ft. "
        "Formation penetrated: {formation}. {issues}. "
        "Recommendations: {recommendations}.",
    ],
    "completion": [
        "Completion report for well {well} at depth {depth} ft in {formation}. "
        "Perforated interval: {perf_interval}. Equipment used: {equipment}. "
        "Initial production: {oil_rate} bbl/day oil, {gas_rate} MMcf/day gas. "
        "Completion type: {completion_type}.",

        "Well {well} completion in {formation} formation at {depth} ft. "
        "Flowing tubing head pressure: {fthp} psi. {issues}. "
        "Equipment installed: {equipment}. Initial test rates: {oil_rate} BOPD, {gas_rate} MMscf/d.",

        "Post-completion report for {well}. {completion_type} completion. "
        "Target formation: {formation} at {depth} ft. {issues}. "
        "Recommendations: {recommendations}.",
    ],
    "workover": [
        "Workover report for well {well}. Previous depth {depth} ft in {formation}. "
        "Objective: {workover_objective}. Equipment used: {equipment}. "
        "Issues encountered: {issues}. Post-workover production: {oil_rate} BOPD.",

        "Workover operations at {well} targeting {formation} at {depth} ft. "
        "Well was {well_status} prior to workover. {issues}. "
        "Equipment deployed: {equipment}. Result: {result}.",

        "Well {workover_objective} at {well}. Target interval {formation}, {depth} ft. "
        "{issues}. Equipment: {equipment}. Recommendations: {recommendations}.",
    ],
    "production": [
        "Monthly production report for well {well}. Average production: {oil_rate} BOPD, "
        "{gas_rate} MMscf/d, {water_rate} BWPD. Reservoir pressure: {pressure} psi. "
        "Cumulative production: {cumulative} MMBO. Issues: {issues}.",

        "Production test for {well} in {formation} at {depth} ft. "
        "Flowing bottom hole pressure: {fbhp} psi. Oil rate: {oil_rate} BOPD. "
        "Gas oil ratio: {gor} scf/bbl. Water cut: {water_cut} percent. {issues}.",

        "Well test results for {well}. Reservoir: {formation}. Depth: {depth} ft. "
        "Productivity index: {pi} bbl/psi. {issues}. "
        "Recommendations: {recommendations}.",
    ],
}


def _rand_depth():
    return random.randint(3000, 18000)


def _rand_rate(low, high):
    return round(random.uniform(low, high), 1)


def generate_report(report_type=None):
    if report_type is None:
        report_type = random.choice(list(REPORT_TEMPLATES.keys()))

    templates = REPORT_TEMPLATES[report_type]
    template = random.choice(templates)

    well = random.choice(WELL_NAMES)
    depth = _rand_depth()
    formation = random.choice(FORMATIONS)
    equipment = random.choice(EQUIPMENT)
    issues = random.choice(ISSUES)
    recommendations = random.choice(RECOMMENDATIONS)

    oil_rate = _rand_rate(50, 3000)
    gas_rate = _rand_rate(0.5, 50)
    water_rate = _rand_rate(100, 5000)

    values = {
        "well": well,
        "depth": str(depth),
        "formation": formation,
        "equipment": equipment,
        "issues": issues,
        "recommendations": recommendations,
        "mud_weight": str(round(random.uniform(8.5, 18), 1)),
        "mud_type": random.choice(["water-based mud", "oil-based mud", "synthetic mud"]),
        "whp": str(random.randint(500, 5000)),
        "rop": str(round(random.uniform(5, 80), 1)),
        "casing_depth": str(random.randint(2000, depth - 500)),
        "casing_size": random.choice(["7", "9 5/8", "5 1/2"]),
        "time": str(random.randint(15, 90)),
        "perf_interval": f"{depth - random.randint(50, 300)}-{depth}",
        "completion_type": random.choice(["open hole", "cased and perforated", "slotted liner", "frac-pack"]),
        "fthp": str(random.randint(500, 8000)),
        "fbhp": str(random.randint(1000, 6000)),
        "pressure": str(random.randint(1500, 8000)),
        "oil_rate": str(oil_rate),
        "gas_rate": str(gas_rate),
        "water_rate": str(water_rate),
        "gor": str(random.randint(200, 3000)),
        "water_cut": str(round(random.uniform(5, 85), 1)),
        "pi": str(round(random.uniform(0.5, 25), 2)),
        "cumulative": str(round(random.uniform(1, 100), 1)),
        "well_status": random.choice(["inactive", "plugged", "damaged", "low producer"]),
        "workover_objective": random.choice([
            "repair casing", "recomplete in new zone", "stimulate formation",
            "install artificial lift", "perform plug and abandon", "remediate cement"
        ]),
        "result": random.choice(["successful", "partially successful", "unsuccessful"]),
    }

    text = template
    for key, val in values.items():
        text = text.replace("{" + key + "}", val)

    structured = {
        "report_type": report_type,
        "well_name": well,
        "depth_ft": depth,
        "formation": formation,
        "equipment": equipment,
        "issues": issues,
        "recommendations": recommendations,
        "oil_rate_bopd": oil_rate,
        "gas_rate_mmscfd": gas_rate,
    }
    return text, structured


def generate_dataset(n=500):
    data = []
    for _ in range(n):
        rtype = random.choice(list(REPORT_TEMPLATES.keys()))
        text, structured = generate_report(rtype)
        data.append({"text": text, "structured": structured})
    return data


def save_dataset(path="outputs/training_data.json", n=500):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = generate_dataset(n)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return dataset
