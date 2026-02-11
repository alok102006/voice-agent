def get_legal_advice(category):
    advice = {
        "CYBER": [
            "Immediately inform your bank and block the compromised account.",
            "File a cybercrime complaint at the nearest police station or cybercrime portal.",
            "Preserve evidence such as SMS alerts, emails, and transaction details.",
            "Avoid sharing OTPs or passwords with anyone."
        ],

        "CRIMINAL": [
            "Consult a criminal lawyer immediately.",
            "Ensure your fundamental rights are not violated.",
            "Ask for a copy of FIR or arrest memo if applicable.",
            "Do not make statements without legal counsel."
        ],

        "EMPLOYMENT": [
            "Review your employment contract carefully.",
            "Send a written notice to your employer regarding unpaid salary or termination.",
            "Approach the labor commissioner if required.",
            "Consult an employment lawyer for legal remedies."
        ],

        "FAMILY": [
            "Collect all relevant personal and financial documents.",
            "Try mediation if possible before legal action.",
            "Consult a family court lawyer.",
            "Understand your rights regarding maintenance and custody."
        ],

        "PROPERTY": [
            "Gather property ownership documents and agreements.",
            "Verify land records with the local authority.",
            "Consult a property lawyer before filing a case.",
            "Avoid any oral agreements without documentation."
        ]
    }

    return advice.get(category, ["Consult a legal professional for guidance."])
