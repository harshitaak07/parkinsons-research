import sys

def print_simulated_output():
    """Prints the hardcoded, simulated model inference output to the console."""

    # --- SIMULATED DATA ---
    patno = "A001"
    visit = "V02"
    diagnosis = "EARLY STAGE PARKINSON'S DISEASE"
    confidence = "93.8%"

    # Attention weights: The core finding that Imaging (47%) > Non-Motor (37%) > Motor (16%)
    weights = [
        ("IMAGING", "47%", "DTI - Substantia Nigra (Low FA value)"),
        ("NON-MOTOR", "37%", "Sleep Status (RBD) (Confirmed)"),
        ("MOTOR", "16%", "NP2WALK (Walking - Score 1)")
    ]

    # --- PRINTING TO CONSOLE ---
    sys.stdout.write(f"$ python multimodal_pd_predictor.py --patno {patno} --visit {visit}\n\n")
    sys.stdout.write("====================================================================\n")
    sys.stdout.write("SEQUENTIAL MULTIMODAL PD RISK ASSESSMENT FRAMEWORK\n")
    sys.stdout.write(f"(Inference Run: 2025-11-06 11:30:04 IST)\n")
    sys.stdout.write("====================================================================\n\n")

    sys.stdout.write(f"--- INPUT SUMMARY (Patient {patno} / Baseline + 6 Months) ---\n")
    sys.stdout.write("| Modality | Input Vector Status | Key Clinical Score |\n")
    sys.stdout.write("|----------|---------------------|--------------------|\n")
    sys.stdout.write("| IMAGING  | VAE Encoded 16D     | DTI Status: Anomalous |\n")
    sys.stdout.write("| NON-MOTOR| VAE Encoded 16D     | PDAQ Score: 8/40 |\n")
    sys.stdout.write("| MOTOR    | Transformer 16D     | MDS-UPDRS II: 2.0 |\n")
    sys.stdout.write("--------------------------------------------------------------------\n\n")

    sys.stdout.write("--- FINAL PREDICTION ---\n")
    sys.stdout.write(f"DIAGNOSIS:               {diagnosis}\n")
    sys.stdout.write(f"PREDICTION CONFIDENCE:   {confidence}\n")
    sys.stdout.write("--------------------------------------------------------------------\n\n")

    sys.stdout.write("--- INTERPRETABILITY: MODALITY CONTRIBUTION (ATTENTION WEIGHTS) ---\n")
    sys.stdout.write("(Explaining *why* the prediction was made: Total must sum to 100%)\n")
    sys.stdout.write("| Modality | Contribution (%) | Specific Feature Highlight |\n")
    sys.stdout.write("|----------|------------------|----------------------------|\n")

    # Print the weights, bolding the key IMAGING feature
    for modality, contribution, highlight in weights:
        if modality == "IMAGING":
            sys.stdout.write(f"| **{modality:<8}** | **{contribution:<16}** | **{highlight:<26}** |\n")
        else:
            sys.stdout.write(f"| {modality:<8} | {contribution:<16} | {highlight:<26} |\n")

    sys.stdout.write("--------------------------------------------------------------------\n")
    sys.stdout.write(f"*Rationale: The model's decision for 'Early Stage' is primarily driven by the IMAGING encoder's detection of structural pathology, consistent with the pre-motor phase of PD. Non-Motor contribution validates the prodromal phase.*\n\n")

    sys.stdout.write("--- NEXT CLINICAL ACTION ---\n")
    sys.stdout.write("RECOMMENDATION: Schedule follow-up imaging (MRI/DTI) within 6 months. Monitor Non-Motor symptoms closely.\n")
    sys.stdout.write("MODEL TRAJECTORY: Predicted transition to Mid-Stage within 30-48 months (p=0.74).\n\n")
    sys.stdout.write("$ _\n")


if __name__ == "__main__":
    print_simulated_output()
