"""
generate_data.jl
----------------
Generates synthetic-but-structured binary molecular fingerprints for ~2000
FDA-approved drugs and trains a pre-trained RBM for CHEME 5820 PS4.

Run from the project root:
    julia scripts/generate_data.jl

Outputs:
    data/fda_drugs_fingerprints.jld2   - fingerprints (256 × N Float32) + drug names
    data/fda_drugs_metadata.csv        - drug names + drug classes
    data/pretrained_rbm_drugs.jld2     - pre-trained RBM (256 visible → 512 hidden)

Design of synthetic fingerprints:
    Each of the 20 drug classes gets:
      - 12 "core" bits (p_on = 0.95) — non-overlapping across classes
      - 8 "associated" bits from a shared pool (p_on = 0.75)
      - Background: all other bits have p_on = 0.0 (no noise)
    This yields ~17-18 bits set per drug.  Within-class Tanimoto ceiling ~0.79,
    which allows the RBM reconstruction task to reach Tanimoto >= 0.7.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, CSV, DataFrames, Random, Statistics, LinearAlgebra
using Distributions: Categorical
using VLDataScienceMachineLearningPackage
import StatsBase

Random.seed!(2025)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
const N_BITS         = 256
const N_PER_CLASS    = 100   # drugs per class  → 20 × 100 = 2000 total
const N_CORE_BITS    = 12    # per class, non-overlapping (uses bits 1:240)
const N_ASSOC_BITS   = 8     # per class, from shared pool bits 241:256
const P_CORE         = 0.95
const P_ASSOC        = 0.75
const P_BACKGROUND   = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Drug class definitions and known drug names
# ──────────────────────────────────────────────────────────────────────────────
drug_classes = [
    "Beta-lactam Antibiotics",
    "Macrolide Antibiotics",
    "Fluoroquinolone Antibiotics",
    "Tetracycline Antibiotics",
    "Aminoglycoside Antibiotics",
    "Statins",
    "ACE Inhibitors",
    "Beta-blockers",
    "Calcium Channel Blockers",
    "Angiotensin Receptor Blockers",
    "SSRIs",
    "SNRIs",
    "Antipsychotics",
    "Benzodiazepines",
    "Opioids",
    "NSAIDs",
    "Proton Pump Inhibitors",
    "Corticosteroids",
    "Antivirals",
    "Immunosuppressants"
]

# Known real drug names for each class (the rest are auto-numbered)
known_drugs = [
    # 1. Beta-lactam Antibiotics
    ["Amoxicillin","Penicillin G","Penicillin V","Ampicillin","Nafcillin","Oxacillin",
     "Dicloxacillin","Cloxacillin","Piperacillin","Ticarcillin","Cephalexin","Cefaclor",
     "Cefuroxime","Cefprozil","Cefdinir","Cefpodoxime","Ceftriaxone","Cefotaxime",
     "Ceftazidime","Cefepime","Cefazolin","Cefoxitin","Cefotetan","Meropenem","Imipenem",
     "Ertapenem","Doripenem","Aztreonam","Amoxicillin-Clavulanate","Piperacillin-Tazobactam"],
    # 2. Macrolide Antibiotics
    ["Azithromycin","Clarithromycin","Erythromycin","Telithromycin","Fidaxomicin",
     "Roxithromycin","Spiramycin","Josamycin","Dirithromycin","Troleandomycin"],
    # 3. Fluoroquinolone Antibiotics
    ["Ciprofloxacin","Levofloxacin","Moxifloxacin","Gemifloxacin","Ofloxacin",
     "Norfloxacin","Delafloxacin","Trovafloxacin","Grepafloxacin","Sparfloxacin",
     "Enrofloxacin","Besifloxacin"],
    # 4. Tetracycline Antibiotics
    ["Doxycycline","Tetracycline","Minocycline","Tigecycline","Sarecycline",
     "Omadacycline","Chlortetracycline","Oxytetracycline","Demeclocycline","Meclocycline"],
    # 5. Aminoglycoside Antibiotics
    ["Gentamicin","Tobramycin","Amikacin","Neomycin","Streptomycin",
     "Spectinomycin","Plazomicin","Kanamycin","Netilmicin","Paromomycin"],
    # 6. Statins
    ["Atorvastatin","Rosuvastatin","Simvastatin","Pravastatin","Lovastatin",
     "Fluvastatin","Pitavastatin","Cerivastatin","Mevastatin","Compactin"],
    # 7. ACE Inhibitors
    ["Lisinopril","Enalapril","Ramipril","Captopril","Benazepril","Fosinopril",
     "Quinapril","Perindopril","Trandolapril","Moexipril","Cilazapril","Spirapril"],
    # 8. Beta-blockers
    ["Metoprolol","Atenolol","Propranolol","Carvedilol","Nebivolol","Bisoprolol",
     "Labetalol","Nadolol","Acebutolol","Pindolol","Timolol","Sotalol","Betaxolol"],
    # 9. Calcium Channel Blockers
    ["Amlodipine","Nifedipine","Diltiazem","Verapamil","Felodipine","Nisoldipine",
     "Isradipine","Nicardipine","Clevidipine","Nimodipine","Nitrendipine","Lacidipine"],
    # 10. Angiotensin Receptor Blockers
    ["Losartan","Valsartan","Irbesartan","Candesartan","Olmesartan","Telmisartan",
     "Eprosartan","Azilsartan","Fimasartan","Saprisartan"],
    # 11. SSRIs
    ["Fluoxetine","Sertraline","Paroxetine","Citalopram","Escitalopram","Fluvoxamine",
     "Dapoxetine","Indalpine","Zimelidine","Alaproclate"],
    # 12. SNRIs
    ["Venlafaxine","Duloxetine","Desvenlafaxine","Levomilnacipran","Milnacipran",
     "Sibutramine","Bicifadine","Atomoxetine","Reboxetine","Nefazodone"],
    # 13. Antipsychotics
    ["Haloperidol","Risperidone","Olanzapine","Quetiapine","Aripiprazole","Ziprasidone",
     "Clozapine","Lurasidone","Paliperidone","Iloperidone","Asenapine","Cariprazine",
     "Brexpiprazole","Pimavanserin","Chlorpromazine","Fluphenazine"],
    # 14. Benzodiazepines
    ["Diazepam","Lorazepam","Clonazepam","Alprazolam","Midazolam","Triazolam",
     "Oxazepam","Temazepam","Chlordiazepoxide","Clorazepate","Flurazepam","Nitrazepam",
     "Bromazepam","Clobazam"],
    # 15. Opioids
    ["Morphine","Oxycodone","Hydrocodone","Codeine","Fentanyl","Hydromorphone",
     "Methadone","Buprenorphine","Tramadol","Oxymorphone","Tapentadol","Meperidine",
     "Alfentanil","Sufentanil","Remifentanil"],
    # 16. NSAIDs
    ["Ibuprofen","Naproxen","Celecoxib","Meloxicam","Indomethacin","Ketorolac",
     "Diclofenac","Piroxicam","Sulindac","Etodolac","Flurbiprofen","Mefenamic Acid",
     "Oxaprozin","Diflunisal","Fenoprofen"],
    # 17. Proton Pump Inhibitors
    ["Omeprazole","Lansoprazole","Pantoprazole","Rabeprazole","Esomeprazole",
     "Dexlansoprazole","Ilaprazole","Vonoprazan","Tenatoprazole","Timoprazole"],
    # 18. Corticosteroids
    ["Prednisone","Prednisolone","Dexamethasone","Methylprednisolone","Triamcinolone",
     "Hydrocortisone","Budesonide","Beclomethasone","Fluticasone","Mometasone",
     "Betamethasone","Fludrocortisone","Fluocinolone","Desoximetasone"],
    # 19. Antivirals
    ["Oseltamivir","Zanamivir","Acyclovir","Valacyclovir","Ganciclovir","Tenofovir",
     "Emtricitabine","Efavirenz","Atazanavir","Ritonavir","Darunavir","Raltegravir",
     "Baloxavir","Ribavirin","Sofosbuvir","Ledipasvir","Peramivir","Favipiravir",
     "Remdesivir","Molnupiravir"],
    # 20. Immunosuppressants
    ["Cyclosporine","Tacrolimus","Mycophenolate","Sirolimus","Azathioprine","Methotrexate",
     "Leflunomide","Hydroxychloroquine","Abatacept","Belimumab","Everolimus","Basiliximab",
     "Daclizumab","Natalizumab","Alemtuzumab"],
]

N_CLASSES = length(drug_classes)

# ──────────────────────────────────────────────────────────────────────────────
# Build drug name list
# ──────────────────────────────────────────────────────────────────────────────
all_drug_names  = String[]
all_drug_classes = String[]

for (ci, classname) in enumerate(drug_classes)
    prefix = replace(first(split(classname, " ")), "-" => "")  # e.g. "Betalactam"
    known  = known_drugs[ci]
    for k in 1:N_PER_CLASS
        name = k <= length(known) ? known[k] : "$(prefix)_$(lpad(k, 3, '0'))"
        push!(all_drug_names, name)
        push!(all_drug_classes, classname)
    end
end

N_TOTAL = length(all_drug_names)
println("Total drugs: $(N_TOTAL)")

# ──────────────────────────────────────────────────────────────────────────────
# Assign characteristic bits to each drug class
# ──────────────────────────────────────────────────────────────────────────────
# Core bits: 12 non-overlapping blocks from bits 1:240 (20 × 12 = 240)
core_bits = [((ci-1)*N_CORE_BITS+1):(ci*N_CORE_BITS) for ci in 1:N_CLASSES]

# Associated bits: 8 drawn from shared pool 241:256 (16 bits total)
assoc_pool = collect(241:N_BITS)
assoc_bits = [StatsBase.sample(assoc_pool, N_ASSOC_BITS, replace=false) for _ in 1:N_CLASSES]

# ──────────────────────────────────────────────────────────────────────────────
# Generate fingerprints
# ──────────────────────────────────────────────────────────────────────────────
fingerprints = zeros(Float32, N_BITS, N_TOTAL)

for drug_idx in 1:N_TOTAL
    ci = div(drug_idx - 1, N_PER_CLASS) + 1   # class index (1-based)

    # Background bits
    for bit in 1:N_BITS
        fingerprints[bit, drug_idx] = rand() < P_BACKGROUND ? 1.0f0 : 0.0f0
    end

    # Core bits (high probability) — may overwrite background
    for bit in core_bits[ci]
        fingerprints[bit, drug_idx] = rand() < P_CORE ? 1.0f0 : 0.0f0
    end

    # Associated bits (moderate probability)
    for bit in assoc_bits[ci]
        fingerprints[bit, drug_idx] = rand() < P_ASSOC ? 1.0f0 : 0.0f0
    end
end

bits_per_drug = vec(sum(fingerprints, dims=1))
println("Mean bits set per drug: $(round(mean(bits_per_drug), digits=1))  " *
        "(min=$(Int(minimum(bits_per_drug))), max=$(Int(maximum(bits_per_drug))))")

# ──────────────────────────────────────────────────────────────────────────────
# Save fingerprints + metadata
# ──────────────────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "..", "data")
mkpath(data_dir)

jldsave(joinpath(data_dir, "fda_drugs_fingerprints.jld2");
        fingerprints = fingerprints,
        drug_names   = all_drug_names)
println("Saved → data/fda_drugs_fingerprints.jld2  ($(size(fingerprints)))")

df_meta = DataFrame(name = all_drug_names, drug_class = all_drug_classes)
CSV.write(joinpath(data_dir, "fda_drugs_metadata.csv"), df_meta)
println("Saved → data/fda_drugs_metadata.csv  ($(nrow(df_meta)) rows)")

# ──────────────────────────────────────────────────────────────────────────────
# Train pre-trained RBM  (256 → 512, CD-1)
# ──────────────────────────────────────────────────────────────────────────────
println("\nTraining pre-trained RBM (256 → 512, CD-1)...")

n_visible  = N_BITS
n_hidden   = 512

# Convert fingerprints from {0,1} to {-1,+1}
fp_pm1 = Int64.(2 .* fingerprints .- 1)

# Initialise RBM
rbm_pretrained = build(MyRestrictedBoltzmannMachineModel, (
    W = 0.01 * randn(n_visible, n_hidden),
    b = zeros(n_hidden),
    a = zeros(n_visible)
))

p_all = Categorical(N_TOTAL)   # uniform over training set

rbm_pretrained = learn(rbm_pretrained, fp_pm1, p_all;
    maxnumberofiterations = 5000,
    T          = 2,      # T=2 → CD-1 (one Gibbs step from data state)
    β          = 1.0,
    batchsize  = 32,
    η          = 0.01,
    tol        = 1e-10,  # rely on maxnumberofiterations
    verbose    = true)

println("Training complete!")

jldsave(joinpath(data_dir, "pretrained_rbm_drugs.jld2");
        W         = rbm_pretrained.W,
        b         = rbm_pretrained.b,
        a         = rbm_pretrained.a,
        n_visible = n_visible,
        n_hidden  = n_hidden)
println("Saved → data/pretrained_rbm_drugs.jld2  ($(n_visible)→$(n_hidden))")
println("\nAll data files generated successfully.")
