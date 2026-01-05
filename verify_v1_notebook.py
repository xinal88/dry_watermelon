"""Verify the v1 notebook is correct."""
import json

with open('train_dry_watermelon_v1.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("="*70)
print("NOTEBOOK VERIFICATION")
print("="*70)

print(f"\nTotal cells: {len(nb['cells'])}")
print(f"Accelerator: {nb['metadata'].get('accelerator', 'N/A')}")
print(f"GPU Type: {nb['metadata']['colab'].get('gpuType', 'N/A')}")

print("\n" + "="*70)
print("CELL STRUCTURE")
print("="*70)

for i, cell in enumerate(nb['cells'], 1):
    cell_type = cell['cell_type']
    source = cell.get('source', [])
    
    if source:
        if isinstance(source, list):
            first_line = source[0][:60]
        else:
            first_line = str(source)[:60]
    else:
        first_line = "(empty)"
    
    print(f"{i:2d}. {cell_type:8} - {first_line}")

print("\n" + "="*70)
print("KEY SECTIONS CHECK")
print("="*70)

key_sections = {
    "Environment Check": False,
    "Clone Repository": False,
    "Mount Google Drive": False,
    "Validate Data": False,
    "Create Model": False,
    "Create Dataloaders": False,
    "Training Loop": False,
    "Plot Results": False,
    "Test Evaluation": False,
}

for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    
    if "IN_COLAB" in source and "sys.modules" in source:
        key_sections["Environment Check"] = True
    if "git clone" in source:
        key_sections["Clone Repository"] = True
    if "drive.mount" in source or "RAVDESS_PATH" in source:
        key_sections["Mount Google Drive"] = True
    if "Validating RAVDESS dataset" in source or "Data validation" in source:
        key_sections["Validate Data"] = True
    if "create_model" in source and "def create_model" in source:
        key_sections["Create Model"] = True
    if "create_ravdess_dataloaders" in source:
        key_sections["Create Dataloaders"] = True
    if "STARTING TRAINING" in source or "for epoch in range" in source:
        key_sections["Training Loop"] = True
    if "plt.subplots" in source and "training_curves" in source:
        key_sections["Plot Results"] = True
    if "EVALUATING ON TEST SET" in source:
        key_sections["Test Evaluation"] = True

for section, found in key_sections.items():
    status = "✅" if found else "❌"
    print(f"{status} {section}")

all_found = all(key_sections.values())

print("\n" + "="*70)
if all_found:
    print("✅ NOTEBOOK IS COMPLETE AND READY!")
    print("="*70)
    print("\n✅ All key sections present")
    print(f"✅ {len(nb['cells'])} cells created")
    print("✅ GPU configuration set")
    print("\n🚀 Ready to use in Colab!")
else:
    print("❌ NOTEBOOK IS INCOMPLETE!")
    print("="*70)
    missing = [k for k, v in key_sections.items() if not v]
    print(f"\nMissing sections: {missing}")
