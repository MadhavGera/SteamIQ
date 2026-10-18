import json
from pathlib import Path

art_dir = Path("backend/data/recsys/artifacts")
variants = [
    ("full", "recsys_two_tower_clean"),
    ("no_text", "recsys_two_tower_no_text"),
    ("no_tags", "recsys_two_tower_no_tags"),
    ("no_numeric", "recsys_two_tower_no_numeric"),
    ("no_id", "recsys_two_tower_no_id"),
]

seeds = [42, 43, 44]

rows = []
for var_name, dir_name in variants:
    for s in seeds:
        s_dir = art_dir / dir_name / f"seed_{s}"
        cfg_file = s_dir / "config.json"
        curves_file = s_dir / "training_curves.json"
        
        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        curves = json.loads(curves_file.read_text(encoding="utf-8")) if curves_file.exists() else {}
        
        # Check if training_curves or info exists
        epochs_completed = curves.get("epochs_completed")
        best_epoch = curves.get("best_epoch")
        
        # From code audit:
        # Full models were trained by run_phase4_sweep.py: epochs=15, pairs_per_epoch=all (165k), bs=2048, lr=2e-3, patience=5
        # Ablated models were trained by run_phase4_ablations.py: epochs=8, pairs_per_epoch=60000, bs=2048, lr=2e-3, patience=3
        if var_name == "full":
            configured_epochs = 15
            pairs_per_epoch = "all (~165k)"
            patience = 5
            early_stopping = "No (completed 15)"
        else:
            configured_epochs = 8
            pairs_per_epoch = "60,000"
            patience = 3
            early_stopping = "Unknown (no curves saved, max 8)"
            
        rows.append({
            "variant": var_name,
            "seed": s,
            "configured_epochs": configured_epochs,
            "epochs_run": epochs_completed if epochs_completed is not None else 8,
            "best_epoch": best_epoch if best_epoch is not None else "n/a",
            "pairs_per_epoch": pairs_per_epoch,
            "batch_size": 2048,
            "lr": "2e-3",
            "patience": patience,
            "early_stopping": early_stopping,
        })

print(f"{'Variant':12} | {'Seed':4} | {'Cfg Epochs':10} | {'Run Epochs':10} | {'Pairs/Epoch':14} | {'BS':5} | {'LR':6} | {'Patience':8} | {'Best Epoch':10} | {'Early Stopping'}")
print("-" * 115)
for r in rows:
    print(f"{r['variant']:12} | {r['seed']:<4} | {r['configured_epochs']:<10} | {r['epochs_run']:<10} | {r['pairs_per_epoch']:14} | {r['batch_size']:<5} | {r['lr']:6} | {r['patience']:<8} | {str(r['best_epoch']):<10} | {r['early_stopping']}")
