import os

with open('pmsa_bn_poc/kaggle_radiography_single_cell.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Configuration updates
code = code.replace(
    'SOURCE = "Montgomery"     # Montgomery County (MC), USA',
    'SOURCE = "Shenzhen"       # Shenzhen Hospital (SZ), China'
)
code = code.replace(
    'TARGET = "Shenzhen"       # Shenzhen Hospital (SZ), China',
    'TARGET = "Montgomery"     # Montgomery County (MC), USA'
)
code = code.replace('cxr_best_model.pth', 'cxr_sz2mc_best_model.pth')
code = code.replace('cxr_source_stats.pth', 'cxr_sz2mc_source_stats.pth')
code = code.replace(
    'Source : Montgomery County (MC), USA  (Eureka / Del Medical Flat Panel)',
    'Source : Shenzhen Hospital (SZ), China (Philips Digital Radiography)'
)
code = code.replace(
    'Target : Shenzhen Hospital (SZ), China (Philips Digital Radiography)',
    'Target : Montgomery County (MC), USA (Eureka / Del Medical Flat Panel)'
)
code = code.replace(
    'PMSA-BN Chest X-Ray Benchmark Pipeline (Montgomery → Shenzhen)',
    'PMSA-BN Chest X-Ray Reverse Benchmark (Shenzhen → Montgomery)'
)
code = code.replace('fig0_cxr_learning_curves.png', 'fig0_cxr_sz2mc_learning_curves.png')
code = code.replace('fig1_cxr_benchmark_comparison.png', 'fig1_cxr_sz2mc_benchmark_comparison.png')
code = code.replace('fig2_cxr_strip_plots.png', 'fig2_cxr_sz2mc_strip_plots.png')
code = code.replace('fig3_cxr_qualitative_examples.png', 'fig3_cxr_sz2mc_qualitative_examples.png')
code = code.replace('fig4_cxr_ablation.png', 'fig4_cxr_sz2mc_ablation.png')
code = code.replace('cxr_benchmark_results.json', 'cxr_sz2mc_benchmark_results.json')

# Update build_cxr_pairs
old_split = '''    # Deterministic split for Montgomery (Source): 85% train, 15% validation
    rng = random.Random(CFG.SEED)
    rng.shuffle(montgomery_pairs)
    n_val = max(15, int(0.15 * len(montgomery_pairs)))
    mc_val   = montgomery_pairs[:n_val]
    mc_train = montgomery_pairs[n_val:]

    return {
        "mc_train": mc_train,
        "mc_val":   mc_val,
        "sz_test":  shenzhen_pairs,
        "cxr_dir":  cxr_dir,
        "msk_dir":  msk_dir
    }'''

new_split = '''    # Deterministic split for Shenzhen (Source): 85% train, 15% validation
    rng = random.Random(CFG.SEED)
    rng.shuffle(shenzhen_pairs)
    n_val = max(50, int(0.15 * len(shenzhen_pairs)))
    sz_val   = shenzhen_pairs[:n_val]
    sz_train = shenzhen_pairs[n_val:]

    return {
        "mc_train": sz_train,
        "mc_val":   sz_val,
        "sz_test":  montgomery_pairs,
        "cxr_dir":  cxr_dir,
        "msk_dir":  msk_dir
    }'''

assert old_split in code, "Old split block not found in source file!"
code = code.replace(old_split, new_split)

with open('pmsa_bn_poc/kaggle_radiography_sz_to_mc.py', 'w', encoding='utf-8') as f:
    f.write(code)

with open('pmsa_bn_poc/benchmark_radiography_sz_to_mc.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Generated kaggle_radiography_sz_to_mc.py and benchmark_radiography_sz_to_mc.py successfully!")
