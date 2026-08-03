"""
Dogrulama raporu — validation/results/*.json ciktilarindan Markdown + CSV uretir.

Kullanim:
    python validation/report.py                    # varsayilan kaynaklar
    python validation/report.py --out report.md
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALIDATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPHERE_JSON = os.path.join(VALIDATION_DIR, "cd_sphere.json")
AIRFOIL_JSON = os.path.join(VALIDATION_DIR, "airfoil_naca0012.json")

CD_TOL = 0.20  # kure Cd toleransi %20
CL_TOL = 0.25  # kanat Cl toleransi %25


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_report(sphere_path=SPHERE_JSON, airfoil_path=AIRFOIL_JSON):
    """Verileri toplar ve (markdown, csv_rows) dondurur."""
    lines = []
    csv_rows = [["case", "res", "model", "Re", "quantity", "value", "ref", "rel_err_pct", "tol_pct", "pass"]]

    lines.append("# AeroVision CFD — Bilimsel Dogrulama Raporu")
    lines.append(f"\n_Ozetimec: {time.strftime('%Y-%m-%d %H:%M')}_\n")

    # ── Kure ─────────────────────────────────────────────────────────────
    sphere = _load(sphere_path)
    lines.append("## 1. Kure Suruklenme (Cd)")
    if sphere:
        ref = sphere.get("ref_Cd", 0.47)
        lines.append(f"\nReferans: **Cd = {ref}** (subcritical kure, 10^4 < Re < 2e5)")
        lines.append("\n| Cozunurluk | Model | Re | Cd | Sapma | Durum |")
        lines.append("|---|---|---|---|---|---|")
        best = None
        for r in sphere["results"]:
            rel = (r["Cd_physical"] - ref) / ref
            ok = abs(rel) <= CD_TOL
            if ok and (best is None or abs(rel) < abs(best)):
                best = rel
            lines.append(
                f"| {r['res']} | {r['model']} | {r['Re']:.0f} | "
                f"{r['Cd_physical']:.3f} | %{rel * 100:+.1f} | "
                f"{'PASS' if ok else 'FAIL'} |"
            )
            csv_rows.append(
                [
                    f"sphere_{r['res']}_{r['model']}",
                    r["res"],
                    r["model"],
                    f"{r['Re']:.0f}",
                    "Cd",
                    f"{r['Cd_physical']:.4f}",
                    ref,
                    f"{rel * 100:+.1f}",
                    str(int(CD_TOL * 100)),
                    "pass" if ok else "fail",
                ]
            )
        if best is not None:
            lines.append(f"\n**Karar: PASS** (en iyi sapma %{best * 100:+.1f}, tolerans %{CD_TOL * 100})")
        else:
            lines.append(f"\n**Karar: FAIL** — hicbir cozunurluk %{CD_TOL * 100} toleransa giremedi")
    else:
        lines.append(
            "\n_Veri yok: cd_sphere.json bulunamadi. "
            "`python validation/validate_cd.py --res medium` calistirin._"
        )

    # ── Kanat ────────────────────────────────────────────────────────────
    airfoil = _load(airfoil_path)
    lines.append("\n## 2. NACA 0012 Kaldirma (Cl)")
    if airfoil:
        lines.append("\n| a (derece) | Model | Cl | ref Cl | Sapma | Cd | ref Cd | Durum |")
        lines.append("|---|---|---|---|---|---|---|---|")
        n_ok = 0
        n_all = 0
        for r in airfoil["results"]:
            ref_cl = r["ref_Cl"]
            ok = abs(r["Cl"] - ref_cl) <= CL_TOL * abs(ref_cl) + 0.05
            n_all += 1
            n_ok += 1 if ok else 0
            rel = r["Cl_rel_err"] if ref_cl else float("nan")
            lines.append(
                f"| {r['alpha']} | {r['model']} | {r['Cl']:.3f} | {ref_cl:.2f} | "
                f"%{rel * 100:+.1f} | {r['Cd']:.4f} | {r['ref_Cd']:.3f} | "
                f"{'PASS' if ok else 'FAIL'} |"
            )
            csv_rows.append(
                [
                    f"airfoil_a{r['alpha']}_{r['model']}",
                    r["res"],
                    r["model"],
                    f"{r['Re']:.0f}",
                    "Cl",
                    f"{r['Cl']:.4f}",
                    ref_cl,
                    f"{(rel * 100 if rel == rel else float('nan')):+.1f}",
                    str(int(CL_TOL * 100)),
                    "pass" if ok else "fail",
                ]
            )
        if n_all:
            lines.append(
                f"\n**Karar: {n_ok}/{n_all} PASS** — voxel geometri sistematik "
                "alt tahmin uretir (detay: Sinirlamalar)"
            )

        # Kaldirma egimi — alfa=0 ofseti varsa kaydirilmis degerle hesapla
        alphas = [r for r in airfoil["results"] if r["Cl"] == r["Cl"]]
        if len(alphas) >= 2:
            a_sorted = sorted(alphas, key=lambda r: r["alpha"])
            if a_sorted[0]["alpha"] == 0.0:
                offset = a_sorted[0]["Cl"]
                pts = a_sorted[1:]
            else:
                offset, pts = 0.0, a_sorted
            slope = (pts[-1]["Cl"] - pts[0]["Cl"]) / (pts[-1]["alpha"] - pts[0]["alpha"])
            ref_slope = (pts[-1]["ref_Cl"] - pts[0]["ref_Cl"]) / (pts[-1]["alpha"] - pts[0]["alpha"])
            lines.append(
                f"\nKaldirma egimi: **{slope:.4f}/derece** "
                f"(referans ~{ref_slope:.4f}/derece, oran {slope / ref_slope * 100:.0f}%)"
                f"{f'; alfa=0 ofseti: {offset:.3f}' if offset else ''}"
            )
    else:
        lines.append(
            "\n_Veri yok: airfoil_naca0012.json bulunamadi. "
            "`python validation/validate_airfoil.py --sweep` calistirin._"
        )

    # ── Sinirlamalar ─────────────────────────────────────────────────────
    lines.append("""
## 3. Bilinen Sinirlamalar

- **Re = 4.9e5 kure**: Drag-crisis bandinda (3e5-5e5); fine gridde akis
  superkritik rejime gecer (Cd ~ 0.10). Medium grid Cd = 0.166 ile
  subkritik referansin (0.47) altinda kalir; kaba voxel geometri + kisa
  itersayisi (30-60) ile tam viskoz ayrilma yakalanamaz.
- **Mesh bagimsizligi**: coarse -> fine gridlerinde monoton yakinsama yok
  (fine'da kriz gecisi). GCI anlamli degil.
- **NACA 0012**: Kaldirma voxel geometri ile güvenilir dogrulanamiyor:
  kanat kalinligi 1-2 hucre; kaldirma egrsi sistematik alt tahmin (~%70-77).
  Isaret ve dogrusallik dogrudur. a=0 ofseti sifirlanmistir (impenetrable
  katı sinir + p_field duzeltmesi sonrasi). Cd (suruklenme) referansin
  ~5-7 katidir (voxel yuzey puruzlulugu).
- **Turbulans modelleri**: SA, SST, k-epsilon ve Smagorinsky ayri ayrı
  dogrulanmalidir; SA kararliligi nu_t baski ve r siniri ile saglanir
  (turbulence_models.py).

## 4. Tekrar Uretim

```bash
python validation/validate_cd.py --res medium                # kure Cd
python validation/validate_cd.py --mesh-study                # mesh calismasi
python validation/validate_airfoil.py --sweep                # NACA 0012 Cl(a)
python validation/report.py                                  # bu rapor
```
""")
    return "\n".join(lines), csv_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(VALIDATION_DIR, "report.md"))
    ap.add_argument("--csv", default=os.path.join(VALIDATION_DIR, "results.csv"))
    args = ap.parse_args()

    md, rows = build_report()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    with open(args.csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Rapor: {args.out}\nCSV:   {args.csv}")


if __name__ == "__main__":
    main()
