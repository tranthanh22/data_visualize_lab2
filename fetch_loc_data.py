"""
Fetch WDI data từ World Bank API cho phần Lộc:
- Phân tích tương quan đa chỉ số
- Phân cụm quốc gia
- Modeling / dự đoán xu hướng

Phạm vi: 11 nước Đông Nam Á, 2000–2024 (25 năm)
Output: ./datafile/loc_sea_data.csv (long format)
"""

import wbgapi as wb
import pandas as pd
import os

# ── Cấu hình ──────────────────────────────────────────────────────────────────

SEA_COUNTRIES = ["KHM", "VNM", "THA", "MYS", "IDN",
                 "SGP", "PHL", "MMR", "LAO", "BRN", "TLS"]

COUNTRY_NAMES = {
    "KHM": "Cambodia",
    "VNM": "Viet Nam",
    "THA": "Thailand",
    "MYS": "Malaysia",
    "IDN": "Indonesia",
    "SGP": "Singapore",
    "PHL": "Philippines",
    "MMR": "Myanmar",
    "LAO": "Lao PDR",
    "BRN": "Brunei Darussalam",
    "TLS": "Timor-Leste",
}

YEAR_START = 2000
YEAR_END   = 2024

INDICATORS = {
    # Kinh tế
    "NY.GDP.PCAP.KD":       "GDP per capita (constant 2015 US$)",
    "NY.GDP.MKTP.KD.ZG":    "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG":       "Inflation CPI (annual %)",
    "SL.UEM.TOTL.ZS":       "Unemployment rate (%)",
    "BX.KLT.DINV.WD.GD.ZS":"FDI net inflows (% GDP)",
    "NE.EXP.GNFS.ZS":       "Exports (% GDP)",
    "NE.GDI.TOTL.ZS":       "Gross capital formation (% GDP)",
    "NE.CON.GOVT.ZS":       "Gov final consumption (% GDP)",  # thay GC.XPD.TOTL.GD.ZS

    # Xã hội
    "SP.DYN.LE00.IN":       "Life expectancy at birth",
    "SP.URB.TOTL.IN.ZS":    "Urban population (%)",
    "SP.POP.TOTL":          "Total population",
    "SE.XPD.TOTL.GD.ZS":   "Education expenditure (% GDP)",
    "SH.XPD.CHEX.GD.ZS":   "Health expenditure (% GDP)",
    "SP.DYN.IMRT.IN":       "Infant mortality rate",

    # Môi trường & công nghệ
    "EG.FEC.RNEW.ZS":       "Renewable energy (% total final energy)",
    "IT.NET.USER.ZS":       "Internet users (% population)",
}

# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_all():
    frames = []
    total = len(INDICATORS)

    for i, (code, label) in enumerate(INDICATORS.items(), 1):
        print(f"[{i:02d}/{total}] {code} — {label}")
        try:
            df = wb.data.DataFrame(
                code,
                economy=SEA_COUNTRIES,
                time=range(YEAR_START, YEAR_END + 1),
                numericTimeKeys=True,
            )
            # wb trả về: index=economy code, columns=year  →  melt về long
            df = df.reset_index()
            # Dùng cột đầu tiên làm country code (tránh hardcode tên cột)
            economy_col = df.columns[0]
            df = df.melt(
                id_vars=[economy_col],
                var_name="Year",
                value_name="Value",
            )
            df.columns = ["Country Code", "Year", "Value"]
            df["Country Name"] = df["Country Code"].map(COUNTRY_NAMES)
            df["Series Code"] = code
            df["Series Name"] = label
            frames.append(df)
        except Exception as e:
            print(f"  !! Lỗi: {e}")

    if not frames:
        raise RuntimeError("Không fetch được dữ liệu nào.")

    result = pd.concat(frames, ignore_index=True)
    result["Year"] = pd.to_numeric(result["Year"], errors="coerce")
    result["Value"] = pd.to_numeric(result["Value"], errors="coerce")
    result = result[["Country Name", "Country Code", "Series Name", "Series Code", "Year", "Value"]]
    return result


def main():
    os.makedirs("./datafile", exist_ok=True)
    out_path = "./datafile/loc_sea_data.csv"

    print("=== Bắt đầu fetch dữ liệu World Bank ===")
    print(f"  Quốc gia : {len(SEA_COUNTRIES)} nước Đông Nam Á")
    print(f"  Năm      : {YEAR_START} – {YEAR_END}")
    print(f"  Chỉ số   : {len(INDICATORS)} indicators\n")

    df = fetch_all()

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== Xong! Đã lưu {len(df):,} dòng → {out_path} ===")

    # Tóm tắt coverage
    print("\n--- Coverage (% không-NaN theo indicator) ---")
    coverage = (
        df.groupby("Series Name")["Value"]
        .apply(lambda s: round(s.notna().mean() * 100, 1))
        .sort_values()
    )
    print(coverage.to_string())


if __name__ == "__main__":
    main()

