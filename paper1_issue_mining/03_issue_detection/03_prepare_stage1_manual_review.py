"""
Prepare Stage-1 human-review workbook after project reorganization.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config_local import PAPER1_ANALYSIS_ROOT

INPUT_FILE = PAPER1_ANALYSIS_ROOT / "stage1_review_sample_balanced_llm.csv"
OUTPUT_FILE = PAPER1_ANALYSIS_ROOT / "stage1_manual_review_2000.xlsx"

def main():
    print("=" * 70)
    print("Preparing Stage-1 manual-review workbook")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [
        "stage1_id",
        "review_text",
        "llm_is_actionable_issue",
        "llm_evidence",
        "llm_confidence",
        "llm_annotation_status",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    failed = df[df["llm_annotation_status"] != "success"]
    if len(failed) > 0:
        raise RuntimeError(f"{len(failed)} rows have non-success annotations.")

    review_df = pd.DataFrame({
        "stage1_id": df["stage1_id"],
        "review_text": df["review_text"].fillna("").astype(str),
        "llm_is_actionable_issue": pd.to_numeric(
            df["llm_is_actionable_issue"], errors="coerce"
        ).astype("Int64"),
        "llm_evidence": df["llm_evidence"].fillna("").astype(str),
        "llm_confidence": pd.to_numeric(df["llm_confidence"], errors="coerce"),
        "manual_is_actionable_issue": pd.NA,
        "manual_notes": "",
        "manual_reviewed": 0,
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    writer = pd.ExcelWriter(
        OUTPUT_FILE,
        engine="xlsxwriter",
        engine_kwargs={"options": {
            "strings_to_formulas": False,
            "strings_to_urls": False,
        }},
    )

    review_df.to_excel(writer, sheet_name="Manual Review", index=False)

    workbook = writer.book
    worksheet = writer.sheets["Manual Review"]

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })
    text_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top",
        "border": 1,
    })
    locked_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top",
        "border": 1,
        "bg_color": "#F7F7F7",
    })
    editable_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top",
        "border": 1,
        "bg_color": "#FFF2CC",
    })
    center_fmt = workbook.add_format({
        "align": "center",
        "valign": "top",
        "border": 1,
    })
    confidence_fmt = workbook.add_format({
        "num_format": "0.00",
        "align": "center",
        "border": 1,
    })

    for col_num, value in enumerate(review_df.columns):
        worksheet.write(0, col_num, value, header_fmt)

    worksheet.set_column("A:A", 10, center_fmt)
    worksheet.set_column("B:B", 80, text_fmt)
    worksheet.set_column("C:C", 20, center_fmt)
    worksheet.set_column("D:D", 55, locked_fmt)
    worksheet.set_column("E:E", 15, confidence_fmt)
    worksheet.set_column("F:F", 24, editable_fmt)
    worksheet.set_column("G:G", 40, editable_fmt)
    worksheet.set_column("H:H", 18, editable_fmt)

    worksheet.freeze_panes(1, 2)
    worksheet.autofilter(0, 0, len(review_df), len(review_df.columns) - 1)
    worksheet.set_default_row(36)

    last_row = len(review_df)
    worksheet.data_validation(
        1, 5, last_row, 5,
        {
            "validate": "list",
            "source": [0, 1],
            "input_title": "人工判断",
            "input_message": "1=包含 actionable issue；0=不包含。",
            "error_title": "只能填写 0 或 1",
            "error_message": "请选择 0 或 1。",
        }
    )
    worksheet.data_validation(
        1, 7, last_row, 7,
        {
            "validate": "list",
            "source": [0, 1],
            "input_title": "复核状态",
            "input_message": "完成本行复核后改为 1。",
            "error_title": "只能填写 0 或 1",
            "error_message": "请选择 0 或 1。",
        }
    )

    instructions = workbook.add_worksheet("Instructions")
    instructions.set_column("A:A", 18)
    instructions.set_column("B:B", 100)

    title_fmt = workbook.add_format({"bold": True, "font_size": 14})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

    instructions.write("A1", "Stage 1 人工复核说明", title_fmt)

    rows = [
        ("目标", "判断完整 Steam review 中是否至少包含一个 actionable issue。"),
        ("1 = Actionable", "评论包含至少一个具体、可理解、开发者或产品团队可能采取行动的问题、缺陷、不足、明确改进请求或具体关切。"),
        ("0 = No Issue", "只有表扬、一般偏好、泛泛不满、情绪表达、玩笑，或没有足够明确的可行动问题。"),
        ("正面评论", "推荐/正面评论也可能包含 actionable issue。"),
        ("负面评论", "非推荐/负面评论不自动等于 actionable；必须存在具体问题。"),
        ("Feature Request", "明确提出开发者可以理解的功能、选项、模式、支持或改进请求，可以标为 1。"),
        ("人工操作", "先看 review_text，再参考 LLM 标签和 evidence。填写 manual_is_actionable_issue；边界案例写 manual_notes；完成后把 manual_reviewed 改为 1。"),
        ("不确定案例", "如确实难以判断，请在 manual_notes 以 UNCERTAIN: 开头说明，但仍给出当前最佳 0/1 判断。"),
    ]

    for i, (label, text) in enumerate(rows, start=2):
        instructions.write(i, 0, label, header_fmt)
        instructions.write(i, 1, text, wrap_fmt)

    writer.close()

    print(f"Rows: {len(review_df)}")
    print()
    print("LLM label distribution:")
    print(
        review_df["llm_is_actionable_issue"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print(f"Saved:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
