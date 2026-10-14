from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


APP_DIRECTORY = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Return Risk | Order Check",
    page_icon="R",
    layout="centered",
)

model_files = sorted(APP_DIRECTORY.glob("*.joblib"), key=lambda path: path.name.lower())
data_files = sorted(APP_DIRECTORY.glob("*.csv"), key=lambda path: path.name.lower())

if not model_files:
    st.error(f"ไม่พบไฟล์โมเดล .joblib ในโฟลเดอร์: {APP_DIRECTORY}")
    st.stop()

control_column, theme_column = st.columns(2)
with control_column:
    selected_model_path = st.selectbox(
        "เลือกโมเดล",
        options=model_files,
        format_func=lambda path: path.stem,
    )

with theme_column:
    theme = st.selectbox("ธีม", options=["ขาวดำ - สว่าง", "ขาวดำ - มืด"])

if theme == "ขาวดำ - มืด":
    background = "#000000"
    surface = "#111111"
    text_color = "#ffffff"
    muted_color = "#e5e5e5"
    border_color = "#aaaaaa"
    input_background = "#161616"
    button_background = "#ffffff"
    button_text = "#000000"
else:
    background = "#ffffff"
    surface = "#ffffff"
    text_color = "#111111"
    muted_color = "#333333"
    border_color = "#777777"
    input_background = "#ffffff"
    button_background = "#111111"
    button_text = "#ffffff"

st.markdown(
    f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {background};
        color: {text_color};
    }}
    .block-container {{
        max-width: 880px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}
    h1, h2, h3, p, label, [data-testid="stMetricValue"],
    [data-testid="stCaptionContainer"] {{
        color: {text_color} !important;
        font-family: "Sarabun", "Noto Sans Thai", sans-serif;
    }}
    [data-testid="stCaptionContainer"] {{
        color: {muted_color} !important;
    }}
    div[data-testid="stForm"] {{
        border: 1px solid {border_color};
        border-radius: 8px;
        background: {surface};
        padding: 1.25rem 1.4rem;
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {{
        background: {input_background};
        color: {text_color};
        border-color: {border_color};
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {{
        color: {text_color} !important;
    }}
    div.stButton > button[kind="primary"] {{
        background: {button_background};
        border: 1px solid {button_background};
        color: {button_text};
        min-height: 2.8rem;
        font-weight: 650;
    }}
    div.stButton > button[kind="primary"]:hover {{
        opacity: 0.85;
        color: {button_text};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(model_path: str) -> dict:
    return joblib.load(model_path)


@st.cache_data(show_spinner="กำลังอ่านข้อมูล...")
def load_data(data_path: str, modified_at: int) -> pd.DataFrame:
    return pd.read_csv(data_path)


@st.cache_data(show_spinner="กำลังประเมินโมเดล...")
def evaluate_model(
    _bundle: dict,
    data: pd.DataFrame,
    model_name: str | None = None,
    _estimator=None,
    cache_key: str = "",
) -> dict:
    target_name = _bundle.get("target_name", "returned")
    feature_columns = _bundle["feature_columns"]
    missing_columns = [
        column
        for column in [*feature_columns, target_name]
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"ไม่พบคอลัมน์: {', '.join(missing_columns)}")

    features = data[feature_columns]
    target = data[target_name].astype(int)
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    train_features = features.loc[train_indices]
    test_features = features.loc[test_indices]
    train_target = target.loc[train_indices]
    test_target = target.loc[test_indices]

    preprocessor = clone(_bundle["preprocessor"])
    train_transformed = preprocessor.fit_transform(train_features, train_target)
    test_transformed = preprocessor.transform(test_features)
    feature_selector = clone(_bundle["feature_selector"])
    train_selected = feature_selector.fit_transform(train_transformed, train_target)
    test_selected = feature_selector.transform(test_transformed)
    model = clone(_estimator if _estimator is not None else _bundle["model"])
    model.fit(train_selected, train_target)
    predictions = model.predict(test_selected)

    return {
        "โมเดล": model_name or _bundle.get("model_name", "โมเดลไม่ระบุชื่อ"),
        "Accuracy": accuracy_score(test_target, predictions),
        "Precision คืนสินค้า": precision_score(
            test_target, predictions, pos_label=1, zero_division=0
        ),
        "Recall คืนสินค้า": recall_score(
            test_target, predictions, pos_label=1, zero_division=0
        ),
        "F1 คืนสินค้า": f1_score(
            test_target, predictions, pos_label=1, zero_division=0
        ),
    }


st.title("ตรวจความเสี่ยงการคืนสินค้า")
st.caption("กรอกข้อมูลคำสั่งซื้อเพื่อประเมินโอกาสการคืนสินค้าด้วยโมเดลที่ฝึกไว้")

try:
    model_bundle = load_model(str(selected_model_path))
except Exception as error:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {error}")
    st.stop()

prediction_tab, comparison_tab, data_tab = st.tabs(
    ["ประเมินคำสั่งซื้อ", "เปรียบเทียบโมเดล", "ดูข้อมูล"]
)

with prediction_tab:
    st.subheader("ข้อมูลคำสั่งซื้อ")
    with st.form("order_prediction"):
        left_column, right_column = st.columns(2)

        with left_column:
            price = st.number_input(
                "ราคา (บาท)", min_value=1.0, max_value=100000.0, value=1295.0, step=100.0
            )
            delivery_days = st.number_input(
                "ระยะเวลาจัดส่ง (วัน)", min_value=1, max_value=365, value=7, step=1
            )
            rating = st.selectbox("คะแนนรีวิว", options=[1, 2, 3, 4, 5], index=2)

        with right_column:
            discount_percent = st.number_input(
                "ส่วนลด (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0
            )
            category = st.selectbox(
                "ประเภทสินค้า", options=["Fashion", "Home", "Electronics"]
            )
            repeat_customer = st.selectbox("ลูกค้าซื้อซ้ำ", options=["No", "Yes"])

        submitted = st.form_submit_button(
            "ประเมินการคืนสินค้า", type="primary", width="stretch"
        )

    if submitted:
        order = pd.DataFrame(
            [{
                "price": price,
                "delivery_days": delivery_days,
                "rating": rating,
                "discount_percent": discount_percent,
                "category": category,
                "repeat_customer": repeat_customer,
            }]
        )
        order = order[model_bundle["feature_columns"]]

        transformed_order = model_bundle["preprocessor"].transform(order)
        selected_order = model_bundle["feature_selector"].transform(transformed_order)
        model = model_bundle["model"]
        prediction = int(model.predict(selected_order)[0])
        probabilities = model.predict_proba(selected_order)[0]
        returned_index = list(model.classes_).index(1)
        return_probability = float(probabilities[returned_index])

        st.divider()
        st.subheader("ผลการประเมิน")
        result_label = model_bundle["class_labels"].get(
            prediction, "คืนสินค้า" if prediction == 1 else "ไม่คืนสินค้า"
        )

        if prediction == 1:
            st.warning(f"ผลทำนาย: {result_label}")
        else:
            st.success(f"ผลทำนาย: {result_label}")

        metric_column, model_column = st.columns(2)
        metric_column.metric("โอกาสคืนสินค้า", f"{return_probability:.1%}")
        model_column.metric("โมเดล", model_bundle["model_name"])
        st.progress(return_probability, text="ความน่าจะเป็นที่โมเดลประเมิน")
        st.caption("ผลนี้เป็นการประเมินจากข้อมูลและโมเดล ไม่ใช่ข้อสรุปเชิงเหตุและผล")

    st.divider()
    st.caption(f"Model artifact: {selected_model_path.name}")

with comparison_tab:
    st.subheader("ผลบนชุดทดสอบ")
    st.caption("แบ่งข้อมูลแบบ stratified 80/20 ด้วย random state 42 และฝึกแต่ละโมเดลใหม่บน train split")

    if not data_files:
        st.warning("ไม่พบไฟล์ CSV สำหรับใช้ประเมินโมเดล")
    else:
        comparison_data_path = st.selectbox(
            "ชุดข้อมูลสำหรับประเมิน",
            options=data_files,
            format_func=lambda path: path.name,
            key="comparison_data_path",
        )
        comparison_model_paths = st.multiselect(
            "โมเดลที่ต้องการเปรียบเทียบ",
            options=model_files,
            default=model_files,
            format_func=lambda path: path.stem,
        )

        if not comparison_model_paths:
            st.info("เลือกไฟล์โมเดลอย่างน้อยหนึ่งไฟล์")
        else:
            try:
                comparison_data = load_data(
                    str(comparison_data_path), comparison_data_path.stat().st_mtime_ns
                )
                metric_rows = []
                errors = []
                evaluated_model_types = []

                for model_path in comparison_model_paths:
                    try:
                        bundle = load_model(str(model_path))
                        metric_rows.append(
                            evaluate_model(
                                bundle,
                                comparison_data,
                                cache_key=(
                                    f"{model_path}:{model_path.stat().st_mtime_ns}"
                                ),
                            )
                        )
                        evaluated_model_types.append(type(bundle["model"]).__name__)
                    except Exception as error:
                        errors.append(f"{model_path.name}: {error}")

                if (
                    comparison_model_paths
                    and "RandomForestClassifier" not in evaluated_model_types
                ):
                    try:
                        baseline_bundle = load_model(str(comparison_model_paths[0]))
                        random_forest = RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        )
                        metric_rows.append(
                            evaluate_model(
                                baseline_bundle,
                                comparison_data,
                                model_name="Random Forest (baseline)",
                                _estimator=random_forest,
                                cache_key=(
                                    f"{comparison_model_paths[0]}:"
                                    f"{comparison_model_paths[0].stat().st_mtime_ns}"
                                ),
                            )
                        )
                    except Exception as error:
                        errors.append(f"Random Forest baseline: {error}")

                if metric_rows:
                    comparison = pd.DataFrame(metric_rows).set_index("โมเดล")
                    comparison = comparison.sort_values("F1 คืนสินค้า", ascending=False)
                    st.dataframe(
                        comparison.style.format("{:.1%}"),
                        width="stretch",
                    )
                    st.bar_chart(comparison[["F1 คืนสินค้า", "Recall คืนสินค้า"]])
                for error in errors:
                    st.warning(f"ประเมินโมเดลไม่สำเร็จ: {error}")
                if len(metric_rows) == 1:
                    st.info("ประเมินได้เพียงหนึ่งโมเดลจากไฟล์และ baseline ที่เลือก")
            except Exception as error:
                st.error(f"อ่านชุดข้อมูลไม่สำเร็จ: {error}")

with data_tab:
    st.subheader("สำรวจชุดข้อมูล")
    if not data_files:
        st.warning("ไม่พบไฟล์ CSV ในโฟลเดอร์ของแอป")
    else:
        selected_data_path = st.selectbox(
            "เลือกชุดข้อมูล",
            options=data_files,
            format_func=lambda path: path.name,
            key="viewer_data_path",
        )
        try:
            data = load_data(str(selected_data_path), selected_data_path.stat().st_mtime_ns)
            filter_columns = st.columns(3)
            search_text = filter_columns[0].text_input("ค้นหาข้อความ")
            category_filter = "ทั้งหมด"
            if "category" in data.columns:
                categories = sorted(data["category"].dropna().astype(str).unique())
                category_filter = filter_columns[1].selectbox(
                    "ประเภทสินค้า", ["ทั้งหมด", *categories]
                )

            target_column = "returned" if "returned" in data.columns else None
            return_filter = "ทั้งหมด"
            if target_column:
                return_filter = filter_columns[2].selectbox(
                    "สถานะคืนสินค้า", ["ทั้งหมด", "คืนสินค้า", "ไม่คืนสินค้า"]
                )

            filtered_data = data.copy()
            if category_filter != "ทั้งหมด":
                filtered_data = filtered_data[
                    filtered_data["category"].astype(str) == category_filter
                ]
            if return_filter != "ทั้งหมด" and target_column:
                target_value = "1" if return_filter == "คืนสินค้า" else "0"
                filtered_data = filtered_data[
                    filtered_data[target_column].astype(str) == target_value
                ]
            if search_text:
                matches = filtered_data.astype(str).apply(
                    lambda column: column.str.contains(
                        search_text, case=False, regex=False, na=False
                    )
                ).any(axis=1)
                filtered_data = filtered_data[matches]

            summary_columns = st.columns(3)
            summary_columns[0].metric("ข้อมูลทั้งหมด", f"{len(data):,} แถว")
            summary_columns[1].metric("ผลหลังกรอง", f"{len(filtered_data):,} แถว")
            if target_column and len(filtered_data):
                return_rate = pd.to_numeric(
                    filtered_data[target_column], errors="coerce"
                ).mean()
                summary_columns[2].metric("อัตราคืนสินค้า", f"{return_rate:.1%}")

            row_limit = st.number_input(
                "จำนวนแถวที่แสดง",
                min_value=1,
                max_value=max(1, len(filtered_data)),
                value=min(100, max(1, len(filtered_data))),
                step=1,
            )
            st.dataframe(
                filtered_data.head(int(row_limit)),
                width="stretch",
                hide_index=True,
            )
        except Exception as error:
            st.error(f"อ่านชุดข้อมูลไม่สำเร็จ: {error}")