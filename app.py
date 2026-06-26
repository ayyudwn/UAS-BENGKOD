from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "customer_churn_deployment.joblib"


# ============================================================
# MEMUAT MODEL DAN KOMPONEN DEPLOYMENT
# ============================================================

@st.cache_resource
def load_deployment_artifact(model_path: Path) -> dict[str, Any]:
    """Memuat model dan seluruh komponen preprocessing dari file joblib."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"File {model_path.name} tidak ditemukan. "
            "Pastikan customer_churn_deployment.joblib berada "
            "di folder yang sama dengan app.py."
        )

    loaded_artifact = joblib.load(model_path)

    if not isinstance(loaded_artifact, dict):
        raise TypeError(
            "Isi customer_churn_deployment.joblib harus berupa dictionary."
        )

    required_keys = {
        "model",
        "preprocessor",
        "iqr_bounds",
        "input_feature_columns",
        "processed_feature_columns",
    }

    missing_keys = sorted(required_keys.difference(loaded_artifact.keys()))

    if missing_keys:
        raise KeyError(
            "Komponen berikut tidak ditemukan di dalam file model: "
            + ", ".join(missing_keys)
        )

    return loaded_artifact


try:
    artifact = load_deployment_artifact(MODEL_PATH)
except Exception as error:
    st.error("Aplikasi belum dapat memuat model.")
    st.code(str(error))
    st.info(
        "Taruh customer_churn_deployment.joblib di folder yang sama "
        "dengan app.py, lalu jalankan ulang aplikasi."
    )
    st.stop()


model = artifact["model"]
preprocessor = artifact["preprocessor"]
iqr_bounds = artifact["iqr_bounds"]

input_feature_columns = list(
    artifact["input_feature_columns"]
)

processed_feature_columns = list(
    artifact["processed_feature_columns"]
)

reference_date = artifact.get("reference_date")
target_column = artifact.get("target_column", "churn")
model_name = artifact.get(
    "model_name",
    type(model).__name__,
)

dropped_low_impact_features = artifact.get(
    "dropped_low_impact_features",
    [],
)

imputation_strategy = artifact.get(
    "imputation_strategy",
    None,
)

smote_information = artifact.get(
    "smote_information",
    {},
)

evaluation = artifact.get(
    "evaluation",
    {},
)


# ============================================================
# FUNGSI PENDUKUNG PREPROCESSING
# ============================================================

def get_transformer_columns(transformer_name: str) -> list[str]:
    """Mengambil daftar kolom dari transformer yang sudah di-fit."""
    for name, _, columns in preprocessor.transformers_:
        if name != transformer_name:
            continue

        if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            return list(columns)

        return []

    return []


def get_category_options() -> dict[str, list[Any]]:
    """Mengambil kategori dari OneHotEncoder hasil training."""
    categorical_columns = get_transformer_columns("cat")

    try:
        encoder = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["encoder"]
        )

        categories = encoder.categories_

        return {
            column: [
                value.item() if hasattr(value, "item") else value
                for value in values
            ]
            for column, values in zip(
                categorical_columns,
                categories,
            )
        }
    except (KeyError, AttributeError):
        return {}


def get_numeric_defaults() -> dict[str, float]:
    """
    Mengambil nilai imputasi dari transformer numerik.

    Notebook final menggunakan dua transformer:
    - num_mean untuk fitur dengan distribusi relatif simetris
    - num_median untuk fitur dengan distribusi lebih miring
    """
    defaults: dict[str, float] = {}

    for transformer_name in ["num_mean", "num_median"]:
        try:
            columns = get_transformer_columns(
                transformer_name
            )

            statistics = (
                preprocessor
                .named_transformers_[transformer_name]
                .named_steps["imputer"]
                .statistics_
            )

            for column, value in zip(columns, statistics):
                if pd.notna(value):
                    defaults[column] = float(value)

        except (KeyError, AttributeError):
            continue

    return defaults


category_options = get_category_options()
numeric_defaults = get_numeric_defaults()


FALLBACK_CATEGORIES = {
    "gender": ["Female", "Male", "Other"],
    "country": ["Bangladesh", "Germany", "India", "UK", "USA"],
    "city": [
        "Berlin",
        "Delhi",
        "Dhaka",
        "Hamburg",
        "London",
        "Mumbai",
        "New York",
    ],
    "acquisition_channel": [
        "Email",
        "Facebook Ads",
        "Google Ads",
        "Organic",
        "Referral",
    ],
    "device_type": [
        "Desktop",
        "Mobile",
        "Tablet",
    ],
    "subscription_type": [
        "Annual",
        "Monthly",
    ],
    "payment_method": [
        "BKash",
        "Card",
        "PayPal",
        "SEPA",
        "UPI",
    ],
}


def choices_for(column: str) -> list[Any]:
    """Mengambil pilihan kategori dari encoder atau nilai fallback."""
    choices = category_options.get(
        column,
        FALLBACK_CATEGORIES.get(column, []),
    )

    return list(choices)


def numeric_default(column: str, fallback: float) -> float:
    """Mengambil nilai default dari hasil imputasi data training."""
    value = numeric_defaults.get(column, fallback)

    if pd.isna(value):
        return float(fallback)

    return float(value)


def apply_iqr_capping(
    data: pd.DataFrame,
    bounds: dict[str, Any],
) -> pd.DataFrame:
    """Menerapkan batas IQR yang dihitung dari data train."""
    capped_data = data.copy()

    for column, limits in bounds.items():
        if column not in capped_data.columns:
            continue

        if not isinstance(limits, dict):
            continue

        lower = limits.get("lower")
        upper = limits.get("upper")

        capped_data[column] = pd.to_numeric(
            capped_data[column],
            errors="coerce",
        )

        capped_data[column] = capped_data[column].clip(
            lower=lower,
            upper=upper,
        )

    return capped_data


def prepare_input(
    raw_input: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menyamakan input aplikasi dengan alur pada notebook final:
    feature engineering, IQR capping, imputasi, encoding, dan scaling.
    """
    input_copy = raw_input.copy()

    signup_date = pd.to_datetime(
        input_copy.pop("signup_date"),
        errors="coerce",
    )

    last_purchase_date = pd.to_datetime(
        input_copy.pop("last_purchase_date"),
        errors="coerce",
    )

    coupon_used = bool(
        input_copy.pop("coupon_used")
    )

    if pd.isna(signup_date) or pd.isna(last_purchase_date):
        raise ValueError(
            "Tanggal pendaftaran dan pembelian terakhir harus valid."
        )

    saved_reference_date = pd.to_datetime(
        reference_date,
        errors="coerce",
    )

    if pd.isna(saved_reference_date):
        saved_reference_date = last_purchase_date

    # Sesuai notebook final, selisih absolut digunakan karena
    # terdapat beberapa urutan tanggal yang tidak konsisten.
    customer_tenure_days = abs(
        int((last_purchase_date - signup_date).days)
    )

    days_since_last_purchase = int(
        (saved_reference_date - last_purchase_date).days
    )

    engineered_input = {
        **input_copy,
        "has_coupon_code": int(coupon_used),
        "customer_tenure_days": customer_tenure_days,
        "days_since_last_purchase": days_since_last_purchase,
    }

    raw_dataframe = pd.DataFrame(
        [engineered_input]
    )

    missing_columns = [
        column
        for column in input_feature_columns
        if column not in raw_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Fitur berikut belum tersedia pada input aplikasi: "
            + ", ".join(missing_columns)
        )

    # Mengikuti urutan fitur yang dipakai saat training.
    model_input = raw_dataframe.reindex(
        columns=input_feature_columns
    )

    # Penanganan outlier menggunakan batas dari data train.
    capped_input = apply_iqr_capping(
        model_input,
        iqr_bounds,
    )

    # Mean/median, modus, OneHotEncoder, dan StandardScaler.
    transformed_input = preprocessor.transform(
        capped_input
    )

    if hasattr(model, "feature_names_in_"):
        final_columns = list(
            model.feature_names_in_
        )
    else:
        final_columns = processed_feature_columns

    if len(final_columns) != transformed_input.shape[1]:
        raise ValueError(
            "Jumlah fitur hasil preprocessing tidak sesuai dengan model. "
            "Buat ulang customer_churn_deployment.joblib menggunakan "
            "notebook final terbaru."
        )

    processed_dataframe = pd.DataFrame(
        transformed_input,
        columns=final_columns,
    )

    return model_input, processed_dataframe


def predict_churn(
    processed_input: pd.DataFrame,
) -> tuple[int, float | None]:
    """Menghasilkan kelas prediksi dan probabilitas churn."""
    prediction = int(
        model.predict(processed_input)[0]
    )

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            processed_input
        )[0]

        classes = list(model.classes_)

        if 1 in classes:
            churn_index = classes.index(1)
            probability = float(
                probabilities[churn_index]
            )

    return prediction, probability


def format_metric(metric_name: str) -> str:
    """Memformat metrik evaluasi menjadi empat angka desimal."""
    value = evaluation.get(metric_name)

    if value is None:
        return "-"

    return f"{float(value):.4f}"


def normalize_imputation_table(
    value: Any,
) -> pd.DataFrame:
    """Mengubah ringkasan imputasi dari artifact menjadi DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])

    return pd.DataFrame()


# ============================================================
# VALIDASI FITUR DEPLOYMENT
# ============================================================

SUPPORTED_ENGINEERED_FEATURES = {
    "gender",
    "age",
    "country",
    "city",
    "acquisition_channel",
    "device_type",
    "subscription_type",
    "is_premium_user",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "total_spent",
    "support_tickets",
    "refund_requested",
    "delivery_delay_days",
    "payment_method",
    "satisfaction_score",
    "nps_score",
    "marketing_spend_per_user",
    "lifetime_value",
    "last_3_month_purchase_freq",
    "has_coupon_code",
    "customer_tenure_days",
    "days_since_last_purchase",
}

unsupported_features = sorted(
    set(input_feature_columns)
    - SUPPORTED_ENGINEERED_FEATURES
)

if unsupported_features:
    st.error(
        "Ada fitur model yang belum tersedia pada form aplikasi."
    )
    st.code(", ".join(unsupported_features))
    st.stop()


# ============================================================
# TAMPILAN UTAMA
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Aplikasi ini memprediksi apakah pelanggan berpotensi berhenti "
    "menggunakan layanan atau tetap menggunakan layanan."
)

st.caption(
    f"Target: 0 = tidak churn dan 1 = churn. "
    f"Model deployment: {model_name}."
)

prediction_tab, model_tab, guide_tab = st.tabs(
    [
        "Prediksi Pelanggan",
        "Informasi Model",
        "Panduan Fitur",
    ]
)


# ============================================================
# TAB PREDIKSI
# ============================================================

with prediction_tab:
    st.subheader("Form Data Pelanggan")

    st.write(
        "Nama setiap input mengikuti nama kolom pada dataset agar lebih mudah "
        "dicocokkan dengan data asli. Input tetap diproses menggunakan alur "
        "yang sama dengan notebook final, yaitu feature engineering, IQR capping, "
        "imputasi mean atau median, imputasi modus, OneHotEncoder, "
        "StandardScaler, dan prediksi model."
    )

    with st.form("customer_churn_form"):
        st.markdown("### 1. Data Demografis dan Akun")

        demographic_col1, demographic_col2, demographic_col3 = st.columns(3)

        with demographic_col1:
            gender_choices = (
                ["Tidak diketahui"]
                + choices_for("gender")
            )

            gender_value = st.selectbox(
                "gender",
                gender_choices,
            )

            gender = (
                np.nan
                if gender_value == "Tidak diketahui"
                else gender_value
            )

            age_unknown = st.checkbox(
                "age tidak diketahui"
            )

            age = st.number_input(
                "age",
                min_value=0.0,
                max_value=100.0,
                value=min(
                    max(
                        numeric_default("age", 35.0),
                        0.0,
                    ),
                    100.0,
                ),
                step=1.0,
                disabled=age_unknown,
            )

            if age_unknown:
                age = np.nan

        with demographic_col2:
            country = st.selectbox(
                "country",
                choices_for("country"),
            )

            city = st.selectbox(
                "city",
                choices_for("city"),
            )

            is_premium_user = st.selectbox(
                "is_premium_user",
                options=[0, 1],
                format_func=lambda value: (
                    "Ya" if value == 1 else "Tidak"
                ),
            )

        with demographic_col3:
            acquisition_channel = st.selectbox(
                "acquisition_channel",
                choices_for("acquisition_channel"),
            )

            device_type = st.selectbox(
                "device_type",
                choices_for("device_type"),
            )

            subscription_type = st.selectbox(
                "subscription_type",
                choices_for("subscription_type"),
            )

        st.markdown("### 2. Riwayat Pelanggan")

        date_col1, date_col2, date_col3 = st.columns(3)

        default_reference_date = pd.to_datetime(
            reference_date,
            errors="coerce",
        )

        if pd.isna(default_reference_date):
            default_reference_date = pd.Timestamp.today().normalize()

        default_last_purchase = (
            default_reference_date.date()
        )

        default_signup = (
            default_reference_date
            - pd.Timedelta(days=365)
        ).date()

        with date_col1:
            signup_date = st.date_input(
                "signup_date",
                value=default_signup,
            )

        with date_col2:
            last_purchase_date = st.date_input(
                "last_purchase_date",
                value=default_last_purchase,
                max_value=default_reference_date.date(),
            )

        with date_col3:
            coupon_used = st.selectbox(
                "coupon_code",
                options=[False, True],
                format_func=lambda value: (
                    "Ada" if value else "Tidak ada"
                ),
            )

            payment_method = st.selectbox(
                "payment_method",
                choices_for("payment_method"),
            )

        st.markdown("### 3. Aktivitas Penggunaan Layanan")

        activity_col1, activity_col2, activity_col3 = st.columns(3)

        with activity_col1:
            total_visits = st.number_input(
                "total_visits",
                min_value=0,
                max_value=1000,
                value=int(
                    round(
                        numeric_default(
                            "total_visits",
                            15,
                        )
                    )
                ),
                step=1,
            )

            avg_session_time = st.number_input(
                "avg_session_time",
                min_value=0.0,
                max_value=1000.0,
                value=numeric_default(
                    "avg_session_time",
                    8.0,
                ),
                step=0.1,
            )

        with activity_col2:
            pages_per_session = st.number_input(
                "pages_per_session",
                min_value=0.0,
                max_value=1000.0,
                value=numeric_default(
                    "pages_per_session",
                    4.0,
                ),
                step=0.1,
            )

            email_open_rate = st.slider(
                "email_open_rate",
                min_value=0.0,
                max_value=1.0,
                value=min(
                    max(
                        numeric_default(
                            "email_open_rate",
                            0.5,
                        ),
                        0.0,
                    ),
                    1.0,
                ),
                step=0.01,
            )

        with activity_col3:
            email_click_rate = st.slider(
                "email_click_rate",
                min_value=0.0,
                max_value=1.0,
                value=min(
                    max(
                        numeric_default(
                            "email_click_rate",
                            0.25,
                        ),
                        0.0,
                    ),
                    1.0,
                ),
                step=0.01,
            )

            last_3_month_purchase_freq = st.number_input(
                "last_3_month_purchase_freq",
                min_value=0,
                max_value=1000,
                value=int(
                    round(
                        numeric_default(
                            "last_3_month_purchase_freq",
                            7,
                        )
                    )
                ),
                step=1,
            )

        st.markdown("### 4. Transaksi dan Pemasaran")

        transaction_col1, transaction_col2 = st.columns(2)

        with transaction_col1:
            total_spent_unknown = st.checkbox(
                "total_spent tidak diketahui"
            )

            total_spent = st.number_input(
                "total_spent",
                min_value=0.0,
                max_value=10_000_000.0,
                value=max(
                    numeric_default(
                        "total_spent",
                        500.0,
                    ),
                    0.0,
                ),
                step=10.0,
                disabled=total_spent_unknown,
            )

            if total_spent_unknown:
                total_spent = np.nan

            marketing_spend_per_user = st.number_input(
                "marketing_spend_per_user",
                min_value=0.0,
                max_value=10_000_000.0,
                value=max(
                    numeric_default(
                        "marketing_spend_per_user",
                        17.5,
                    ),
                    0.0,
                ),
                step=1.0,
            )

        with transaction_col2:
            lifetime_value = st.number_input(
                "lifetime_value",
                min_value=0.0,
                max_value=10_000_000.0,
                value=max(
                    numeric_default(
                        "lifetime_value",
                        1200.0,
                    ),
                    0.0,
                ),
                step=10.0,
            )

            refund_requested = st.selectbox(
                "refund_requested",
                options=[0, 1],
                format_func=lambda value: (
                    "Ya" if value == 1 else "Tidak"
                ),
            )

        st.markdown("### 5. Layanan dan Kepuasan")

        service_col1, service_col2, service_col3 = st.columns(3)

        with service_col1:
            support_tickets = st.number_input(
                "support_tickets",
                min_value=0,
                max_value=1000,
                value=int(
                    round(
                        numeric_default(
                            "support_tickets",
                            2,
                        )
                    )
                ),
                step=1,
            )

            delivery_delay_days = st.number_input(
                "delivery_delay_days",
                min_value=0,
                max_value=1000,
                value=int(
                    round(
                        numeric_default(
                            "delivery_delay_days",
                            3,
                        )
                    )
                ),
                step=1,
            )

        with service_col2:
            satisfaction_unknown = st.checkbox(
                "satisfaction_score tidak diketahui"
            )

            satisfaction_score = st.number_input(
                "satisfaction_score",
                min_value=1.0,
                max_value=5.0,
                value=min(
                    max(
                        numeric_default(
                            "satisfaction_score",
                            4.0,
                        ),
                        1.0,
                    ),
                    5.0,
                ),
                step=1.0,
                disabled=satisfaction_unknown,
            )

            if satisfaction_unknown:
                satisfaction_score = np.nan

        with service_col3:
            nps_score = st.number_input(
                "nps_score",
                min_value=0,
                max_value=10,
                value=int(
                    min(
                        max(
                            round(
                                numeric_default(
                                    "nps_score",
                                    5,
                                )
                            ),
                            0,
                        ),
                        10,
                    )
                ),
                step=1,
            )

        submitted = st.form_submit_button(
            "Prediksi Churn",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        raw_input = {
            "gender": gender,
            "age": age,
            "country": country,
            "city": city,
            "signup_date": signup_date,
            "last_purchase_date": last_purchase_date,
            "acquisition_channel": acquisition_channel,
            "device_type": device_type,
            "subscription_type": subscription_type,
            "is_premium_user": is_premium_user,
            "total_visits": total_visits,
            "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session,
            "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate,
            "total_spent": total_spent,
            "coupon_used": coupon_used,
            "support_tickets": support_tickets,
            "refund_requested": refund_requested,
            "delivery_delay_days": delivery_delay_days,
            "payment_method": payment_method,
            "satisfaction_score": satisfaction_score,
            "nps_score": nps_score,
            "marketing_spend_per_user": marketing_spend_per_user,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq,
        }

        try:
            model_input, processed_input = prepare_input(
                raw_input
            )

            prediction, churn_probability = predict_churn(
                processed_input
            )

            st.divider()
            st.subheader("Hasil Prediksi")

            if prediction == 1:
                st.error(
                    "Pelanggan diprediksi **CHURN** atau berpotensi "
                    "berhenti menggunakan layanan."
                )
            else:
                st.success(
                    "Pelanggan diprediksi **TIDAK CHURN** atau tetap "
                    "menggunakan layanan."
                )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Hasil Kelas",
                    (
                        "Churn (1)"
                        if prediction == 1
                        else "Tidak Churn (0)"
                    ),
                )

            with result_col2:
                st.metric(
                    "Model",
                    model_name,
                )

            with result_col3:
                if churn_probability is not None:
                    st.metric(
                        "Probabilitas Churn",
                        f"{churn_probability * 100:.2f}%",
                    )
                else:
                    st.metric(
                        "Probabilitas Churn",
                        "Tidak tersedia",
                    )

            if churn_probability is not None:
                st.progress(
                    min(
                        max(churn_probability, 0.0),
                        1.0,
                    ),
                    text=(
                        "Probabilitas churn: "
                        f"{churn_probability * 100:.2f}%"
                    ),
                )

            with st.expander(
                "Lihat data setelah feature engineering"
            ):
                st.dataframe(
                    model_input,
                    use_container_width=True,
                )

            prediction_result = model_input.copy()

            prediction_result["prediction"] = prediction

            prediction_result["prediction_label"] = (
                "Churn"
                if prediction == 1
                else "Tidak Churn"
            )

            prediction_result[
                "churn_probability"
            ] = churn_probability

            st.download_button(
                label="Unduh Hasil Prediksi",
                data=prediction_result.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "hasil_prediksi_customer_churn.csv"
                ),
                mime="text/csv",
            )

        except Exception as error:
            st.error(
                "Prediksi belum berhasil diproses."
            )
            st.code(str(error))


# ============================================================
# TAB INFORMASI MODEL
# ============================================================

with model_tab:
    st.subheader("Informasi Model Deployment")

    st.info(
        f"Model yang digunakan: **{model_name}**"
    )

    dataset_evaluation = evaluation.get(
        "dataset",
        "Test asli tanpa SMOTE",
    )

    st.caption(
        f"Metrik berikut berasal dari {dataset_evaluation}."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Accuracy",
            format_metric("accuracy"),
        )

    with metric_col2:
        st.metric(
            "Precision",
            format_metric("precision"),
        )

    with metric_col3:
        st.metric(
            "Recall",
            format_metric("recall"),
        )

    with metric_col4:
        st.metric(
            "F1-Score",
            format_metric("f1_score"),
        )

    st.markdown("### Tahapan Pemrosesan Input")

    st.markdown(
        """
1. Mengubah informasi kupon menjadi `has_coupon_code`.
2. Mengubah tanggal menjadi `customer_tenure_days` dan `days_since_last_purchase`.
3. Menghapus tiga fitur dengan pengaruh terendah.
4. Membatasi outlier menggunakan IQR capping dari data train.
5. Mengisi missing value numerik menggunakan mean atau median sesuai distribusi fitur.
6. Mengisi missing value kategorikal menggunakan modus.
7. Mengubah kategori menggunakan OneHotEncoder.
8. Melakukan scaling fitur numerik menggunakan StandardScaler.
9. Melakukan prediksi menggunakan model terbaik hasil hyperparameter tuning.
        """
    )

    if dropped_low_impact_features:
        st.markdown(
            "### Tiga Fitur yang Dihapus"
        )

        st.write(
            ", ".join(
                str(feature)
                for feature in dropped_low_impact_features
            )
        )

    st.markdown("### Informasi SMOTE")

    st.info(
        "SMOTE hanya digunakan pada data train dan validation saat "
        "pelatihan. Data test dan input pengguna pada aplikasi tidak "
        "diseimbangkan menggunakan SMOTE."
    )

    if smote_information:
        smote_col1, smote_col2, smote_col3 = st.columns(3)

        with smote_col1:
            st.metric(
                "Metode",
                str(
                    smote_information.get(
                        "method",
                        "SMOTE",
                    )
                ),
            )

        with smote_col2:
            applied_to = smote_information.get(
                "applied_to",
                ["train", "validation"],
            )

            st.metric(
                "Diterapkan pada",
                ", ".join(applied_to),
            )

        with smote_col3:
            test_balanced = smote_information.get(
                "test_balanced",
                False,
            )

            st.metric(
                "Test diseimbangkan",
                "Ya" if test_balanced else "Tidak",
            )

    imputation_table = normalize_imputation_table(
        imputation_strategy
    )

    if not imputation_table.empty:
        with st.expander(
            "Lihat strategi imputasi setiap fitur"
        ):
            st.dataframe(
                imputation_table,
                use_container_width=True,
                hide_index=True,
            )

    if reference_date is not None:
        reference_date_display = pd.to_datetime(
            reference_date,
            errors="coerce",
        )

        if pd.notna(reference_date_display):
            st.caption(
                "Tanggal referensi untuk menghitung jarak pembelian "
                f"terakhir: {reference_date_display.date()}."
            )


# ============================================================
# TAB PANDUAN FITUR
# ============================================================

with guide_tab:
    st.subheader("Panduan Pengisian Fitur")

    feature_guide = pd.DataFrame(
        [
            [
                "age",
                "Usia pelanggan dalam tahun.",
            ],
            [
                "acquisition_channel",
                "Sumber awal pelanggan, seperti Email, Organic, atau iklan.",
            ],
            [
                "total_visits",
                "Jumlah kunjungan pelanggan ke layanan.",
            ],
            [
                "email_open_rate",
                "Proporsi email yang dibuka dalam rentang 0 sampai 1.",
            ],
            [
                "email_click_rate",
                "Proporsi klik email dalam rentang 0 sampai 1.",
            ],
            [
                "total_spent",
                "Total pengeluaran pelanggan selama menggunakan layanan.",
            ],
            [
                "support_tickets",
                "Jumlah tiket bantuan yang pernah dibuat pelanggan.",
            ],
            [
                "satisfaction_score",
                "Tingkat kepuasan pelanggan dalam rentang 1 sampai 5.",
            ],
            [
                "nps_score",
                "Nilai rekomendasi pelanggan dalam rentang 0 sampai 10.",
            ],
            [
                "lifetime_value",
                "Perkiraan nilai pelanggan selama menggunakan layanan.",
            ],
            [
                "last_3_month_purchase_freq",
                "Jumlah pembelian dalam tiga bulan terakhir.",
            ],
        ],
        columns=[
            "Fitur",
            "Keterangan",
        ],
    )

    st.dataframe(
        feature_guide,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Hasil aplikasi merupakan prediksi berdasarkan pola pada "
        "dataset penelitian. Hasil ini tidak dapat dianggap sebagai "
        "keputusan bisnis mutlak dan tetap perlu dipertimbangkan "
        "bersama informasi pelanggan lainnya."
    )
