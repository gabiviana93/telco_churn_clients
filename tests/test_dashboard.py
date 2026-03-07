"""
Testes do Dashboard
===================

Testa funções utilitárias e lógica de negócio do dashboard Streamlit,
isolando a camada de apresentação (st.*) via mocks.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers para importar funções do dashboard sem disparar st.set_page_config
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """Dict que também suporta acesso por atributo, como st.session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as err:
            raise AttributeError(name) from err


@pytest.fixture(autouse=True)
def _patch_streamlit():
    """Mock do Streamlit para evitar erros de runtime fora do navegador."""
    mock_st = MagicMock()
    mock_st.session_state = _AttrDict()
    mock_st.cache_resource = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    mock_st.cache_data = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


@pytest.fixture
def dashboard():
    """Importa o módulo dashboard com Streamlit mockado."""
    import importlib

    import scripts.dashboard as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures de dados
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_customer():
    """Dados de um cliente para predição."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1397.47,
    }


@pytest.fixture
def metrics_json_data():
    """Estrutura completa de reports/metrics.json."""
    return {
        "best_model": "catboost",
        "f1_score": 0.6345,
        "roc_auc": 0.8422,
        "precision": 0.5463,
        "recall": 0.7567,
        "accuracy": 0.7686,
        "auprc": 0.6537,
        "individual_results": {
            "lightgbm": {
                "f1_score": 0.6201,
                "roc_auc": 0.8415,
                "precision": 0.5361,
                "recall": 0.7353,
            },
            "xgboost": {
                "f1_score": 0.6323,
                "roc_auc": 0.8425,
                "precision": 0.5444,
                "recall": 0.7540,
            },
            "catboost": {
                "f1_score": 0.6345,
                "roc_auc": 0.8422,
                "precision": 0.5463,
                "recall": 0.7567,
            },
        },
    }


@pytest.fixture
def reference_df():
    """DataFrame de referência para simulação de drift."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(18, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        }
    )


# ===========================================================================
# _encode_categoricals
# ===========================================================================


class TestEncodeCategoricals:
    def test_converts_object_columns_to_codes(self, dashboard):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
        result = dashboard._encode_categoricals(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        # Códigos categóricos devem ser inteiros
        assert result[:, 0].dtype in (np.int8, np.int16, np.int32, np.int64)

    def test_preserves_numeric_columns(self, dashboard):
        df = pd.DataFrame({"num": [10.5, 20.5, 30.5]})
        result = dashboard._encode_categoricals(df)
        np.testing.assert_array_almost_equal(result[:, 0], [10.5, 20.5, 30.5])

    def test_empty_dataframe(self, dashboard):
        df = pd.DataFrame({"a": pd.Series([], dtype="object")})
        result = dashboard._encode_categoricals(df)
        assert result.shape == (0, 1)


# ===========================================================================
# _get_preprocessor_feature_names
# ===========================================================================


class TestGetPreprocessorFeatureNames:
    def test_returns_generic_names_when_no_preprocessor(self, dashboard):
        names = dashboard._get_preprocessor_feature_names(None, 5)
        assert names == [f"Feature {i}" for i in range(5)]

    def test_uses_get_feature_names_out(self, dashboard):
        mock_prep = MagicMock()
        mock_prep.get_feature_names_out.return_value = ["a", "b", "c"]
        # Força o import de get_feature_names a falhar
        with patch("src.interpret.get_feature_names", side_effect=Exception):
            names = dashboard._get_preprocessor_feature_names(mock_prep, 3)
        assert names == ["a", "b", "c"]


# ===========================================================================
# get_available_models
# ===========================================================================


class TestGetAvailableModels:
    def test_returns_converted_models(self, dashboard):
        models = dashboard.get_available_models()
        assert isinstance(models, list)
        if models:
            m = models[0]
            assert "name" in m
            assert "path" in m
            assert "size_mb" in m
            assert "modified" in m
            assert "is_default" in m
            assert isinstance(m["path"], Path)

    def test_sorted_by_modification_date_descending(self, dashboard):
        models = dashboard.get_available_models()
        if len(models) >= 2:
            dates = [m["modified"] for m in models]
            assert dates == sorted(dates, reverse=True)


# ===========================================================================
# predict_locally
# ===========================================================================


class TestPredictLocally:
    def test_with_proba_model(self, dashboard, sample_customer):
        """Modelo com predict_proba retorna probabilidade e risco."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        result = dashboard.predict_locally(mock_model, sample_customer)

        assert result is not None
        assert result["probability"] == pytest.approx(0.7)
        assert result["prediction"] == 1  # 0.7 >= 0.5
        assert "churn_risk" in result

    def test_with_proba_model_below_threshold(self, dashboard, sample_customer):
        """Modelo com probabilidade abaixo do threshold padrão retorna 0."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        result = dashboard.predict_locally(mock_model, sample_customer)

        assert result is not None
        assert result["prediction"] == 0
        assert result["probability"] == pytest.approx(0.3)

    def test_with_package_dict(self, dashboard, sample_customer):
        """Pacote com threshold customizado é respeitado."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.55, 0.45]])

        package = {"model": mock_model, "threshold": 0.4}
        result = dashboard.predict_locally(package, sample_customer)

        assert result is not None
        assert result["prediction"] == 1  # 0.45 >= 0.4

    def test_with_package_threshold_above(self, dashboard, sample_customer):
        """Probabilidade abaixo do threshold customizado retorna 0."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.55, 0.45]])

        package = {"model": mock_model, "threshold": 0.5}
        result = dashboard.predict_locally(package, sample_customer)

        assert result is not None
        assert result["prediction"] == 0  # 0.45 < 0.5

    def test_without_proba_model(self, dashboard, sample_customer):
        """Modelo sem predict_proba usa predict direto."""
        mock_model = MagicMock(spec=[])  # sem predict_proba
        mock_model.predict = MagicMock(return_value=np.array([1]))
        # spec=[] garante que hasattr(model, 'predict_proba') é False

        result = dashboard.predict_locally(mock_model, sample_customer)

        assert result is not None
        assert result["prediction"] == 1
        assert result["probability"] == 1.0

    def test_none_model_in_package(self, dashboard, sample_customer):
        """Pacote sem modelo retorna None."""
        package = {"model": None}

        result = dashboard.predict_locally(package, sample_customer)
        assert result is None

    def test_with_feature_engineer(self, dashboard, sample_customer):
        """Feature engineer do pacote é aplicado antes da predição."""
        mock_fe = MagicMock()
        mock_fe.transform.return_value = pd.DataFrame([sample_customer])
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])

        package = {"model": mock_model, "feature_engineer": mock_fe, "threshold": 0.5}
        result = dashboard.predict_locally(package, sample_customer)

        mock_fe.transform.assert_called_once()
        assert result is not None
        assert result["prediction"] == 1


# ===========================================================================
# _load_metrics_from_report
# ===========================================================================


class TestLoadMetricsFromReport:
    def test_returns_catboost_metrics_for_catboost_model(
        self, dashboard, metrics_json_data, tmp_path, _patch_streamlit
    ):
        """Quando modelo selecionado é catboost, retorna métricas do catboost."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_json_data))

        _patch_streamlit.session_state = _AttrDict(selected_model_name="model_catboost")

        with patch.object(dashboard, "REPORTS_DIR", tmp_path):
            result = dashboard._load_metrics_from_report()

        assert result["f1_score"] == pytest.approx(0.6345)
        assert result["roc_auc"] == pytest.approx(0.8422)

    def test_returns_xgboost_metrics_for_xgboost_model(
        self, dashboard, metrics_json_data, tmp_path, _patch_streamlit
    ):
        """Quando modelo selecionado é xgboost, retorna métricas do xgboost."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_json_data))

        _patch_streamlit.session_state = _AttrDict(selected_model_name="model_xgboost")

        with patch.object(dashboard, "REPORTS_DIR", tmp_path):
            result = dashboard._load_metrics_from_report()

        assert result["f1_score"] == pytest.approx(0.6323)
        assert result["roc_auc"] == pytest.approx(0.8425)

    def test_returns_lightgbm_metrics_for_lightgbm_model(
        self, dashboard, metrics_json_data, tmp_path, _patch_streamlit
    ):
        """Quando modelo selecionado é lightgbm, retorna métricas do lightgbm."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_json_data))

        _patch_streamlit.session_state = _AttrDict(selected_model_name="model_lightgbm")

        with patch.object(dashboard, "REPORTS_DIR", tmp_path):
            result = dashboard._load_metrics_from_report()

        assert result["f1_score"] == pytest.approx(0.6201)

    def test_fallback_to_toplevel_for_default_model(
        self, dashboard, metrics_json_data, tmp_path, _patch_streamlit
    ):
        """Modelo sem match em individual_results retorna top-level."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_json_data))

        _patch_streamlit.session_state = _AttrDict(selected_model_name="model")

        with patch.object(dashboard, "REPORTS_DIR", tmp_path):
            result = dashboard._load_metrics_from_report()

        assert result["f1_score"] == pytest.approx(0.6345)

    def test_no_metrics_file_with_package(self, dashboard, tmp_path, _patch_streamlit):
        """Sem metrics.json, usa métricas do pacote do modelo."""
        _patch_streamlit.session_state = {}

        mock_package = {"metrics": {"f1_score": 0.55, "roc_auc": 0.80}}
        with (
            patch.object(dashboard, "REPORTS_DIR", tmp_path),
            patch.object(dashboard, "load_model", return_value=mock_package),
        ):
            result = dashboard._load_metrics_from_report()

        assert result["f1_score"] == pytest.approx(0.55)

    def test_no_metrics_file_no_model_returns_empty(self, dashboard, tmp_path):
        """Sem metrics.json e sem modelo, retorna dict vazio."""
        with (
            patch.object(dashboard, "REPORTS_DIR", tmp_path),
            patch.object(dashboard, "load_model", return_value=None),
        ):
            result = dashboard._load_metrics_from_report()

        assert result == {}


# ===========================================================================
# _severity_color & _severity_icon
# ===========================================================================


class TestSeverityHelpers:
    @pytest.mark.parametrize(
        "severity,expected_color",
        [
            ("none", "#00cc66"),
            ("low", "#ffcc00"),
            ("moderate", "#ff8800"),
            ("high", "#ff4b4b"),
        ],
    )
    def test_severity_color(self, dashboard, severity, expected_color):
        assert dashboard._severity_color(severity) == expected_color

    def test_severity_color_unknown(self, dashboard):
        assert dashboard._severity_color("unknown") == "#888888"

    @pytest.mark.parametrize(
        "severity,expected_icon",
        [
            ("none", "\u2705"),
            ("low", "\U0001f7e1"),
            ("moderate", "\U0001f7e0"),
            ("high", "\U0001f534"),
        ],
    )
    def test_severity_icon(self, dashboard, severity, expected_icon):
        assert dashboard._severity_icon(severity) == expected_icon

    def test_severity_icon_unknown(self, dashboard):
        assert dashboard._severity_icon("unknown") == "\u2753"


# ===========================================================================
# _simulate_production_data
# ===========================================================================


class TestSimulateProductionData:
    def test_returns_correct_shape(self, dashboard, reference_df):
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=0.5,
            sample_size=50,
            numeric_features=["tenure", "MonthlyCharges", "TotalCharges"],
            categorical_features=["Contract", "InternetService"],
        )
        assert len(result) == 50
        assert set(result.columns) == set(reference_df.columns)

    def test_zero_drift_preserves_categoricals(self, dashboard, reference_df):
        """Sem drift, categorias devem permanecer inalteradas (apenas reamostragem)."""
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=0.0,
            sample_size=100,
            numeric_features=["tenure", "MonthlyCharges"],
            categorical_features=["Contract"],
        )
        # Valores categóricos devem estar no conjunto original
        original_contracts = set(reference_df["Contract"].unique())
        assert set(result["Contract"].unique()).issubset(original_contracts)

    def test_high_drift_modifies_numerics(self, dashboard, reference_df):
        """Drift alto deve alterar a distribuição numérica."""
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=2.0,
            sample_size=200,
            numeric_features=["MonthlyCharges"],
            categorical_features=[],
        )
        # A média deve diferir significativamente com drift=2.0
        orig_mean = reference_df["MonthlyCharges"].mean()
        new_mean = result["MonthlyCharges"].mean()
        assert abs(new_mean - orig_mean) > 1.0  # algum deslocamento

    def test_sample_size_capped_at_dataframe_length(self, dashboard, reference_df):
        """Se sample_size > len(df), usa len(df)."""
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=0.5,
            sample_size=10_000,
            numeric_features=["tenure"],
            categorical_features=[],
        )
        assert len(result) == len(reference_df)

    def test_reset_index(self, dashboard, reference_df):
        """Resultado deve ter index contínuo (reset_index)."""
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=0.5,
            sample_size=50,
            numeric_features=["tenure"],
            categorical_features=[],
        )
        assert list(result.index) == list(range(len(result)))

    def test_categorical_drift_above_threshold(self, dashboard, reference_df):
        """Drift > 0.3 deve aplicar swaps categóricos."""
        result = dashboard._simulate_production_data(
            reference_df,
            drift_intensity=0.8,
            sample_size=200,
            numeric_features=[],
            categorical_features=["Contract"],
        )
        # Deve retornar sem erro e manter categorias válidas
        original_contracts = set(reference_df["Contract"].unique())
        assert set(result["Contract"].unique()).issubset(original_contracts)


# ===========================================================================
# _get_active_model_name
# ===========================================================================


class TestGetActiveModelName:
    def test_returns_session_state_when_set(self, dashboard, _patch_streamlit):
        _patch_streamlit.session_state = _AttrDict(selected_model_name="model_xgboost")
        assert dashboard._get_active_model_name() == "model_xgboost"

    def test_returns_default_stem_when_no_session(self, dashboard, _patch_streamlit):
        _patch_streamlit.session_state = {}
        result = dashboard._get_active_model_name()
        assert result == Path(dashboard.MODEL_PATH).stem


# ===========================================================================
# predict_via_api (com mock de requests)
# ===========================================================================


class TestPredictViaApi:
    def test_successful_prediction(self, dashboard, sample_customer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": {
                "churn_prediction": 1,
                "churn_probability": 0.85,
                "churn_risk": "HIGH",
            }
        }

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.predict_via_api(sample_customer)

        assert result is not None
        assert result["prediction"] == 1
        assert result["probability"] == 0.85
        assert result["churn_risk"] == "HIGH"

    def test_api_error_returns_none(self, dashboard, sample_customer):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.predict_via_api(sample_customer)

        assert result is None

    def test_connection_error_returns_none(self, dashboard, sample_customer):
        import requests as real_requests

        with patch(
            "scripts.dashboard.requests.post",
            side_effect=real_requests.ConnectionError("refused"),
        ):
            result = dashboard.predict_via_api(sample_customer)

        assert result is None


# ===========================================================================
# check_api_health
# ===========================================================================


class TestCheckApiHealth:
    def test_healthy_api(self, dashboard):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            assert dashboard.check_api_health() is True

    def test_unhealthy_api(self, dashboard):
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            assert dashboard.check_api_health() is False

    def test_connection_refused(self, dashboard):
        import requests as real_requests

        with patch(
            "scripts.dashboard.requests.get",
            side_effect=real_requests.ConnectionError(),
        ):
            assert dashboard.check_api_health() is False


# ===========================================================================
# explain_via_api
# ===========================================================================


class TestExplainViaApi:
    def test_successful_explanation(self, dashboard, sample_customer):
        explanation = {"features": ["tenure"], "shap_values": [0.15]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"explanation": explanation}

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.explain_via_api(sample_customer)

        assert result == explanation

    def test_failed_explanation_returns_none(self, dashboard, sample_customer):
        mock_response = MagicMock()
        mock_response.status_code = 422

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.explain_via_api(sample_customer)

        assert result is None


# ===========================================================================
# get_feature_importance_via_api
# ===========================================================================


class TestGetFeatureImportanceViaApi:
    def test_success(self, dashboard):
        importances = [{"feature": "tenure", "importance": 0.25}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"feature_importances": importances}

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            result = dashboard.get_feature_importance_via_api(top_n=10)

        assert result == importances

    def test_failure_returns_none(self, dashboard):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            result = dashboard.get_feature_importance_via_api()

        assert result is None


# ===========================================================================
# get_global_shap_via_api
# ===========================================================================


class TestGetGlobalShapViaApi:
    def test_success(self, dashboard):
        shap_data = {"features": ["tenure"], "mean_abs_shap": [0.12]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = shap_data

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            result = dashboard.get_global_shap_via_api(sample_size=50, top_n=10)

        assert result == shap_data

    def test_failure_returns_none(self, dashboard):
        with patch("scripts.dashboard.requests.get", side_effect=Exception("timeout")):
            result = dashboard.get_global_shap_via_api()

        assert result is None


# ===========================================================================
# detect_drift_via_api
# ===========================================================================


class TestDetectDriftViaApi:
    def test_success(self, dashboard, reference_df):
        drift_result = {"overall_severity": "low", "features_with_drift": 2}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = drift_result

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.detect_drift_via_api(reference_df)

        assert result == drift_result

    def test_failure_returns_none(self, dashboard, reference_df):
        with patch("scripts.dashboard.requests.post", side_effect=Exception("err")):
            result = dashboard.detect_drift_via_api(reference_df)

        assert result is None


# ===========================================================================
# list_models_via_api & switch_model_via_api
# ===========================================================================


class TestModelsViaApi:
    def test_list_models_success(self, dashboard):
        data = {
            "models": [{"name": "model_catboost"}],
            "active_model": "model_catboost",
            "total": 1,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data

        with patch("scripts.dashboard.requests.get", return_value=mock_response):
            result = dashboard.list_models_via_api()

        assert result == data

    def test_list_models_failure(self, dashboard):
        import requests as real_requests

        with patch(
            "scripts.dashboard.requests.get",
            side_effect=real_requests.ConnectionError("err"),
        ):
            result = dashboard.list_models_via_api()

        assert result is None

    def test_switch_model_success(self, dashboard):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.switch_model_via_api("model_xgboost")

        assert result is True

    def test_switch_model_failure(self, dashboard):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("scripts.dashboard.requests.post", return_value=mock_response):
            result = dashboard.switch_model_via_api("nonexistent")

        assert result is False

    def test_switch_model_connection_error(self, dashboard):
        import requests as real_requests

        with patch(
            "scripts.dashboard.requests.post",
            side_effect=real_requests.ConnectionError(),
        ):
            result = dashboard.switch_model_via_api("model_catboost")

        assert result is False


# ===========================================================================
# load_model (routing logic)
# ===========================================================================


class TestLoadModel:
    def test_uses_selected_path_when_set(self, dashboard, _patch_streamlit):
        mock_package = {"model": MagicMock()}
        _patch_streamlit.session_state = _AttrDict(selected_model_path=Path("/fake/model.joblib"))

        with patch.object(dashboard, "load_model_by_path", return_value=mock_package) as mock_load:
            result = dashboard.load_model()

        mock_load.assert_called_once_with(Path("/fake/model.joblib"))
        assert result == mock_package

    def test_uses_default_path_when_no_selection(self, dashboard, _patch_streamlit):
        mock_package = {"model": MagicMock()}
        _patch_streamlit.session_state = _AttrDict()

        with patch.object(dashboard, "load_model_by_path", return_value=mock_package) as mock_load:
            result = dashboard.load_model()

        mock_load.assert_called_once_with(Path(dashboard.MODEL_PATH))
        assert result == mock_package

    def test_returns_none_on_file_not_found(self, dashboard, _patch_streamlit):
        _patch_streamlit.session_state = _AttrDict()

        with patch.object(dashboard, "load_model_by_path", return_value=None):
            result = dashboard.load_model()

        assert result is None


# ── Testes de _rank_best_model ────────────────────────────────────────────────


class TestRankBestModel:
    """Testes para lógica de ranking composto por vitórias + desempate F1."""

    @pytest.fixture()
    def dashboard(self):
        import scripts.dashboard as mod

        return mod

    def test_clear_winner(self, dashboard):
        """Modelo que vence todas as métricas é selecionado."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B", "C"],
                "F1-Score": [0.90, 0.80, 0.70],
                "AUC-ROC": [0.95, 0.85, 0.75],
                "AUPRC": [0.88, 0.78, 0.68],
                "Recall": [0.92, 0.82, 0.72],
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        assert best_idx == 0
        assert wins[0] == 4

    def test_tiebreak_by_f1(self, dashboard):
        """Empate em vitórias é desempatado por F1-Score."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "F1-Score": [0.70, 0.80],  # B vence F1
                "AUC-ROC": [0.90, 0.85],  # A vence AUC
                "AUPRC": [0.60, 0.75],  # B vence AUPRC
                "Recall": [0.95, 0.80],  # A vence Recall
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        # A=2, B=2 → desempate F1 → B (0.80 > 0.70)
        assert wins[0] == 2
        assert wins[1] == 2
        assert best_idx == 1

    def test_three_way_tie(self, dashboard):
        """Empate triplo desempatado por F1."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B", "C"],
                "F1-Score": [0.60, 0.70, 0.65],
                "AUC-ROC": [0.80, 0.70, 0.75],
                "AUPRC": [0.50, 0.55, 0.60],
            }
        )
        metrics = ["F1-Score", "AUC-ROC", "AUPRC"]
        best_idx, wins = dashboard._rank_best_model(df, metrics)
        # A=1(AUC), B=1(F1), C=1(AUPRC) → desempate F1 → B
        assert best_idx == 1

    def test_nan_metric_ignored(self, dashboard):
        """Métricas NaN não contam como vitória."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "F1-Score": [0.80, 0.70],
                "AUC-ROC": [float("nan"), 0.90],
                "AUPRC": [0.75, float("nan")],
                "Recall": [0.85, 0.80],
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        # A vence F1, AUPRC, Recall (3); B vence AUC (1)
        assert best_idx == 0
        assert wins[0] == 3
        assert wins[1] == 1

    def test_all_nan_column_skipped(self, dashboard):
        """Coluna inteiramente NaN não gera nenhuma vitória."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "F1-Score": [0.80, 0.70],
                "AUC-ROC": [float("nan"), float("nan")],
                "AUPRC": [0.65, 0.60],
                "Recall": [0.75, 0.80],
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        # A vence F1 e AUPRC (2); B vence Recall (1); AUC skipped
        assert wins.sum() == 3
        assert best_idx == 0

    def test_missing_metric_column(self, dashboard):
        """Coluna ausente no DataFrame é ignorada sem erro."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "F1-Score": [0.80, 0.70],
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        assert best_idx == 0
        assert wins[0] == 1

    def test_single_model(self, dashboard):
        """Com apenas um modelo, ele é o vencedor."""
        df = pd.DataFrame(
            {
                "Nome": ["Solo"],
                "F1-Score": [0.75],
                "AUC-ROC": [0.80],
                "AUPRC": [0.70],
                "Recall": [0.85],
            }
        )
        best_idx, wins = dashboard._rank_best_model(df)
        assert best_idx == 0
        assert wins[0] == 4

    def test_custom_metrics(self, dashboard):
        """Ranking com lista customizada de métricas."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "F1-Score": [0.70, 0.80],
                "AUC-ROC": [0.90, 0.85],
                "AUPRC": [0.60, 0.75],
                "Recall": [0.95, 0.80],
            }
        )
        # Usando apenas AUPRC → B vence
        best_idx, wins = dashboard._rank_best_model(df, ["AUPRC"])
        assert best_idx == 1
        assert wins[1] == 1
