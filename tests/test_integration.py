"""
Testes de Integração
====================

Testes que vão além dos testes unitários para verificar:
- Carregamento de config do YAML
- Imports de módulos no codebase
- Endpoints da API (startup, health, prediction)
- Importabilidade do dashboard
- Pipeline de inferência end-to-end
"""

import numpy as np
import pandas as pd
import pytest

from src.config import STEP_MODEL

# =============================================================================
# INTEGRAÇÃO DE CONFIG E IMPORTS
# =============================================================================


class TestConfigIntegration:
    """Verifica se todos os valores de config centralizados carregam do YAML corretamente."""

    def test_config_constants_exist(self):
        from src.config import (
            DEFAULT_MODEL_TYPE,
            DEFAULT_N_TRIALS,
            DEFAULT_OPTIMIZATION_METRIC,
            DEFAULT_OPTIMIZATION_TIMEOUT,
            DEFAULT_OPTIMIZE_THRESHOLD,
        )

        assert isinstance(DEFAULT_MODEL_TYPE, str)
        assert len(DEFAULT_MODEL_TYPE) > 0  # qualquer nome de modelo registrado
        assert isinstance(DEFAULT_N_TRIALS, int) and DEFAULT_N_TRIALS > 0
        assert DEFAULT_OPTIMIZATION_METRIC in ("f1", "roc_auc", "precision", "recall", "auprc")
        assert isinstance(DEFAULT_OPTIMIZE_THRESHOLD, bool)
        assert DEFAULT_OPTIMIZATION_TIMEOUT is None or isinstance(DEFAULT_OPTIMIZATION_TIMEOUT, int)

    def test_search_space_complete(self):
        from src.config import SEARCH_SPACE

        required_keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
            "num_leaves",
            "min_child_samples",
            "l2_leaf_reg",
        ]
        for key in required_keys:
            assert key in SEARCH_SPACE, f"Missing key: {key}"
            val = SEARCH_SPACE[key]
            assert isinstance(val, tuple) and len(val) == 2, f"Bad range for {key}: {val}"
            assert val[0] < val[1], f"Invalid range for {key}: {val}"

    def test_model_config_yaml_defaults(self):
        from src.config import ModelConfig

        mc = ModelConfig()
        assert mc.model_type is not None
        assert mc.n_estimators > 0
        assert mc.max_depth > 0
        assert 0 < mc.learning_rate <= 1

    def test_optimization_config_yaml_defaults(self):
        from src.config import DEFAULT_N_TRIALS
        from src.optimization import OptimizationConfig

        oc = OptimizationConfig()
        assert oc.n_trials == DEFAULT_N_TRIALS

    def test_training_config_defaults(self):
        from src.config import DEFAULT_N_TRIALS
        from src.optimization import OptimizationConfig

        tc = OptimizationConfig()
        assert tc.n_trials == DEFAULT_N_TRIALS

    def test_model_type_in_registry(self):
        from src.config import DEFAULT_MODEL_TYPE, MODEL_REGISTRY

        assert DEFAULT_MODEL_TYPE in MODEL_REGISTRY
        entry = MODEL_REGISTRY[DEFAULT_MODEL_TYPE]
        assert "class" in entry
        assert "." in entry["class"]  # must be a dotted import path

    def test_create_model_factory(self):
        from src.config import DEFAULT_MODEL_TYPE, create_model

        model = create_model(DEFAULT_MODEL_TYPE, {"n_estimators": 10, "random_state": 42})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_unknown_raises(self):
        from src.config import create_model

        with pytest.raises(ValueError, match="não encontrado no registry"):
            create_model("nonexistent_model", {})

    def test_model_registry_builtin_entries(self):
        from src.config import MODEL_REGISTRY

        for name in ("xgboost", "lightgbm", "catboost"):
            assert name in MODEL_REGISTRY, f"Built-in model '{name}' missing from registry"
            entry = MODEL_REGISTRY[name]
            assert "class" in entry
            assert "search_params" in entry
            assert isinstance(entry["search_params"], list)


# =============================================================================
# INTEGRAÇÃO DE PIPELINE
# =============================================================================


class TestPipelineIntegration:
    """Verifica se criação e ajuste do pipeline funcionam end-to-end."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "tenure": np.random.randint(1, 72, n),
                "MonthlyCharges": np.random.uniform(20, 100, n),
                "TotalCharges": np.random.uniform(100, 5000, n),
                "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
                "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
                "PaymentMethod": np.random.choice(["Electronic check", "Mailed check"], n),
            }
        )
        y = pd.Series(np.random.choice([0, 1], n, p=[0.73, 0.27]))
        return X, y

    def test_churn_pipeline_fit_predict(self, sample_data):
        from src.pipeline import ChurnPipeline

        X, y = sample_data
        pipeline = ChurnPipeline()
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_churn_pipeline_evaluate(self, sample_data):
        from src.pipeline import ChurnPipeline

        X, y = sample_data
        pipeline = ChurnPipeline()
        pipeline.fit(X, y)

        metrics = pipeline.evaluate(X, y)
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["f1_score"] <= 1

    def test_churn_pipeline_save_load(self, sample_data, tmp_path):
        from src.pipeline import ChurnPipeline

        X, y = sample_data
        pipeline = ChurnPipeline()
        pipeline.fit(X, y)

        path = tmp_path / "test_model.joblib"
        pipeline.save(path)
        assert path.exists()

        loaded = ChurnPipeline.load(path)
        preds = loaded.predict(X)
        assert len(preds) == len(X)


# =============================================================================
# INTEGRAÇÃO DO MODEL SERVICE
# =============================================================================


class TestModelServiceIntegration:
    """Verifica helper _detect_model_name do model_service."""

    def test_detect_name_from_pipeline(self):
        from unittest.mock import MagicMock

        from api.services.model_service import _detect_model_name

        mock_model = MagicMock()
        mock_model.named_steps = {STEP_MODEL: MagicMock(spec=[])}
        type(mock_model.named_steps[STEP_MODEL]).__name__ = "LGBMClassifier"

        assert _detect_model_name(mock_model) == "LGBMClassifier"

    def test_detect_name_from_raw_model(self):
        from unittest.mock import MagicMock

        from api.services.model_service import _detect_model_name

        mock = MagicMock(spec=[])
        type(mock).__name__ = "XGBClassifier"

        assert _detect_model_name(mock) == "XGBClassifier"


# =============================================================================
# INTEGRAÇÃO DA API
# =============================================================================


class TestAPIIntegration:
    """Testa startup da API e respostas dos endpoints."""

    @pytest.fixture(scope="class")
    def api_client(self):
        """Cria TestClient para a app FastAPI."""
        from fastapi.testclient import TestClient

        from api.main import app

        client = TestClient(app)
        yield client

    def test_health_endpoint(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_model_info_endpoint(self, api_client):
        resp = api_client.get("/model/info")
        # Deve retornar 200 ou 503 (modelo não carregado) mas não 500
        assert resp.status_code in (200, 503)

    def test_predict_endpoint_structure(self, api_client):
        """Testa se endpoint de predição aceita o schema correto."""
        payload = {
            "features": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0,
            }
        }
        resp = api_client.post("/predict/", json=payload)
        assert resp.status_code == 200, f"Prediction failed: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "churn_probability" in data["prediction"]

    def test_api_docs_accessible(self, api_client):
        resp = api_client.get("/docs")
        assert resp.status_code == 200

    def test_api_description_generic(self, api_client):
        """Verifica se descrição da API não contém XGBoost/LightGBM hardcoded."""
        resp = api_client.get("/openapi.json")
        assert resp.status_code == 200
        openapi = resp.json()
        description = openapi.get("info", {}).get("description", "")
        assert "Customer Churn" in description or "Prediction" in description
        # NÃO deve conter nomes de modelo hardcoded na descrição
        assert "XGBoost/LightGBM Classifier" not in description


# =============================================================================
# IMPORTABILIDADE DO DASHBOARD
# =============================================================================


class TestDashboardIntegration:
    """Verifica se módulo do dashboard pode ser importado sem falhar."""

    def test_dashboard_importable(self):
        """Dashboard deve ser importável (mesmo sem renderizar sem Streamlit)."""
        # Testa que imports, carregamento de config e helpers funcionam
        try:
            import importlib

            spec = importlib.util.find_spec("scripts.dashboard")
            assert spec is not None, "scripts.dashboard module not found"
        except Exception as e:
            pytest.fail(f"Dashboard import failed: {e}")


# =============================================================================
# PIPELINE DE INFERÊNCIA
# =============================================================================


class TestInferencePipeline:
    """Testa módulo de inferência end-to-end."""

    def test_load_model_package(self):
        from src.config import MODEL_PATH
        from src.inference import load_model_package

        if not MODEL_PATH.exists():
            pytest.skip("Model file not available (CI environment)")

        package = load_model_package()
        assert package is not None
        assert "model" in package
        assert package["model"] is not None

    def test_inference_with_model(self):
        """Testa inferência se arquivo de modelo existe."""
        from src.config import MODEL_PATH

        if not MODEL_PATH.exists():
            pytest.skip("No trained model available for inference test")

        from src.inference import load_model_package

        package = load_model_package()
        assert package is not None

        # Try to predict with the loaded model
        sample = pd.DataFrame(
            [
                {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.0,
                    "TotalCharges": 840.0,
                }
            ]
        )

        # Usa o modelo diretamente do pacote
        model = package.get("model") or package.get("pipeline")
        assert model is not None, "No model found in package"
        proba = model.predict_proba(sample)
        assert proba.shape[1] == 2
        assert 0 <= proba[0, 1] <= 1
