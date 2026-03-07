"""
Testes da API
=============

Suite de testes abrangente para a API FastAPI de predição de churn.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Cria cliente de teste."""
    return TestClient(app)


@pytest.fixture
def sample_customer_data():
    """Dados de cliente de exemplo para testes."""
    return {
        "customer_id": "TEST-001",
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
        },
    }


class TestRootEndpoint:
    """Testes para endpoint raiz."""

    def test_root_returns_api_info(self, client):
        """Testa se endpoint raiz retorna informações da API."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_root_contains_documentation_links(self, client):
        """Testa se raiz contém links de documentação."""
        response = client.get("/")
        data = response.json()

        assert "documentation" in data
        assert "swagger" in data["documentation"]
        assert "redoc" in data["documentation"]


class TestHealthEndpoints:
    """Testes para endpoints de health check."""

    def test_health_endpoint(self, client):
        """Testa endpoint principal de health."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_liveness_probe(self, client):
        """Testa endpoint de liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, client):
        """Testa endpoint de readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        if response.status_code == 200:
            assert data["status"] == "ready"
        else:
            assert data["status"] == "not ready"


class TestPredictionEndpoints:
    """Testes para endpoints de predição."""

    def test_single_prediction_valid_input(self, client, sample_customer_data):
        """Testa predição individual com entrada válida."""
        response = client.post("/predict/", json=sample_customer_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "churn_prediction" in data["prediction"]
        assert "churn_probability" in data["prediction"]
        assert "churn_risk" in data["prediction"]
        assert "confidence" in data["prediction"]

    def test_single_prediction_without_customer_id(self, client, sample_customer_data):
        """Testa predição sem customer_id opcional."""
        data = sample_customer_data.copy()
        del data["customer_id"]

        response = client.post("/predict/", json=data)
        assert response.status_code == 200

    def test_single_prediction_invalid_tenure(self, client, sample_customer_data):
        """Testa predição com valor de tenure inválido."""
        data = sample_customer_data.copy()
        data["features"]["tenure"] = -1  # Inválido

        response = client.post("/predict/", json=data)
        assert response.status_code == 422  # Validation error

    def test_single_prediction_invalid_gender(self, client, sample_customer_data):
        """Testa predição com valor de gender inválido."""
        data = sample_customer_data.copy()
        data["features"]["gender"] = "Invalid"

        response = client.post("/predict/", json=data)
        assert response.status_code == 422

    def test_batch_prediction_valid_input(self, client, sample_customer_data):
        """Testa predição em lote com entrada válida."""
        batch_data = {
            "customers": [sample_customer_data, {**sample_customer_data, "customer_id": "TEST-002"}]
        }

        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2
        assert "processing_time_ms" in data

    def test_batch_prediction_empty_list(self, client):
        """Testa predição em lote com lista vazia."""
        batch_data = {"customers": []}

        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Testes para endpoints de informação do modelo."""

    def test_model_info_endpoint(self, client):
        """Testa endpoint de info do modelo."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "features_count" in data

    def test_feature_importance_endpoint(self, client):
        """Testa endpoint de importância de features."""
        response = client.get("/interpret/feature-importance")
        # Retorna 200 se modelo carregado com importância, 404 se não disponível
        assert response.status_code in (200, 404)

        if response.status_code == 200:
            data = response.json()
            assert "feature_importances" in data


class TestDocumentation:
    """Testes para documentação da API."""

    def test_openapi_json_available(self, client):
        """Testa se schema JSON OpenAPI está disponível."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_swagger_docs_available(self, client):
        """Testa se Swagger UI está disponível."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Testa se ReDoc está disponível."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestModelsManagementEndpoints:
    """Testes para endpoints de gerenciamento de modelos (GET /models/ e POST /models/switch)."""

    def test_list_models_returns_payload_structure(self, client):
        """Testa se GET /models/ retorna estrutura esperada com modelos disponíveis."""
        response = client.get("/models/")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "active_model" in data
        assert "total" in data
        assert isinstance(data["models"], list)
        assert data["total"] == len(data["models"])
        assert data["total"] > 0

        # Verifica campos de cada modelo
        model = data["models"][0]
        assert "name" in model
        assert "filename" in model
        assert "size_mb" in model
        assert "is_default" in model
        assert model["filename"].endswith(".joblib")

    def test_list_models_active_model_is_stem(self, client):
        """Testa se active_model é o stem do arquivo (sem extensão)."""
        response = client.get("/models/")
        data = response.json()

        active = data["active_model"]
        all_names = [m["name"] for m in data["models"]]
        assert active in all_names, f"active_model '{active}' não está na lista de nomes"

    def test_switch_model_not_found(self, client):
        """Testa 404 ao tentar trocar para modelo inexistente."""
        response = client.post("/models/switch", json={"model_name": "nonexistent_model_xyz"})
        assert response.status_code == 404

        data = response.json()
        assert data["detail"]["error_code"] == "MODEL_NOT_FOUND"

    def test_switch_model_success(self, client):
        """Testa troca bem-sucedida para um modelo disponível."""
        # Descobre um modelo válido via GET
        list_resp = client.get("/models/")
        models = list_resp.json()["models"]
        target = models[0]["name"]

        response = client.post("/models/switch", json={"model_name": target})
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["active_model"] == target
        assert "estimator_name" in data
        assert "message" in data

    def test_switch_model_reflected_in_list(self, client):
        """Testa se a troca é refletida no GET /models/ subsequente."""
        list_resp = client.get("/models/")
        models = list_resp.json()["models"]
        if len(models) < 2:
            pytest.skip("Apenas um modelo disponível, não é possível testar troca")

        target = models[-1]["name"]
        switch_resp = client.post("/models/switch", json={"model_name": target})
        assert switch_resp.status_code == 200

        verify_resp = client.get("/models/")
        assert verify_resp.json()["active_model"] == target

    def test_switch_model_missing_body(self, client):
        """Testa 422 quando body está ausente no POST /models/switch."""
        response = client.post("/models/switch")
        assert response.status_code == 422


class TestEdgeCases:
    """Testes para casos extremos e tratamento de erros."""

    def test_prediction_with_boundary_values(self, client, sample_customer_data):
        """Testa predição com valores limite."""
        data = sample_customer_data.copy()
        data["features"]["tenure"] = 0
        data["features"]["MonthlyCharges"] = 18.0
        data["features"]["TotalCharges"] = 0

        response = client.post("/predict/", json=data)
        assert response.status_code == 200

    def test_prediction_with_max_tenure(self, client, sample_customer_data):
        """Testa predição com tenure máximo."""
        data = sample_customer_data.copy()
        data["features"]["tenure"] = 72

        response = client.post("/predict/", json=data)
        assert response.status_code == 200

    def test_nonexistent_endpoint(self, client):
        """Testa 404 para endpoint inexistente."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
