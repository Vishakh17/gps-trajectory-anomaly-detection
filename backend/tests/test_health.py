def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data


def test_data_summary_endpoint(client):
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["has_raw_upload"], bool)
    assert isinstance(data["has_processed_gps"], bool)
    assert isinstance(data["has_features"], bool)
    assert isinstance(data["has_trained_model"], bool)
    if data["vehicles"] is not None:
        assert isinstance(data["vehicles"], list)
        assert all(isinstance(v, str) for v in data["vehicles"])
    if data["raw_rows"] is not None:
        assert isinstance(data["raw_rows"], int)
    if data["processed_rows"] is not None:
        assert isinstance(data["processed_rows"], int)
    if data["feature_rows"] is not None:
        assert isinstance(data["feature_rows"], int)
