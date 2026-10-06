from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_predict(monkeypatch):

    def fake_predict(text):
        return [
            [
                {
                    "label": "food",
                    "score": 0.95,
                }
            ]
        ]

    def fake_save_prediction(
        db,
        text,
        predicted_label,
        confidence,
        model_version,
    ):
        return SimpleNamespace(
            id=100,
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
            model_version=model_version,
        )

    monkeypatch.setattr(
        main.model_manager,
        "predict",
        fake_predict,
    )

    monkeypatch.setattr(
        main,
        "save_prediction",
        fake_save_prediction,
    )

    response = client.post(
        "/predict",
        json={
            "text": "I had a delicious pizza for dinner."
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "prediction_id": 100,
        "label": "food",
        "score": 0.95,
        "model_version": main.model_manager.model_version,
    }