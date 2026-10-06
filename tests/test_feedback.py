from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_feedback_like(monkeypatch):

    fake_prediction = SimpleNamespace(
        id=100,
        text="I had a delicious pizza for dinner.",
        predicted_label="food",
        confidence=0.95,
        model_version="v009",
    )

    fake_feedback = SimpleNamespace(
        id=1,
        prediction_id=100,
        feedback="like",
        corrected_label="food",
    )

    def fake_get_prediction(db, prediction_id):
        return fake_prediction

    def fake_save_feedback(db, prediction, feedback):
        return fake_feedback

    monkeypatch.setattr(
        main,
        "get_prediction",
        fake_get_prediction,
    )

    monkeypatch.setattr(
        main,
        "save_feedback",
        fake_save_feedback,
    )

    response = client.post(
        "/feedback",
        json={
            "prediction_id": 100,
            "feedback": "like",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "feedback_id": 1,
        "prediction_id": 100,
        "feedback": "like",
        "corrected_label": "food",
        "retraining_triggered": False,
        "retraining_started": False,
    }
    
def test_feedback_dislike_triggers_retraining(monkeypatch):

    fake_prediction = SimpleNamespace(
        id=101,
        text="The chips were too small.",
        predicted_label="food",
        confidence=0.70,
        model_version="v009",
    )

    fake_feedback = SimpleNamespace(
        id=2,
        prediction_id=101,
        feedback="dislike",
        corrected_label="not_food",
    )

    def fake_get_prediction(db, prediction_id):
        return fake_prediction

    def fake_save_feedback(db, prediction, feedback):
        return fake_feedback

    def fake_check_retraining_trigger(db):
        return True

    def fake_start_retraining(model_manager):
        return True

    monkeypatch.setattr(
        main,
        "get_prediction",
        fake_get_prediction,
    )

    monkeypatch.setattr(
        main,
        "save_feedback",
        fake_save_feedback,
    )

    monkeypatch.setattr(
        main,
        "check_retraining_trigger",
        fake_check_retraining_trigger,
    )

    monkeypatch.setattr(
        main,
        "start_retraining",
        fake_start_retraining,
    )

    response = client.post(
        "/feedback",
        json={
            "prediction_id": 101,
            "feedback": "dislike",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "feedback_id": 2,
        "prediction_id": 101,
        "feedback": "dislike",
        "corrected_label": "not_food",
        "retraining_triggered": True,
        "retraining_started": True,
    }

def test_feedback_prediction_not_found(monkeypatch):

    def fake_get_prediction(db, prediction_id):
        return None

    monkeypatch.setattr(
        main,
        "get_prediction",
        fake_get_prediction,
    )

    response = client.post(
        "/feedback",
        json={
            "prediction_id": 999999,
            "feedback": "like",
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Prediction Not Found!"
    }